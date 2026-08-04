/**
 * field.js — miroir JS de `sonic_gravity/field.py`.
 *
 * ⚠️ Ces deux fichiers doivent rester d'accord au chiffre près : le modèle est
 * entraîné en Python sur le potentiel Python, et appliqué ici. Une divergence
 * ne produirait aucune erreur, juste une gravité qui ne correspond pas à ce qui
 * a été appris. `tests/test_sonic_gravity_parity.py` compare les deux sur des
 * vecteurs figés — le même patron que `console_fx.py` / `mixEngine.ts`.
 *
 * Les constantes ne sont PAS recopiées : elles viennent de `spec.json`, lu par
 * les deux implémentations.
 */

// Une VALEUR en décibels vaut 10·log10(x) ; sa DÉRIVÉE vaut DB_SCALE / x.
// Les confondre gonfle la force sans rien casser de visible (déjà payé).
const DB_VALUE = 10.0;
const DB_SCALE = 10.0 / Math.log(10.0);
const EPS = 1e-12;

/**
 * Perte de Huber : carré près de la cible, linéaire au-delà.
 *
 * Le carré nu laissait un break dominer tout le champ — pendant un passage où
 * seul le piano joue, les bandes graves sortent à −10 σ et le potentiel monte
 * à 88. Aucun fader ne peut créer du grave que la basse ne joue pas.
 */
function huber(z, delta) {
  const a = Math.abs(z);
  return a <= delta ? z * z : 2 * delta * a - delta * delta;
}

/** Dérivée de `huber` : bornée à 2·delta, donc jamais de force qui panique. */
function dhuber(z, delta) {
  return Math.abs(z) <= delta ? 2 * z : 2 * delta * Math.sign(z);
}

export function bandEdges(nBands, fLo, fHi) {
  const out = new Float64Array(nBands + 1);
  const r = Math.log(fHi / fLo) / nBands;
  for (let i = 0; i <= nBands; i++) out[i] = fLo * Math.exp(r * i);
  return out;
}

/**
 * Pour chaque bande, la liste des bins qui la composent — miroir de
 * `field.band_gather`.
 *
 * ⚠️ Aucune bande ne peut ressortir vide. Une bande plus étroite que la
 * résolution de la FFT ne reçoit aucun bin, tomberait au plancher (−174 dB) et
 * deviendrait un puits qui écrase tout le potentiel. Elle adopte donc le bin
 * dont le centre en est le plus proche, quitte à le partager avec sa voisine.
 */
export function bandGather(nBins, sampleRate, fftSize, edges) {
  const df = sampleRate / fftSize;
  const nBands = edges.length - 1;
  const out = [];
  for (let b = 0; b < nBands; b++) {
    const lo = edges[b];
    const hi = edges[b + 1];
    const idx = [];
    for (let k = 0; k < nBins; k++) {
      const f = k * df;
      if (f >= lo && f < hi) idx.push(k);
    }
    if (idx.length === 0) {
      const center = Math.sqrt(lo * hi);
      let best = 0;
      let bestD = Infinity;
      for (let k = 0; k < nBins; k++) {
        const d = Math.abs(Math.log(Math.max(k * df, 1e-6) / center));
        if (d < bestD) {
          bestD = d;
          best = k;
        }
      }
      idx.push(best);
    }
    out.push(Int32Array.from(idx));
  }
  return out;
}

/** Somme les puissances de bin par bande, selon la table de `bandGather`. */
export function bandsFromGather(binPower, gather, out) {
  const dst = out || new Float64Array(gather.length);
  for (let b = 0; b < gather.length; b++) {
    const idx = gather[b];
    let s = 0;
    for (let k = 0; k < idx.length; k++) s += binPower[idx[k]];
    dst[b] = s;
  }
  return dst;
}

/**
 * Une scène = les spectres PRÉ-fader de toutes les tranches à un instant.
 *
 * Tout ce qui ne dépend pas des gains (formes spectrales, recouvrements par
 * paire, énergies totales) est calculé UNE fois ici. C'est ce qui permet
 * d'évaluer le potentiel des centaines de fois par image pour tracer le profil
 * du champ le long de chaque fader, sans que la boucle coûte quoi que ce soit.
 */
export class Scene {
  /**
   * @param {object} spec  contenu de spec.json
   * @param {Float64Array[]} P  puissances par bande, une entrée par tranche
   */
  constructor(spec, P) {
    this.spec = spec;
    this.P = P;
    this.n = P.length;
    this.nBands = spec.bands.n;

    this.targetDb = Float64Array.from(spec.target.db);
    this.sigmaDb = Float64Array.from(spec.target.sigma);
    this.weight = Float64Array.from(spec.target.weight);
    this.levelDb = spec.target.levelDb;
    this.levelSigma = spec.target.levelSigma;
    this.wTilt = spec.weights.tilt;
    this.wMask = spec.weights.mask;
    this.wLevel = spec.weights.level;
    this.wAnchor = spec.weights.anchor ?? 0;
    this.anchorSigma = spec.weights.anchorSigma ?? 0.25;
    this.tiltHuber = spec.weights.tiltHuber ?? 3.0;

    // Énergie totale et forme spectrale de chaque tranche (sans gain).
    this.total = new Float64Array(this.n);
    this.shape = [];
    for (let i = 0; i < this.n; i++) {
      let s = 0;
      for (let b = 0; b < this.nBands; b++) s += P[i][b];
      this.total[i] = s;
      const q = new Float64Array(this.nBands);
      const inv = 1 / Math.max(s, EPS);
      for (let b = 0; b < this.nBands; b++) q[b] = P[i][b] * inv;
      this.shape.push(q);
    }

    // Poids par bande, pondérés par la matière RÉELLEMENT disponible (miroir de
    // `field._band_weights`). Le champ ne juge que ce sur quoi les faders ont
    // prise : pendant un break, aucune position ne peut ramener du grave que
    // personne ne joue. Mesuré à plein gain, donc constant vis-à-vis de la
    // dérivation — et calculé UNE fois par scène, jamais dans la boucle.
    {
      let availSum = 0;
      const avail = new Float64Array(this.nBands);
      for (let b = 0; b < this.nBands; b++) {
        let s2 = EPS;
        for (let i = 0; i < this.n; i++) s2 += P[i][b];
        avail[b] = s2;
        availSum += s2;
      }
      this.wEff = new Float64Array(this.nBands);
      for (let b = 0; b < this.nBands; b++) {
        const share = avail[b] / availSum;
        const targetLin = Math.pow(10, this.targetDb[b] / DB_VALUE);
        this.wEff[b] = this.weight[b] * Math.min(1, share / targetLin);
      }
    }

    // Recouvrement par paire : intersection d'histogrammes des formes.
    // Indépendant des gains, donc calculé une seule fois.
    this.O = [];
    for (let i = 0; i < this.n; i++) this.O.push(new Float64Array(this.n));
    for (let i = 0; i < this.n; i++) {
      for (let j = i + 1; j < this.n; j++) {
        let o = 0;
        const qi = this.shape[i];
        const qj = this.shape[j];
        for (let b = 0; b < this.nBands; b++) o += Math.min(qi[b], qj[b]);
        this.O[i][j] = o;
        this.O[j][i] = o;
      }
    }

    // Tampons réutilisés — la boucle tourne à 60 Hz, on n'y alloue rien.
    this._M = new Float64Array(this.nBands);
    this._u = new Float64Array(this.n);
    this._e = new Float64Array(this.n);
    this._root = new Float64Array(this.n);
    this._c = new Float64Array(this.nBands);
    this._dVdM = new Float64Array(this.nBands);
    this._grad = new Float64Array(this.n);
    this._dmask = new Float64Array(this.n);
  }

  /** Mix par bande : M[b] = Σ (g·a)² P[i][b]. */
  _mix(g, a) {
    const { _M: M, _u: u, P, n, nBands } = this;
    M.fill(EPS);
    for (let i = 0; i < n; i++) {
      const ga = g[i] * (a ? a[i] : 1);
      u[i] = ga * ga;
      if (u[i] === 0) continue;
      const Pi = P[i];
      const ui = u[i];
      for (let b = 0; b < nBands; b++) M[b] += ui * Pi[b];
    }
    return M;
  }

  /** Terme de masquage + sa dérivée par rapport à l'énergie de chaque tranche. */
  _mask(u, wantGrad) {
    const { _e: e, _root: root, _dmask: dmask, O, total, n } = this;
    let D = EPS;
    for (let i = 0; i < n; i++) {
      e[i] = u[i] * total[i];
      root[i] = Math.sqrt(Math.max(e[i], 0) + EPS);
      D += e[i];
    }
    let N = 0;
    for (let i = 0; i < n; i++) {
      const Oi = O[i];
      for (let j = i + 1; j < n; j++) N += Oi[j] * root[i] * root[j];
    }
    const v = N / D;
    if (wantGrad) {
      for (let i = 0; i < n; i++) {
        const Oi = O[i];
        let s = 0;
        for (let j = 0; j < n; j++) s += Oi[j] * root[j];
        const dN = 0.5 * (s / root[i]);
        dmask[i] = (dN * D - N) / (D * D);
      }
    }
    return v;
  }

  /**
   * V(g) et le détail de ses termes.
   * @param {Float64Array} anchor positions posées par la main (rappel), ou null
   */
  potential(g, a, anchor) {
    const M = this._mix(g, a);
    const { nBands, targetDb, sigmaDb } = this;
    let S = 0;
    for (let b = 0; b < nBands; b++) S += M[b];

    let tilt = 0;
    for (let b = 0; b < nBands; b++) {
      const sdb = DB_VALUE * Math.log10(M[b] / S);
      const d = (sdb - targetDb[b]) / sigmaDb[b];
      tilt += this.wEff[b] * huber(d, this.tiltHuber);
    }
    tilt /= nBands;

    const mask = this._mask(this._u, false);

    const level = DB_VALUE * Math.log10(S);
    const dl = (level - this.levelDb) / this.levelSigma;
    const lvl = dl * dl;

    // Rappel vers l'intention : SOMME sur les tranches, chacune paie son écart
    // en entier (miroir de field.py — moyenner le noyait face au terme spectral).
    let anc = 0;
    if (anchor && this.wAnchor) {
      for (let i = 0; i < this.n; i++) {
        const d = (g[i] - anchor[i]) / this.anchorSigma;
        anc += d * d;
      }
    }

    const total = this.wTilt * tilt + this.wMask * mask + this.wLevel * lvl + this.wAnchor * anc;
    return { total, tilt, mask, level: lvl, anchor: anc, levelDb: level };
  }

  /** Juste la valeur — chemin chaud du tracé de profil. */
  value(g, a, anchor) {
    return this.potential(g, a, anchor).total;
  }

  /** dV/dg, dérivé à la main (démonstration et vérification dans field.py). */
  gradient(g, a, anchor) {
    const M = this._mix(g, a);
    const { nBands, n, targetDb, sigmaDb, _c: c, _dVdM: dVdM, _grad: grad, P, total } = this;

    let S = 0;
    for (let b = 0; b < nBands; b++) S += M[b];

    let cSum = 0;
    for (let b = 0; b < nBands; b++) {
      const sdb = DB_VALUE * Math.log10(M[b] / S);
      const d = (sdb - targetDb[b]) / sigmaDb[b];
      c[b] = ((1 / nBands) * this.wEff[b] * dhuber(d, this.tiltHuber)) / sigmaDb[b];
      cSum += c[b];
    }

    const level = DB_VALUE * Math.log10(S);
    const dLevel = ((2 * (level - this.levelDb)) / (this.levelSigma * this.levelSigma)) * (DB_SCALE / S);

    for (let b = 0; b < nBands; b++) {
      dVdM[b] = this.wTilt * DB_SCALE * (c[b] / M[b] - cSum / S) + this.wLevel * dLevel;
    }

    this._mask(this._u, true);

    for (let i = 0; i < n; i++) {
      const ai = a ? a[i] : 1;
      const Pi = P[i];
      let dV_du = 0;
      for (let b = 0; b < nBands; b++) dV_du += Pi[b] * dVdM[b];
      dV_du += this.wMask * this._dmask[i] * total[i];
      grad[i] = dV_du * (2 * g[i] * ai * ai);
      if (anchor && this.wAnchor) {
        grad[i] += (this.wAnchor * 2 * (g[i] - anchor[i])) / (this.anchorSigma * this.anchorSigma);
      }
    }
    return grad;
  }

  /** La gravité ressentie : −dV/dg. */
  force(g, a, anchor) {
    const grad = this.gradient(g, a, anchor);
    const out = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) out[i] = -grad[i];
    return out;
  }

  /**
   * Profil du potentiel le long d'un fader, les autres tenus fixes.
   * C'est le « relief » dessiné derrière la course : la vallée est visible
   * avant même de lâcher le fader.
   */
  profile(ch, g, a, anchor, steps = 40) {
    const out = new Float64Array(steps);
    const probe = Float64Array.from(g);
    for (let k = 0; k < steps; k++) {
      probe[ch] = k / (steps - 1);
      out[k] = this.value(probe, a, anchor);
    }
    probe[ch] = g[ch];
    return out;
  }
}
