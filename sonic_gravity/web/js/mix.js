/**
 * mix.js — le monde sur lequel agit la gravité : un vrai mix multipiste.
 *
 * Une tranche = un stem en boucle → analyseur PRÉ-fader → gain → bus master.
 *
 * ⚠️ L'analyseur est PRÉ-fader, et ce n'est pas un détail : le champ a besoin
 * du spectre de la tranche INDÉPENDAMMENT de son gain. C'est ce qui permet
 * d'écrire M[b] = Σ (g·a)²·P[i][b] et d'en tirer une dérivée exacte. Un
 * analyseur post-fader donnerait une mesure qui bouge avec l'inconnue qu'on
 * cherche, et le gradient serait faux.
 */

import { bandGather, bandsFromGather } from "./field.js";

// Constante de temps du lissage des spectres. Le champ doit décrire l'équilibre
// du morceau, pas chaque coup de caisse claire : sans ce lissage les faders
// frémissent à chaque transitoire et la gravité paraît nerveuse.
const SMOOTH_TAU = 0.25;

// Sous ce niveau relatif au plus fort passage entendu, il n'y a plus de
// musique : bouclage, break, ou simplement pas encore démarré. Le champ est
// alors GELÉ sur sa dernière valeur valide.
//
// ⚠️ Sans cette porte, un spectre vide est comparé au gabarit comme s'il
// s'agissait d'un mix : toutes les bandes tombent au plancher, le z-score
// explose et les faders reçoivent une secousse. Mesuré le 2026-08-04 sur les
// fondus de boucle : V passait de 2,3 à 12,3 toutes les 24 secondes.
const SILENCE_FLOOR = 1e-3; // −30 dB sous le pic
// …mais un seuil RELATIF ne suffit pas : au démarrage, le pic est initialisé
// sur le silence lui-même, donc le bruit de plancher des analyseurs passe pour
// de la musique. Pendant les ~120 ms qui précèdent le premier échantillon
// audio, le potentiel était calculé sur ce plancher et sortait à 88,9. Il faut
// donc AUSSI un plancher absolu, dérivé du `minDecibels` de l'analyseur.
const FLOOR_MARGIN = 8; // combien de fois le plancher théorique avant d'y croire

export class MixWorld {
  constructor(spec) {
    this.spec = spec;
    this.faderExponent = spec.faderExponent ?? 1;
    this.ctx = null;
    this.tracks = [];
    this.started = false;
    this.gather = null;
    this.P = []; // puissances par bande, lissées — l'entrée du champ
    this.silent = true; // vrai tant qu'on n'entend rien d'exploitable
    this._raw = [];
    this._scratch = null;
    this._lastSample = 0;
    this._peakEnergy = 0;
    this._absFloor = 0;
  }

  get size() {
    return this.tracks.length;
  }

  /** Charge le manifeste et décode tous les stems. */
  async load(manifestUrl = "audio/manifest.json") {
    const manifest = await (await fetch(manifestUrl)).json();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx({ latencyHint: "interactive" });

    this.master = this.ctx.createGain();
    this.master.gain.value = 0.9;
    // Protection d'écrêtage seulement : le mix doit pouvoir sonner mal (c'est
    // le sujet), mais jamais saturer les enceintes de celui qui essaie.
    this.limiter = this.ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -2;
    this.limiter.knee.value = 0;
    this.limiter.ratio.value = 20;
    this.limiter.attack.value = 0.002;
    this.limiter.release.value = 0.12;
    this.masterAnalyser = this.ctx.createAnalyser();
    this.masterAnalyser.fftSize = 2048;
    this.masterAnalyser.smoothingTimeConstant = 0.7;
    this.master.connect(this.limiter);
    this.limiter.connect(this.masterAnalyser);
    this.masterAnalyser.connect(this.ctx.destination);

    const fftSize = this.spec.analysis?.fftSize ?? 8192;
    const buffers = await Promise.all(
      manifest.tracks.map(async (t) => {
        const bytes = await (await fetch(t.file)).arrayBuffer();
        return this.ctx.decodeAudioData(bytes);
      })
    );

    this.tracks = manifest.tracks.map((t, i) => {
      const analyser = this.ctx.createAnalyser();
      analyser.fftSize = fftSize;
      analyser.smoothingTimeConstant = 0.5;
      const gain = this.ctx.createGain();
      gain.gain.value = 0;
      analyser.connect(gain);
      gain.connect(this.master);
      return {
        id: t.id,
        label: t.id,
        buffer: buffers[i],
        analyser,
        gain,
        source: null,
        bins: new Float32Array(analyser.frequencyBinCount),
        muted: false,
        soloed: false,
        level: 0,
      };
    });

    const nBands = this.spec.bands.n;
    const edges = new Float64Array(nBands + 1);
    const r = Math.log(this.spec.bands.fHi / this.spec.bands.fLo) / nBands;
    for (let i = 0; i <= nBands; i++) edges[i] = this.spec.bands.fLo * Math.exp(r * i);
    this.gather = bandGather(fftSize / 2, this.ctx.sampleRate, fftSize, edges);

    // Énergie qu'un analyseur rend quand il n'entend RIEN : tous ses bins au
    // plancher. En dessous de quelques fois cette valeur, il n'y a pas de son.
    const a0 = this.tracks[0].analyser;
    this._absFloor = a0.frequencyBinCount * Math.pow(10, a0.minDecibels / 10) * FLOOR_MARGIN;

    this.P = this.tracks.map(() => new Float64Array(nBands).fill(1e-9));
    this._raw = this.tracks.map(() => new Float64Array(nBands));
    this._scratch = new Float64Array(fftSize / 2);
    this.manifest = manifest;
    return manifest;
  }

  /**
   * Démarre toutes les pistes au même instant.
   * Un décalage même faible déphaserait le mix et fausserait tout ce que le
   * champ mesure.
   */
  async start() {
    if (this.started) return;
    if (this.ctx.state === "suspended") await this.ctx.resume();
    const at = this.ctx.currentTime + 0.12;
    for (const t of this.tracks) {
      const src = this.ctx.createBufferSource();
      src.buffer = t.buffer;
      src.loop = true;
      src.connect(t.analyser);
      src.start(at);
      t.source = src;
    }
    this.started = true;
    this._lastSample = 0;
  }

  stop() {
    for (const t of this.tracks) {
      try {
        t.source?.stop();
      } catch {
        /* déjà arrêtée */
      }
      t.source = null;
    }
    this.started = false;
  }

  /** Applique les positions de fader (0..1) au moteur audio. */
  applyGains(values, audible) {
    const now = this.ctx.currentTime;
    for (let i = 0; i < this.tracks.length; i++) {
      const t = this.tracks[i];
      // Loi de fader lue dans spec.json, JAMAIS écrite en dur : c'est la même
      // constante que celle dont le champ tire son gradient. Le jour où elles
      // ont divergé (moteur en g², champ en g), le champ décrivait un mix que
      // personne n'entendait, sans qu'aucun test ne bronche.
      const g = audible[i] ? Math.pow(values[i], this.faderExponent) : 0;
      // Rampe courte : sans elle, un fader poussé à 60 Hz produit un escalier
      // de gains audible en zipper noise.
      t.gain.gain.setTargetAtTime(g, now, 0.012);
    }
  }

  /** Relit les spectres et met à jour `P` (lissé). Retourne false si on gèle. */
  sample(now) {
    if (!this.started) return false;
    const dt = this._lastSample ? (now - this._lastSample) / 1000 : 0;
    this._lastSample = now;
    const alpha = dt > 0 ? 1 - Math.exp(-dt / SMOOTH_TAU) : 1;

    // Première passe : y a-t-il de la musique ? On mesure PRÉ-fader, donc la
    // porte ne dépend pas de la position des faders — baisser tout le mix ne
    // doit pas être confondu avec un silence du morceau.
    let energy = 0;
    for (const t of this.tracks) {
      t.analyser.getFloatFrequencyData(t.bins);
      let e = 0;
      for (let k = 0; k < t.bins.length; k++) {
        const db = t.bins[k];
        if (db > -180) e += Math.pow(10, db / 10);
      }
      t.energy = e;
      energy += e;
    }
    this._peakEnergy = Math.max(this._peakEnergy * 0.9995, energy);
    this.silent =
      energy < this._absFloor || energy < this._peakEnergy * SILENCE_FLOOR;
    if (this.silent) {
      for (const t of this.tracks) t.level *= 0.7; // les vu-mètres retombent
      return false; // P reste sur sa dernière valeur : le champ ne bouge pas
    }

    for (let i = 0; i < this.tracks.length; i++) {
      const t = this.tracks[i];
      // dB → puissance linéaire. Le plancher de l'analyseur (−100 dB par
      // défaut) devient 1e-10 : négligeable, mais jamais zéro, donc le log du
      // potentiel reste défini.
      const s = this._scratch;
      for (let k = 0; k < s.length; k++) {
        const db = t.bins[k];
        s[k] = db <= -180 ? 0 : Math.pow(10, db / 10);
      }
      const raw = bandsFromGather(s, this.gather, this._raw[i]);
      const P = this.P[i];
      for (let b = 0; b < P.length; b++) {
        P[b] += (raw[b] - P[b]) * alpha;
        if (P[b] < 1e-12) P[b] = 1e-12;
      }
      t.level = Math.sqrt(t.energy);
    }
    return true;
  }

  /** Audibilité de chaque tranche : mute, et solo posé ailleurs. */
  audibility(out) {
    const anySolo = this.tracks.some((t) => t.soloed);
    const dst = out || new Float64Array(this.tracks.length);
    for (let i = 0; i < this.tracks.length; i++) {
      const t = this.tracks[i];
      dst[i] = t.muted || (anySolo && !t.soloed) ? 0 : 1;
    }
    return dst;
  }

  /** Niveau du bus master, en dBFS approché. */
  masterLevel() {
    const n = this.masterAnalyser.frequencyBinCount;
    if (!this._masterBins || this._masterBins.length !== n) this._masterBins = new Float32Array(n);
    this.masterAnalyser.getFloatFrequencyData(this._masterBins);
    let p = 0;
    for (let k = 0; k < n; k++) {
      const db = this._masterBins[k];
      if (db > -180) p += Math.pow(10, db / 10);
    }
    return 10 * Math.log10(Math.max(p, 1e-12));
  }
}
