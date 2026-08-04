/**
 * learned.js — the world model at runtime, forward AND backward, by hand.
 *
 * The console needs `∂V/∂g` sixty times a second. With a learned model in the
 * chain, that derivative has to travel back through the network — so this file
 * implements the backward pass explicitly rather than pulling in an autodiff
 * runtime. Three reasons, in order of importance:
 *
 *   1. The gradient is the product, not a by-product. Shipping an inference-only
 *      runtime (ONNX, TF.js) would give predictions and no derivative, which is
 *      the one thing this system cannot do without.
 *   2. It is small. Five dense layers and a mean pool — the backward pass below
 *      is about a hundred lines, and it is checked against PyTorch by
 *      `tests/test_learned_parity.py` to 1e-6.
 *   3. No network fetch, no WASM, no cold start. The whole model is one JSON.
 *
 * This constraint shaped the architecture: mean pooling was chosen over max
 * precisely because its backward pass is a division, not a gather.
 */

const DB = 10.0;
const DB_SCALE = 10.0 / Math.log(10.0);
const EPS = 1e-12;

const silu = (x) => x / (1 + Math.exp(-x));
/** d/dx of x·σ(x) = σ(x)·(1 + x·(1−σ(x))). */
function dsilu(x) {
  const s = 1 / (1 + Math.exp(-x));
  return s * (1 + x * (1 - s));
}

/** y = Wᵀx + b, with W stored row-major as [in][out] (the export layout). */
function dense(W, b, x, out) {
  const nOut = b.length;
  const y = out || new Float64Array(nOut);
  for (let j = 0; j < nOut; j++) y[j] = b[j];
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = W[i];
    for (let j = 0; j < nOut; j++) y[j] += xi * row[j];
  }
  return y;
}

/** dx = W·dy — the transpose of the forward matmul. */
function denseBackward(W, dy, nIn) {
  const dx = new Float64Array(nIn);
  for (let i = 0; i < nIn; i++) {
    const row = W[i];
    let s = 0;
    for (let j = 0; j < dy.length; j++) s += row[j] * dy[j];
    dx[i] = s;
  }
  return dx;
}

export class LearnedMix {
  constructor(weights) {
    this.w = weights;
    this.nBands = weights.nBands;
    this.twoP = 2 * weights.pLaw;
    this.report = weights.report || null;
  }

  static async load(url = "model.json") {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`no model at ${url}`);
    return new LearnedMix(await res.json());
  }

  /**
   * Predict the mix spectrum SHAPE, in dB, normalised so the bands sum to one.
   *
   * @param {Float64Array[]} P  per-channel band powers, pre-fader
   * @param {Float64Array} g    fader positions
   * @param {Float64Array} a    audibility (0/1)
   * @returns {{shape: Float64Array, cache: object}}
   */
  forward(P, g, a) {
    const { nBands, w } = this;
    const n = P.length;

    // ── baseline, and the quantities its own derivative will need ──────────
    const M = new Float64Array(nBands).fill(EPS);
    const u = new Float64Array(n);
    const totals = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const ga = g[i] * (a ? a[i] : 1);
      u[i] = ga === 0 ? 0 : Math.pow(ga, this.twoP);
      const Pi = P[i];
      let tot = 0;
      for (let b = 0; b < nBands; b++) {
        tot += Pi[b];
        M[b] += u[i] * Pi[b];
      }
      totals[i] = tot;
    }
    let S = 0;
    for (let b = 0; b < nBands; b++) S += M[b];
    const baseN = new Float64Array(nBands);
    for (let b = 0; b < nBands; b++) baseN[b] = DB * Math.log10(M[b] / S);

    // ── per-channel encoder ────────────────────────────────────────────────
    const feats = [];
    const z1 = [];
    const h1 = [];
    const z2 = [];
    const h2 = [];
    const pooled = new Float64Array(w.width);
    let present = 0;

    for (let i = 0; i < n; i++) {
      const f = new Float64Array(nBands + 2);
      const inv = 1 / Math.max(totals[i], EPS);
      for (let b = 0; b < nBands; b++) f[b] = Math.log10(P[i][b] * inv + EPS);
      f[nBands] = g[i];
      f[nBands + 1] = Math.log10(totals[i] + EPS) / 10;
      feats.push(f);

      const a1 = dense(w.enc[0].W, w.enc[0].b, f);
      const o1 = new Float64Array(a1.length);
      for (let k = 0; k < a1.length; k++) o1[k] = silu(a1[k]);
      const a2 = dense(w.enc[1].W, w.enc[1].b, o1);
      const o2 = new Float64Array(a2.length);
      for (let k = 0; k < a2.length; k++) o2[k] = silu(a2[k]);

      z1.push(a1);
      h1.push(o1);
      z2.push(a2);
      h2.push(o2);

      // Masked channels contribute nothing to the pool — the Python side
      // averages over PRESENT channels only, and a mismatch here would be
      // invisible except as a model that quietly disagrees with its training.
      const on = a ? a[i] : 1;
      if (on) {
        present++;
        for (let k = 0; k < o2.length; k++) pooled[k] += o2[k];
      }
    }
    const denom = Math.max(present, 1);
    for (let k = 0; k < pooled.length; k++) pooled[k] /= denom;

    // ── decoder ────────────────────────────────────────────────────────────
    const decIn = new Float64Array(w.width + nBands);
    decIn.set(pooled, 0);
    for (let b = 0; b < nBands; b++) decIn[w.width + b] = baseN[b] / 10;

    const z3 = dense(w.dec[0].W, w.dec[0].b, decIn);
    const d1 = new Float64Array(z3.length);
    for (let k = 0; k < z3.length; k++) d1[k] = silu(z3[k]);
    const z4 = dense(w.dec[1].W, w.dec[1].b, d1);
    const d2 = new Float64Array(z4.length);
    for (let k = 0; k < z4.length; k++) d2[k] = silu(z4[k]);
    const delta = dense(w.dec[2].W, w.dec[2].b, d2);

    const shape = new Float64Array(nBands);
    for (let b = 0; b < nBands; b++) shape[b] = baseN[b] + delta[b];

    return {
      shape,
      cache: { P, g, a, n, M, S, u, totals, baseN, feats, z1, h1, z2, h2,
               present: denom, decIn, z3, d1, z4, d2 },
    };
  }

  /**
   * Backward: given ∂V/∂shape, return ∂V/∂g.
   *
   * Two paths reach the faders and BOTH must be summed — forgetting either
   * gives a gradient that looks plausible and points slightly wrong:
   *   • through the baseline (g → u → M → baseN → shape)
   *   • through the encoder  (g is a per-channel input feature)
   * plus a third, subtle one: the decoder also reads `baseN`, so the baseline
   * path re-enters through the decoder's input as well.
   */
  backward(cache, dShape) {
    const { nBands, w, twoP } = this;
    const { P, g, a, n, M, S, u, totals, feats, z1, h1, z2, h2, present, z3, d1, z4, d2 } = cache;

    // ── decoder ────────────────────────────────────────────────────────────
    const dDelta = dShape;
    const dD2 = denseBackward(w.dec[2].W, dDelta, d2.length);
    for (let k = 0; k < dD2.length; k++) dD2[k] *= dsilu(z4[k]);
    const dD1 = denseBackward(w.dec[1].W, dD2, d1.length);
    for (let k = 0; k < dD1.length; k++) dD1[k] *= dsilu(z3[k]);
    const dDecIn = denseBackward(w.dec[0].W, dD1, w.width + nBands);

    // ∂V/∂baseN: directly (shape = baseN + delta) and through the decoder input.
    const dBaseN = new Float64Array(nBands);
    for (let b = 0; b < nBands; b++) dBaseN[b] = dShape[b] + dDecIn[w.width + b] / 10;

    // ── pooling → per-channel encoder ──────────────────────────────────────
    const dPooled = new Float64Array(w.width);
    for (let k = 0; k < w.width; k++) dPooled[k] = dDecIn[k] / present;

    const dG = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (a && !a[i]) continue; // masked out of the pool, so no path back
      const dH2 = Float64Array.from(dPooled);
      for (let k = 0; k < dH2.length; k++) dH2[k] *= dsilu(z2[i][k]);
      const dH1 = denseBackward(w.enc[1].W, dH2, h1[i].length);
      for (let k = 0; k < dH1.length; k++) dH1[k] *= dsilu(z1[i][k]);
      const dFeat = denseBackward(w.enc[0].W, dH1, feats[i].length);
      // Only the fader component of the feature vector depends on g; the
      // spectral shape and the energy are properties of the audio.
      dG[i] += dFeat[nBands];
    }

    // ── baseline path: baseN[b] = 10·log10(M[b]/S) ─────────────────────────
    // ∂baseN[b]/∂M[j] = DB_SCALE·(δ_bj/M[b] − 1/S)
    let dSum = 0;
    for (let b = 0; b < nBands; b++) dSum += dBaseN[b];
    for (let i = 0; i < n; i++) {
      const ai = a ? a[i] : 1;
      if (!ai) continue;
      const Pi = P[i];
      let acc = 0;
      for (let b = 0; b < nBands; b++) acc += (dBaseN[b] * Pi[b]) / M[b];
      const dV_du = DB_SCALE * (acc - (dSum * totals[i]) / S);
      // d/dg of (g·a)^(2p) = 2p·a·g^(2p−1)
      dG[i] += dV_du * twoP * ai * Math.pow(Math.max(g[i], 1e-12), twoP - 1);
    }
    return dG;
  }
}
