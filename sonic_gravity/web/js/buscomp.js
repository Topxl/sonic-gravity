/**
 * buscomp.js — multiband bus compressor, mirror of `sonic_gravity/bus.py`.
 *
 * This is the world the recurrent model was trained in. Running that model in
 * a browser whose bus behaves differently would be running it in a world it
 * has never seen — so this file has to agree with the Python reference, and
 * `tests/test_bus_parity.py` checks that it does, sample for sample.
 *
 * Pure DSP, no AudioWorklet API: the processor wrapper lives in
 * `bus-worklet.js`. Splitting them is what makes the parity test possible at
 * all — node has no AudioContext.
 */

const EPS = 1e-12;

/**
 * Second-order Butterworth section via the bilinear transform.
 * Same coefficients `scipy.signal.butter(2, …, output="sos")` produces; the
 * parity test pins them.
 */
export function butter2(fc, sr, highpass) {
  const k = Math.tan((Math.PI * fc) / sr);
  const k2 = k * k;
  const norm = 1 / (1 + Math.SQRT2 * k + k2);
  const a1 = 2 * (k2 - 1) * norm;
  const a2 = (1 - Math.SQRT2 * k + k2) * norm;
  return highpass
    ? { b0: norm, b1: -2 * norm, b2: norm, a1, a2 }
    : { b0: k2 * norm, b1: 2 * k2 * norm, b2: k2 * norm, a1, a2 };
}

/** Transposed direct form II — the same structure `sosfilt` uses. */
export class Biquad {
  constructor(c) {
    this.c = c;
    this.z1 = 0;
    this.z2 = 0;
  }
  reset() {
    this.z1 = 0;
    this.z2 = 0;
  }
  process(x) {
    const { b0, b1, b2, a1, a2 } = this.c;
    const y = b0 * x + this.z1;
    this.z1 = b1 * x - a1 * y + this.z2;
    this.z2 = b2 * x - a2 * y;
    return y;
  }
}

/** Gain reduction curve in dB (≤ 0), soft knee. Mirror of `bus.static_curve`. */
export function staticCurve(levelDb, comp) {
  const over = levelDb - comp.thresholdDb;
  const slope = 1 - 1 / comp.ratio;
  const k = Math.max(comp.kneeDb, 1e-9);
  if (over <= -k / 2) return 0;
  if (over >= k / 2) return -slope * over;
  const x = over + k / 2;
  return -(slope * x * x) / (2 * k);
}

/**
 * One band: peak detector with attack/release, at block rate.
 *
 * ⚠️ The detector runs per BLOCK, exactly as the Python reference does. Running
 * it per sample here would be "more accurate" and would put the browser in a
 * different world from the training data — which is the one thing that must
 * not happen.
 */
export class BandCompressor {
  constructor(comp, sr, block = 32) {
    this.comp = comp;
    this.block = block;
    this.setSampleRate(sr);
    this.state = 0; // current gain reduction, dB
    this.prev = 0; // previous block's, so the gain can ramp instead of step
  }

  setSampleRate(sr) {
    this.sr = sr;
    const rate = sr / this.block;
    this.attack = 1 - Math.exp(-1 / Math.max((this.comp.attackMs * 1e-3) * rate, 1));
    this.release = 1 - Math.exp(-1 / Math.max((this.comp.releaseMs * 1e-3) * rate, 1));
  }

  /** Advance the detector over one block and return its gain reduction in dB. */
  detect(buf, from, len) {
    let peak = 0;
    for (let i = 0; i < len; i++) {
      const v = Math.abs(buf[from + i]);
      if (v > peak) peak = v;
    }
    const target = staticCurve(20 * Math.log10(peak + EPS), this.comp);
    // Attack while clamping down, release while letting go — the asymmetry is
    // what makes the mix depend on its recent past.
    const coeff = target < this.state ? this.attack : this.release;
    this.prev = this.state;
    this.state += coeff * (target - this.state);
    return this.state;
  }
}

export const DEFAULT_CROSSOVERS = [200, 2000];

/** The three-band bus the recurrent model was trained against. */
export const DEFAULT_BANDS = [
  { thresholdDb: -26, ratio: 8, attackMs: 5, releaseMs: 700, kneeDb: 6 },
  { thresholdDb: -24, ratio: 4, attackMs: 15, releaseMs: 250, kneeDb: 6 },
  { thresholdDb: -30, ratio: 3, attackMs: 3, releaseMs: 80, kneeDb: 6 },
];

export class MultibandBus {
  /**
   * @param {number} sr
   * @param {number} channels  how many audio channels to process (detector is mono)
   */
  constructor(sr, bands = DEFAULT_BANDS, crossovers = DEFAULT_CROSSOVERS, channels = 1) {
    this.sr = sr;
    this.crossovers = crossovers;
    this.bands = bands;
    this.channels = channels;
    this.comps = bands.map((b) => new BandCompressor(b, sr));
    // One filter bank per channel, plus one for the mono detector: each needs
    // its own state, and sharing them would leak one channel into another.
    this.banks = Array.from({ length: channels + 1 }, () => this._makeBank());
    this.scratch = Array.from({ length: channels + 1 }, () =>
      bands.map(() => new Float32Array(2048))
    );
    this.gainReduction = bands.map(() => 0);
  }

  _makeBank() {
    // Linkwitz-Riley 4th order = the same 2nd-order section applied twice, so
    // the bands sum back to the original when nothing is compressing.
    return this.crossovers.map((fc) => ({
      lowA: new Biquad(butter2(fc, this.sr, false)),
      lowB: new Biquad(butter2(fc, this.sr, false)),
      highA: new Biquad(butter2(fc, this.sr, true)),
      highB: new Biquad(butter2(fc, this.sr, true)),
    }));
  }

  /** Split one signal into bands using bank `bi`, into that bank's scratch. */
  _split(x, bi, len) {
    const bank = this.banks[bi];
    const buf = this.scratch[bi];
    for (let b = 0; b < buf.length; b++) {
      if (buf[b].length < len) buf[b] = new Float32Array(len);
    }
    let rest = x;
    for (let s = 0; s < bank.length; s++) {
      const f = bank[s];
      const low = buf[s];
      const high = buf[s + 1];
      for (let i = 0; i < len; i++) {
        const v = rest[i];
        low[i] = f.lowB.process(f.lowA.process(v));
        high[i] = f.highB.process(f.highA.process(v));
      }
      rest = high;
    }
    if (bank.length === 0) {
      for (let i = 0; i < len; i++) buf[0][i] = x[i];
    }
    return buf.slice(0, bank.length + 1);
  }

  /**
   * Compress `chans` (array of Float32Array) into `outs`.
   *
   * ⚠️ Detection is MONO — the level comes from the average of all channels and
   * the same reduction is applied to each. Detecting per channel would let the
   * two sides compress by different amounts, so the stereo image would wander
   * whenever the bus works. It is also what the training data did.
   *
   * ⚠️ Detection re-runs every 32 samples, NOT once per audio block: the
   * worklet hands over 128 at a time, but the Python reference detects on 32,
   * and a detector running four times slower is a different compressor.
   */
  process(chans, outs, len) {
    const nb = this.comps.length;
    const nc = chans.length;

    if (!this._mono || this._mono.length < len) this._mono = new Float32Array(len);
    const mono = this._mono;
    if (nc === 1) {
      mono.set(chans[0].subarray(0, len));
    } else {
      for (let i = 0; i < len; i++) {
        let s = 0;
        for (let c = 0; c < nc; c++) s += chans[c][i];
        mono[i] = s / nc;
      }
    }

    const detBands = this._split(mono, nc, len);
    const chanBands = [];
    for (let c = 0; c < nc; c++) chanBands.push(this._split(chans[c], c, len));

    for (let c = 0; c < nc; c++) for (let i = 0; i < len; i++) outs[c][i] = 0;

    const step = this.comps[0].block;
    for (let off = 0; off < len; off += step) {
      const n = Math.min(step, len - off);
      for (let b = 0; b < nb; b++) {
        const comp = this.comps[b];
        const gr = comp.detect(detBands[b], off, n);
        this.gainReduction[b] = gr;
        // Ramp from the previous block's reduction to this one's — the same
        // causal form as the Python reference, so the two stay identical.
        for (let i = 0; i < n; i++) {
          const g = Math.pow(10, (comp.prev + (gr - comp.prev) * (i / n)) / 20);
          for (let c = 0; c < nc; c++) outs[c][off + i] += chanBands[c][b][off + i] * g;
        }
      }
    }
    return this.gainReduction;
  }
}
