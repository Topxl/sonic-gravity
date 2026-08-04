"""Build the dataset the learned world model trains on — and measure whether
there is anything to learn in the first place.

The analytic field assumes **decorrelated phases**:

    M[b] = Σᵢ gᵢ^(2p) · P[i,b]

Real summation does not work that way. Adding signals adds *amplitudes*, so the
true mix spectrum carries cross terms:

    |Σ aᵢXᵢ|² = Σ aᵢ²|Xᵢ|²  +  Σᵢ≠ⱼ aᵢaⱼ·Re(XᵢXⱼ*)
                └ the analytic model ┘  └── everything it ignores ──┘

Those cross terms are not noise. Separated stems bleed into each other, and
bleed is correlated by construction: the same snare hit lives in `drums` and,
faintly, in `other`. Where two channels share content, raising one does not add
power the way the analytic model predicts.

So this module renders the **actual summed audio** for randomised fader
positions and measures its real spectrum. It reports the analytic model's error
before any training happens: if that error turned out to be negligible, the
honest conclusion would be that a learned model has nothing to add here, and
this file would say so.

    python -m sonic_gravity.dataset --decompositions /path/to/stems --limit 40
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import soundfile as sf

from .field import band_edges, band_gather, bands_from_gather, load_spec

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "data"

WINDOW_S = 1.5  # long enough for a stable spectrum, short enough to stay in one section
FFT = 8192
EPS = 1e-12


def _welch(x: np.ndarray, fft: int = FFT) -> np.ndarray:
    """Average periodogram, Hann window, 50 % overlap."""
    if len(x) < fft:
        x = np.pad(x, (0, fft - len(x)))
    win = np.hanning(fft)
    hop = fft // 2
    frames = max(1, (len(x) - fft) // hop)
    acc = np.zeros(fft // 2, dtype=np.float64)
    for k in range(frames):
        seg = x[k * hop : k * hop + fft] * win
        acc += np.abs(np.fft.rfft(seg)[: fft // 2]) ** 2
    return acc / frames


def _sample_gains(n: int, rng: np.random.Generator) -> np.ndarray:
    """Fader positions to render.

    Half the samples sit near the neutral position, where the console actually
    lives and where the field's accuracy matters most; the rest spread wide so
    the model sees the whole travel and does not extrapolate blindly.
    """
    if rng.random() < 0.5:
        return np.clip(rng.normal(0.75, 0.09, size=n), 0.05, 1.0)
    return rng.uniform(0.05, 1.0, size=n)


def build(
    decomp: pathlib.Path,
    limit: int,
    windows: int,
    configs: int,
    seed: int,
    out: pathlib.Path,
) -> dict:
    spec = load_spec()
    edges = band_edges(spec.n_bands, spec.f_lo, spec.f_hi)
    p_law = spec.fader_exponent
    rng = np.random.default_rng(seed)

    dirs = sorted(d for d in decomp.iterdir() if (d / "stems").is_dir())
    rng.shuffle(dirs)
    dirs = dirs[:limit]
    if not dirs:
        sys.exit(f"no stem folders under {decomp}")

    rows_P, rows_g, rows_M, rows_track = [], [], [], []
    gather_cache: dict[int, list] = {}

    for ti, d in enumerate(dirs):
        files = sorted((d / "stems").glob("*.wav"))
        if len(files) < 3:
            continue
        try:
            info = sf.info(files[0])
        except Exception:
            continue
        sr = info.samplerate
        if info.duration < WINDOW_S * 4:
            continue
        if sr not in gather_cache:
            gather_cache[sr] = band_gather(FFT // 2, sr, FFT, edges)
        gather = gather_cache[sr]

        n_win = int(sr * WINDOW_S)
        for _ in range(windows):
            start = int(rng.uniform(5.0, max(6.0, info.duration - WINDOW_S - 2)) * sr)
            chans = []
            for f in files:
                try:
                    x, _ = sf.read(f, frames=n_win, start=start, dtype="float32")
                except Exception:
                    continue
                if x.ndim > 1:
                    x = x.mean(axis=1)
                chans.append(x.astype(np.float64))
            if len(chans) < 3:
                continue
            X = np.stack(chans)

            # Drop channels that are silent in this window: they carry no
            # information, and a fader over silence has no effect to learn.
            keep = np.sqrt((X**2).mean(axis=1)) > 1e-4
            if keep.sum() < 3:
                continue
            X = X[keep]

            P = np.stack([bands_from_gather(_welch(x), gather) for x in X]) + EPS

            for _ in range(configs):
                g = _sample_gains(len(X), rng)
                # THE point of this dataset: sum the actual waveforms, with
                # their actual phases, exactly as the audio engine does.
                amp = g**p_law
                mix = (amp[:, None] * X).sum(axis=0)
                M = bands_from_gather(_welch(mix), gather) + EPS

                rows_P.append(P.astype(np.float32))
                rows_g.append(g.astype(np.float32))
                rows_M.append(M.astype(np.float32))
                rows_track.append(ti)

        if (ti + 1) % 5 == 0:
            print(f"  {ti + 1}/{len(dirs)} tracks, {len(rows_M)} samples")

    if not rows_M:
        sys.exit("no usable samples")

    # Channel counts vary per window, so samples are stored flat with an index.
    counts = np.array([len(p) for p in rows_P], dtype=np.int32)
    P_flat = np.concatenate(rows_P, axis=0)
    g_flat = np.concatenate(rows_g, axis=0)
    M_arr = np.stack(rows_M)
    track = np.array(rows_track, dtype=np.int32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, P=P_flat, g=g_flat, M=M_arr, counts=counts, track=track,
        fader_exponent=np.float32(p_law), n_bands=np.int32(spec.n_bands),
    )

    report = baseline_error(P_flat, g_flat, M_arr, counts, p_law)
    report.update({"samples": int(len(M_arr)), "tracks": int(track.max() + 1),
                   "file": str(out)})
    return report


def analytic_mix(P: np.ndarray, g: np.ndarray, p_law: float) -> np.ndarray:
    """The baseline: decorrelated-phase power summation."""
    u = g**(2.0 * p_law)
    return u @ P


def baseline_error(P_flat, g_flat, M, counts, p_law: float) -> dict:
    """How wrong is the analytic model on real audio?

    Measured in dB per band, on the *shape* of the spectrum (normalised to sum
    one), because that is what the potential actually consumes — an overall
    level offset is absorbed by the level term and is not what we want to fix.
    """
    errs = []
    off = 0
    for i, c in enumerate(counts):
        P = P_flat[off : off + c]
        g = g_flat[off : off + c]
        off += c
        hat = analytic_mix(P, g, p_law) + EPS
        true = M[i] + EPS
        # Compare shapes: level is the master's business, not the field's.
        e = 10 * np.log10(hat / hat.sum()) - 10 * np.log10(true / true.sum())
        errs.append(e)
    E = np.stack(errs)
    absE = np.abs(E)
    return {
        "mae_db": float(absE.mean()),
        "rmse_db": float(np.sqrt((E**2).mean())),
        "p50_db": float(np.percentile(absE, 50)),
        "p95_db": float(np.percentile(absE, 95)),
        "max_db": float(absE.max()),
        "bias_db": float(E.mean()),
        "worst_bands": [int(b) for b in np.argsort(-absE.mean(axis=0))[:6]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decompositions", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("SG_DECOMPOSITIONS", ROOT / "decompositions")))
    ap.add_argument("--limit", type=int, default=40, help="how many tracks")
    ap.add_argument("--windows", type=int, default=12, help="time windows per track")
    ap.add_argument("--configs", type=int, default=14, help="fader configurations per window")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=pathlib.Path, default=OUT / "mixes.npz")
    args = ap.parse_args()

    if not args.decompositions.is_dir():
        sys.exit(f"no stem folders under {args.decompositions} (set --decompositions)")

    print(f"building from {args.decompositions}")
    report = build(args.decompositions, args.limit, args.windows, args.configs,
                   args.seed, args.out)

    print(f"\n{report['samples']} samples from {report['tracks']} tracks → {report['file']}")
    print("\nAnalytic baseline error on real summed audio (spectrum shape):")
    print(f"  MAE      {report['mae_db']:.3f} dB")
    print(f"  RMSE     {report['rmse_db']:.3f} dB")
    print(f"  median   {report['p50_db']:.3f} dB")
    print(f"  p95      {report['p95_db']:.3f} dB")
    print(f"  max      {report['max_db']:.3f} dB")
    print(f"  bias     {report['bias_db']:+.3f} dB")
    print(f"  worst bands: {report['worst_bands']}")
    (args.out.parent / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
