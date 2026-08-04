"""Sequential dataset: fader *trajectories* through a compressed bus.

The instantaneous dataset (`dataset.py`) asks a static question — given these
faders, what is the mix? With a bus compressor in the path the question stops
being static, and that is the point of this file:

  • gain reduction depends on what the signal did milliseconds ago, so the same
    fader position gives a different mix depending on how it was reached;
  • lowering one channel changes the reduction, hence **every** other channel's
    contribution — the faders are no longer independent;
  • after a transient the bus keeps recovering for as long as the release.

So the faders have to *move*. Trajectories mix held positions, slow ramps and
abrupt jumps, because a model that only ever saw smooth motion would never
learn what happens when a hand yanks a fader — which is exactly when the
compressor's memory dominates.

Frames hop by 46 ms under a 186 ms window: shorter than the 150 ms release, so
consecutive frames actually share compressor state. Frames spaced wider than
the release would carry no memory to learn.

    python -m sonic_gravity.dataset_seq --decompositions /path/to/stems --limit 30
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import soundfile as sf

from .bus import CROSSOVERS, BusComp, apply as compress, apply_multiband
from .field import band_edges, band_gather, bands_from_gather, load_spec

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).resolve().parent / "data"

FFT = 8192
HOP = 2048  # 46 ms at 44.1 kHz — well under the compressor's release
SEQ_S = 8.0
EPS = 1e-12


def _trajectory(n_ch: int, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Fader positions over time, as a hand might actually move them.

    Three regimes on purpose. Held positions are where the console spends its
    life; ramps are ordinary gestures; jumps are where the compressor's memory
    is most visible, and a dataset without them would let the model believe the
    world is quasi-static.
    """
    g = np.empty((n_frames, n_ch), dtype=np.float32)
    for c in range(n_ch):
        t = 0
        value = float(np.clip(rng.normal(0.75, 0.1), 0.1, 1.0))
        while t < n_frames:
            kind = rng.choice(["hold", "ramp", "jump"], p=[0.45, 0.35, 0.20])
            span = int(rng.integers(4, 26))
            span = min(span, n_frames - t)
            if kind == "hold":
                g[t : t + span, c] = value
            elif kind == "ramp":
                target = float(np.clip(value + rng.normal(0, 0.22), 0.05, 1.0))
                g[t : t + span, c] = np.linspace(value, target, span)
                value = target
            else:
                target = float(np.clip(rng.uniform(0.05, 1.0), 0.05, 1.0))
                g[t : t + span, c] = target
                value = target
            t += span
    return g


def _stft_bands(x: np.ndarray, gather, n_frames: int) -> np.ndarray:
    """Band powers per frame, sliding an FFT window with hop HOP."""
    win = np.hanning(FFT)
    out = np.empty((n_frames, len(gather)), dtype=np.float32)
    for k in range(n_frames):
        seg = x[k * HOP : k * HOP + FFT]
        if len(seg) < FFT:
            seg = np.pad(seg, (0, FFT - len(seg)))
        spec = np.abs(np.fft.rfft(seg * win)[: FFT // 2]) ** 2
        out[k] = bands_from_gather(spec, gather) + EPS
    return out


def build(decomp: pathlib.Path, limit: int, seqs: int, seed: int,
          comp, out: pathlib.Path, multiband: bool = False) -> dict:
    spec = load_spec()
    edges = band_edges(spec.n_bands, spec.f_lo, spec.f_hi)
    p_law = spec.fader_exponent
    rng = np.random.default_rng(seed)

    dirs = sorted(d for d in decomp.iterdir() if (d / "stems").is_dir())
    rng.shuffle(dirs)
    dirs = dirs[:limit]
    if not dirs:
        sys.exit(f"no stem folders under {decomp}")

    P_seqs, g_seqs, M_seqs, tracks = [], [], [], []
    gather_cache: dict[int, list] = {}

    for ti, d in enumerate(dirs):
        files = sorted((d / "stems").glob("*.wav"))
        try:
            info = sf.info(files[0])
        except Exception:
            continue
        sr = info.samplerate
        if info.duration < SEQ_S * 3:
            continue
        if sr not in gather_cache:
            gather_cache[sr] = band_gather(FFT // 2, sr, FFT, edges)
        gather = gather_cache[sr]

        n_samp = int(sr * SEQ_S)
        n_frames = (n_samp - FFT) // HOP

        for _ in range(seqs):
            start = int(rng.uniform(5.0, max(6.0, info.duration - SEQ_S - 2)) * sr)
            chans = []
            for f in files:
                try:
                    x, _ = sf.read(f, frames=n_samp, start=start, dtype="float32")
                except Exception:
                    continue
                if x.ndim > 1:
                    x = x.mean(axis=1)
                chans.append(x.astype(np.float64))
            if len(chans) < 3:
                continue
            X = np.stack(chans)
            keep = np.sqrt((X**2).mean(axis=1)) > 1e-4
            if keep.sum() < 3:
                continue
            X = X[keep]
            n_ch = len(X)

            g = _trajectory(n_ch, n_frames, rng)

            # Fader curve at sample rate: the trajectory is per frame, but the
            # audio must move continuously or every frame boundary becomes a
            # click that the spectrum would faithfully record.
            centres = np.arange(n_frames) * HOP + FFT / 2
            idx = np.arange(n_samp)
            amp = np.stack([
                np.interp(idx, centres, g[:, c], left=g[0, c], right=g[-1, c]) ** p_law
                for c in range(n_ch)
            ])

            mix = (amp * X).sum(axis=0)
            mixed = apply_multiband(mix, sr, comp) if multiband else compress(mix, sr, comp)

            P = np.stack([_stft_bands(x, gather, n_frames) for x in X])  # (C,T,N)
            M = _stft_bands(mixed, gather, n_frames)                      # (T,N)

            P_seqs.append(P.astype(np.float32).transpose(1, 0, 2))        # (T,C,N)
            g_seqs.append(g)
            M_seqs.append(M.astype(np.float32))
            tracks.append(ti)

        if (ti + 1) % 5 == 0:
            print(f"  {ti + 1}/{len(dirs)} tracks, {len(M_seqs)} sequences")

    if not M_seqs:
        sys.exit("no usable sequences")

    # Channel counts differ between sequences; store flat with an index.
    counts = np.array([p.shape[1] for p in P_seqs], dtype=np.int32)
    n_frames = M_seqs[0].shape[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        P=np.concatenate([p.reshape(n_frames, -1) for p in P_seqs], axis=1),
        g=np.concatenate(g_seqs, axis=1),
        M=np.stack(M_seqs),
        counts=counts,
        track=np.array(tracks, dtype=np.int32),
        n_bands=np.int32(spec.n_bands),
        n_frames=np.int32(n_frames),
        fader_exponent=np.float32(p_law),
        multiband=np.int32(1 if multiband else 0),
        comp=np.array([[c.threshold_db, c.ratio, c.attack_ms, c.release_ms, c.knee_db]
                       for c in (comp if multiband else [comp])], dtype=np.float32),
    )
    return {"sequences": len(M_seqs), "tracks": int(max(tracks) + 1),
            "frames_per_seq": int(n_frames), "file": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decompositions", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get("SG_DECOMPOSITIONS", ROOT / "decompositions")))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seqs", type=int, default=4, help="sequences per track")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=-18.0)
    ap.add_argument("--ratio", type=float, default=4.0)
    ap.add_argument("--attack", type=float, default=10.0)
    ap.add_argument("--release", type=float, default=150.0)
    ap.add_argument("--out", type=pathlib.Path, default=OUT / "sequences.npz")
    ap.add_argument("--multiband", action="store_true",
                    help="three-band bus: the only version whose SHAPE is time-dependent")
    args = ap.parse_args()

    if not args.decompositions.is_dir():
        sys.exit(f"no stem folders under {args.decompositions}")

    if args.multiband:
        # Low band slow and heavy (it ducks on kicks), top band fast and light —
        # so each band's gain reduction follows its own history and the mix's
        # SHAPE becomes time-dependent, which a wideband bus never makes it.
        comp = [
            BusComp(threshold_db=-26, ratio=8, attack_ms=5, release_ms=700),
            BusComp(threshold_db=-24, ratio=4, attack_ms=15, release_ms=250),
            BusComp(threshold_db=-30, ratio=3, attack_ms=3, release_ms=80),
        ]
        print(f"multiband bus, crossovers {CROSSOVERS[0]:.0f}/{CROSSOVERS[1]:.0f} Hz:")
        for c in comp:
            print(f"  {c.threshold_db:.0f} dB, {c.ratio:.0f}:1, {c.attack_ms:.0f}/{c.release_ms:.0f} ms")
    else:
        comp = BusComp(threshold_db=args.threshold, ratio=args.ratio,
                       attack_ms=args.attack, release_ms=args.release)
        print(f"wideband bus: {comp.threshold_db} dB, {comp.ratio}:1, "
              f"{comp.attack_ms}/{comp.release_ms} ms")
    report = build(args.decompositions, args.limit, args.seqs, args.seed, comp,
                   args.out, multiband=args.multiband)
    print(f"\n{report['sequences']} sequences × {report['frames_per_seq']} frames "
          f"from {report['tracks']} tracks → {report['file']}")
    (args.out.parent / "sequences.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
