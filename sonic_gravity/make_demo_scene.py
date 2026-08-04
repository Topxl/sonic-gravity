"""Synthesise a royalty-free multitrack scene for the demo.

The gravity field needs a real multitrack mix to act on. Shipping one made of
commercial stems would put a copyright violation in a public repository, so the
default scene is generated from scratch: six tracks, no samples, no external
material, nothing to clear.

It is deliberately *imperfect*. Two tracks (`keys` and `lead`) sit in the same
midrange region and fight each other, and the pad is mixed too loud — otherwise
the field would have nothing to correct and the demo would prove nothing.

    python -m sonic_gravity.make_demo_scene
    python -m sonic_gravity.make_demo_scene --bars 8 --bpm 96

Use `prepare_demo.py` instead to run the field on your own separated stems.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

OUT = pathlib.Path(__file__).resolve().parent / "web" / "audio"
SR = 48000

# A minor: the scene is four bars of Am - F - C - G, the most ordinary
# progression there is. The point is the mixing problem, not the music.
ROOT = 55.0  # A1
CHORDS = [  # semitones above root, per bar
    [0, 3, 7, 12],  # Am
    [-4, 0, 5, 8],  # F
    [3, 7, 10, 15],  # C
    [-2, 2, 5, 10],  # G
]


def _hz(semitones: float) -> float:
    return ROOT * 2.0 ** (semitones / 12.0)


def _env(n: int, attack: float, decay: float, sustain: float = 0.0, release: float = 0.0) -> np.ndarray:
    """Simple ADSR in samples, always ending at zero so loops never click."""
    a = max(1, int(attack * SR))
    d = max(1, int(decay * SR))
    r = max(1, int(release * SR))
    s = max(0, n - a - d - r)
    out = np.concatenate([
        np.linspace(0, 1, a),
        np.linspace(1, sustain, d),
        np.full(s, sustain),
        np.linspace(sustain, 0, r),
    ])
    return out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)))


def _lowpass(x: np.ndarray, cutoff: float, resonance: float = 0.7) -> np.ndarray:
    """One-pole-per-stage lowpass, applied twice. Good enough for a demo scene."""
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += alpha * (x[i] - acc)
        y[i] = acc
    z = np.empty_like(y)
    acc = 0.0
    for i in range(len(y)):
        acc += alpha * (y[i] - acc)
        z[i] = acc
    return z * (1.0 + resonance)


def _saw(freq: float, n: int, detune: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    f = freq * (1.0 + detune)
    # Band-limited enough by summing a fixed number of harmonics: a naive saw
    # aliases badly at 48 kHz and would put fake energy in the top bands, which
    # is exactly what the field measures.
    out = np.zeros(n)
    k = 1
    while f * k < SR / 2.5 and k <= 20:
        out += np.sin(2 * np.pi * f * k * t) / k
        k += 1
    return out * (2 / np.pi)


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n) * 0.5


def build(bars: int, bpm: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    spb = 60.0 / bpm  # seconds per beat
    bar_n = int(spb * 4 * SR)
    total = bar_n * bars
    t = np.arange(total) / SR

    tracks = {name: np.zeros(total) for name in
              ("drums", "bass", "keys", "lead", "pad", "air")}

    for bar in range(bars):
        chord = CHORDS[bar % len(CHORDS)]
        b0 = bar * bar_n

        # ── drums ──────────────────────────────────────────────────────────
        for beat in range(4):
            at = b0 + int(beat * spb * SR)
            n = int(0.28 * SR)
            seg = np.arange(n) / SR
            # Kick: pitch sweeping down from 110 Hz to 45 Hz.
            f = 45 + 65 * np.exp(-seg * 26)
            kick = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.09, 0.0, 0.14)
            tracks["drums"][at:at + n] += kick[: max(0, total - at)] * 0.9

            if beat % 2 == 1:  # snare on 2 and 4
                n2 = int(0.19 * SR)
                sn = (_noise(n2, rng) * 0.7 + np.sin(2 * np.pi * 190 * np.arange(n2) / SR) * 0.3)
                sn *= _env(n2, 0.001, 0.06, 0.0, 0.11)
                tracks["drums"][at:at + n2] += sn[: max(0, total - at)] * 0.5

            for half in (0, 1):  # hats on eighths
                ah = at + int(half * spb * SR / 2)
                n3 = int(0.05 * SR)
                if ah + n3 > total:
                    continue
                h = _noise(n3, rng)
                h = h - _lowpass(h, 6000)  # crude highpass
                tracks["drums"][ah:ah + n3] += h * _env(n3, 0.0005, 0.02, 0.0, 0.028) * 0.28

        # ── bass: root note, one per bar plus an offbeat push ───────────────
        for at_beat, dur in ((0.0, spb * 2.2), (2.5, spb * 1.2)):
            at = b0 + int(at_beat * spb * SR)
            n = min(int(dur * SR), total - at)
            if n <= 0:
                continue
            f = _hz(chord[0])
            raw = _saw(f, n) * _env(n, 0.006, 0.1, 0.75, 0.12)
            tracks["bass"][at:at + n] += _lowpass(raw, 220) * 0.55

        # ── keys: the full chord, sustained, sitting right in the midrange ──
        n = min(int(spb * 3.6 * SR), total - b0)
        keys = np.zeros(n)
        for st in chord:
            keys += _saw(_hz(st + 24), n, detune=rng.uniform(-0.002, 0.002))
        keys *= _env(n, 0.02, 0.25, 0.55, 0.5) / len(chord)
        tracks["keys"][b0:b0 + n] += _lowpass(keys, 2200) * 0.5

        # ── lead: arpeggio in the SAME midrange as the keys, on purpose ─────
        for step in range(8):
            at = b0 + int(step * spb / 2 * SR)
            n = min(int(spb * 0.45 * SR), total - at)
            if n <= 0:
                continue
            st = chord[step % len(chord)] + 24
            v = _saw(_hz(st), n) * _env(n, 0.004, 0.08, 0.4, 0.1)
            tracks["lead"][at:at + n] += _lowpass(v, 3000) * 0.42

        # ── pad: slow, wide, and mixed too loud — the field should notice ───
        n = min(bar_n, total - b0)
        pad = np.zeros(n)
        for st in chord:
            for d in (-0.004, 0.004):
                pad += np.sin(2 * np.pi * _hz(st + 12) * (1 + d) * np.arange(n) / SR)
        pad *= _env(n, 0.4, 0.3, 0.7, 0.5) / (len(chord) * 2)
        tracks["pad"][b0:b0 + n] += pad * 0.62

    # ── air: a continuous top-end texture, the least useful track ──────────
    air = _noise(total, rng)
    air = air - _lowpass(air, 4500)
    tracks["air"] = air * (0.16 + 0.06 * np.sin(2 * np.pi * 0.15 * t))

    return tracks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--bpm", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bitrate", default="112k")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required to encode the demo scene")

    tracks = build(args.bars, args.bpm, args.seed)
    duration = len(next(iter(tracks.values()))) / SR

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.opus"):
        old.unlink()

    entries = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, mono in tracks.items():
            peak = float(np.abs(mono).max())
            if peak < 1e-6:
                continue
            # Leave the relative balance alone — the whole point is that this
            # mix is not already correct. Only guard against clipping.
            x = mono / max(peak, 1.0)
            # A touch of stereo width so the scene is not a mono block.
            stereo = np.stack([x, np.roll(x, 13)], axis=1)
            wav = pathlib.Path(tmp) / f"{name}.wav"
            sf.write(wav, stereo, SR, subtype="PCM_16")

            dst = OUT / f"{name}.opus"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(wav),
                 "-c:a", "libopus", "-b:a", args.bitrate, "-ar", "48000", "-ac", "2", str(dst)],
                check=True,
            )
            rms = float(np.sqrt((x**2).mean()))
            entries.append({"id": name, "file": f"audio/{name}.opus",
                            "rms": round(rms, 5), "bytes": dst.stat().st_size})
            print(f"  {name:6s} rms={rms:.4f}  {dst.stat().st_size // 1024} kB")

    order = ["drums", "bass", "keys", "lead", "pad", "air"]
    entries.sort(key=lambda e: order.index(e["id"]) if e["id"] in order else 99)

    (OUT / "manifest.json").write_text(json.dumps({
        "source": "Synthetic scene (public domain)",
        "hash": f"synth-{args.seed}",
        "tempo": args.bpm,
        "start": 0.0,
        "duration": round(duration, 3),
        "synthetic": True,
        "tracks": entries,
    }, indent=2) + "\n")
    print(f"\n{len(entries)} tracks, {duration:.1f}s → {OUT}/manifest.json")


if __name__ == "__main__":
    main()
