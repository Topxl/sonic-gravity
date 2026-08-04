"""Prépare le matériel sonore d'une scène de gravité.

Prend une décomposition en stems (`decompositions/<hash>/stems/*.wav`) et en
extrait une boucle courte, encodée pour le web, avec son manifeste.

Pourquoi une boucle courte : la surface doit tourner en boucle fermée pendant
qu'on manipule les faders, sans que la structure du morceau change sous les
pieds du modèle. Un couplet qui devient un refrain déplacerait le champ de
gravité pour une raison qui n'a rien à voir avec l'action de l'utilisateur —
autant enlever cette variable tant qu'on met le monde au point.

    python -m sonic_gravity.prepare_demo --list
    python -m sonic_gravity.prepare_demo e9cb4ebd80a11f31 --start 60 --dur 24
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Où vivent les stems séparés. Ce paquet est autonome : il ne suppose aucun
# dépôt parent. Réglable par --decompositions ou SG_DECOMPOSITIONS.
DECOMP = pathlib.Path(os.environ.get("SG_DECOMPOSITIONS", ROOT / "decompositions"))
OUT = pathlib.Path(__file__).resolve().parent / "web" / "audio"

# Un stem sous ce RMS sur la fenêtre retenue n'est pas une piste : c'est du
# silence que le séparateur n'a pas su attribuer. Lui donner une tranche
# donnerait un fader sans effet, donc un gradient nul et une colonne morte.
MIN_RMS = 0.01

# Ordre d'affichage sur la surface : grave à gauche, aigu à droite, voix au
# centre. C'est la disposition d'une vraie console, pas l'ordre alphabétique
# du séparateur.
STEM_ORDER = ["drums", "bass", "piano", "guitar", "other", "residual", "vocals"]


def scan() -> list[tuple[int, str, str, list[str]]]:
    """Liste les décompositions par nombre de stems réellement actifs."""
    found = []
    for d in sorted(DECOMP.iterdir()):
        stems = d / "stems"
        if not stems.is_dir():
            continue
        try:
            name = json.loads((d / "manifest.json").read_text()).get("source_filename", "?")
        except Exception:
            name = "?"
        active = []
        for f in sorted(stems.glob("*.wav")):
            try:
                info = sf.info(f)
                start = int(info.samplerate * min(45.0, max(0.0, info.duration - 25.0)))
                x, _ = sf.read(f, frames=info.samplerate * 15, start=start, dtype="float32")
                if float(np.sqrt((x**2).mean())) > MIN_RMS:
                    active.append(f.stem)
            except Exception:
                pass
        if active:
            found.append((len(active), name, d.name, active))
    found.sort(reverse=True)
    return found


def _loudest_window(stems: list[pathlib.Path], dur: float) -> float:
    """Cherche le passage le plus dense : celui où le plus de stems jouent.

    Un extrait pris au hasard tombe souvent sur une intro où trois tranches sur
    six sont muettes — le mix n'a alors rien à arbitrer.
    """
    info = sf.info(stems[0])
    sr = info.samplerate
    total = info.duration
    if total <= dur + 2:
        return 0.0
    step = 4.0
    best, best_score = 0.0, -1.0
    t = 5.0
    while t + dur < total - 2:
        score = 0.0
        for f in stems:
            x, _ = sf.read(f, frames=int(sr * min(dur, 8.0)), start=int(sr * t), dtype="float32")
            rms = float(np.sqrt((x**2).mean()))
            # On compte les stems PRÉSENTS, sans laisser la batterie (toujours
            # la plus forte) décider seule du passage retenu.
            if rms > MIN_RMS:
                score += 1.0 + min(rms, 0.2)
        if score > best_score:
            best_score, best = score, t
        t += step
    return best


def prepare(track_hash: str, start: float | None, dur: float, bitrate: str) -> None:
    src = DECOMP / track_hash / "stems"
    if not src.is_dir():
        sys.exit(f"stems introuvables : {src}")
    try:
        meta = json.loads((DECOMP / track_hash / "manifest.json").read_text())
    except Exception:
        meta = {}

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg est requis pour encoder les boucles")

    files = sorted(src.glob("*.wav"))
    if not files:
        sys.exit("aucun stem")

    if start is None:
        start = _loudest_window(files, dur)
        print(f"fenêtre retenue automatiquement : {start:.1f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.opus"):
        old.unlink()

    tracks = []
    for f in files:
        info = sf.info(f)
        x, sr = sf.read(f, frames=int(info.samplerate * dur), start=int(info.samplerate * start), dtype="float32")
        rms = float(np.sqrt((x**2).mean()))
        if rms <= MIN_RMS:
            print(f"  – {f.stem:9s} muet sur la fenêtre, écarté")
            continue

        dst = OUT / f"{f.stem}.opus"
        # On encode depuis le WAV source plutôt que depuis le tableau numpy :
        # ffmpeg fait le rééchantillonnage et le fondu proprement, et la boucle
        # doit être sans clic à la jonction (fondu de 40 ms aux deux bouts).
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(f),
            "-af", f"afade=t=in:st=0:d=0.04,afade=t=out:st={dur - 0.04:.3f}:d=0.04",
            "-c:a", "libopus", "-b:a", bitrate, "-ar", "48000", "-ac", "2",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
        tracks.append({
            "id": f.stem,
            "file": f"audio/{f.stem}.opus",
            "rms": round(rms, 5),
            "bytes": dst.stat().st_size,
        })
        print(f"  ✓ {f.stem:9s} rms={rms:.4f}  {dst.stat().st_size // 1024} Ko")

    order = {name: i for i, name in enumerate(STEM_ORDER)}
    tracks.sort(key=lambda t: order.get(t["id"], 99))

    manifest = {
        "source": meta.get("source_filename", track_hash),
        "hash": track_hash,
        "tempo": meta.get("tempo"),
        "start": round(start, 3),
        "duration": dur,
        "tracks": tracks,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n{len(tracks)} tranches → {OUT}/manifest.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hash", nargs="?", help="hash d'une décomposition")
    ap.add_argument("--list", action="store_true", help="lister les décompositions utilisables")
    ap.add_argument("--start", type=float, default=None, help="début de la boucle (s) ; auto si absent")
    ap.add_argument("--dur", type=float, default=24.0, help="durée de la boucle (s)")
    ap.add_argument("--bitrate", default="112k")
    ap.add_argument("--decompositions", type=pathlib.Path, default=None,
                    help="dossier des stems séparés (défaut : $SG_DECOMPOSITIONS)")
    args = ap.parse_args()

    global DECOMP
    if args.decompositions:
        DECOMP = args.decompositions

    if not DECOMP.is_dir():
        sys.exit(
            f"aucun dossier de stems en {DECOMP}\n"
            "→ pour la démo sans matériel externe : python -m sonic_gravity.make_demo_scene\n"
            "→ pour tes propres stems : --decompositions <dossier>"
        )

    if args.list or not args.hash:
        for n, name, h, active in scan()[:25]:
            print(f"{n} {h} {name[:50]:50s} {','.join(active)}")
        return
    prepare(args.hash, args.start, args.dur, args.bitrate)


if __name__ == "__main__":
    main()
