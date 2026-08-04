"""Mesure le gabarit spectral du champ de gravité sur de VRAIS morceaux.

Sans ça, le potentiel comparerait le mix à une règle inventée (« un bon mix
suit une pente rose »), et la gravité pousserait la main vers une idée que
personne n'a jamais vérifiée. Ici le gabarit est la répartition spectrale
médiane d'un échantillon de la bibliothèque, avec sa dispersion par bande :
le potentiel devient une distance au **domaine des morceaux qui existent**.

La dispersion compte autant que la médiane : les bandes où les vrais morceaux
sont très dispersés (le grave extrême, l'aigu au-delà de 12 kHz, qui dépendent
du genre et du mastering) pèsent naturellement moins dans le z-score, sans
qu'on ait à décréter un poids à la main.

    python -m sonic_gravity.fit_target --limit 200
    python -m sonic_gravity.fit_target --limit 400 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import soundfile as sf

from .field import EPS, SPEC_PATH, band_edges, band_gather, bands_from_gather

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Fenêtre d'analyse par morceau. On lit au milieu : les intros et les fins
# sont rarement représentatives de l'équilibre du morceau.
WINDOW_S = 20.0
FFT = 8192  # doit rester la résolution de spec.analysis.fftSize


AUDIO_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".aiff"}


def _library_files(limit: int, seed: int, audio_dir: pathlib.Path | None) -> list[pathlib.Path]:
    """Les morceaux sur lesquels mesurer le gabarit.

    Le paquet est autonome : la source par défaut est un dossier passé en
    argument (ou SG_LIBRARY). `library_source` n'est tenté qu'en dernier, et
    seulement s'il se trouve être importable — c'est le cas dans le dépôt d'où
    ce projet est issu, jamais ailleurs.
    """
    files: list[pathlib.Path] = []

    base = audio_dir or (pathlib.Path(os.environ["SG_LIBRARY"]) if os.environ.get("SG_LIBRARY") else None)
    if base is not None:
        if not base.is_dir():
            sys.exit(f"dossier introuvable : {base}")
        print(f"source : {base}")
        files = [p for p in base.rglob("*") if p.suffix.lower() in AUDIO_EXTS]

    if not files:
        try:
            sys.path.insert(0, str(ROOT))
            import library_source

            found, label = library_source.resolve_library(allow_pi=False)
            if found is not None:
                print(f"source : {label} ({found})")
                files = [p for p, _ in library_source.iter_audio(found)]
        except Exception:
            pass

    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:limit]


def _spectrum(path: pathlib.Path, gather, n_bands: int, sr_expected: float, edges) -> np.ndarray | None:
    """Répartition spectrale d'un morceau, en dB, normalisée (somme des parts = 1)."""
    try:
        info = sf.info(path)
    except Exception:
        return None
    if info.duration < WINDOW_S + 4:
        return None

    start = int(info.samplerate * (info.duration - WINDOW_S) / 2)
    try:
        x, sr = sf.read(path, frames=int(info.samplerate * WINDOW_S), start=start, dtype="float32")
    except Exception:
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    if not np.isfinite(x).all() or float(np.sqrt((x**2).mean())) < 1e-4:
        return None

    # La table bin → bande est calculée pour un taux d'échantillonnage donné.
    # Un morceau à 44,1 kHz ne peut pas emprunter celle d'un 48 kHz : les bornes
    # de bande tomberaient sur les mauvais bins et le gabarit serait biaisé.
    if abs(sr - sr_expected) > 1:
        gather = band_gather(FFT // 2, sr, FFT, edges)

    # Welch : moyenne des périodogrammes, fenêtre de Hann, recouvrement 50 %.
    win = np.hanning(FFT).astype(np.float32)
    hop = FFT // 2
    n_frames = (len(x) - FFT) // hop
    if n_frames < 4:
        return None
    acc = np.zeros(FFT // 2, dtype=np.float64)
    for k in range(n_frames):
        seg = x[k * hop : k * hop + FFT] * win
        acc += np.abs(np.fft.rfft(seg)[: FFT // 2]) ** 2
    acc /= n_frames

    bands = bands_from_gather(acc, gather) + EPS
    share = bands / bands.sum()
    return 10.0 * np.log10(share)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=200, help="nombre de morceaux à mesurer")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=pathlib.Path, default=SPEC_PATH)
    ap.add_argument("--audio-dir", type=pathlib.Path, default=None,
                    help="dossier de morceaux à mesurer (défaut : $SG_LIBRARY)")
    args = ap.parse_args()

    spec = json.loads(SPEC_PATH.read_text())
    n_bands = int(spec["bands"]["n"])
    edges = band_edges(n_bands, float(spec["bands"]["fLo"]), float(spec["bands"]["fHi"]))
    gather = band_gather(FFT // 2, 44100.0, FFT, edges)

    files = _library_files(args.limit, args.seed, args.audio_dir)
    if not files:
        sys.exit(
            "aucun morceau à mesurer — indique une source avec --audio-dir <dossier>\n"
            "(le gabarit livré dans spec.json est déjà mesuré sur 220 morceaux)"
        )
    print(f"{len(files)} morceaux à mesurer…")

    rows = []
    for i, f in enumerate(files, 1):
        s = _spectrum(f, gather, n_bands, 44100.0, edges)
        if s is not None:
            rows.append(s)
        if i % 25 == 0:
            print(f"  {i}/{len(files)}  ({len(rows)} retenus)")

    if len(rows) < 20:
        sys.exit(f"trop peu de morceaux exploitables ({len(rows)})")

    X = np.stack(rows)
    # Médiane et MAD : un morceau au mastering extrême ne doit pas déplacer le
    # gabarit à lui seul. 1,4826 ramène la MAD à un écart-type gaussien.
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med), axis=0)
    # Plancher de dispersion : une bande anormalement homogène rendrait le
    # z-score explosif et la gravité brutale sur une seule bande.
    sigma = np.maximum(mad, 1.5)

    spec["target"]["db"] = [round(float(v), 3) for v in med]
    spec["target"]["sigma"] = [round(float(v), 3) for v in sigma]
    spec["target"]["weight"] = [1.0] * n_bands
    spec["target"]["fitted"] = True
    spec["target"]["nTracks"] = len(rows)
    args.out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")

    centers = np.sqrt(edges[:-1] * edges[1:])
    print(f"\ngabarit mesuré sur {len(rows)} morceaux → {args.out}")
    print(f"{'Hz':>7}  {'médiane dB':>10}  {'σ dB':>6}")
    for b in range(0, n_bands, 4):
        print(f"{centers[b]:7.0f}  {med[b]:10.2f}  {sigma[b]:6.2f}")


if __name__ == "__main__":
    main()
