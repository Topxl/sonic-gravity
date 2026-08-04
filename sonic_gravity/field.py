"""Le champ de gravité sonore : potentiel V et son gradient analytique.

C'est la pièce centrale, et la seule qui doit être EXACTE au sens strict : la
force envoyée aux faders est `-dV/dg`, donc une erreur de signe ou de facteur
ne se voit pas dans un test de forme — elle se voit en poussant la main de
l'utilisateur vers le mauvais mix.

Le modèle
---------
Chaque tranche `i` est mesurée PRÉ-fader : `P[i, b]` = puissance de la tranche
dans la bande `b`, indépendante de son gain. Le mix vaut alors

    M[b] = somme_i (g_i · a_i)² · P[i, b]

avec `g_i` le fader (0..1) et `a_i` l'audibilité (0 si coupée, ou tue par un
solo ailleurs). ⚠️ C'est une approximation à **phases décorrélées** : deux
sources cohérentes s'additionnent en amplitude, pas en puissance. Elle tient
bien sur un mix multipiste réel (les stems d'un séparateur ne sont pas en
phase), elle tomberait sur deux copies du même signal. C'est exactement la
non-linéarité que le monde appris (`model.py`) doit rattraper : la référence
analytique n'est pas la vérité, c'est la ligne de base à battre.

Le potentiel a trois termes, tous sans dimension et de l'ordre de l'unité :

1. `tilt`  — écart de la répartition spectrale au gabarit des vrais morceaux,
             en z-score par bande (gabarit mesuré par `fit_target.py`), passé
             dans une **perte de Huber**. Le carré nu laissait un break dominer
             tout le champ : mesuré le 2026-08-04 sur un passage de Clocks où
             seul le piano joue, les sept bandes graves sortaient à −10 σ et le
             potentiel montait à 88. Or aucun fader ne peut créer du grave que
             la basse et la batterie ne jouent pas — s'acharner sur une bande
             sans matière disponible n'est pas une correction, c'est du bruit.
             Huber garde le carré près de la cible (où la correction est fine)
             et devient linéaire au-delà (gradient borné, jamais de panique).
2. `mask`  — masquage : deux sources fortes qui occupent la même zone du
             spectre se battent. Terme **invariant d'échelle** (normalisé par
             l'énergie totale), donc son gradient ne dit jamais « baisse tout »,
             il dit « baisse CELUI qui se bat avec les autres ».
3. `level` — écart au niveau visé, qui empêche l'effondrement vers le silence
             et l'écrêtage.
4. `anchor`— rappel vers la position que la MAIN a posée. Sans lui, le minimum
             du potentiel est un mix entièrement recalculé par la machine, et
             les faders partent en butée (mesuré le 2026-08-04 : batterie 1,00,
             voix 0,997, « other » 0,118 en trois secondes). Ce terme est ce qui
             fait la différence entre un exosquelette et un pilote automatique :
             la gravité corrige AUTOUR de l'intention, elle ne la remplace pas.
             Poids à zéro = la machine reprend tout le mix, ce qui reste un mode
             de démonstration valable, mais pas le défaut.

Le gradient est dérivé à la main plus bas ; `tests/test_field_gradient.py` le
compare à des différences finies — c'est la seule preuve qui compte.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

SPEC_PATH = pathlib.Path(__file__).resolve().parent / "spec.json"

# ⚠️ Deux constantes voisines qu'il ne faut pas confondre — la première erreur
# de ce module a été d'écrire l'une pour l'autre, ce qui gonflait la force de
# 1,7× à 2,8× selon la scène sans jamais produire d'absurdité visible.
#   • une VALEUR en décibels vaut 10·log10(x)
#   • sa DÉRIVÉE vaut d(10·log10 x)/dx = 10/(x·ln10) = DB_SCALE / x
DB_VALUE = 10.0
DB_SCALE = 10.0 / np.log(10.0)

# Plancher de puissance. Une bande vide donnerait -inf en dB et un gradient
# infini ; ce plancher est à −120 dB relatif, très en dessous de tout signal
# utile, mais il rend la dérivée finie partout.
EPS = 1e-12


def band_edges(n_bands: int, f_lo: float, f_hi: float) -> np.ndarray:
    """Bornes des bandes, espacées logarithmiquement (n_bands + 1 valeurs)."""
    return np.geomspace(f_lo, f_hi, n_bands + 1)


def bin_band_map(n_bins: int, sample_rate: float, fft_size: int, edges: np.ndarray) -> np.ndarray:
    """Table bin FFT → index de bande (−1 = hors plage).

    ⚠️ Une bande plus étroite que la résolution de la FFT ne reçoit AUCUN bin.
    Elle vaudrait alors le plancher, soit −174 dB, et comme le gabarit est mesuré
    avec la même table, cette bande fantôme deviendrait un puits qui domine tout
    le potentiel dès que le mix y a la moindre énergie. Payé le 2026-08-04 : la
    bande 34-39 Hz sortait à −173,9 dB sur 220 morceaux.

    Le repli est donc obligatoire, pas optionnel : toute bande restée vide
    adopte le bin dont le centre en est le plus proche. Le bin est alors compté
    dans deux bandes voisines — c'est assumé, parce que le gabarit et le mix
    subissent EXACTEMENT le même traitement, donc le z-score reste comparable.
    Ce qu'il ne faut jamais faire, c'est laisser une bande à zéro.

    Cette table ne SAIT pas dupliquer un bin (elle associe un bin à une seule
    bande) : c'est pourquoi la fonction canonique est `band_gather`, qui liste
    les bins de chaque bande et peut en partager un entre deux voisines. On ne
    garde celle-ci que pour inspecter la couverture.
    """
    freqs = np.arange(n_bins) * (sample_rate / fft_size)
    mapping = np.full(n_bins, -1, dtype=np.int32)
    for b in range(len(edges) - 1):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        mapping[sel] = b
    return mapping


def band_gather(n_bins: int, sample_rate: float, fft_size: int, edges: np.ndarray) -> list[np.ndarray]:
    """Pour chaque bande, la liste des bins qui la composent.

    Représentation explicite, préférée à la table bin → bande : elle autorise
    un bin à servir DEUX bandes voisines, ce dont on a besoin dans le grave où
    les bandes sont plus étroites que la résolution. Aucune bande ne peut être
    vide en sortie — c'est vérifié par les tests.
    """
    freqs = np.arange(n_bins) * (sample_rate / fft_size)
    centers = np.sqrt(edges[:-1] * edges[1:])
    out = []
    for b in range(len(edges) - 1):
        idx = np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(np.log(np.maximum(freqs, 1e-6) / centers[b]))))])
        out.append(idx.astype(np.int32))
    return out


def bands_from_gather(bin_power: np.ndarray, gather: list[np.ndarray]) -> np.ndarray:
    """Somme les puissances de bin par bande. `bin_power` : (n_bins,) ou (n_ch, n_bins)."""
    x = np.atleast_2d(bin_power)
    out = np.empty((x.shape[0], len(gather)), dtype=np.float64)
    for b, idx in enumerate(gather):
        out[:, b] = x[:, idx].sum(axis=1)
    return out if bin_power.ndim > 1 else out[0]


def _huber(z: np.ndarray, delta: float) -> np.ndarray:
    """Carré près de zéro, linéaire au-delà de `delta` — et continue en delta.

    Se réduit exactement à `z**2` quand `delta` est infini, ce qui permet de
    retrouver l'ancien comportement en réglant `tiltHuber` très haut.
    """
    a = np.abs(z)
    return np.where(a <= delta, z**2, 2.0 * delta * a - delta**2)


def _dhuber(z: np.ndarray, delta: float) -> np.ndarray:
    """Dérivée de `_huber` : bornée à 2·delta, donc jamais de force qui panique."""
    return np.where(np.abs(z) <= delta, 2.0 * z, 2.0 * delta * np.sign(z))


@dataclass
class Spec:
    n_bands: int
    f_lo: float
    f_hi: float
    target_db: np.ndarray  # gabarit de répartition, en dB (somme des parts = 1)
    sigma_db: np.ndarray  # dispersion par bande, en dB
    weight: np.ndarray  # poids par bande
    level_db: float
    level_sigma: float
    w_tilt: float
    w_mask: float
    w_level: float
    w_anchor: float
    anchor_sigma: float
    tilt_huber: float
    fitted: bool
    n_tracks: int

    @property
    def edges(self) -> np.ndarray:
        return band_edges(self.n_bands, self.f_lo, self.f_hi)

    @property
    def centers(self) -> np.ndarray:
        e = self.edges
        return np.sqrt(e[:-1] * e[1:])


def load_spec(path: pathlib.Path | None = None) -> Spec:
    d = json.loads((path or SPEC_PATH).read_text())
    b, t, w = d["bands"], d["target"], d["weights"]
    return Spec(
        n_bands=int(b["n"]), f_lo=float(b["fLo"]), f_hi=float(b["fHi"]),
        target_db=np.asarray(t["db"], dtype=np.float64),
        sigma_db=np.asarray(t["sigma"], dtype=np.float64),
        weight=np.asarray(t["weight"], dtype=np.float64),
        level_db=float(t["levelDb"]), level_sigma=float(t["levelSigma"]),
        w_tilt=float(w["tilt"]), w_mask=float(w["mask"]), w_level=float(w["level"]),
        w_anchor=float(w.get("anchor", 0.0)), anchor_sigma=float(w.get("anchorSigma", 0.25)),
        tilt_huber=float(w.get("tiltHuber", 3.0)),
        fitted=bool(t.get("fitted", False)), n_tracks=int(t.get("nTracks", 0)),
    )


class Field:
    """Potentiel de gravité et gradient, pour un jeu de tranches donné."""

    def __init__(self, spec: Spec | None = None):
        self.spec = spec or load_spec()

    # ── Potentiel ───────────────────────────────────────────────────────────
    def potential(
        self,
        P: np.ndarray,
        g: np.ndarray,
        audible: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float]]:
        """V(g) et le détail de ses termes.

        P : (n_ch, n_bands) puissances PRÉ-fader.
        g : (n_ch,) positions de fader 0..1.
        audible : (n_ch,) 0/1 — mute et solo.
        anchor : (n_ch,) positions posées par la main ; None = pas de rappel.
        """
        s = self.spec
        P = np.asarray(P, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        a = np.ones(len(g)) if audible is None else np.asarray(audible, dtype=np.float64)

        u = (g * a) ** 2  # contribution en puissance de chaque tranche
        M = u @ P + EPS  # (n_bands,) mix
        S = float(M.sum())

        # 1. Gabarit spectral, en z-score par bande.
        s_db = DB_VALUE * np.log10(M / S)
        d = (s_db - s.target_db) / s.sigma_db
        w_eff = self._band_weights(P)
        v_tilt = float((w_eff * _huber(d, s.tilt_huber)).sum() / s.n_bands)

        # 2. Masquage entre tranches audibles.
        v_mask, _ = self._mask_term(P, u)

        # 3. Niveau.
        level = DB_VALUE * np.log10(S)
        v_level = float(((level - s.level_db) / s.level_sigma) ** 2)

        # 4. Rappel vers l'intention de la main.
        # ⚠️ SOMME et non moyenne. Chaque tranche paie son propre écart en
        # entier : moyenner le divisait par le nombre de tranches, alors que le
        # terme spectral, lui, agit sur toutes les bandes à la fois. Le rappel
        # perdait donc l'arbitrage 6 contre 1 et deux faders tombaient de 0,75 à
        # 0,01 malgré l'ancrage (mesuré le 2026-08-04).
        v_anchor = 0.0
        if anchor is not None and s.w_anchor:
            dg = (g - np.asarray(anchor, dtype=np.float64)) / s.anchor_sigma
            v_anchor = float((dg**2).sum())

        total = (
            s.w_tilt * v_tilt + s.w_mask * v_mask + s.w_level * v_level + s.w_anchor * v_anchor
        )
        return total, {
            "tilt": v_tilt, "mask": v_mask, "level": v_level, "anchor": v_anchor,
            "levelDb": float(level), "total": float(total),
        }

    def _band_weights(self, P: np.ndarray) -> np.ndarray:
        """Poids par bande, pondérés par la matière RÉELLEMENT disponible.

        Le champ ne doit juger que ce sur quoi les faders ont prise. Pendant un
        break au piano, la basse et la batterie ne jouent pas : aucune position
        de fader ne peut ramener du 45 Hz, et pénaliser son absence revient à
        demander à un fader de créer un signal qui n'existe pas.

        On compare donc la répartition atteignable à PLEIN GAIN au gabarit :
        une bande dix fois sous ce que le gabarit attend ne pèse qu'un dixième.

        ⚠️ Ce poids ne dépend PAS des gains (il est mesuré à g = 1), donc il est
        constant vis-à-vis de la dérivation : le gradient garde exactement la
        même forme, on ne fait que repondérer les bandes.
        """
        s = self.spec
        avail = P.sum(axis=0) + EPS
        share = avail / avail.sum()
        target_lin = 10.0 ** (s.target_db / DB_VALUE)
        return s.weight * np.minimum(1.0, share / target_lin)

    def _shapes(self, P: np.ndarray) -> np.ndarray:
        """Forme spectrale de chaque tranche : distribution normalisée, sans gain."""
        tot = P.sum(axis=1, keepdims=True)
        return P / np.maximum(tot, EPS)

    def _mask_term(self, P: np.ndarray, u: np.ndarray) -> tuple[float, np.ndarray]:
        """Terme de masquage et son gradient par rapport à l'énergie `e_i`.

        Recouvrement = intersection d'histogrammes des formes spectrales (donc
        indépendant des gains), pondéré par la moyenne géométrique des énergies
        et normalisé par l'énergie totale.
        """
        q = self._shapes(P)
        e = u * P.sum(axis=1)  # énergie effective de chaque tranche
        n = len(e)

        # O = recouvrement par paire, calculé une fois (ne dépend pas de g).
        O = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                o = float(np.minimum(q[i], q[j]).sum())
                O[i, j] = O[j, i] = o

        D = float(e.sum()) + EPS
        root = np.sqrt(np.maximum(e, 0.0) + EPS)
        # N = somme_{i<j} O_ij sqrt(e_i e_j)
        N = 0.5 * float(root @ O @ root)
        v = N / D

        # dN/de_i = (1/2) (1/sqrt(e_i)) * somme_j O_ij sqrt(e_j)
        dN = 0.5 * (O @ root) / root
        dv_de = (dN * D - N) / (D * D)
        return v, dv_de

    # ── Gradient ────────────────────────────────────────────────────────────
    def gradient(
        self,
        P: np.ndarray,
        g: np.ndarray,
        audible: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        """dV/dg — dérivé à la main, vérifié par différences finies dans les tests."""
        s = self.spec
        P = np.asarray(P, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        a = np.ones(len(g)) if audible is None else np.asarray(audible, dtype=np.float64)

        u = (g * a) ** 2
        du_dg = 2.0 * g * a**2  # du_i/dg_i
        M = u @ P + EPS
        S = float(M.sum())

        # ── dV_tilt/dM ──
        s_db = DB_VALUE * np.log10(M / S)
        d = (s_db - s.target_db) / s.sigma_db
        w_eff = self._band_weights(P)
        c = (1.0 / s.n_bands) * w_eff * _dhuber(d, s.tilt_huber) / s.sigma_db  # dV_tilt/ds_db
        # ds_db[k]/dM[j] = DB_SCALE * (delta_kj/M[k] - 1/S)
        dtilt_dM = DB_SCALE * (c / M - c.sum() / S)

        # ── dV_level/dM ──
        level = DB_VALUE * np.log10(S)
        dlevel_dM = 2.0 * (level - s.level_db) / (s.level_sigma**2) * DB_SCALE / S

        dV_dM = s.w_tilt * dtilt_dM + s.w_level * np.full_like(M, dlevel_dM)

        # ── chaîne vers les gains ──
        # dV/du_i = somme_b dV/dM[b] * P[i, b]   (+ masquage, via e_i)
        dV_du = P @ dV_dM

        _, dmask_de = self._mask_term(P, u)
        dV_du += s.w_mask * dmask_de * P.sum(axis=1)  # e_i = u_i * somme_b P[i,b]

        grad = dV_du * du_dg

        if anchor is not None and s.w_anchor:
            a_ref = np.asarray(anchor, dtype=np.float64)
            grad = grad + s.w_anchor * 2.0 * (g - a_ref) / (s.anchor_sigma**2)
        return grad

    def force(
        self,
        P: np.ndarray,
        g: np.ndarray,
        audible: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        """La gravité : `-dV/dg`, ce qui pousse chaque fader vers un meilleur mix."""
        return -self.gradient(P, g, audible, anchor)
