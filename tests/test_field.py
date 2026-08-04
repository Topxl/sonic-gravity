"""Le gradient du champ de gravité est-il le vrai gradient du potentiel ?

C'est la seule vérification qui compte pour ce module : la force envoyée aux
faders est `-dV/dg`. Une erreur de signe ou de facteur ne casse aucun test de
forme — elle pousse simplement la main de l'utilisateur vers le mauvais mix,
en silence. On compare donc la dérivée écrite à la main à des différences
finies centrées, sur des configurations tirées au hasard.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sonic_gravity.field import (
    EPS,
    SPEC_PATH,
    Field,
    band_edges,
    band_gather,
    bands_from_gather,
    load_spec,
)

N_BANDS = load_spec().n_bands


def _random_scene(rng: np.random.Generator, n_ch: int = 6, n_bands: int = N_BANDS):
    """Des tranches au spectre plausible : chacune concentrée autour d'une zone."""
    centers = rng.uniform(0, n_bands, size=n_ch)
    widths = rng.uniform(3.0, 12.0, size=n_ch)
    b = np.arange(n_bands)
    P = np.stack([
        np.exp(-0.5 * ((b - c) / w) ** 2) * 10 ** rng.uniform(-3, -1)
        for c, w in zip(centers, widths)
    ])
    P += 1e-9  # aucun stem n'est parfaitement vide hors de sa zone
    g = rng.uniform(0.15, 0.95, size=n_ch)
    return P, g


def _finite_diff(field: Field, P, g, audible, anchor=None, h: float = 1e-6) -> np.ndarray:
    out = np.zeros_like(g)
    for i in range(len(g)):
        gp, gm = g.copy(), g.copy()
        gp[i] += h
        gm[i] -= h
        vp, _ = field.potential(P, gp, audible, anchor)
        vm, _ = field.potential(P, gm, audible, anchor)
        out[i] = (vp - vm) / (2 * h)
    return out


@pytest.mark.parametrize("seed", range(12))
def test_gradient_matches_finite_differences(seed: int):
    rng = np.random.default_rng(seed)
    field = Field()
    P, g = _random_scene(rng)

    got = field.gradient(P, g)
    want = _finite_diff(field, P, g, None)

    # Échelle de référence : le gradient lui-même. On veut 4 chiffres, ce qui
    # est bien au-delà de ce dont la boucle temps réel a besoin, mais ce qui
    # attrape toute erreur de dérivation.
    scale = max(float(np.abs(want).max()), 1e-9)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4 * scale), (
        f"analytique {got}\nfinies     {want}"
    )


@pytest.mark.parametrize("seed", range(6))
def test_gradient_with_muted_channels(seed: int):
    """Une tranche coupée ne reçoit aucune force : elle ne contribue pas au mix."""
    rng = np.random.default_rng(100 + seed)
    field = Field()
    P, g = _random_scene(rng)
    audible = np.ones(len(g))
    audible[rng.integers(0, len(g))] = 0.0
    audible[rng.integers(0, len(g))] = 0.0

    got = field.gradient(P, g, audible)
    want = _finite_diff(field, P, g, audible)
    scale = max(float(np.abs(want).max()), 1e-9)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4 * scale)
    assert np.all(np.abs(got[audible == 0]) < 1e-12)


def test_force_is_opposite_of_gradient():
    rng = np.random.default_rng(7)
    field = Field()
    P, g = _random_scene(rng)
    assert np.allclose(field.force(P, g), -field.gradient(P, g))


def test_descending_the_gradient_lowers_the_potential():
    """Suivre la force doit améliorer le mix — sinon la surface ment à la main."""
    rng = np.random.default_rng(3)
    field = Field()
    P, g = _random_scene(rng)

    v0, _ = field.potential(P, g)
    step = 0.02 / max(float(np.abs(field.gradient(P, g)).max()), 1e-9)
    for _ in range(25):
        g = np.clip(g + step * field.force(P, g), 0.0, 1.0)
    v1, _ = field.potential(P, g)
    assert v1 < v0


def test_mask_term_is_scale_invariant():
    """Monter tous les faders ensemble ne change pas le masquage.

    C'est ce qui fait que la gravité dit « baisse CELUI qui se bat », et jamais
    « baisse tout » : sans cette invariance, le terme de masquage aurait un
    minimum trivial au silence.
    """
    rng = np.random.default_rng(11)
    field = Field()
    P, g = _random_scene(rng)
    u = g**2
    v_a, _ = field._mask_term(P, u)
    v_b, _ = field._mask_term(P, u * 4.0)
    assert v_a == pytest.approx(v_b, rel=1e-9)


def test_identical_sources_mask_more_than_disjoint_ones():
    """Deux tranches au même spectre se masquent ; deux tranches disjointes non."""
    field = Field()
    n = field.spec.n_bands
    b = np.arange(n)
    low = np.exp(-0.5 * ((b - 6) / 3.0) ** 2) + 1e-9
    high = np.exp(-0.5 * ((b - (n - 5)) / 3.0) ** 2) + 1e-9

    same = np.stack([low, low.copy()])
    apart = np.stack([low, high])
    u = np.array([0.5, 0.5])

    v_same, _ = field._mask_term(same, u)
    v_apart, _ = field._mask_term(apart, u)
    assert v_same > 4 * v_apart


@pytest.mark.parametrize(
    "sample_rate,fft_size",
    [(48000.0, 8192), (44100.0, 8192), (48000.0, 4096), (44100.0, 2048)],
)
def test_no_band_is_ever_empty(sample_rate: float, fft_size: int):
    """AUCUNE bande ne doit rester vide, à aucune résolution.

    Régression du 2026-08-04. Une bande plus étroite que la résolution de la
    FFT ne recevait aucun bin, tombait au plancher, et le gabarit mesuré sur
    220 morceaux portait un −173,9 dB à 37 Hz. Comme le z-score divise par σ,
    cette bande fantôme devenait un puits qui écrasait les 39 autres dès que le
    mix avait la moindre énergie dans le grave.

    Le test balaie aussi des résolutions volontairement trop grossières : c'est
    précisément là que le repli doit tenir, et l'ancienne version tolérait les
    trous « tant qu'ils sont dans le grave ». C'est cette tolérance qui a laissé
    passer le défaut.
    """
    spec = load_spec()
    edges = band_edges(spec.n_bands, spec.f_lo, spec.f_hi)
    gather = band_gather(fft_size // 2, sample_rate, fft_size, edges)

    assert len(gather) == spec.n_bands
    empty = [b for b, idx in enumerate(gather) if idx.size == 0]
    assert not empty, f"bandes sans aucun bin : {empty}"

    flat = np.ones((1, fft_size // 2))
    got = bands_from_gather(flat, gather)[0]
    assert got.min() > 0, f"bandes à zéro : {np.where(got == 0)[0]}"


def test_spec_bands_fit_the_analysis_resolution():
    """La bande la plus grave doit être plus large qu'un bin, sans compter sur le repli.

    Le repli est un filet, pas une méthode : s'il faut l'invoquer sur un tiers
    du spectre, c'est que le découpage en bandes ne correspond pas à la FFT
    choisie, et les bandes graves deviennent alors des copies les unes des
    autres — donc un grave compté plusieurs fois dans le potentiel.
    """
    d = json.loads(SPEC_PATH.read_text())
    fft_size = int(d.get("analysis", {}).get("fftSize", 8192))
    spec = load_spec()
    edges = band_edges(spec.n_bands, spec.f_lo, spec.f_hi)

    bin_hz = 48000.0 / fft_size
    assert edges[1] - edges[0] > bin_hz, (
        f"bande la plus grave {edges[1] - edges[0]:.1f} Hz < résolution {bin_hz:.1f} Hz"
    )

    gather = band_gather(fft_size // 2, 48000.0, fft_size, edges)
    shared = sum(1 for idx in gather if idx.size == 1)
    assert shared <= spec.n_bands // 6, f"{shared} bandes ne tiennent qu'à un bin"


def test_potential_parts_sum_to_total():
    rng = np.random.default_rng(5)
    field = Field()
    P, g = _random_scene(rng)
    total, parts = field.potential(P, g)
    s = field.spec
    recomposed = s.w_tilt * parts["tilt"] + s.w_mask * parts["mask"] + s.w_level * parts["level"]
    assert recomposed == pytest.approx(total, rel=1e-12)


def test_silence_does_not_explode():
    """Tous les faders à zéro : le potentiel reste fini et le gradient aussi."""
    field = Field()
    P, _ = _random_scene(np.random.default_rng(1))
    g = np.zeros(P.shape[0])
    v, _ = field.potential(P, g)
    grad = field.gradient(P, g)
    assert np.isfinite(v)
    assert np.all(np.isfinite(grad))
    assert EPS > 0


# ── Ancrage : la gravité corrige autour de l'intention, elle ne la remplace pas ──


@pytest.mark.parametrize("seed", range(8))
def test_gradient_with_anchor_matches_finite_differences(seed: int):
    rng = np.random.default_rng(500 + seed)
    field = Field()
    P, g = _random_scene(rng)
    anchor = rng.uniform(0.2, 0.9, size=len(g))

    got = field.gradient(P, g, None, anchor)
    want = _finite_diff(field, P, g, None, anchor)
    scale = max(float(np.abs(want).max()), 1e-9)
    assert np.allclose(got, want, rtol=1e-4, atol=1e-4 * scale), (
        f"analytique {got}\nfinies     {want}"
    )


def test_anchor_keeps_the_optimum_off_the_end_stops():
    """Régression du 2026-08-04 : sans ancrage, les faders partaient en butée.

    Mesuré en direct sur Clocks : batterie 1,000, voix 0,997, « other » 0,118
    trois secondes après le démarrage. Le minimum du potentiel était un mix
    entièrement recalculé, ce qui rend l'outil inutilisable — on ne mixe plus,
    on regarde la machine mixer. Avec le rappel, la descente doit rester au
    voisinage de ce que la main a posé.
    """
    rng = np.random.default_rng(21)
    field = Field()
    P, _ = _random_scene(rng)
    anchor = np.full(P.shape[0], 0.75)

    g = anchor.copy()
    for _ in range(400):
        grad = field.gradient(P, g, None, anchor)
        step = 0.01 / max(float(np.abs(grad).max()), 1e-9)
        g = np.clip(g - step * grad, 0.0, 1.0)

    at_stops = int(np.sum((g <= 1e-6) | (g >= 1 - 1e-6)))
    assert at_stops == 0, f"{at_stops} faders en butée : {g}"
    assert np.abs(g - anchor).max() < 0.35, f"dérive trop loin de l'intention : {g - anchor}"


def test_anchor_still_allows_a_real_correction():
    """Le rappel ne doit pas figer la console : une tranche fautive doit bouger.

    Sans cette contrepartie, il suffirait de mettre le poids d'ancrage très
    haut pour que tous les tests passent — et la gravité ne servirait à rien.
    """
    field = Field()
    n = field.spec.n_bands
    b = np.arange(n)
    # Deux tranches au spectre identique : elles se masquent l'une l'autre.
    twin = np.exp(-0.5 * ((b - 14) / 4.0) ** 2) + 1e-9
    other = np.exp(-0.5 * ((b - 30) / 5.0) ** 2) + 1e-9
    P = np.stack([twin, twin.copy(), other])
    anchor = np.full(3, 0.75)

    g = anchor.copy()
    for _ in range(400):
        grad = field.gradient(P, g, None, anchor)
        step = 0.01 / max(float(np.abs(grad).max()), 1e-9)
        g = np.clip(g - step * grad, 0.0, 1.0)

    assert np.abs(g - anchor).max() > 0.02, "l'ancrage a figé la console"


def test_anchor_absent_behaves_as_before():
    """`anchor=None` doit rendre exactement le champ sans rappel."""
    rng = np.random.default_rng(33)
    field = Field()
    P, g = _random_scene(rng)
    v_none, parts = field.potential(P, g, None, None)
    v_zero, _ = field.potential(P, g)
    assert v_none == pytest.approx(v_zero, rel=1e-12)
    assert parts["anchor"] == 0.0


# ── Huber : ne pas laisser un break dominer le champ ────────────────────────


def test_huber_is_continuous_and_reduces_to_the_square():
    from sonic_gravity.field import _dhuber, _huber

    z = np.linspace(-8, 8, 401)
    d = 3.0
    h = _huber(z, d)
    # Continuité à la jonction, dans les deux sens.
    assert _huber(np.array([d]), d)[0] == pytest.approx(d**2)
    assert _huber(np.array([-d]), d)[0] == pytest.approx(d**2)
    # Près de zéro, c'est exactement le carré.
    small = np.linspace(-1, 1, 51)
    assert np.allclose(_huber(small, d), small**2)
    # Croissance linéaire au-delà, donc dérivée bornée.
    assert np.abs(_dhuber(z, d)).max() == pytest.approx(2 * d)
    # Monotone en |z| : un mix plus loin de la cible n'est jamais mieux noté.
    assert np.all(np.diff(h[z >= 0]) >= -1e-12)


def test_huber_gradient_matches_finite_differences_across_the_knee():
    """La jonction est le seul endroit où la dérivée pourrait être fausse."""
    from sonic_gravity.field import _dhuber, _huber

    d = 3.0
    for z0 in (-4.5, -3.05, -2.95, -0.5, 0.0, 2.95, 3.05, 4.5):
        h = 1e-7
        num = (_huber(np.array([z0 + h]), d)[0] - _huber(np.array([z0 - h]), d)[0]) / (2 * h)
        assert _dhuber(np.array([z0]), d)[0] == pytest.approx(num, abs=1e-4)


def test_a_missing_band_cannot_dominate_the_potential():
    """Régression du 2026-08-04 : un break au piano faisait monter V à 88.

    On simule le cas exactement : toutes les tranches muettes sauf une, dans
    le haut du spectre. Le grave est alors absent — non par mauvais mixage,
    mais parce que rien ne le joue. Le potentiel doit rester dans un ordre de
    grandeur exploitable, sans quoi la jauge ne veut plus rien dire et la force
    s'acharne sur des faders qui ne commandent rien.
    """
    field = Field()
    n = field.spec.n_bands
    b = np.arange(n)
    quiet = np.full(n, 1e-12)
    piano = np.exp(-0.5 * ((b - n * 0.6) / 4.0) ** 2) * 1e-2 + 1e-12
    P = np.stack([quiet, quiet.copy(), piano, quiet.copy()])
    g = np.full(4, 0.75)

    total, parts = field.potential(P, g)
    assert total < 12, f"un break domine encore le champ : V={total:.1f}"
    # Et la force reste bornée : pas de panique sur les faders muets.
    assert np.abs(field.force(P, g)).max() < 1e3
