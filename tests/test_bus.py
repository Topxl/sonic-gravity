"""The bus compressor defines what the world remembers, so it has to be right.

Everything about the temporal side of this project rests on this file: if the
compressor does not actually depend on the recent past, there is no memory to
model and any recurrent result would be measuring noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from sonic_gravity.bus import (
    CROSSOVERS,
    BusComp,
    apply,
    apply_multiband,
    gain_reduction,
    static_curve,
)

SR = 44100


def test_static_curve_is_continuous_and_monotone():
    comp = BusComp(threshold_db=-18.0, ratio=4.0, knee_db=6.0)
    lv = np.linspace(-60, 0, 2000)
    gr = static_curve(lv, comp)

    assert np.all(gr <= 1e-12), "a compressor never adds gain"
    assert np.all(np.diff(gr) <= 1e-9), "more level must never mean less reduction"
    # No step at either end of the knee — the whole system differentiates this.
    assert np.abs(np.diff(gr)).max() < 0.05
    # Well below threshold, nothing happens; well above, the slope is 1 − 1/ratio.
    assert static_curve(np.array([-40.0]), comp)[0] == pytest.approx(0.0, abs=1e-9)
    slope = (static_curve(np.array([-6.0]), comp) - static_curve(np.array([-8.0]), comp))[0] / 2
    assert slope == pytest.approx(-(1 - 1 / comp.ratio), rel=1e-6)


def test_knee_has_no_kink_in_its_derivative():
    """A hard corner sits exactly where a bus compressor lives, and the force
    sent to the faders is a derivative."""
    comp = BusComp(knee_db=6.0)
    lv = np.linspace(-30, -6, 4000)
    d2 = np.diff(static_curve(lv, comp), n=2)
    assert np.abs(d2).max() < 1e-3, "second derivative spikes: the knee is not smooth"


def test_attack_is_faster_than_release():
    """The asymmetry IS the memory: without it the past would not matter."""
    comp = BusComp(threshold_db=-30.0, ratio=8.0, attack_ms=5.0, release_ms=400.0)
    n = int(SR * 1.2)
    x = np.zeros(n)
    x[: n // 3] = 0.9  # loud, then silence
    gr = gain_reduction(x, SR, comp)

    onset = gr[: len(gr) // 3]
    tail = gr[len(gr) // 3 :]
    # Clamps down quickly…
    assert onset.min() < -5, f"barely compressed: {onset.min():.2f} dB"
    settle = np.argmax(onset < 0.9 * onset.min())
    # …and lets go slowly: still recovering well after the signal stopped.
    assert tail[0] < -1, "released instantly — no memory at all"
    assert settle < len(tail) // 2, "attack should be much shorter than release"


def test_the_mix_depends_on_its_recent_past():
    """Same instantaneous input, different history → different output.

    This is the property a recurrent model would exist to capture. If it did not
    hold, the temporal experiment would be measuring nothing.
    """
    comp = BusComp(threshold_db=-28.0, ratio=6.0, attack_ms=5.0, release_ms=400.0)
    n = int(SR * 0.5)
    tone = 0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)

    quiet_before = np.concatenate([np.zeros(n), tone])
    loud_before = np.concatenate([0.95 * np.ones(n) * np.sin(
        2 * np.pi * 60 * np.arange(n) / SR), tone])

    gr_q = gain_reduction(quiet_before, SR, comp)
    gr_l = gain_reduction(loud_before, SR, comp)
    # Look at the same instant of the SAME tone, reached two different ways.
    at = len(gr_q) // 2 + 2
    assert abs(gr_q[at] - gr_l[at]) > 1.0, (
        f"history left no trace: {gr_q[at]:.2f} vs {gr_l[at]:.2f} dB"
    )


def test_multiband_split_reconstructs_the_signal():
    """Linkwitz-Riley bands must sum back to the original.

    A plain Butterworth pair leaves a bump at each crossover, and the field
    would read that artefact as a spectral defect the mix does not have.
    """
    from sonic_gravity.bus import _split

    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR // 2) * 0.1
    total = sum(_split(x, SR))
    # Crossovers are phase-shifting, so compare energy rather than samples.
    e_in = float(np.sqrt((x**2).mean()))
    e_out = float(np.sqrt((total**2).mean()))
    assert e_out == pytest.approx(e_in, rel=0.12), f"{e_in:.4f} → {e_out:.4f}"


def test_multiband_changes_the_SHAPE_where_wideband_does_not():
    """The reason multiband exists in this project, as a measurement.

    A wideband bus moves the LEVEL (−18 dB on a typical mix) but barely the
    shape (0.16 dB). The potential consumes only the shape, so a wideband bus
    is nearly invisible to the field — and any temporal effect stays buried
    under the model's own error. Per-band detectors change that by an order of
    magnitude.
    """
    from sonic_gravity.field import (
        band_edges,
        band_gather,
        bands_from_gather,
        load_spec,
    )

    spec = load_spec()
    fft = 8192
    gather = band_gather(fft // 2, SR, fft, band_edges(spec.n_bands, spec.f_lo, spec.f_hi))
    win = np.hanning(fft)

    def shape_db(x):
        p = bands_from_gather(np.abs(np.fft.rfft(x[:fft] * win)[: fft // 2]) ** 2, gather) + 1e-12
        return 10 * np.log10(p / p.sum())

    rng = np.random.default_rng(0)
    t = np.arange(fft) / SR
    x = (0.9 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 6)
         + 0.3 * np.sin(2 * np.pi * 440 * t)
         + 0.15 * rng.standard_normal(fft))

    wide = apply(x, SR, BusComp(threshold_db=-22, ratio=6, release_ms=600))
    multi = apply_multiband(x, SR, [
        BusComp(threshold_db=-26, ratio=8, attack_ms=5, release_ms=700),
        BusComp(threshold_db=-24, ratio=4, attack_ms=15, release_ms=250),
        BusComp(threshold_db=-30, ratio=3, attack_ms=3, release_ms=80),
    ])

    ref = shape_db(x)
    d_wide = np.abs(shape_db(wide) - ref).mean()
    d_multi = np.abs(shape_db(multi) - ref).mean()
    assert d_wide < 0.5, f"wideband should barely touch the shape, got {d_wide:.3f} dB"
    assert d_multi > 5 * d_wide, f"multiband {d_multi:.3f} vs wideband {d_wide:.3f} dB"


def test_crossovers_are_ordered_and_inside_the_audible_range():
    assert list(CROSSOVERS) == sorted(CROSSOVERS)
    assert 20 < CROSSOVERS[0] and CROSSOVERS[-1] < SR / 2


def test_silence_is_left_alone():
    comp = BusComp()
    x = np.zeros(SR // 10)
    assert np.allclose(apply(x, SR, comp), 0.0)
    assert np.all(np.isfinite(gain_reduction(x, SR, comp)))
