"""Bus compressor — the reason the world has a memory at all.

Without it there is nothing temporal to model. Weighted summation is
instantaneous: the mix spectrum at time t depends only on the faders at time t,
and a recurrent model would have nothing to remember.

A compressor changes that completely. Its gain reduction depends on what the
signal did *milliseconds ago*, so:

  • lowering one fader changes the reduction, which changes **every** channel's
    contribution — the faders stop being independent;
  • the same fader position sounds different depending on how it was reached;
  • a kick can duck the whole mix for 100 ms after it has stopped.

That is pumping, and it is central to how records are actually mixed. It is
also precisely the kind of state an action-conditioned world model exists to
carry.

Deliberately a **textbook feed-forward peak compressor**, not an emulation of
`DynamicsCompressorNode`: this file is the reference the browser must mirror
(same pattern as the field), and a spec no one can read from the source would
guarantee the two drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class BusComp:
    """Bus compressor settings, in the units engineers actually use."""

    threshold_db: float = -18.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 150.0
    knee_db: float = 6.0
    makeup_db: float = 0.0

    def coeffs(self, sr: float) -> tuple[float, float]:
        """One-pole smoothing coefficients for attack and release.

        The usual convention: the time constant is how long the detector takes
        to cover 1 − 1/e of the distance to its target.
        """
        a = 1.0 - np.exp(-1.0 / max(self.attack_ms * 1e-3 * sr, 1.0))
        r = 1.0 - np.exp(-1.0 / max(self.release_ms * 1e-3 * sr, 1.0))
        return float(a), float(r)


def static_curve(level_db: np.ndarray, comp: BusComp) -> np.ndarray:
    """Gain reduction in dB (≤ 0) for a given input level, with a soft knee.

    The knee matters here beyond taste: a hard corner makes the derivative
    discontinuous exactly where a bus compressor spends most of its time, and
    the whole system is built on a gradient.
    """
    over = level_db - comp.threshold_db
    slope = 1.0 - 1.0 / comp.ratio
    k = max(comp.knee_db, 1e-9)

    below = over <= -k / 2
    above = over >= k / 2
    # Quadratic blend across the knee, matching value and slope at both ends.
    knee = slope * (over + k / 2) ** 2 / (2 * k)
    gr = np.where(below, 0.0, np.where(above, slope * over, knee))
    return -gr


def gain_reduction(x: np.ndarray, sr: float, comp: BusComp, block: int = 32) -> np.ndarray:
    """Per-BLOCK gain reduction in dB, from peak detection with attack/release.

    The detector runs at block rate (32 samples ≈ 0.7 ms at 44.1 kHz), not per
    sample. That is not an approximation of convenience — real compressors,
    `DynamicsCompressorNode` included, detect on blocks. It also turns a
    66 000-iteration Python loop into a 2 000-iteration one, which is the
    difference between a dataset that builds in minutes and one that does not
    build at all.

    Returns one value per block; `apply` interpolates back to sample rate so no
    step discontinuity is introduced into the audio.
    """
    n_blocks = max(1, len(x) // block)
    trimmed = x[: n_blocks * block].reshape(n_blocks, block)
    peak = np.abs(trimmed).max(axis=1)
    level_db = 20.0 * np.log10(peak + EPS)
    target = static_curve(level_db, comp)

    # Coefficients at block rate: the time constants are in seconds, so the
    # effective sample rate of the detector is sr / block.
    a, r = comp.coeffs(sr / block)

    gr = np.empty(n_blocks)
    state = 0.0
    for n in range(n_blocks):
        t = target[n]
        # Attack while clamping down, release while letting go — the asymmetry
        # is the whole reason the mix depends on its recent past.
        state += (a if t < state else r) * (t - state)
        gr[n] = state
    return gr


def apply(x: np.ndarray, sr: float, comp: BusComp, block: int = 32) -> np.ndarray:
    """Compress a mono signal.

    The gain ramps linearly from the previous block's reduction to this one's,
    across the block. A per-block step would put broadband clicks straight into
    the spectrum being measured.

    ⚠️ A ramp, not an interpolation between block CENTRES. Both are smooth, but
    only the ramp can be reproduced by a real-time processor, which cannot see
    the next block's reduction before computing this block. The browser mirror
    (`web/js/buscomp.js`) has to match this exactly, so the reference is the
    causal version.
    """
    gr = gain_reduction(x, sr, comp, block)
    n_blocks = len(gr)
    gr_full = np.empty(len(x))
    prev = 0.0
    for n in range(n_blocks):
        a, b = n * block, min((n + 1) * block, len(x))
        gr_full[a:b] = np.linspace(prev, gr[n], b - a, endpoint=False)
        prev = gr[n]
    gr_full[n_blocks * block :] = prev
    return x * 10.0 ** ((gr_full + comp.makeup_db) / 20.0)


# ── Multiband ───────────────────────────────────────────────────────────────
# A wideband compressor moves the LEVEL, not the shape: measured on a typical
# mix it shifts level by −18 dB while changing the spectrum's shape by 0.157 dB
# on average. The field consumes only the shape, so a wideband bus is nearly
# invisible to it — and any temporal effect is buried six times under the
# model's own residual error.
#
# A multiband bus is different in kind. Each band's gain reduction follows its
# OWN history, so the spectrum's shape becomes genuinely time-dependent: a kick
# ducks the low band for hundreds of milliseconds while the top stays open. That
# is both what mastering chains actually do and the only version of this world
# where memory can possibly pay for itself.

CROSSOVERS = (200.0, 2000.0)


def _split(x: np.ndarray, sr: float, crossovers=CROSSOVERS) -> list[np.ndarray]:
    """Split into bands with 4th-order Linkwitz-Riley crossovers.

    Linkwitz-Riley because the bands must sum back to the original when no
    compression is applied; a plain Butterworth pair leaves a peak at each
    crossover that the field would read as a spectral defect that isn't there.
    """
    from scipy.signal import butter, sosfilt

    bands = []
    prev = x
    for fc in crossovers:
        lo_sos = butter(2, fc / (sr / 2), btype="low", output="sos")
        hi_sos = butter(2, fc / (sr / 2), btype="high", output="sos")
        # Cascading the same 2nd-order section twice gives Linkwitz-Riley 4th.
        low = sosfilt(lo_sos, sosfilt(lo_sos, prev))
        high = sosfilt(hi_sos, sosfilt(hi_sos, prev))
        bands.append(low)
        prev = high
    bands.append(prev)
    return bands


def apply_multiband(x: np.ndarray, sr: float, comps: list[BusComp],
                    crossovers=CROSSOVERS, block: int = 32) -> np.ndarray:
    """Compress each band on its own detector, then sum."""
    bands = _split(x, sr, crossovers)
    assert len(bands) == len(comps), "one compressor per band"
    return sum(apply(b, sr, c, block) for b, c in zip(bands, comps))
