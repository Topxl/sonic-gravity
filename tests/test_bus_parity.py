"""Does the browser's bus behave exactly like the one the model trained in?

The recurrent model learned the dynamics of `bus.py`. If `buscomp.js` differs —
different detector rate, different filter, a step where there should be a ramp —
then the browser is a different world, and a model running there is being asked
a question it was never trained on. Nothing would crash; the console would just
be subtly wrong.

So the two are compared sample for sample on the same input.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from sonic_gravity.bus import CROSSOVERS, BusComp, apply, apply_multiband, static_curve

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUS_JS = ROOT / "sonic_gravity" / "web" / "js" / "buscomp.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node absent")

SR = 44100

BRIDGE = """
import { MultibandBus, staticCurve, butter2 } from %(mod)s;
const input = JSON.parse(await new Promise((res) => {
  let s = ""; process.stdin.on("data", (d) => (s += d)); process.stdin.on("end", () => res(s));
}));

if (input.mode === "curve") {
  process.stdout.write(JSON.stringify(input.levels.map((l) => staticCurve(l, input.comp))));
} else if (input.mode === "coeffs") {
  process.stdout.write(JSON.stringify(butter2(input.fc, input.sr, input.hp)));
} else {
  const bus = new MultibandBus(input.sr, input.bands, input.crossovers, 1);
  const x = Float32Array.from(input.x);
  const out = new Float32Array(x.length);
  // Feed it the way an AudioWorklet would: 128 samples at a time.
  const B = input.blockSize;
  const tmp = new Float32Array(B);
  for (let off = 0; off < x.length; off += B) {
    const n = Math.min(B, x.length - off);
    bus.process([x.subarray(off, off + n)], [tmp], n);
    out.set(tmp.subarray(0, n), off);
  }
  process.stdout.write(JSON.stringify(Array.from(out)));
}
"""


def _js(payload: dict):
    script = BRIDGE % {"mod": json.dumps(BUS_JS.as_uri())}
    proc = subprocess.run(["node", "--input-type=module", "-e", script],
                          input=json.dumps(payload), capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        raise AssertionError(f"node failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


BANDS_PY = [
    BusComp(threshold_db=-26, ratio=8, attack_ms=5, release_ms=700),
    BusComp(threshold_db=-24, ratio=4, attack_ms=15, release_ms=250),
    BusComp(threshold_db=-30, ratio=3, attack_ms=3, release_ms=80),
]
BANDS_JS = [
    {"thresholdDb": -26, "ratio": 8, "attackMs": 5, "releaseMs": 700, "kneeDb": 6},
    {"thresholdDb": -24, "ratio": 4, "attackMs": 15, "releaseMs": 250, "kneeDb": 6},
    {"thresholdDb": -30, "ratio": 3, "attackMs": 3, "releaseMs": 80, "kneeDb": 6},
]


def test_static_curve_agrees():
    levels = np.linspace(-60, 0, 200)
    comp = BANDS_PY[0]
    got = np.asarray(_js({"mode": "curve", "levels": levels.tolist(), "comp": BANDS_JS[0]}))
    assert np.allclose(got, static_curve(levels, comp), atol=1e-9)


@pytest.mark.parametrize("fc,hp", [(200, False), (200, True), (2000, False), (2000, True)])
def test_filter_coefficients_agree_with_scipy(fc: int, hp: bool):
    from scipy.signal import butter

    sos = butter(2, fc / (SR / 2), btype="high" if hp else "low", output="sos")[0]
    js = _js({"mode": "coeffs", "fc": fc, "sr": SR, "hp": hp})
    assert np.allclose([sos[0], sos[1], sos[2], sos[4], sos[5]],
                       [js["b0"], js["b1"], js["b2"], js["a1"], js["a2"]], atol=1e-12)


@pytest.mark.parametrize("block_size", [128, 256])
def test_multiband_output_matches_python(block_size: int):
    """The whole chain, fed the way a worklet feeds it.

    Block size is varied on purpose: the worklet hands over 128 samples, the
    detector runs on 32, and a mirror that quietly tied its detector to the
    audio block size would pass at one size and fail at the other.
    """
    rng = np.random.default_rng(0)
    n = SR // 2
    t = np.arange(n) / SR
    x = (0.8 * np.sin(2 * np.pi * 55 * t) * np.exp(-((t % 0.5) * 8))
         + 0.25 * np.sin(2 * np.pi * 700 * t)
         + 0.1 * rng.standard_normal(n)).astype(np.float32)

    want = apply_multiband(x.astype(np.float64), SR, BANDS_PY)
    got = np.asarray(_js({
        "mode": "bus", "sr": SR, "x": x.tolist(), "bands": BANDS_JS,
        "crossovers": list(CROSSOVERS), "blockSize": block_size,
    }))

    # float32 in the browser against float64 here, through cascaded IIR
    # filters — agreement is to single precision, not to the last bit.
    rms = float(np.sqrt((want**2).mean()))
    err = float(np.sqrt(((got - want) ** 2).mean()))
    assert err < 2e-3 * rms, f"mirror diverges: {err:.3e} vs signal rms {rms:.3e}"
    assert np.corrcoef(got, want)[0, 1] > 0.9999


def test_wideband_single_band_also_matches():
    """The one-band path is what the analytic dataset used; keep it honest too."""
    rng = np.random.default_rng(3)
    n = SR // 4
    x = (0.5 * np.sin(2 * np.pi * 110 * np.arange(n) / SR)
         + 0.2 * rng.standard_normal(n)).astype(np.float32)
    comp = BusComp(threshold_db=-22, ratio=6, attack_ms=10, release_ms=150)

    want = apply(x.astype(np.float64), SR, comp)
    got = np.asarray(_js({
        "mode": "bus", "sr": SR, "x": x.tolist(),
        "bands": [{"thresholdDb": -22, "ratio": 6, "attackMs": 10,
                   "releaseMs": 150, "kneeDb": 6}],
        "crossovers": [], "blockSize": 128,
    }))
    rms = float(np.sqrt((want**2).mean()))
    assert float(np.sqrt(((got - want) ** 2).mean())) < 2e-3 * rms
