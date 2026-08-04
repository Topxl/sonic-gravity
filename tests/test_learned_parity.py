"""Does the hand-written JavaScript backward pass match PyTorch's autograd?

The browser computes `∂V/∂g` through the learned model with a backward pass
written by hand. Nothing checks that by itself: a wrong path — or a forgotten
one — still produces a plausible-looking force that simply points somewhere
slightly wrong. The console would feel fine and mix worse.

So the JS forward is compared to the PyTorch forward, and the JS gradient to
`autograd`, on random inputs.

Skipped when node or torch is absent; it must run wherever both exist.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEARNED_JS = ROOT / "sonic_gravity" / "web" / "js" / "learned.js"
WEIGHTS = ROOT / "sonic_gravity" / "web" / "model.json"

pytestmark = [
    pytest.mark.skipif(shutil.which("node") is None, reason="node absent"),
    pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained model — run sonic_gravity.train"),
]

BRIDGE = """
import { LearnedMix } from %(mod)s;
const input = JSON.parse(await new Promise((res) => {
  let s = ""; process.stdin.on("data", (d) => (s += d)); process.stdin.on("end", () => res(s));
}));
const model = new LearnedMix(input.weights);
const P = input.P.map((r) => Float64Array.from(r));
const g = Float64Array.from(input.g);
const a = Float64Array.from(input.a);
const { shape, cache } = model.forward(P, g, a);
const dG = model.backward(cache, Float64Array.from(input.dShape));
process.stdout.write(JSON.stringify({ shape: Array.from(shape), dG: Array.from(dG) }));
"""


def _js(payload: dict) -> dict:
    script = BRIDGE % {"mod": json.dumps(LEARNED_JS.as_uri())}
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(payload), capture_output=True, text=True, timeout=90,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def _torch_model(weights: dict):
    """Rebuild the exported network in torch, from the JSON alone."""
    from sonic_gravity.model import MixWorldModel

    m = MixWorldModel(weights["nBands"], width=weights["width"],
                      hidden=weights["hidden"], p_law=weights["pLaw"])

    def load(lin, blob):
        lin.weight.data = torch.tensor(np.array(blob["W"]).T, dtype=torch.float32)
        lin.bias.data = torch.tensor(np.array(blob["b"]), dtype=torch.float32)

    load(m.enc[0], weights["enc"][0])
    load(m.enc[2], weights["enc"][1])
    load(m.dec[0], weights["dec"][0])
    load(m.dec[2], weights["dec"][1])
    load(m.dec[4], weights["dec"][2])
    return m.eval().double()


def _scene(rng, n_ch, n_bands):
    b = np.arange(n_bands)
    P = np.stack([
        np.exp(-0.5 * ((b - rng.uniform(0, n_bands)) / rng.uniform(3, 12)) ** 2)
        * 10 ** rng.uniform(-3, -1) + 1e-9
        for _ in range(n_ch)
    ])
    return P, rng.uniform(0.15, 0.95, size=n_ch)


@pytest.mark.parametrize("seed,n_ch", [(0, 6), (1, 4), (2, 8), (3, 5)])
def test_forward_and_backward_match_pytorch(seed: int, n_ch: int):
    weights = json.loads(WEIGHTS.read_text())
    n_bands = weights["nBands"]
    rng = np.random.default_rng(seed)
    P, g = _scene(rng, n_ch, n_bands)
    a = np.ones(n_ch)
    if seed % 2:
        a[rng.integers(0, n_ch)] = 0.0
    dShape = rng.normal(0, 1, size=n_bands)

    js = _js({"weights": weights, "P": P.tolist(), "g": g.tolist(),
              "a": a.tolist(), "dShape": dShape.tolist()})

    m = _torch_model(weights)
    Pt = torch.tensor(P, dtype=torch.float64).unsqueeze(0)
    # `requires_grad` must be set on the tensor that is actually a leaf: calling
    # unsqueeze afterwards makes it a view, and `.grad` then stays None.
    gt = torch.tensor(g[None, :], dtype=torch.float64, requires_grad=True)
    at = torch.tensor(a, dtype=torch.float64).unsqueeze(0)
    shape, _ = m(Pt, gt, at)
    shape.backward(torch.tensor(dShape, dtype=torch.float64).unsqueeze(0))

    want_shape = shape.detach().numpy()[0]
    got_shape = np.asarray(js["shape"])
    assert np.allclose(got_shape, want_shape, atol=1e-6), (
        f"forward diverges\njs {got_shape[:5]}\npy {want_shape[:5]}"
    )

    want_grad = gt.grad.numpy()[0]
    got_grad = np.asarray(js["dG"])
    scale = max(float(np.abs(want_grad).max()), 1e-9)
    assert np.allclose(got_grad, want_grad, rtol=1e-5, atol=1e-6 * scale), (
        f"gradient diverges\njs {got_grad}\npy {want_grad}"
    )


def test_masked_channels_get_no_gradient():
    """A muted channel is out of the mix: its fader must feel nothing."""
    weights = json.loads(WEIGHTS.read_text())
    n_bands = weights["nBands"]
    rng = np.random.default_rng(11)
    P, g = _scene(rng, 6, n_bands)
    a = np.ones(6)
    a[2] = 0.0
    a[4] = 0.0

    js = _js({"weights": weights, "P": P.tolist(), "g": g.tolist(),
              "a": a.tolist(), "dShape": rng.normal(0, 1, size=n_bands).tolist()})
    dG = np.asarray(js["dG"])
    assert abs(dG[2]) < 1e-12 and abs(dG[4]) < 1e-12, f"muted channels pulled: {dG}"


def test_untrained_model_reproduces_the_analytic_baseline():
    """With a zeroed output layer the model must BE the baseline, exactly.

    This is what makes the residual design safe: the learned field starts where
    the analytic one ends, so switching it on can never be a regression before
    training has earned anything.
    """
    from sonic_gravity.field import load_spec
    from sonic_gravity.model import MixWorldModel, analytic_log_mix, normalise_shape

    spec = load_spec()
    m = MixWorldModel(spec.n_bands, p_law=spec.fader_exponent).double().eval()
    rng = np.random.default_rng(5)
    P, g = _scene(rng, 6, spec.n_bands)
    Pt = torch.tensor(P).unsqueeze(0)
    gt = torch.tensor(g).unsqueeze(0)
    at = torch.ones(1, 6, dtype=torch.float64)

    pred, base = m(Pt, gt, at)
    want = normalise_shape(analytic_log_mix(Pt, gt, spec.fader_exponent))
    assert torch.allclose(pred, want, atol=1e-9)
    assert torch.allclose(base, want, atol=1e-9)


# ── The learned field as a whole: is its gradient still the true gradient? ──

FIELD_JS = ROOT / "sonic_gravity" / "web" / "js" / "field.js"
SPEC = ROOT / "sonic_gravity" / "spec.json"

FIELD_BRIDGE = """
import { Scene } from %(field)s;
import { LearnedMix } from %(learned)s;
const input = JSON.parse(await new Promise((res) => {
  let s = ""; process.stdin.on("data", (d) => (s += d)); process.stdin.on("end", () => res(s));
}));
const P = input.P.map((r) => Float64Array.from(r));
const scene = new Scene(input.spec, P);
if (input.useLearned) scene.setPredictor(new LearnedMix(input.weights));
const g = Float64Array.from(input.g);
const a = Float64Array.from(input.a);
const anchor = input.anchor ? Float64Array.from(input.anchor) : null;

const grad = Array.from(scene.gradient(g, a, anchor));
// Centred finite differences on the potential the console actually reads.
const h = 1e-6, fd = [];
for (let i = 0; i < g.length; i++) {
  const gp = Float64Array.from(g), gm = Float64Array.from(g);
  gp[i] += h; gm[i] -= h;
  fd.push((scene.potential(gp, a, anchor).total - scene.potential(gm, a, anchor).total) / (2 * h));
}
process.stdout.write(JSON.stringify({ grad, fd, V: scene.potential(g, a, anchor).total }));
"""


def _field_js(payload: dict) -> dict:
    script = FIELD_BRIDGE % {"field": json.dumps(FIELD_JS.as_uri()),
                             "learned": json.dumps(LEARNED_JS.as_uri())}
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(payload), capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


@pytest.mark.parametrize("use_learned", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_field_gradient_is_exact_with_and_without_the_model(seed: int, use_learned: bool):
    """The force must remain −∂V/∂g once the network sits in the chain.

    Wiring a model into the middle of a gradient is where sign errors and
    forgotten paths live, and none of them announce themselves: the console
    keeps working and simply pulls the wrong way. So the assembled field is
    checked against finite differences of the potential the console reads —
    with the model on, and with it off.
    """
    weights = json.loads(WEIGHTS.read_text())
    spec = json.loads(SPEC.read_text())
    n_bands = spec["bands"]["n"]
    rng = np.random.default_rng(seed + 40)
    P, g = _scene(rng, 6, n_bands)
    a = np.ones(6)
    a[rng.integers(0, 6)] = 0.0
    anchor = rng.uniform(0.3, 0.9, size=6)

    out = _field_js({
        "spec": spec, "weights": weights, "useLearned": use_learned,
        "P": P.tolist(), "g": g.tolist(), "a": a.tolist(), "anchor": anchor.tolist(),
    })
    grad = np.asarray(out["grad"])
    fd = np.asarray(out["fd"])
    scale = max(float(np.abs(fd).max()), 1e-9)
    assert np.allclose(grad, fd, rtol=2e-4, atol=2e-4 * scale), (
        f"learned={use_learned}\nanalytic {grad}\nfinite   {fd}"
    )


def test_the_learned_field_changes_the_forces_it_produces():
    """If both fields gave the same force, the model would be decoration."""
    weights = json.loads(WEIGHTS.read_text())
    spec = json.loads(SPEC.read_text())
    rng = np.random.default_rng(77)
    P, g = _scene(rng, 6, spec["bands"]["n"])
    a = np.ones(6)
    common = {"spec": spec, "weights": weights, "P": P.tolist(),
              "g": g.tolist(), "a": a.tolist(), "anchor": None}

    base = np.asarray(_field_js({**common, "useLearned": False})["grad"])
    learned = np.asarray(_field_js({**common, "useLearned": True})["grad"])
    rel = np.abs(learned - base).max() / max(np.abs(base).max(), 1e-9)
    assert rel > 1e-3, f"the learned field is indistinguishable from the baseline ({rel:.2e})"


# ── Recurrent runtime: one GRU step, forward and backward ───────────────────

SEQ_WEIGHTS = ROOT / "sonic_gravity" / "web" / "model_seq.json"

REC_BRIDGE = """
import { RecurrentMix } from %(mod)s;
const input = JSON.parse(await new Promise((res) => {
  let s = ""; process.stdin.on("data", (d) => (s += d)); process.stdin.on("end", () => res(s));
}));
const model = new RecurrentMix(input.weights);
model.h = Float64Array.from(input.h);
const P = input.P.map((r) => Float64Array.from(r));
const g = Float64Array.from(input.g);
const a = Float64Array.from(input.a);
const { shape, cache } = model.forward(P, g, a);
const dG = model.backward(cache, Float64Array.from(input.dShape));
process.stdout.write(JSON.stringify({
  shape: Array.from(shape), dG: Array.from(dG), h: Array.from(cache.gru.h),
}));
"""


def _rec_js(payload: dict) -> dict:
    script = REC_BRIDGE % {"mod": json.dumps(LEARNED_JS.as_uri())}
    proc = subprocess.run(["node", "--input-type=module", "-e", script],
                          input=json.dumps(payload), capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        raise AssertionError(f"node failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def _torch_recurrent(weights: dict):
    from sonic_gravity.model import RecurrentMixModel

    m = RecurrentMixModel(weights["nBands"], width=weights["width"],
                          hidden=weights["hidden"], state=weights["state"],
                          p_law=weights["pLaw"])

    def load(lin, blob):
        lin.weight.data = torch.tensor(np.array(blob["W"]).T, dtype=torch.float32)
        lin.bias.data = torch.tensor(np.array(blob["b"]), dtype=torch.float32)

    load(m.enc[0], weights["enc"][0])
    load(m.enc[2], weights["enc"][1])
    load(m.dec[0], weights["dec"][0])
    load(m.dec[2], weights["dec"][1])
    load(m.dec[4], weights["dec"][2])
    gw = weights["gru"]
    m.gru.weight_ih.data = torch.tensor(np.array(gw["Wih"]).T, dtype=torch.float32)
    m.gru.weight_hh.data = torch.tensor(np.array(gw["Whh"]).T, dtype=torch.float32)
    m.gru.bias_ih.data = torch.tensor(np.array(gw["bih"]), dtype=torch.float32)
    m.gru.bias_hh.data = torch.tensor(np.array(gw["bhh"]), dtype=torch.float32)
    return m.eval().double()


@pytest.mark.skipif(not SEQ_WEIGHTS.exists(), reason="no recurrent model trained")
@pytest.mark.parametrize("seed,n_ch", [(0, 6), (1, 4), (2, 8)])
def test_recurrent_step_matches_pytorch(seed: int, n_ch: int):
    """One step of the GRU, forward and backward, against autograd.

    The runtime never backpropagates through time — at 60 Hz the hidden state
    is the past, and the past does not depend on the fader being moved now. So
    only this single step has to be right, and if it is wrong the console pulls
    in a direction nothing else would ever flag.
    """
    weights = json.loads(SEQ_WEIGHTS.read_text())
    n_bands = weights["nBands"]
    rng = np.random.default_rng(seed + 200)
    P, g = _scene(rng, n_ch, n_bands)
    a = np.ones(n_ch)
    if seed % 2:
        a[rng.integers(0, n_ch)] = 0.0
    h0 = rng.normal(0, 0.4, size=weights["state"])
    dShape = rng.normal(0, 1, size=n_bands)

    js = _rec_js({"weights": weights, "P": P.tolist(), "g": g.tolist(),
                  "a": a.tolist(), "h": h0.tolist(), "dShape": dShape.tolist()})

    m = _torch_recurrent(weights)
    Pt = torch.tensor(P, dtype=torch.float64).unsqueeze(0)
    gt = torch.tensor(g[None, :], dtype=torch.float64, requires_grad=True)
    at = torch.tensor(a, dtype=torch.float64).unsqueeze(0)
    ht = torch.tensor(h0[None, :], dtype=torch.float64)
    pred, _, h_new = m.step(Pt, gt, at, ht)
    pred.backward(torch.tensor(dShape, dtype=torch.float64).unsqueeze(0))

    assert np.allclose(js["shape"], pred.detach().numpy()[0], atol=1e-6), "forward diverges"
    assert np.allclose(js["h"], h_new.detach().numpy()[0], atol=1e-6), "GRU state diverges"

    want = gt.grad.numpy()[0]
    got = np.asarray(js["dG"])
    scale = max(float(np.abs(want).max()), 1e-9)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6 * scale), (
        f"gradient diverges\njs {got}\npy {want}"
    )


@pytest.mark.skipif(not SEQ_WEIGHTS.exists(), reason="no recurrent model trained")
def test_hidden_state_actually_changes_the_prediction():
    """If the state made no difference, the memory would be decoration."""
    weights = json.loads(SEQ_WEIGHTS.read_text())
    rng = np.random.default_rng(9)
    P, g = _scene(rng, 6, weights["nBands"])
    a = np.ones(6)
    common = {"weights": weights, "P": P.tolist(), "g": g.tolist(), "a": a.tolist(),
              "dShape": rng.normal(0, 1, size=weights["nBands"]).tolist()}

    zero = _rec_js({**common, "h": [0.0] * weights["state"]})
    warm = _rec_js({**common, "h": rng.normal(0, 0.6, size=weights["state"]).tolist()})
    diff = np.abs(np.asarray(zero["shape"]) - np.asarray(warm["shape"])).max()
    assert diff > 1e-3, f"the hidden state does nothing ({diff:.2e})"


REC_FIELD_BRIDGE = """
import { Scene } from %(field)s;
import { RecurrentMix } from %(learned)s;
const input = JSON.parse(await new Promise((res) => {
  let s = ""; process.stdin.on("data", (d) => (s += d)); process.stdin.on("end", () => res(s));
}));
const P = input.P.map((r) => Float64Array.from(r));
const scene = new Scene(input.spec, P);
const model = new RecurrentMix(input.weights);
model.h = Float64Array.from(input.h);   // a warm state, as at runtime
scene.setPredictor(model);
const g = Float64Array.from(input.g);
const a = Float64Array.from(input.a);
const anchor = input.anchor ? Float64Array.from(input.anchor) : null;
const grad = Array.from(scene.gradient(g, a, anchor));
const h = 1e-6, fd = [];
for (let i = 0; i < g.length; i++) {
  const gp = Float64Array.from(g), gm = Float64Array.from(g);
  gp[i] += h; gm[i] -= h;
  fd.push((scene.potential(gp, a, anchor).total - scene.potential(gm, a, anchor).total) / (2 * h));
}
process.stdout.write(JSON.stringify({ grad, fd }));
"""


@pytest.mark.skipif(not SEQ_WEIGHTS.exists(), reason="no recurrent model trained")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_field_gradient_is_exact_with_the_recurrent_model(seed: int):
    """The assembled field must still return −∂V/∂g with a GRU in the chain.

    The hidden state is held fixed while differentiating, which is exactly what
    happens on the console: the past is given, only the action varies. Finite
    differences of the potential the console reads are the check.
    """
    weights = json.loads(SEQ_WEIGHTS.read_text())
    spec = json.loads(SPEC.read_text())
    rng = np.random.default_rng(seed + 300)
    P, g = _scene(rng, 6, spec["bands"]["n"])
    a = np.ones(6)
    a[rng.integers(0, 6)] = 0.0
    anchor = rng.uniform(0.3, 0.9, size=6)
    h = rng.normal(0, 0.4, size=weights["state"])

    script = REC_FIELD_BRIDGE % {"field": json.dumps((ROOT / "sonic_gravity" / "web" / "js" / "field.js").as_uri()),
                                 "learned": json.dumps(LEARNED_JS.as_uri())}
    proc = subprocess.run(["node", "--input-type=module", "-e", script],
                          input=json.dumps({"spec": spec, "weights": weights, "P": P.tolist(),
                                            "g": g.tolist(), "a": a.tolist(),
                                            "anchor": anchor.tolist(), "h": h.tolist()}),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)

    grad, fd = np.asarray(out["grad"]), np.asarray(out["fd"])
    scale = max(float(np.abs(fd).max()), 1e-9)
    assert np.allclose(grad, fd, rtol=2e-4, atol=2e-4 * scale), (
        f"analytic {grad}\nfinite   {fd}"
    )
