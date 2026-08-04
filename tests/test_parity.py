"""Le champ Python et le champ JavaScript disent-ils la même chose ?

Le modèle est entraîné en Python, sur le potentiel Python ; il est appliqué
dans le navigateur, sur le potentiel JavaScript. Si les deux divergent, rien ne
plante et rien ne s'affiche : la gravité ne correspond simplement plus à ce qui
a été appris, et l'écart se découvre des semaines plus tard en écoutant.

C'est le même patron que `console_fx.py` / `mixEngine.ts` dans le projet
Yantra, où la parité est vérifiée à 0,0005 dB près contre le navigateur.

Ignoré si node n'est pas installé — la vérification ne doit pas bloquer un
poste qui n'a pas d'environnement JS, mais elle doit tourner là où il y en a un.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from sonic_gravity.field import SPEC_PATH, Field

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIELD_JS = ROOT / "sonic_gravity" / "web" / "js" / "field.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node absent")

# Le pont : on instancie la même scène des deux côtés et on compare. Écrit ici
# plutôt que dans un fichier du paquet web — ce code ne sert qu'au test, il n'a
# rien à faire dans ce qui est servi au navigateur.
BRIDGE = """
import { Scene } from %(field)s;
const input = JSON.parse(await new Promise((res) => {
  let s = ""; process.stdin.on("data", (d) => (s += d)); process.stdin.on("end", () => res(s));
}));
const P = input.P.map((row) => Float64Array.from(row));
const scene = new Scene(input.spec, P);
const g = Float64Array.from(input.g);
const a = input.audible ? Float64Array.from(input.audible) : null;
const anchor = input.anchor ? Float64Array.from(input.anchor) : null;
const parts = scene.potential(g, a, anchor);
const grad = Array.from(scene.gradient(g, a, anchor));
process.stdout.write(JSON.stringify({ parts, grad }));
"""


def _js_eval(payload: dict) -> dict:
    script = BRIDGE % {"field": json.dumps(FIELD_JS.as_uri())}
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node a échoué :\n{proc.stderr}")
    return json.loads(proc.stdout)


def _scene(rng: np.random.Generator, n_ch: int, n_bands: int):
    b = np.arange(n_bands)
    P = np.stack([
        np.exp(-0.5 * ((b - rng.uniform(0, n_bands)) / rng.uniform(3.0, 12.0)) ** 2)
        * 10 ** rng.uniform(-3, -1)
        + 1e-9
        for _ in range(n_ch)
    ])
    return P, rng.uniform(0.15, 0.95, size=n_ch)


@pytest.mark.parametrize("seed", range(5))
def test_potential_and_gradient_agree_between_python_and_javascript(seed: int):
    spec = json.loads(SPEC_PATH.read_text())
    field = Field()
    rng = np.random.default_rng(seed)
    P, g = _scene(rng, 6, field.spec.n_bands)
    anchor = rng.uniform(0.3, 0.9, size=6)
    audible = np.ones(6)
    if seed % 2:
        audible[seed % 6] = 0.0

    got = _js_eval({
        "spec": spec,
        "P": P.tolist(),
        "g": g.tolist(),
        "audible": audible.tolist(),
        "anchor": anchor.tolist(),
    })

    v_py, parts_py = field.potential(P, g, audible, anchor)
    grad_py = field.gradient(P, g, audible, anchor)

    assert got["parts"]["total"] == pytest.approx(v_py, rel=1e-9), "potentiel divergent"
    for key in ("tilt", "mask", "level", "anchor"):
        assert got["parts"][key] == pytest.approx(parts_py[key], rel=1e-9), f"terme « {key} » divergent"

    scale = max(float(np.abs(grad_py).max()), 1e-12)
    assert np.allclose(np.asarray(got["grad"]), grad_py, rtol=1e-9, atol=1e-9 * scale), (
        f"gradient divergent\njs {got['grad']}\npy {grad_py.tolist()}"
    )


def test_band_layout_agrees_between_python_and_javascript():
    """Les bandes doivent contenir les MÊMES bins des deux côtés.

    Un découpage qui diffère d'un seul bin dans le grave suffirait à décaler le
    gabarit, et l'écart serait invisible : les deux implémentations resteraient
    cohérentes avec elles-mêmes.
    """
    from sonic_gravity.field import band_edges, band_gather

    spec = json.loads(SPEC_PATH.read_text())
    fft_size = int(spec.get("analysis", {}).get("fftSize", 8192))
    n_bands = int(spec["bands"]["n"])

    script = """
import { bandEdges, bandGather } from %(field)s;
const input = JSON.parse(await new Promise((res) => {
  let s = ""; process.stdin.on("data", (d) => (s += d)); process.stdin.on("end", () => res(s));
}));
const edges = bandEdges(input.n, input.fLo, input.fHi);
const gather = bandGather(input.fftSize / 2, input.sr, input.fftSize, edges);
process.stdout.write(JSON.stringify(gather.map((a) => Array.from(a))));
""" % {"field": json.dumps(FIELD_JS.as_uri())}

    payload = {
        "n": n_bands,
        "fLo": spec["bands"]["fLo"],
        "fHi": spec["bands"]["fHi"],
        "fftSize": fft_size,
        "sr": 48000.0,
    }
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(payload), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)

    edges = band_edges(n_bands, float(spec["bands"]["fLo"]), float(spec["bands"]["fHi"]))
    py = [idx.tolist() for idx in band_gather(fft_size // 2, 48000.0, fft_size, edges)]

    assert len(js) == len(py) == n_bands
    for b, (a, c) in enumerate(zip(js, py)):
        assert a == c, f"bande {b} : js {a[:4]}… ≠ py {c[:4]}…"
