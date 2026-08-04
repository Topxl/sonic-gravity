# Sonic Gravity

**A mixing console whose motorised faders push back when your hand moves toward a
bad mix, and slide into place on their own when you let go.**

Not a music generator. An *exoskeleton for the mixing engineer's hand*: the
machine never decides the mix, it makes the shape of the problem physically
felt.

Target hardware is a **Behringer X-Touch** (9 motorised faders, MCU protocol).
Until that hardware arrives, the surface is reproduced on screen behind
**exactly the same API**, so swapping in the real console replaces one file and
nothing else.

![The virtual X-Touch](docs/screenshot.jpg)

The gradient behind each fader is the gravity field: green is a valley, red is a
hill, and the white curve is the potential along the fader's travel. You can see
where the fader will settle before you release it.

---

## Why this is not a gadget

Today's AI music tools are *generative and remote*: type a prompt, wait, receive
a finished file. Nothing about that is real-time, embodied, or under continuous
control.

This is the opposite problem, and it is the one that matters for physical
systems:

- **Continuous, noisy signals** rather than discrete tokens.
- **Closed loop** at 60 Hz: measurement → prediction → force → actuator →
  measurement.
- **Action-conditioned prediction.** The whole system rests on answering *"what
  would happen if I pushed this fader?"* **without pushing it.** You cannot
  measure that live across nine faders — you would have to move each one and
  listen. That question is exactly what an action-conditioned world model
  exists to answer.

## Run it

```bash
pip install -r requirements.txt
python -m sonic_gravity.make_demo_scene   # synthesises a royalty-free 6-track scene
python -m sonic_gravity.serve             # → http://127.0.0.1:8700
```

Then: **Écouter** (play), **Dérégler** (scramble), **Tout lâcher** (release all).
The faders fall back into the valley. Grab one with the mouse and it lags behind
your finger, on the side the field is pushing. On a phone, the Vibration API
turns that lag into something you can feel.

To run the field on your own separated stems instead:

```bash
python -m sonic_gravity.prepare_demo --decompositions /path/to/stems --list
python -m sonic_gravity.prepare_demo <id>
```

## How it works

```
audio → pre-fader spectra → field → force → motors → faders → audio
```

Every channel is measured **pre-fader**: `P[i,b]` is the power of channel `i` in
band `b`, independent of its gain. The mix is then

```
M[b] = Σᵢ (gᵢ·aᵢ)² · P[i,b]
```

which makes the potential **analytically differentiable** with respect to the
faders. The force sent to each motor is `−∂V/∂gᵢ`, computed in closed form.

### The potential

| Term | What it measures | Why it is shaped that way |
|---|---|---|
| `tilt` | Distance to the spectral profile of real records, as a per-band z-score, through a **Huber loss** | The profile is **measured** on 220 tracks, not postulated. Huber stops a quiet passage from dominating the field |
| `mask` | Masking between channels (histogram intersection of spectral shapes, weighted by energy) | **Scale-invariant**, so its gradient says "lower the one that is fighting", never "lower everything" |
| `level` | Distance to a target level, **deliberately low weight** | Level is the master fader's job, not the channels'. At high weight the optimum ran into the top end stop |
| `anchor` | Pull back toward where the hand last left the fader | This is what separates an exoskeleton from an autopilot. A "Liberté" slider releases it and hands the whole mix to the machine |

Each band is additionally weighted by the **material actually available**:
during a passage where only one instrument plays, no fader can produce low end
that nobody is playing, so those bands stop counting. That weighting is measured
at full gain, so it is constant with respect to differentiation and leaves the
gradient's form untouched.

### Does it work?

Six scramble-then-release trials, each scored **on the same audio frame** so the
music cannot drift and flatter the result:

| | scrambled | after gravity |
|---|---|---|
| potential `V` | 12.3 – 17.1 | 1.19 – 1.94 |
| mix term only (anchor removed) | 1.43 – 2.67 | 1.16 – 1.91 |
| fader spread | 0.53 – 0.73 | 0.02 – 0.05 |

**6 / 6 improved**, including with the anchor term removed — so it is the mix
itself that gets better, not merely the faders returning home.

A sharper test: the synthetic demo scene contains two deliberate defects — `keys`
and `lead` occupy the same midrange and fight, and the pad is mixed too loud.
Released with the anchor at zero, the field **muted `keys` and `pad`** and kept
the other four. It found both planted faults on its own.

Measured live in the browser: **60 fps**, force + gradient in **1 µs**, all six
field reliefs (216 potential evaluations) in **0.13 ms** — under 1 % of a frame
budget.

## What the analytic field does *not* capture

The central approximation is `M[b] = Σ g²·P[i,b]`, i.e. **decorrelated phases**.
That holds well for separated stems; it would break on two copies of the same
signal. Compression, saturation and psychoacoustic masking are not modelled
either.

So this field is **not the ground truth — it is the baseline to beat.** That is
the point of shipping it: the learned model that follows has a quantitative
opponent, not a vibe check.

## Tests

```bash
python -m pytest tests/
```

- **43 field tests.** The gradient is checked against centred finite
  differences over 26 randomised configurations (with mutes, with anchoring, and
  across the Huber knee). This is the only proof that counts: a sign or factor
  error breaks no structural test — it just pushes the operator's hand toward
  the worse mix, silently. One such bug was caught this way: `DB_SCALE` (the
  factor in a decibel's *derivative*) written where `10` (the factor in its
  *value*) belonged, inflating every force by 1.7×–2.8×.
- **6 parity tests, Python ↔ JavaScript**, to 1e-9: potential, each term,
  gradient, and band layout. The model is trained on one implementation and
  applied in the other; a divergence would not crash anything, it would surface
  weeks later as a console that no longer matches what it learned.

Four other real defects were found by measuring rather than by looking, and each
is documented at the site of its fix:

1. A **phantom band at −174 dB** in the measured profile: at 37 Hz no FFT bin
   fell inside it. With a z-score divisor it would have buried the other 39.
2. **Faders driving into the end stops**, because the potential had no term
   representing the operator's intent.
3. **`V` exploding to 88** during a musical break — penalising the absence of
   low end that no channel was playing.
4. A **silence gate keyed to the running peak**, which at startup was
   initialised on silence itself, so the analyser noise floor read as music.

## Layout

| File | Role |
|---|---|
| `sonic_gravity/field.py` | **The core**: potential, analytic gradient, band layout |
| `sonic_gravity/web/js/field.js` | JS mirror, held in step by the parity tests |
| `sonic_gravity/spec.json` | **Single source** of constants, read by both implementations |
| `sonic_gravity/fit_target.py` | Measures the spectral profile over a music library |
| `sonic_gravity/make_demo_scene.py` | Synthesises the royalty-free demo scene |
| `sonic_gravity/prepare_demo.py` | Builds a scene from your own separated stems |
| `sonic_gravity/web/js/mix.js` | Web Audio: stems → pre-fader analysers → bus, silence gate |
| `sonic_gravity/web/js/motor.js` | Motorised-fader physics + haptics |
| `sonic_gravity/web/js/surface.js` | The on-screen X-Touch — same API as the MCU driver |
| `sonic_gravity/web/js/app.js` | The closed loop |

## Moving to hardware

`surface.js` exposes `setFader` · `setMeter` · `setVpotRing` · `setMuteLed` … and
emits `{type:"fader"|"mute"|"solo"|"touch"…}`: the MCU vocabulary, the same one
a Web MIDI driver uses to drive real motorised faders over pitch-bend. Attaching
the physical console is a substitution, not a rewrite.

One concept has no hardware counterpart in the other direction: the drawn gap
between finger and fader cap. On screen, resistance can only be *seen*. On the
X-Touch it is *felt* — which is the entire point, and the one thing this
on-screen demo can only approximate.

## Not done yet

- **The learned world model.** The field is analytic today. Next is an
  action-conditioned latent predictor trained self-supervised on interaction
  traces (`serve.py --record`): spectrum encoder → latent, predictor
  `(zₜ, aₜ) → ẑₜ₊₁`, potential head, with the force becoming the gradient taken
  *through* the predictor. The decisive detail is collapse prevention (VICReg,
  or an asymmetric predictor with an EMA target) — without it the encoder learns
  a constant and predicts nothing perfectly. Success criterion is already
  defined: beat the analytic baseline above.
- V-Pot rings move but are not bound to any parameter.
- The master fader is outside the field.
- On phones the button hit target is 66×36 px, below the 44 px guideline; the
  4 px gutter caps vertical growth and the width compensates.
- **Source comments are in French.** The README is not; the code has not been
  translated yet.

## License

MIT — see [LICENSE](LICENSE). The demo scene is synthesised from scratch by
`make_demo_scene.py` and contains no third-party audio.
