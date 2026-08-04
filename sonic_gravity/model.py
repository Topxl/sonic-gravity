"""The learned world model: action-conditioned prediction of the mix spectrum.

What it predicts
----------------
Given the pre-fader spectra of every channel and a set of fader positions, it
predicts the spectrum the mix will actually have. That is what lets the console
answer *"what would happen if I pushed this fader?"* without pushing it — the
question the whole system rests on, and the one you cannot measure live across
nine faders.

Three design decisions carry the whole thing:

**It predicts a residual, not the spectrum.** The analytic model already gets
within ~0.6 dB. Asking a network to rediscover power summation from scratch
would waste its capacity on the easy part and risk doing *worse* than the
baseline. So the network only learns what the baseline misses:

    log M̂ = log M_analytic + Δ(P, g)

Initialised near zero, the model starts *exactly* at the baseline and can only
depart from it where the data pays for it.

**It is permutation-invariant and channel-count agnostic** (deep sets). A mix is
a *set* of channels, not a list: swapping bass and drums must not change the
prediction. And the console must handle 4 stems or 9 without retraining. Per
channel encoder → mean pooling → decoder.

**Mean pooling, not max.** Max would give marginally more capacity, but its
backward pass is a gather that must then be re-implemented by hand in
JavaScript for the browser runtime. Mean keeps the JS gradient a few lines and
impossible to get subtly wrong. The whole point is a gradient that runs at
60 Hz in a browser; an architecture that is awkward to differentiate there is
the wrong architecture, however good it looks in Python.

Why no latent-collapse machinery
--------------------------------
JEPA-style training needs VICReg or an EMA target because encoder *and* target
are learned together, so both can collapse to a constant. Here the target is a
**measured spectrum** — fixed, not learned. There is nothing to collapse into,
so adding that machinery would be cargo cult. It becomes necessary the moment
the target itself is learned, which is the natural next step for modelling
compressor dynamics through time.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch is a training-only dependency
    torch = None
    nn = object

EPS = 1e-12
DB = 10.0


def analytic_log_mix(P: "torch.Tensor", g: "torch.Tensor", p_law: float) -> "torch.Tensor":
    """Baseline in log domain, differentiable. P: (B,C,N) · g: (B,C) → (B,N)."""
    u = g.clamp_min(1e-6) ** (2.0 * p_law)
    return torch.log10(torch.einsum("bc,bcn->bn", u, P) + EPS) * DB


def normalise_shape(log_spec: "torch.Tensor") -> "torch.Tensor":
    """Remove overall level: the field consumes the SHAPE of the spectrum.

    Level is the master fader's job. Letting the model spend capacity on
    absolute level would teach it the one thing the potential deliberately
    ignores.
    """
    return log_spec - torch.logsumexp(log_spec * (np.log(10) / DB), dim=-1, keepdim=True) * (
        DB / np.log(10)
    )


class MixWorldModel(nn.Module):
    """Predicts the residual between the real mix spectrum and the analytic one."""

    def __init__(self, n_bands: int, width: int = 64, hidden: int = 96, p_law: float = 2.0):
        super().__init__()
        self.n_bands = n_bands
        self.p_law = p_law
        self.width = width
        self.hidden = hidden

        # Per-channel encoder. Inputs per channel: its spectral SHAPE (n_bands),
        # its fader position, and its energy relative to the mix — the three
        # things that decide how much it contributes and where.
        self.enc = nn.Sequential(
            nn.Linear(n_bands + 2, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        # Decoder sees the pooled context plus the baseline's own prediction,
        # so it can condition the correction on what the baseline just claimed.
        self.dec = nn.Sequential(
            nn.Linear(width + n_bands, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_bands),
        )
        # Start AT the baseline: the last layer outputs zeros, so an untrained
        # model is exactly the analytic field and training can only earn its way
        # away from it.
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def features(self, P: "torch.Tensor", g: "torch.Tensor", mask: "torch.Tensor"):
        """Per-channel inputs. P: (B,C,N) linear power · g: (B,C) · mask: (B,C)."""
        total = P.sum(dim=-1, keepdim=True).clamp_min(EPS)
        shape = torch.log10(P / total + EPS) * DB / 10.0  # log shape, ~O(1)
        energy = torch.log10(total.squeeze(-1) + EPS) / 10.0
        return torch.cat([shape, g.unsqueeze(-1), energy.unsqueeze(-1)], dim=-1)

    def forward(self, P, g, mask, ablate_context: bool = False):
        """`ablate_context` blanks the per-channel pathway.

        The model then sees only what the baseline already predicted, so it can
        learn a correction to the baseline's own shape but nothing conditional
        on which channels are playing or where their faders sit. That ablation
        is the control: whatever the full model gains over it is, by
        construction, information carried by (P, g).
        """
        base = analytic_log_mix(P * mask.unsqueeze(-1), g, self.p_law)
        base_n = normalise_shape(base)

        h = self.enc(self.features(P, g, mask))
        h = h * mask.unsqueeze(-1)
        # Mean over PRESENT channels only — padding must not dilute the context.
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        if ablate_context:
            pooled = torch.zeros_like(pooled)

        delta = self.dec(torch.cat([pooled, base_n / 10.0], dim=-1))
        return base_n + delta, base_n

    # ── Export ──────────────────────────────────────────────────────────────
    def to_weights(self) -> dict:
        """Flat, framework-free weights for the browser runtime."""
        # Rounded to 5 decimals: the residual it predicts is under a decibel,
        # so the sixth digit changes nothing audible and the JSON shrinks ~3×.
        def layer(lin):
            return {"W": np.round(lin.weight.detach().cpu().numpy().T, 5).tolist(),
                    "b": np.round(lin.bias.detach().cpu().numpy(), 5).tolist()}

        return {
            "nBands": self.n_bands,
            "width": self.width,
            "hidden": self.hidden,
            "pLaw": self.p_law,
            "enc": [layer(self.enc[0]), layer(self.enc[2])],
            "dec": [layer(self.dec[0]), layer(self.dec[2]), layer(self.dec[4])],
            "activation": "silu",
        }


def collate(P_flat, g_flat, counts, idx, n_bands: int):
    """Pack variable-channel samples into a padded batch with a mask."""
    offsets = np.concatenate([[0], np.cumsum(counts)])
    c_max = int(counts[idx].max())
    B = len(idx)
    P = np.zeros((B, c_max, n_bands), dtype=np.float32)
    g = np.zeros((B, c_max), dtype=np.float32)
    mask = np.zeros((B, c_max), dtype=np.float32)
    for k, i in enumerate(idx):
        a, b = offsets[i], offsets[i + 1]
        c = b - a
        P[k, :c] = P_flat[a:b]
        g[k, :c] = g_flat[a:b]
        mask[k, :c] = 1.0
    return P, g, mask
