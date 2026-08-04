"""Train the recurrent model, and test whether memory actually buys anything.

Three predictors on the same unseen tracks:

  1. **analytic** — closed-form summation, which knows nothing about the bus
     compressor and so cannot be right about a compressed mix;
  2. **memoryless** — the same network with its hidden state re-zeroed every
     frame: identical parameters, identical capacity, no access to the past;
  3. **recurrent** — the same network with its state carried forward.

(1) → (2) measures what a learned correction is worth. (2) → (3) measures what
*memory* is worth, and that is the claim being tested. Comparing the recurrent
model only against the analytic baseline would conflate the two and let extra
capacity pass for temporal modelling.

Both learned models are trained separately: evaluating a memory-trained network
with its memory switched off would handicap it rather than ablate it.

    python -m sonic_gravity.train_seq
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from .model import DB, RecurrentMixModel, normalise_shape

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "sequences.npz"
WEIGHTS = HERE / "web" / "model_seq.json"

CHUNK = 48  # truncated BPTT: ~2.2 s, far longer than the compressor's release
# The first frames of a chunk start from a ZEROED hidden state taken mid-
# sequence, which is simply wrong: the compressor was doing something there.
# Scoring them would teach the recurrent model to cope with a false state and
# penalise it for having memory at all — which is exactly what the first run
# measured (recurrent WORSE than memoryless, t = −16.5). They are used to warm
# the state up and excluded from the loss.
WARMUP = 10


def _load(path: pathlib.Path):
    d = np.load(path)
    n_bands = int(d["n_bands"])
    n_frames = int(d["n_frames"])
    counts = d["counts"]
    P_flat, g_flat, M = d["P"], d["g"], d["M"]

    offs = np.concatenate([[0], np.cumsum(counts)])
    seqs = []
    for i, c in enumerate(counts):
        a, b = offs[i], offs[i + 1]
        P = P_flat[:, a * n_bands : b * n_bands].reshape(n_frames, c, n_bands)
        seqs.append((P, g_flat[:, a:b], M[i]))
    return seqs, d["track"], n_bands, float(d["fader_exponent"]), d["comp"]


def _batch(seqs, idx, torch, t0=None, length=None):
    """Pad a set of sequences to a common channel count."""
    c_max = max(seqs[i][0].shape[1] for i in idx)
    T = seqs[idx[0]][0].shape[0] if length is None else length
    B, N = len(idx), seqs[idx[0]][0].shape[2]
    P = np.zeros((B, T, c_max, N), dtype=np.float32)
    g = np.zeros((B, T, c_max), dtype=np.float32)
    mask = np.zeros((B, c_max), dtype=np.float32)
    M = np.zeros((B, T, N), dtype=np.float32)
    for k, i in enumerate(idx):
        Pi, gi, Mi = seqs[i]
        s = 0 if t0 is None else t0
        c = Pi.shape[1]
        P[k, :, :c] = Pi[s : s + T]
        g[k, :, :c] = gi[s : s + T]
        mask[k, :c] = 1.0
        M[k] = Mi[s : s + T]
    return (torch.from_numpy(P), torch.from_numpy(g),
            torch.from_numpy(mask), torch.from_numpy(M))


def _mae(model, torch, seqs, idx, no_memory, batch=8):
    model.eval()
    errs_m, errs_b = [], []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sub = idx[s : s + batch]
            P, g, mask, M = _batch(seqs, sub, torch)
            target = normalise_shape(torch.log10(M + 1e-12) * DB)
            pred, base = model(P, g, mask, no_memory=no_memory)
            # Same exclusion at evaluation: the sequence genuinely starts at
            # h = 0, but the compressor did not — scoring those frames would
            # measure the cold start, not the model.
            errs_m.append((pred - target)[:, WARMUP:].abs().mean(dim=-1).flatten().numpy())
            errs_b.append((base - target)[:, WARMUP:].abs().mean(dim=-1).flatten().numpy())
    return np.concatenate(errs_m), np.concatenate(errs_b)


def train(seqs, track, n_bands, p_law, no_memory: bool, epochs: int, seed: int,
          lr: float, quiet=False, split_seed: int | None = None):
    """`split_seed` separates WHICH tracks are held out from HOW the net is
    initialised.

    Tying them together mixes two sources of variance, and here that mattered
    more than the effect being measured: across three seeds the memory gain read
    +0.155, +0.050 and −0.054 dB. Reporting any single one of those — negative
    or positive — would have been reporting a coin flip.
    """
    import torch

    ids = np.unique(track)
    rng = np.random.default_rng(seed if split_seed is None else split_seed)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * 0.25))
    te_ids, tr_ids = ids[:n_test], ids[n_test:]
    te = np.where(np.isin(track, te_ids))[0]
    tr = np.where(np.isin(track, tr_ids))[0]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)  # batching order follows the init seed
    model = RecurrentMixModel(n_bands, p_law=p_law)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_frames = seqs[0][0].shape[0]

    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        losses = []
        for s in range(0, len(order), 8):
            sub = order[s : s + 8]
            if len(sub) < 2:
                continue
            # Truncated BPTT: a random window per batch, so the model sees every
            # part of every trajectory across epochs without unrolling 168 steps.
            t0 = int(rng.integers(0, max(1, n_frames - CHUNK)))
            P, g, mask, M = _batch(seqs, sub, torch, t0=t0, length=CHUNK)
            target = normalise_shape(torch.log10(M + 1e-12) * DB)
            pred, _ = model(P, g, mask, no_memory=no_memory)
            # Only score frames whose hidden state has had time to become real.
            loss = torch.nn.functional.smooth_l1_loss(
                pred[:, WARMUP:], target[:, WARMUP:], beta=1.0
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        if not quiet and (ep % 10 == 0 or ep == epochs - 1):
            m, b = _mae(model, torch, seqs, te[:16], no_memory)
            print(f"  ep {ep:3d}  loss {np.mean(losses):.4f}  test {m.mean():.4f} dB "
                  f"(analytic {b.mean():.4f})")

    m, b = _mae(model, torch, seqs, te, no_memory)
    return model, m, b, te


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=DATA)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", type=pathlib.Path, default=WEIGHTS)
    ap.add_argument("--seeds", type=int, default=1,
                    help="how many init seeds to average over (the split stays fixed)")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    if not args.data.exists():
        sys.exit(f"no sequences at {args.data} — run: python -m sonic_gravity.dataset_seq")
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("torch is required: pip install torch")

    seqs, track, n_bands, p_law, comp = _load(args.data)
    print(f"{len(seqs)} sequences, {seqs[0][0].shape[0]} frames each, "
          f"{len(np.unique(track))} tracks")
    c = np.atleast_2d(comp)
    kind = "multiband" if len(c) > 1 else "wideband"
    print(f"{kind} bus in the world:")
    for row in c:
        print(f"  {row[0]:.0f} dB, {row[1]:.0f}:1, {row[2]:.0f}/{row[3]:.0f} ms")
    print()

    # One run per init seed, on a FIXED split. The spread across seeds is the
    # error bar that matters — it is what says whether the effect exists at all.
    rows = []
    for k in range(args.seeds):
        seed = args.seed + k
        _, mem_less, analytic, _ = train(seqs, track, n_bands, p_law, True,
                                         args.epochs, seed, args.lr, quiet=True,
                                         split_seed=args.split_seed)
        model, rec, _, _ = train(seqs, track, n_bands, p_law, False,
                                 args.epochs, seed, args.lr, quiet=True,
                                 split_seed=args.split_seed)
        rows.append((analytic.mean(), mem_less.mean(), rec.mean()))
        print(f"  seed {seed}: analytic {rows[-1][0]:.4f} · memoryless {rows[-1][1]:.4f} "
              f"· recurrent {rows[-1][2]:.4f}  (memory {rows[-1][1] - rows[-1][2]:+.4f} dB)")

    arr = np.array(rows)
    analytic_m, mem_m, rec_m = arr.mean(axis=0)
    gains = arr[:, 1] - arr[:, 2]
    gain_learn = analytic_m - mem_m
    gain_mem = float(gains.mean())
    se = float(gains.std(ddof=1) / np.sqrt(len(gains))) if len(gains) > 1 else float("nan")

    print("\n── test set (unseen tracks, compressed bus) ──")
    print(f"  analytic     {analytic_m:.4f} dB")
    print(f"  memoryless   {mem_m:.4f} dB   ({gain_learn:+.4f} from learning)")
    print(f"  recurrent    {rec_m:.4f} dB   ({gain_mem:+.4f} from MEMORY)")
    if len(gains) > 1:
        print(f"  memory gain across {len(gains)} seeds: {gains.round(4).tolist()}")
        print(f"    mean {gain_mem:+.4f} ± {se:.4f} (SE), t = {gain_mem/se:.1f}")

    report = {
        "seeds": int(args.seeds),
        "per_seed_memory_gain_db": [float(v) for v in gains],
        "analytic_mae_db": float(analytic_m),
        "memoryless_mae_db": float(mem_m),
        "recurrent_mae_db": float(rec_m),
        "gain_from_learning_db": float(gain_learn),
        "gain_from_memory_db": float(gain_mem),
        "gain_from_memory_pct": float(100 * gain_mem / mem_m),
        "t_stat_memory": float(gain_mem / se) if se == se else None,
        "memory_gain_se_db": se if se == se else None,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    if len(gains) > 1 and gain_mem <= 2 * se:
        print("\n  ⚠️  memory buys nothing measurable here — report it as such")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights = model.to_weights()
    weights["report"] = report
    args.out.write_text(json.dumps(weights))
    print(f"\nweights → {args.out} ({args.out.stat().st_size / 1024:.0f} kB)")
    (args.data.parent / "recurrent_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
