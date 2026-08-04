"""Train the world model, and prove it actually beats the analytic baseline.

The result that matters is not the loss curve — it is whether the model
predicts *unseen music* better than closed-form power summation. Two guardrails
make that claim mean something:

**Splits are by TRACK, never by sample.** Windows from the same song share
instrumentation, mix balance and bleed; splitting by sample would put near
duplicates on both sides and report a win that evaporates on real new music.

**The model is scored against a bias-corrected baseline, not the raw one.**
The analytic model has a systematic offset (−0.23 dB overall, and a distinct
one per band). A network that learned nothing but that constant would still
"beat" the raw baseline. So the reference is `analytic + per-band constant`,
fitted on the training split alone: whatever the model gains on top of that is
necessarily *conditional* on the channels and the fader positions.

**An ablation runs alongside.** The same model is trained with its per-channel
pathway blanked, so it sees only what the baseline predicted and can learn a
correction to that shape — but nothing conditional on which channels play or
where their faders sit. Whatever the full model gains over the ablation is, by
construction, information carried by (P, g).

⚠️ Two permutation controls were tried first and both were **invalid**, for the
same structural reason. Shuffling targets gave a +2.39 dB "improvement";
re-pairing inputs within each split gave +2.19 dB. Neither was a warning about
the model — the baseline here is a fixed formula, not a fitted predictor. Break
the relation and it stays structured but decorrelated, while a network simply
retreats to the target mean and wins trivially. The two error scales are never
commensurable. A control must remove the *information being tested* while
leaving both predictors on the same problem; that is an ablation, not a
permutation.

    python -m sonic_gravity.train
    python -m sonic_gravity.train --epochs 300 --no-control
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from .model import DB, MixWorldModel, collate, normalise_shape

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data" / "mixes.npz"
WEIGHTS = HERE / "web" / "model.json"


def _split_by_track(track: np.ndarray, seed: int):
    """Group split: a track lives entirely in one fold."""
    ids = np.unique(track)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(n * 0.2))
    n_val = max(1, int(n * 0.15))
    test_ids, val_ids = ids[:n_test], ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    where = lambda sel: np.where(np.isin(track, sel))[0]
    return where(train_ids), where(val_ids), where(test_ids)


def _residuals(model, torch, P_flat, g_flat, M, counts, idx, n_bands, batch=512, ablate=False):
    """Per-sample, per-band residuals for the model and for the raw baseline."""
    model.eval()
    res_m, res_b = [], []
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sub = idx[s : s + batch]
            P, g, mask = collate(P_flat, g_flat, counts, sub, n_bands)
            target = normalise_shape(torch.log10(torch.from_numpy(M[sub]) + 1e-12) * DB)
            pred, base = model(torch.from_numpy(P), torch.from_numpy(g),
                               torch.from_numpy(mask), ablate_context=ablate)
            res_m.append((pred - target).numpy())
            res_b.append((base - target).numpy())
    return np.concatenate(res_m), np.concatenate(res_b)


def _evaluate(model, torch, P_flat, g_flat, M, counts, idx, n_bands, bias=None, batch=512, ablate=False):
    """Mean absolute error in dB: model, raw baseline, bias-corrected baseline.

    `bias` is the per-band constant fitted on TRAIN only. Subtracting it from
    the baseline's residual is what makes the comparison fair — otherwise the
    model gets credit for a constant anyone could have measured.
    """
    res_m, res_b = _residuals(model, torch, P_flat, g_flat, M, counts, idx, n_bands, batch, ablate)
    mae = lambda r: np.abs(r).mean(axis=-1)
    corrected = res_b if bias is None else res_b - bias[None, :]
    return mae(res_m), mae(res_b), mae(corrected)


def _fit_bias(model, torch, P_flat, g_flat, M, counts, idx, n_bands) -> np.ndarray:
    """The per-band constant the baseline is systematically off by.

    Median rather than mean: a few windows have very large cross terms, and the
    reference the model must beat should not be handicapped by them.
    """
    _, res_b = _residuals(model, torch, P_flat, g_flat, M, counts, idx, n_bands)
    return np.median(res_b, axis=0)


def run(data: pathlib.Path, epochs: int, seed: int, lr: float, ablate: bool = False,
        quiet: bool = False) -> dict:
    try:
        import torch
    except ImportError:
        sys.exit("torch is required for training: pip install torch")

    d = np.load(data)
    P_flat, g_flat, M = d["P"], d["g"], d["M"]
    counts, track = d["counts"], d["track"]
    n_bands = int(d["n_bands"])
    p_law = float(d["fader_exponent"])

    tr, va, te = _split_by_track(track, seed)
    if not quiet:
        print(f"{len(tr)} train / {len(va)} val / {len(te)} test samples "
              f"({len(np.unique(track))} tracks, split by track)")

    torch.manual_seed(seed)
    model = MixWorldModel(n_bands, p_law=p_law)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    rng = np.random.default_rng(seed)
    best_val, best_state, patience = np.inf, None, 0
    batch = 256

    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        losses = []
        for s in range(0, len(order), batch):
            sub = order[s : s + batch]
            P, g, mask = collate(P_flat, g_flat, counts, sub, n_bands)
            target = normalise_shape(torch.log10(torch.from_numpy(M[sub]) + 1e-12) * DB)
            pred, _ = model(torch.from_numpy(P), torch.from_numpy(g),
                            torch.from_numpy(mask), ablate_context=ablate)
            # Huber on the residual: a handful of windows have very large cross
            # terms, and L2 would let them steer the whole model.
            loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        vm, vb, _ = _evaluate(model, torch, P_flat, g_flat, M, counts, va, n_bands, ablate=ablate)
        val = float(vm.mean())
        if val < best_val - 1e-5:
            best_val, patience = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if not quiet and (ep % 20 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d}  loss {np.mean(losses):.4f}  val {val:.4f} dB "
                  f"(baseline {vb.mean():.4f})")
        if patience >= 40:
            if not quiet:
                print(f"  early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    bias = _fit_bias(model, torch, P_flat, g_flat, M, counts, tr, n_bands)
    tm, tb, tc = _evaluate(model, torch, P_flat, g_flat, M, counts, te, n_bands, bias, ablate=ablate)
    # Paired comparison against the FAIR reference: same samples, both
    # predictors, the constant offset already granted to the baseline.
    diff = tc - tm
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return {
        "model_mae_db": float(tm.mean()),
        "baseline_mae_db": float(tb.mean()),
        "baseline_debiased_mae_db": float(tc.mean()),
        "improvement_db": float(diff.mean()),
        "improvement_pct": float(100 * diff.mean() / tc.mean()),
        "improvement_se_db": float(se),
        "t_stat": float(diff.mean() / se) if se > 0 else 0.0,
        "won_on_pct": float(100 * (diff > 0).mean()),
        "test_samples": int(len(te)),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "_model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=DATA)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", type=pathlib.Path, default=WEIGHTS)
    ap.add_argument("--no-control", action="store_true", help="skip the permutation control")
    args = ap.parse_args()

    if not args.data.exists():
        sys.exit(f"no dataset at {args.data} — run: python -m sonic_gravity.dataset")

    print("── training on real targets ──")
    real = run(args.data, args.epochs, args.seed, args.lr, ablate=False)
    model = real.pop("_model")

    control = None
    if not args.no_control:
        print("\n── ablation: per-channel pathway blanked ──")
        control = run(args.data, args.epochs, args.seed, args.lr, ablate=True, quiet=True)
        control.pop("_model")
        print(f"  ablated model MAE  {control['model_mae_db']:.4f} dB")
        print(f"  its gain over the fair reference: {control['improvement_db']:+.4f} dB")

    print("\n── test set (unseen tracks) ──")
    print(f"  baseline raw       {real['baseline_mae_db']:.4f} dB")
    print(f"  baseline debiased  {real['baseline_debiased_mae_db']:.4f} dB   ← the fair reference")
    print(f"  model              {real['model_mae_db']:.4f} dB")
    print(f"  improvement    {real['improvement_db']:+.4f} dB "
          f"({real['improvement_pct']:+.1f} %)  t = {real['t_stat']:.1f}")
    print(f"  model wins on  {real['won_on_pct']:.1f} % of samples")
    print(f"  parameters     {real['n_params']}")

    if control is not None:
        margin = control["model_mae_db"] - real["model_mae_db"]
        print(f"\n  ablation − full model = {margin:+.4f} dB "
              f"← what (P, g) is actually worth")
        if margin <= 0.005:
            print("  ⚠️  the ablated model does as well: the gain is NOT conditional on (P, g)")
        else:
            print("  ✓ the per-channel pathway carries the improvement")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights = model.to_weights()
    weights["report"] = {k: v for k, v in real.items()}
    if control:
        weights["report"]["ablation_mae_db"] = control["model_mae_db"]
    args.out.write_text(json.dumps(weights))
    size_kb = args.out.stat().st_size / 1024
    print(f"\nweights → {args.out} ({size_kb:.0f} kB)")


if __name__ == "__main__":
    main()
