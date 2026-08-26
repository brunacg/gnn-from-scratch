"""
Benchmark MLP, GCN, and GAT on Cora across several random seeds.

For each model we run `--trials` independent training runs with different
seeds, capture test accuracy, training time, and any errors raised, then
print a summary table with mean / std / min / max.

Usage
-----
    python scripts/benchmark.py
    python scripts/benchmark.py --trials 10 --epochs 200
    python scripts/benchmark.py --models gcn gat
    python scripts/benchmark.py --patience 20 --grad-clip 5.0
    python scripts/benchmark.py --quiet
"""

from __future__ import annotations

import sys
import os
import gc
import time
import argparse
import traceback
from statistics import mean, pstdev

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn.data.cora  import load_cora, _get_cache_dir, _parse_content, _parse_cites
from gnn.models.gcn import GCN
from gnn.models.gat import GAT
from gnn.models.mlp import MLP
from gnn.train      import train, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MLP/GCN/GAT on Cora across multiple seeds and report accuracy stats."
    )
    p.add_argument("--trials",  type=int, default=5)
    p.add_argument("--epochs",  type=int, default=200)
    p.add_argument("--models",  nargs="+", default=["mlp", "gcn", "gat"],
                   choices=["mlp", "gcn", "gat"])
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--hidden",  type=int, default=64)
    p.add_argument("--lr",      type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=None,
                   help="Early stop after N epochs without val improvement.")
    p.add_argument("--grad-clip", type=float, default=None,
                   help="Clip global gradient L2 norm.")
    p.add_argument("--gat-dropout", type=float, default=0.6)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--n-heads", type=int, default=8,
                   help="GAT attention heads in the hidden layer. "
                        "Paper default is 8; we match it because GATLayer "
                        "is sparse end-to-end and each head only costs "
                        "~|E| floats on Cora (~100 KB).")
    p.add_argument("--n-heads-out", type=int, default=1,
                   help="GAT attention heads in the output layer (averaged).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _load_inputs(verbose: bool):
    """Load Cora plus the raw (un-normalised) adjacency required by GAT."""
    A_hat, X, y, train_mask, val_mask, test_mask = load_cora(verbose=verbose)

    cache = _get_cache_dir(None)
    node_ids, _, _ = _parse_content(cache / "cora/cora.content")
    adj = _parse_cites(cache / "cora/cora.cites", node_ids)
    A_raw = adj.toarray().astype(float)
    return A_hat, A_raw, X, y, train_mask, val_mask, test_mask


def _build_model(name, in_features, n_classes, args, seed):
    if name == "gcn":
        return GCN(in_features, args.hidden, n_classes, args.dropout, seed=seed)
    if name == "gat":
        return GAT(
            in_features, args.hidden, n_classes,
            dropout=args.gat_dropout,
            n_heads=args.n_heads,
            n_heads_out=args.n_heads_out,
            seed=seed,
        )
    if name == "mlp":
        return MLP(in_features, args.hidden, n_classes, args.dropout, seed=seed)
    raise ValueError(f"Unknown model: {name}")


def _build_forward(model, name, A_hat, A_raw, X, y):
    if name == "gcn":
        return lambda mask: model(A_hat, X, y, mask)
    if name == "gat":
        return lambda mask: model(A_raw, X, y, mask)
    return lambda mask: model(X, y, mask)


def _train_one(name, model, inputs, args) -> dict:
    A_hat, A_raw, X, y, train_mask, val_mask, test_mask = inputs
    forward = _build_forward(model, name, A_hat, A_raw, X, y)

    t0 = time.perf_counter()
    history = train(
        model, forward, y, train_mask, val_mask,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        keep_best=True,
        verbose=False,
    )
    train_time = time.perf_counter() - t0

    test_metrics = evaluate(model, forward, y, test_mask)
    val_metrics  = evaluate(model, forward, y, val_mask)

    return {
        "test_acc":   test_metrics["test_acc"],
        "val_acc":    val_metrics["test_acc"],
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "train_time": train_time,
    }


def _summarize(name: str, results: list[dict], errors: list[str]) -> None:
    if not results:
        print(f"\n{name.upper():>4} : all {len(errors)} runs FAILED")
        for i, msg in enumerate(errors):
            print(f"   trial {i+1}: {msg.splitlines()[-1]}")
        return

    accs   = [r["test_acc"] for r in results]
    times  = [r["train_time"] for r in results]
    val    = [r["val_acc"] for r in results]
    epochs = [r["best_epoch"] for r in results]
    n_ok   = len(accs)
    n_fail = len(errors)

    line = (
        f"{name.upper():>4} | runs {n_ok}/{n_ok + n_fail}  "
        f"test_acc {mean(accs)*100:5.2f} +/- {pstdev(accs)*100:4.2f}  "
        f"(min {min(accs)*100:5.2f}, max {max(accs)*100:5.2f})  "
        f"val {mean(val)*100:5.2f}  "
        f"best_epoch {mean(epochs):5.1f}  "
        f"time {mean(times):4.1f}s"
    )
    print(line)
    if errors:
        for msg in errors:
            print(f"      ! error in failed trial: {msg.splitlines()[-1]}")


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    print("Loading Cora ...")
    inputs = _load_inputs(verbose)
    _, _, X, y, *_ = inputs
    in_features = X.shape[1]
    n_classes   = int(y.max()) + 1

    print(
        f"\nBenchmark: trials={args.trials}, epochs={args.epochs}, "
        f"models={args.models}, seed_base={args.seed_base}, "
        f"patience={args.patience}, grad_clip={args.grad_clip}\n"
    )

    summary: dict[str, tuple[list[dict], list[str]]] = {}

    for name in args.models:
        results: list[dict] = []
        errors:  list[str] = []
        for trial in range(args.trials):
            seed = args.seed_base + trial
            model = None
            try:
                model  = _build_model(name, in_features, n_classes, args, seed)
                metrics = _train_one(name, model, inputs, args)
                results.append(metrics)
                if verbose:
                    early = " (early)" if metrics["stopped_early"] else ""
                    print(
                        f"  [{name:>3} trial {trial+1}/{args.trials} seed={seed}] "
                        f"test={metrics['test_acc']*100:5.2f}%  "
                        f"val={metrics['val_acc']*100:5.2f}%  "
                        f"best_epoch={metrics['best_epoch']}{early}  "
                        f"time={metrics['train_time']:4.1f}s",
                        flush=True,
                    )
            except Exception:
                tb = traceback.format_exc()
                errors.append(tb)
                if verbose:
                    print(f"  [{name:>3} trial {trial+1}/{args.trials} seed={seed}] FAILED", flush=True)
                    print(tb, flush=True)
            finally:
                # Drop the autograd graph between trials so N x N buffers do not fragment memory.
                del model
                gc.collect()
        summary[name] = (results, errors)
        # Heap reset between model types: previous GCN/MLP runs leave heap
        # fragmentation that can starve the next model's larger N x N
        # allocations.
        gc.collect()
        gc.collect()

    print("\n" + "=" * 90)
    print("Summary (test accuracy across seeds)")
    print("=" * 90)
    for name in args.models:
        results, errors = summary[name]
        _summarize(name, results, errors)
    print("=" * 90)


if __name__ == "__main__":
    main()
