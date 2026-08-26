"""
CLI entry point -- train a GCN, GAT, or MLP on the Cora dataset.

Usage
-----
    python scripts/train_cora.py
    python scripts/train_cora.py --model gat
    python scripts/train_cora.py --config configs/cora_gcn.yaml
    python scripts/train_cora.py --epochs 200 --lr 0.01 --hidden 64
    python scripts/train_cora.py --save checkpoints/gcn.npz
    python scripts/train_cora.py --load checkpoints/gcn.npz --epochs 0
    python scripts/train_cora.py --plot curves.png
    python scripts/train_cora.py --patience 20 --grad-clip 5.0
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn.data.cora  import load_cora, _get_cache_dir, _parse_content, _parse_cites
from gnn.models.gcn import GCN
from gnn.models.gat import GAT
from gnn.models.mlp import MLP
from gnn.train      import (
    train,
    evaluate,
    plot_history,
    save_checkpoint,
    load_checkpoint,
)


def _load_yaml(path: str) -> dict:
    """Minimal YAML loader (scalars only -- no external deps needed)."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition(":")
            val = val.strip()
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            cfg[key.strip()] = val
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a GCN, GAT, or MLP on the Cora citation network."
    )
    p.add_argument("--config",      type=str,   default=None,  help="Path to YAML config file")
    p.add_argument("--model",       type=str,   default="gcn", choices=["gcn","gat","mlp"],
                   help="Model architecture")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--hidden",      type=int,   default=None)
    p.add_argument("--dropout",     type=float, default=None)
    p.add_argument("--weight-decay",type=float, default=None)
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--log-every",   type=int,   default=None)
    p.add_argument("--patience",    type=int,   default=None,
                   help="Early stop after N epochs without val improvement.")
    p.add_argument("--grad-clip",   type=float, default=None,
                   help="Clip global gradient L2 norm to this value.")
    p.add_argument("--n-heads",     type=int,   default=None,
                   help="GAT: attention heads in the hidden layer.")
    p.add_argument("--n-heads-out", type=int,   default=None,
                   help="GAT: attention heads in the output layer (averaged).")
    p.add_argument("--no-keep-best", action="store_true",
                   help="Do NOT restore best-validation params at the end.")
    p.add_argument("--data-dir",    type=str,   default=None)
    p.add_argument("--save",        type=str,   default=None,  help="Save checkpoint to PATH.npz")
    p.add_argument("--load",        type=str,   default=None,  help="Load checkpoint from PATH.npz")
    p.add_argument("--plot",        type=str,   default=None,  help="Save loss/acc plot to PATH")
    p.add_argument("--show-plot",   action="store_true",
                   help="Display the loss/acc plot interactively (blocks the script).")
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI overrides (CLI wins)."""
    defaults = {
        "model": "gcn", "hidden_dim": 64, "n_classes": 7, "dropout": 0.5,
        "epochs": 200, "lr": 0.01, "weight_decay": 5e-4,
        "log_every": 20, "seed": 42,
        "patience": None, "grad_clip": None, "keep_best": True,
        "n_heads": 8, "n_heads_out": 1,
    }
    cfg = {**defaults}
    if args.config:
        cfg.update(_load_yaml(args.config))
    overrides = {
        "model":        args.model,
        "hidden_dim":   args.hidden,
        "dropout":      args.dropout,
        "epochs":       args.epochs,
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
        "seed":         args.seed,
        "log_every":    args.log_every,
        "patience":     args.patience,
        "grad_clip":    args.grad_clip,
        "n_heads":      args.n_heads,
        "n_heads_out":  args.n_heads_out,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if args.no_keep_best:
        cfg["keep_best"] = False
    return cfg


def _build_forward(model, model_name, A_hat, A_raw, X, y):
    """Return a closure mask -> (log_probs, loss) for the given model."""
    if model_name == "gcn":
        return lambda mask: model(A_hat, X, y, mask)
    if model_name == "gat":
        return lambda mask: model(A_raw, X, y, mask)
    return lambda mask: model(X, y, mask)


def main() -> None:
    args = parse_args()
    cfg  = _build_config(args)

    A_hat, X, y, train_mask, val_mask, test_mask = load_cora(
        data_dir=args.data_dir, verbose=True
    )

    in_features = X.shape[1]
    n_classes   = int(y.max()) + 1
    cfg["n_classes"] = n_classes

    model_name = cfg["model"]
    A_raw = None
    if model_name == "gcn":
        model = GCN(in_features=in_features, hidden_dim=cfg["hidden_dim"],
                    n_classes=n_classes, dropout=cfg["dropout"], seed=cfg["seed"])
    elif model_name == "gat":
        model = GAT(in_features=in_features, hidden_dim=cfg["hidden_dim"],
                    n_classes=n_classes, dropout=cfg["dropout"],
                    n_heads=cfg["n_heads"], n_heads_out=cfg["n_heads_out"],
                    seed=cfg["seed"])
        # GAT takes raw adjacency A, not normalised A_hat
        cache = _get_cache_dir(args.data_dir)
        node_ids, _, _ = _parse_content(cache / "cora/cora.content")
        adj = _parse_cites(cache / "cora/cora.cites", node_ids)
        A_raw = adj.toarray().astype(float)
    else:
        model = MLP(in_features=in_features, hidden_dim=cfg["hidden_dim"],
                    n_classes=n_classes, dropout=cfg["dropout"], seed=cfg["seed"])

    print(f"\n{model}\n")

    if args.load:
        load_checkpoint(model, args.load)

    forward = _build_forward(model, model_name, A_hat, A_raw, X, y)

    if cfg["epochs"] > 0:
        history = train(
            model, forward, y, train_mask, val_mask,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            patience=cfg["patience"],
            grad_clip=cfg["grad_clip"],
            keep_best=cfg["keep_best"],
            verbose=True,
            log_every=cfg["log_every"],
        )

        results = evaluate(model, forward, y, test_mask)

        print("\n" + "=" * 40)
        print(f"  Test loss : {results['test_loss']:.4f}")
        print(f"  Test acc  : {results['test_acc'] * 100:.2f}%")
        print("=" * 40)

        if args.plot:
            plot_history(history, save_path=args.plot)
        elif args.show_plot:
            plot_history(history)
        # else: training metrics are already printed; skip plotting so the
        # script exits cleanly instead of blocking on plt.show().

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        save_checkpoint(model, args.save)


if __name__ == "__main__":
    main()
