"""
Training utilities: AdamW optimiser, generic training loop, and evaluation.

The training loop is model-agnostic: it takes a `forward_fn(mask)` callable
that runs the model on a node mask and returns `(log_probs, loss)`. Any
model exposing the standard `train()/eval()/parameters()/zero_grad()` API
can be trained through the same loop, regardless of how many input arrays
its forward signature actually consumes.

AdamW update rule (Loshchilov & Hutter, 2019, "Decoupled Weight Decay"):
    m_t = b1 * m_{t-1} + (1-b1) * g_t
    v_t = b2 * v_{t-1} + (1-b2) * g_t^2
    theta = theta - lr * (m_t / (1-b1^t)) / (sqrt(v_t / (1-b2^t)) + eps)
    theta = theta - lr * wd * theta            # decoupled, applied to weights, not gradient
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple

from gnn.autograd.tensor import Tensor


ForwardFn = Callable[[np.ndarray], Tuple[Tensor, Tensor]]


class Adam:
    """
    Adam with decoupled weight decay (AdamW).

    Parameters
    ----------
    parameters   : list of Tensor leaves that require grad
    lr           : learning rate
    beta1        : first moment decay
    beta2        : second moment decay
    eps          : numerical stability
    weight_decay : decoupled L2 (applied directly to weights, not gradients)
    """

    def __init__(
        self,
        parameters: list,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = parameters
        self.lr     = lr
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.wd     = weight_decay
        self.t      = 0

        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        self.t += 1
        b1t = 1.0 - self.beta1 ** self.t
        b2t = 1.0 - self.beta2 ** self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g
            p.data -= self.lr * (self.m[i] / b1t) / (np.sqrt(self.v[i] / b2t) + self.eps)
            if self.wd > 0.0:
                p.data -= self.lr * self.wd * p.data

    def clip_grad_norm(self, max_norm: float) -> float:
        """
        Global L2 gradient clipping. Returns the pre-clip global norm so
        the caller can log it.
        """
        sq = 0.0
        for p in self.params:
            if p.grad is not None:
                sq += float((p.grad * p.grad).sum())
        norm = float(np.sqrt(sq))
        if norm > max_norm and norm > 0.0:
            scale = max_norm / norm
            for p in self.params:
                if p.grad is not None:
                    p.grad *= scale
        return norm

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


class TrainingHistory:
    def __init__(self) -> None:
        self.train_loss: list = []
        self.train_acc:  list = []
        self.val_loss:   list = []
        self.val_acc:    list = []
        self.epoch_times: list = []
        self.best_epoch: int = 0
        self.best_val_acc: float = 0.0
        self.stopped_early: bool = False

    def record(self, tl, ta, vl, va, elapsed) -> None:
        self.train_loss.append(tl)
        self.train_acc.append(ta)
        self.val_loss.append(vl)
        self.val_acc.append(va)
        self.epoch_times.append(elapsed)


def accuracy(log_probs: Tensor, labels: np.ndarray, mask: np.ndarray) -> float:
    preds = log_probs.data[mask].argmax(axis=1)
    return float((preds == labels[mask]).mean())


def train(
    model,
    forward_fn: ForwardFn,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: Optional[int] = None,
    grad_clip: Optional[float] = None,
    keep_best: bool = True,
    verbose: bool = True,
    log_every: int = 10,
) -> TrainingHistory:
    """
    Generic training loop.

    Parameters
    ----------
    model        : any model with train()/eval()/parameters()/zero_grad()
    forward_fn   : callable mask -> (log_probs, loss). Captures the model
                   inputs (A, X, y) in a closure.
    y            : integer labels (N,) used for accuracy computation
    train_mask   : boolean mask for training nodes
    val_mask     : boolean mask for validation nodes
    epochs       : maximum number of epochs
    lr           : Adam learning rate
    weight_decay : decoupled L2 weight decay
    patience     : if set, stop early after this many epochs with no val_acc improvement
    grad_clip    : if set, clip global gradient norm to this value
    keep_best    : if True, restore best-validation parameters at the end
    """
    opt     = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainingHistory()

    best_val_acc = -1.0
    best_epoch   = 0
    best_params: Optional[list] = None
    stale        = 0

    if verbose:
        print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  "
              f"{'Val Loss':>9}  {'Val Acc':>8}  {'Time':>6}")
        print("-" * 60)

    last_epoch = 0
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        t0 = time.perf_counter()

        model.train()
        opt.zero_grad()
        log_probs, loss = forward_fn(train_mask)
        loss.backward()
        if grad_clip is not None:
            opt.clip_grad_norm(grad_clip)
        opt.step()

        t_loss = float(loss.data)
        t_acc  = accuracy(log_probs, y, train_mask)

        model.eval()
        val_lp, val_loss_t = forward_fn(val_mask)
        v_loss = float(val_loss_t.data)
        v_acc  = accuracy(val_lp, y, val_mask)

        elapsed = time.perf_counter() - t0
        history.record(t_loss, t_acc, v_loss, v_acc, elapsed)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch
            stale        = 0
            if keep_best:
                best_params = [p.data.copy() for p in model.parameters()]
        else:
            stale += 1

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(
                f"{epoch:>6}  {t_loss:>11.4f}  {t_acc:>10.4f}  "
                f"{v_loss:>9.4f}  {v_acc:>8.4f}  {elapsed:>5.3f}s"
            )

        # Drop references to the per-epoch autograd graph so the (potentially
        # very large -- e.g. multi-head GAT (N x N) attention matrices) tensors
        # are GC'd before the next epoch allocates fresh ones.
        del log_probs, loss, val_lp, val_loss_t

        if patience is not None and stale >= patience:
            history.stopped_early = True
            if verbose:
                print(f"Early stop at epoch {epoch}: no val improvement for {patience} epochs.")
            break

    history.best_epoch   = best_epoch
    history.best_val_acc = best_val_acc

    if keep_best and best_params is not None:
        for p, arr in zip(model.parameters(), best_params):
            p.data = arr
        if verbose:
            print(f"Restored best params from epoch {best_epoch} (val_acc={best_val_acc:.4f}, "
                  f"final epoch={last_epoch}).")

    return history


def evaluate(
    model,
    forward_fn: ForwardFn,
    y: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    """Compute loss and accuracy on the test set using the trained weights."""
    model.eval()
    log_probs, loss = forward_fn(test_mask)
    return {
        "test_loss": float(loss.data),
        "test_acc":  accuracy(log_probs, y, test_mask),
    }


def plot_history(history: TrainingHistory, save_path: Optional[str] = None) -> None:
    """Plot training and validation loss / accuracy curves."""
    epochs = range(1, len(history.train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history.train_loss, label="Train loss")
    ax1.plot(epochs, history.val_loss,   label="Val loss", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Loss curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history.train_acc, label="Train acc")
    ax2.plot(epochs, history.val_acc,   label="Val acc", linestyle="--")
    if history.best_epoch:
        ax2.axvline(history.best_epoch, color="grey", linestyle=":", label=f"Best epoch {history.best_epoch}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy curves")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {save_path}")
    else:
        plt.show()


def save_checkpoint(model, path: str) -> None:
    """Save all model weight arrays to a .npz file (one entry per parameter)."""
    arrays = {f"p{i}": p.data for i, p in enumerate(model.parameters())}
    np.savez(path, **arrays)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(model, path: str) -> None:
    """Load weights from a .npz checkpoint into model in-place."""
    data = np.load(path)
    params = model.parameters()
    keys = sorted(data.files, key=lambda k: int(k[1:]))
    if len(keys) != len(params):
        raise ValueError(
            f"Checkpoint has {len(keys)} arrays but model has {len(params)} parameters."
        )
    for p, key in zip(params, keys):
        p.data = data[key].astype(np.float64)
    print(f"Checkpoint loaded <- {path}")
