"""
Two-layer Graph Convolutional Network.

Architecture:
    X  -> GCNLayer(in_features -> hidden_dim, ReLU, dropout)
    H1 -> GCNLayer(hidden_dim  -> n_classes,  log-softmax)
    Z  -> NLL loss on train_mask
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from gnn.autograd.tensor import Tensor, nll_loss, _accumulate
from gnn.layers.gcn_layer import GCNLayer


def _differentiable_dropout(H: Tensor, scale: "np.ndarray") -> Tensor:
    """Apply a pre-sampled dropout mask while keeping gradient flow."""
    out = Tensor(H.data * scale, _children=(H,), _op="dropout")

    def _backward() -> None:
        if H.requires_grad:
            _accumulate(H, out.grad * scale)

    out._backward = _backward
    out.requires_grad = H.requires_grad
    return out


class GCN:
    """
    Two-layer GCN for semi-supervised node classification.

    Parameters
    ----------
    in_features  : number of input node features  (1433 for Cora)
    hidden_dim   : width of the hidden layer       (default 64)
    n_classes    : number of output classes        (7 for Cora)
    dropout      : drop probability for hidden layer during training
    seed         : RNG seed for reproducibility
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        n_classes: int = 7,
        dropout: float = 0.5,
        seed: Optional[int] = 42,
    ) -> None:
        self.dropout = dropout
        self._training = True
        self._rng = np.random.default_rng(seed)
        self.layer1 = GCNLayer(in_features, hidden_dim, activation="relu", seed=seed)
        self.layer2 = GCNLayer(
            hidden_dim, n_classes, activation="log_softmax",
            seed=(seed + 1 if seed else None)
        )

    def forward(
        self,
        A_hat: np.ndarray,
        X: np.ndarray,
        targets: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> tuple:
        # Tensor.__init__ already coerces to float64; explicit astype here
        # forced a redundant copy of X on every forward pass.
        H1 = self.layer1(A_hat, Tensor(X))

        if self._training and self.dropout > 0.0:
            keep_prob = 1.0 - self.dropout
            # One buffer instead of three (random, bool, scaled):
            scale = self._rng.random(H1.shape)
            kept  = scale < keep_prob
            scale[kept]  = 1.0 / keep_prob
            scale[~kept] = 0.0
            H1 = _differentiable_dropout(H1, scale)

        log_probs = self.layer2(A_hat, H1)
        loss = nll_loss(log_probs, targets, mask) if targets is not None else None
        return log_probs, loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self) -> "GCN":
        self._training = True
        return self

    def eval(self) -> "GCN":
        self._training = False
        return self

    def parameters(self) -> list:
        return self.layer1.parameters() + self.layer2.parameters()

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def n_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"GCN(\n"
            f"  {self.layer1},\n"
            f"  {self.layer2},\n"
            f"  dropout={self.dropout},\n"
            f"  total_params={self.n_parameters():,}\n"
            f")"
        )
