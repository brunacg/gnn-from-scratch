"""
Two-layer MLP baseline (no graph structure).

Used to measure how much the graph topology contributes beyond raw node
features.  A significant gap between MLP and GCN/GAT accuracy demonstrates
the value of neighbourhood aggregation.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from gnn.autograd.tensor import Tensor, relu, log_softmax, nll_loss
from gnn.models.gcn import _differentiable_dropout


class MLP:
    """
    Two-layer MLP: Linear -> ReLU -> Dropout -> Linear -> log-softmax.

    Parameters
    ----------
    in_features : input feature dimension
    hidden_dim  : hidden layer width
    n_classes   : number of output classes
    dropout     : drop probability during training
    seed        : RNG seed
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        n_classes: int = 7,
        dropout: float = 0.5,
        seed: Optional[int] = 42,
    ) -> None:
        self.dropout   = dropout
        self._training = True
        # Two independent RNGs from the same seed: one for weight init, one
        # for dropout. Keeping them separate means dropout sampling never
        # affects (or is affected by) the initial weight distribution.
        self._rng = np.random.default_rng(seed)
        init_rng  = np.random.default_rng(seed)

        def glorot(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return init_rng.uniform(-limit, limit, (fan_in, fan_out))

        self.W1 = Tensor(glorot(in_features, hidden_dim), requires_grad=True)
        self.b1 = Tensor(np.zeros(hidden_dim), requires_grad=True)
        self.W2 = Tensor(glorot(hidden_dim, n_classes), requires_grad=True)
        self.b2 = Tensor(np.zeros(n_classes), requires_grad=True)

    def forward(
        self,
        X: np.ndarray,
        targets: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> tuple:
        H = relu(Tensor(X) @ self.W1 + self.b1)

        if self._training and self.dropout > 0.0:
            # Build the inverted dropout mask in one buffer instead of
            # three (random -> bool -> float64 -> divided). Same idea as
            # GCN/GAT: scale kept positions to 1/keep_prob in place.
            keep_prob = 1.0 - self.dropout
            drop_mask = self._rng.random(H.shape)
            kept = drop_mask < keep_prob
            drop_mask[kept]  = 1.0 / keep_prob
            drop_mask[~kept] = 0.0
            H = _differentiable_dropout(H, drop_mask)

        log_probs = log_softmax(H @ self.W2 + self.b2)
        loss = nll_loss(log_probs, targets, mask) if targets is not None else None
        return log_probs, loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self) -> "MLP":
        self._training = True
        return self

    def eval(self) -> "MLP":
        self._training = False
        return self

    def parameters(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def n_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"MLP(\n"
            f"  Linear({self.W1.shape[0]} -> {self.W1.shape[1]}, ReLU),\n"
            f"  Linear({self.W2.shape[0]} -> {self.W2.shape[1]}, log-softmax),\n"
            f"  dropout={self.dropout},\n"
            f"  total_params={self.n_parameters():,}\n"
            f")"
        )
