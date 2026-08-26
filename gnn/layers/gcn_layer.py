"""
Graph Convolutional Layer.

Implements one step of spectral graph convolution as defined in:

    Kipf, T. N., & Welling, M. (2017).
    Semi-Supervised Classification with Graph Convolutional Networks.
    ICLR 2017.  https://arxiv.org/abs/1609.02907

Forward pass:  H' = activation( A_hat @ H @ W + b )

Weights are initialised with Glorot uniform:
    W ~ U( -sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)) )
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional

from gnn.autograd.tensor import Tensor, relu, log_softmax, fixed_sparse_matmul


Activation = Literal["relu", "log_softmax", "none"]


class GCNLayer:
    """A single graph convolutional layer with learnable W and optional bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Activation = "relu",
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.in_features  = in_features
        self.out_features = out_features
        self.activation   = activation

        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = Tensor(rng.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, A_hat: np.ndarray, H: Tensor) -> Tensor:
        """H' = activation( A_hat @ H @ W + b )

        A_hat is taken as dense for API stability with the notebook
        visualisation code, but the multiplication is dispatched through
        scipy.sparse: fewer than 0.2% of A_hat's entries are nonzero on
        Cora, so going via CSR cuts both the forward and backward
        aggregation cost by ~550x. The CSR conversion is cached on
        id(A_hat) inside `fixed_sparse_matmul`, so it only happens once
        across an entire training run.
        """
        AH  = fixed_sparse_matmul(A_hat, H)
        out = AH @ self.W
        if self.b is not None:
            out = out + self.b
        if self.activation == "relu":
            return relu(out)
        elif self.activation == "log_softmax":
            return log_softmax(out)
        return out

    def __call__(self, A_hat: np.ndarray, H: Tensor) -> Tensor:
        return self.forward(A_hat, H)

    def parameters(self) -> list:
        return [self.W, self.b] if self.b is not None else [self.W]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> str:
        return (
            f"GCNLayer(in={self.in_features}, out={self.out_features}, "
            f"activation='{self.activation}')"
        )
