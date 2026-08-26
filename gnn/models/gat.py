"""
Two-layer Graph Attention Network with multi-head attention, ELU, and the
input-feature + attention dropout regimen from Velickovic et al. (2018).

Architecture (Cora defaults)
----------------------------
    X                                   (N x F)
     -> dropout (training only)
     -> MultiHeadGATLayer(F -> F'/K) x K heads, ELU, concat -> H1   (N x F')
     -> dropout (training only)
     -> MultiHeadGATLayer(F' -> C) x 1 head,  log_softmax,  mean    (N x C)

`hidden_dim` is the *total* hidden width after concatenation, so each
of the K heads in the first layer produces `hidden_dim // n_heads` features.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from gnn.autograd.tensor import Tensor, nll_loss
from gnn.layers.gat_layer import MultiHeadGATLayer
from gnn.models.gcn import _differentiable_dropout


class GAT:
    """
    Two-layer multi-head GAT for semi-supervised node classification.

    Parameters
    ----------
    in_features    : number of input node features
    hidden_dim     : total width of the hidden layer (concatenated across heads)
    n_classes      : number of output classes
    dropout        : drop probability for input features in front of each layer
    n_heads        : attention heads in the first (intermediate) layer
    n_heads_out    : attention heads in the output layer (averaged)
    attn_dropout   : drop probability applied to alpha inside each head
                     (defaults to `dropout` if not specified)
    negative_slope : LeakyReLU slope inside attention scoring
    seed           : RNG seed
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        n_classes: int = 7,
        dropout: float = 0.6,
        n_heads: int = 8,
        n_heads_out: int = 1,
        attn_dropout: Optional[float] = None,
        negative_slope: float = 0.2,
        seed: Optional[int] = 42,
    ) -> None:
        # Paper-faithful default: 8 heads in layer 1, 1 head in layer 2.
        # Now safe because GATLayer is sparse end-to-end -- each head
        # holds (|E|,) attention values rather than an (N, N) matrix, so
        # n_heads=8 on Cora costs ~100 KB per head instead of ~3 GB.
        self.dropout      = dropout
        self.attn_dropout = dropout if attn_dropout is None else attn_dropout
        self._training    = True
        self._rng         = np.random.default_rng(seed)

        per_head_hidden = max(1, hidden_dim // n_heads)
        self._hidden_total = per_head_hidden * n_heads

        self.layer1 = MultiHeadGATLayer(
            in_features=in_features,
            out_features_per_head=per_head_hidden,
            n_heads=n_heads,
            activation="elu",
            concat=True,
            attn_dropout=self.attn_dropout,
            negative_slope=negative_slope,
            seed=seed,
        )
        self.layer2 = MultiHeadGATLayer(
            in_features=self._hidden_total,
            out_features_per_head=n_classes,
            n_heads=n_heads_out,
            activation="log_softmax",
            concat=False,
            attn_dropout=self.attn_dropout,
            negative_slope=negative_slope,
            seed=(seed + 1) if seed is not None else None,
        )

    def _maybe_dropout(self, H: Tensor) -> Tensor:
        if not (self._training and self.dropout > 0.0):
            return H
        keep = 1.0 - self.dropout
        # One float64 buffer for the mask instead of three (random,
        # bool, divided): sample, then scale the kept positions and
        # zero the dropped ones in place.
        drop = self._rng.random(H.shape)
        kept = drop < keep
        drop[kept] = 1.0 / keep
        drop[~kept] = 0.0
        return _differentiable_dropout(H, drop)

    def forward(
        self,
        A: np.ndarray,
        X: np.ndarray,
        targets: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        A       : binary adjacency (N x N) numpy
        X       : node features    (N x F) numpy
        targets : integer labels   (N,)    [for loss]
        mask    : boolean mask     (N,)    [for loss]
        """
        # Tensor.__init__ already coerces to float64; the explicit
        # X.astype(np.float64) here forced a redundant copy on every
        # forward pass (~31 MiB on Cora).
        H = Tensor(X)

        H = self._maybe_dropout(H)
        H1 = self.layer1(A, H)

        H1 = self._maybe_dropout(H1)
        log_probs = self.layer2(A, H1)

        loss = nll_loss(log_probs, targets, mask) if targets is not None else None
        return log_probs, loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self) -> "GAT":
        self._training = True
        self.layer1.train()
        self.layer2.train()
        return self

    def eval(self) -> "GAT":
        self._training = False
        self.layer1.eval()
        self.layer2.eval()
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
            f"GAT(\n"
            f"  {self.layer1},\n"
            f"  {self.layer2},\n"
            f"  feat_dropout={self.dropout}, attn_dropout={self.attn_dropout},\n"
            f"  total_params={self.n_parameters():,}\n"
            f")"
        )
