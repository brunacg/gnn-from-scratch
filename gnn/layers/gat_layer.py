"""
Graph Attention Layer (single-head) and a multi-head wrapper.

Implements the attention mechanism from:

    Velickovic, P. et al. (2018).
    Graph Attention Networks. ICLR 2018.
    https://arxiv.org/abs/1710.10903

Sparse single-head forward pass
-------------------------------
Given a raw adjacency A (binary, unweighted) and node features H, we
work directly with the edge list (src, dst) of A + self-loops -- never
materialising the (N x N) attention matrix. For Cora that drops the
per-head working set from ~5.3 M float64 entries to ~13 K (a ~400x
reduction):

    H_W       = H @ W                                  (N x F')
    e_l, e_r  = H_W @ a_l, H_W @ a_r                   (N x 1) each
    e_edge    = LeakyReLU( e_l[src] + e_r[dst] )       (|E|,)   per-edge logit
    alpha     = softmax_per_source(e_edge, src)        (|E|,)   neighbour-normalised
    alpha     = dropout(alpha)                         (training only)
    H'[i]     = activation( sum_{e: src=i} alpha[e] * H_W[dst[e]] )

The aggregation is implemented as a sparse-times-dense matmul through
scipy.sparse.csr_matrix, which is C-optimised and handles backward by
transposing the same CSR structure.

Multi-head
----------
The MultiHeadGATLayer composes K independent single-head GATLayers and
either concatenates their outputs along the feature axis (intermediate
layers) or averages them (final layer), as in the ICLR 2018 paper.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import List, Literal, Optional, Tuple

from gnn.autograd.tensor import (
    Tensor,
    _accumulate,
    elu,
    hstack,
    leaky_relu,
    log_softmax,
    relu,
)


Activation = Literal["relu", "elu", "log_softmax", "none"]


# Edge-index cache: every head, every forward, every epoch shares one
# (src, dst, indptr, n) tuple per adjacency. Cora's adjacency is fixed
# for the whole run, so this is hit on every call after the first.
_EDGE_INDEX_CACHE: "dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, int]]" = {}
_EDGE_INDEX_CACHE_LIMIT = 4


def _edge_index(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Edge list of A plus self-loops, cached by id(A).

    Returns
    -------
    src    : (|E|,) int32 row indices, sorted ascending
    dst    : (|E|,) int32 column indices, in CSR-aligned order
    indptr : (N+1,) int32 CSR row pointer; row i has edges in [indptr[i], indptr[i+1])
    n      : N
    """
    key = id(A)
    cached = _EDGE_INDEX_CACHE.get(key)
    if cached is not None and cached[3] == A.shape[0]:
        return cached
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"GAT expects a square adjacency, got {A.shape}")

    # Edges + self-loops as a boolean mask. np.where on a 2D bool returns
    # row-major sorted indices, which is exactly the canonical CSR order
    # we want: scipy.sparse.csr_matrix((data, (src, dst))) will preserve
    # the row of `data` if (src, dst) is already row-sorted.
    bool_mask = (A != 0)
    np.fill_diagonal(bool_mask, True)
    src, dst = np.where(bool_mask)
    src = src.astype(np.int32, copy=False)
    dst = dst.astype(np.int32, copy=False)

    # Build CSR row pointer once.
    n = A.shape[0]
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr, src + 1, 1)
    np.cumsum(indptr, out=indptr)

    if len(_EDGE_INDEX_CACHE) >= _EDGE_INDEX_CACHE_LIMIT:
        _EDGE_INDEX_CACHE.pop(next(iter(_EDGE_INDEX_CACHE)))
    _EDGE_INDEX_CACHE[key] = (src, dst, indptr, n)
    return _EDGE_INDEX_CACHE[key]


def _gather_rows(t: Tensor, idx: np.ndarray) -> Tensor:
    """Differentiable row-gather: out[k] = t[idx[k]]."""
    out = Tensor(t.data[idx], _children=(t,), _op="gather")

    def _backward() -> None:
        if t.requires_grad:
            grad = np.zeros_like(t.data)
            # np.add.at handles duplicate indices correctly -- a plain
            # `grad[idx] += out.grad` would lose contributions whenever
            # the same row index appears more than once in `idx`.
            np.add.at(grad, idx, out.grad)
            _accumulate(t, grad)

    out._backward = _backward
    out.requires_grad = t.requires_grad
    return out


def _edge_softmax(scores: Tensor, src: np.ndarray, n: int) -> Tensor:
    """
    Numerically-stable softmax of `scores` per source node (per neighbourhood).

        alpha[e] = exp(scores[e] - max_{e': src[e']=src[e]} scores[e'])
                   / sum_{e': src[e']=src[e]} exp(scores[e'] - max_...)

    `scores` is a Tensor of shape (|E|, 1) (or (|E|,)); the result keeps
    the same shape.
    """
    s = scores.data.reshape(-1)

    # Per-source max for numerical stability.
    src_max = np.full(n, -np.inf, dtype=np.float64)
    np.maximum.at(src_max, src, s)
    shifted = s - src_max[src]
    np.exp(shifted, out=shifted)               # in-place exp on the temp

    src_sum = np.zeros(n, dtype=np.float64)
    np.add.at(src_sum, src, shifted)
    alpha_data = shifted / src_sum[src]        # (|E|,)

    out = Tensor(alpha_data.reshape(scores.data.shape),
                 _children=(scores,), _op="edge_softmax")

    def _backward() -> None:
        if scores.requires_grad:
            g = out.grad.reshape(-1)
            # d alpha_e / d scores_e' = alpha_e * (delta_{e,e'} -
            #                            alpha_e' * 1[src(e)==src(e')])
            # so d L / d scores_e = alpha_e * (g_e - sum_{e': same src} alpha_e' * g_e')
            weighted = np.zeros(n, dtype=np.float64)
            np.add.at(weighted, src, alpha_data * g)
            d = alpha_data * (g - weighted[src])
            _accumulate(scores, d.reshape(scores.data.shape))

    out._backward = _backward
    out.requires_grad = scores.requires_grad
    return out


def _sparse_aggregate(
    alpha: Tensor,
    H_W: Tensor,
    src: np.ndarray,
    dst: np.ndarray,
    indptr: np.ndarray,
    n: int,
) -> Tensor:
    """
    Sparse-times-dense aggregation.

        out[i, :] = sum_{e: src[e] = i} alpha[e] * H_W[dst[e], :]

    Implemented as `csr @ H_W` where `csr` reuses the cached (indices,
    indptr) row layout from `_edge_index` and only its `data` array
    changes. Backward to H_W is `csr.T @ out.grad`; backward to alpha is
    a per-edge dot of the corresponding (src, dst) gradient slices.
    """
    a = alpha.data.reshape(-1)
    csr = sp.csr_matrix((a, dst, indptr), shape=(n, n))
    out_data = csr @ H_W.data
    out = Tensor(out_data, _children=(alpha, H_W), _op="sparse_agg")

    def _backward() -> None:
        if alpha.requires_grad:
            # Per-edge gradient: dL/dalpha[e] = <H_W[dst[e]], out.grad[src[e]]>
            grad_alpha = np.einsum("ef,ef->e", H_W.data[dst], out.grad[src])
            _accumulate(alpha, grad_alpha.reshape(alpha.data.shape))
        if H_W.requires_grad:
            # dL/dH_W[j] = sum_{e: dst[e]=j} alpha[e] * out.grad[src[e]]
            # which is exactly csr.T @ out.grad.
            grad_h = csr.T @ out.grad
            _accumulate(H_W, grad_h)

    out._backward = _backward
    out.requires_grad = alpha.requires_grad or H_W.requires_grad
    return out


class GATLayer:
    """Single-head Graph Attention layer with optional attention dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Activation = "relu",
        negative_slope: float = 0.2,
        attn_dropout: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.in_features    = in_features
        self.out_features   = out_features
        self.activation     = activation
        self.negative_slope = negative_slope
        self.attn_dropout   = attn_dropout

        # We no longer materialise a dense (N, N) attention matrix.
        # last_alpha_edges captures (alpha_per_edge, src, dst, n) so the
        # dense matrix can be reconstructed lazily by the `last_alpha`
        # property if a caller (e.g. the visualisation notebook) wants it.
        self._last_alpha_edges: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, int]
        ] = None

        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W   = Tensor(rng.uniform(-limit, limit, (in_features, out_features)), requires_grad=True)
        self.a_l = Tensor(rng.standard_normal((out_features, 1)) * 0.01, requires_grad=True)
        self.a_r = Tensor(rng.standard_normal((out_features, 1)) * 0.01, requires_grad=True)

        self._training = True
        self._rng = np.random.default_rng(
            None if seed is None else int(seed) ^ 0xC0FFEE
        )

    def forward(self, A: np.ndarray, H: Tensor) -> Tensor:
        """
        A : binary adjacency matrix (N x N), NOT normalised
        H : node features            (N x in_features) Tensor
        """
        src, dst, indptr, n = _edge_index(A)

        # Do not wrap H in a new Tensor: that would detach the graph from layer 1.
        H_W = H @ self.W                                    # (N x F')

        # Attention logits, scored only on real edges (and self-loops):
        #   e_edge[k] = LeakyReLU( a_l^T h_{src[k]} + a_r^T h_{dst[k]} )
        # Two (N, 1) projections then a per-edge gather, then sum +
        # LeakyReLU. We never materialise the (N, N) outer-sum.
        e_l = H_W @ self.a_l                                # (N, 1)
        e_r = H_W @ self.a_r                                # (N, 1)
        e_src_edge = _gather_rows(e_l, src)                 # (|E|, 1)
        e_dst_edge = _gather_rows(e_r, dst)                 # (|E|, 1)
        e_edge     = leaky_relu(e_src_edge + e_dst_edge,
                                self.negative_slope)        # (|E|, 1)

        # Per-source softmax. For each node i, alpha[e] over edges with
        # src[e]=i sums to 1.
        alpha = _edge_softmax(e_edge, src, n)               # (|E|, 1)

        self._last_alpha_edges = (alpha.data.reshape(-1).copy(), src, dst, n)

        # Attention dropout on edge values (was N x N before, now |E|).
        if self._training and self.attn_dropout > 0.0:
            keep = 1.0 - self.attn_dropout
            drop_mask = self._rng.random(alpha.data.shape)
            kept = drop_mask < keep
            drop_mask[kept]  = 1.0 / keep
            drop_mask[~kept] = 0.0
            alpha = _attn_dropout(alpha, drop_mask)

        # Sparse-times-dense aggregation: out[i] = sum_e alpha_e * H_W[dst[e]]
        out = _sparse_aggregate(alpha, H_W, src, dst, indptr, n)

        if self.activation == "relu":
            return relu(out)
        if self.activation == "elu":
            return elu(out)
        if self.activation == "log_softmax":
            return log_softmax(out)
        return out

    def __call__(self, A: np.ndarray, H: Tensor) -> Tensor:
        return self.forward(A, H)

    def train(self) -> "GATLayer":
        self._training = True
        return self

    def eval(self) -> "GATLayer":
        self._training = False
        return self

    def parameters(self) -> list:
        return [self.W, self.a_l, self.a_r]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    @property
    def last_alpha(self) -> Optional[np.ndarray]:
        """
        Dense (N, N) attention matrix from the most recent forward pass,
        reconstructed on demand from the cached edge values.

        Building the dense matrix here costs ~56 MiB on Cora, but it's
        opt-in: if no caller (i.e. the visualisation notebook) accesses
        `last_alpha`, we never materialise it. None of the per-epoch
        training/eval calls hit this path.
        """
        if self._last_alpha_edges is None:
            return None
        a, src, dst, n = self._last_alpha_edges
        full = np.zeros((n, n), dtype=np.float64)
        full[src, dst] = a
        return full

    @property
    def last_alpha_edges(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """Compact edge-level attention from the last forward pass."""
        return self._last_alpha_edges

    def __repr__(self) -> str:
        return (
            f"GATLayer(in={self.in_features}, out={self.out_features}, "
            f"activation='{self.activation}', attn_dropout={self.attn_dropout})"
        )


class MultiHeadGATLayer:
    """
    K independent single-head GAT layers, with their outputs concatenated
    (intermediate layers) or averaged (final layer) along the feature axis.

    Each head has its own (W, a_l, a_r), so a multi-head layer with K heads
    of width F' has K times the parameters of one single-head layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features_per_head: int,
        n_heads: int,
        activation: Activation = "none",
        concat: bool = True,
        attn_dropout: float = 0.0,
        negative_slope: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        self.in_features            = in_features
        self.out_features_per_head  = out_features_per_head
        self.n_heads                = n_heads
        self.concat                 = concat
        self.activation             = activation

        # Each head gets its own deterministic seed derived from the layer seed.
        rng = np.random.default_rng(seed)
        head_seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_heads)]
        self.heads: List[GATLayer] = [
            GATLayer(
                in_features=in_features,
                out_features=out_features_per_head,
                activation="none",
                negative_slope=negative_slope,
                attn_dropout=attn_dropout,
                seed=hs,
            )
            for hs in head_seeds
        ]

    def forward(self, A: np.ndarray, H: Tensor) -> Tensor:
        head_outs = [head(A, H) for head in self.heads]

        if self.concat or len(head_outs) == 1:
            out = head_outs[0] if len(head_outs) == 1 else hstack(head_outs)
        else:
            # Mean across heads (used in the final layer of the original GAT).
            inv_k = 1.0 / len(head_outs)
            out = head_outs[0] * inv_k
            for h in head_outs[1:]:
                out = out + (h * inv_k)

        if self.activation == "relu":
            return relu(out)
        if self.activation == "elu":
            return elu(out)
        if self.activation == "log_softmax":
            return log_softmax(out)
        return out

    def __call__(self, A: np.ndarray, H: Tensor) -> Tensor:
        return self.forward(A, H)

    def train(self) -> "MultiHeadGATLayer":
        for h in self.heads:
            h.train()
        return self

    def eval(self) -> "MultiHeadGATLayer":
        for h in self.heads:
            h.eval()
        return self

    def parameters(self) -> list:
        return [p for head in self.heads for p in head.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    @property
    def out_features(self) -> int:
        if self.concat:
            return self.out_features_per_head * self.n_heads
        return self.out_features_per_head

    @property
    def last_alpha(self) -> Optional[np.ndarray]:
        """
        Mean attention matrix (N, N) across all heads from the most recent
        forward pass, or None if no head has run a forward yet.

        Built lazily from each head's edge-level attention so we only ever
        allocate one (N, N) buffer instead of K of them.
        Inspect a specific head with `layer.heads[i].last_alpha`.
        """
        edge_data = [h.last_alpha_edges for h in self.heads
                     if h.last_alpha_edges is not None]
        if not edge_data:
            return None
        # All heads share the same (src, dst, n) since they see the same A.
        _, src, dst, n = edge_data[0]
        avg = np.zeros((n, n), dtype=np.float64)
        inv_k = 1.0 / len(edge_data)
        for a, _src, _dst, _n in edge_data:
            avg[_src, _dst] += a * inv_k
        return avg

    def __repr__(self) -> str:
        mode = "concat" if self.concat else "mean"
        return (
            f"MultiHeadGATLayer(in={self.in_features}, "
            f"out_per_head={self.out_features_per_head}, n_heads={self.n_heads}, "
            f"mode={mode}, activation='{self.activation}')"
        )


def _attn_dropout(alpha: Tensor, mask: np.ndarray) -> Tensor:
    """
    Multiply alpha entries by `mask` (already inverted: drop=0, keep=1/keep_prob).

    Differentiable so layer-1 attention parameters still receive gradient.
    """
    out = Tensor(alpha.data * mask, _children=(alpha,), _op="attn_dropout")

    def _backward() -> None:
        if alpha.requires_grad:
            _accumulate(alpha, out.grad * mask)

    out._backward = _backward
    out.requires_grad = alpha.requires_grad
    return out


