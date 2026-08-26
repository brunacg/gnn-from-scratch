"""
Matrix-aware autograd engine.

Each Tensor wraps a numpy ndarray and records enough information to run
reverse-mode automatic differentiation (backpropagation).  Only the ops
actually needed by a two-layer GCN are implemented, keeping the code
readable and the math transparent.

Supported ops and their matrix-calculus gradients
--------------------------------------------------
C = A @ B           dL/dA = dL/dC @ B.T      dL/dB = A.T @ dL/dC
C = A + B           dL/dA = dL/dC            dL/dB = sum over broadcast dims
C = relu(A)         dL/dA = dL/dC * (A > 0)
C = log_softmax(A)  dL/dA = dL/dC - softmax(A) * sum(dL/dC, axis=-1, keepdims=True)
L = nll_loss        scalar, gradient flows into log-softmax output
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Callable, Optional, Set, List, Tuple


class Tensor:
    """A numpy array with an attached reverse-mode gradient tape."""

    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Tensor"] = set(_children)
        self._op: str = _op

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@")

        def _backward() -> None:
            if self.requires_grad:
                _accumulate(self, out.grad @ other.data.T)
            if other.requires_grad:
                _accumulate(other, self.data.T @ out.grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __add__(self, other: "Tensor") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward() -> None:
            if self.requires_grad:
                _accumulate(self, _unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                _accumulate(other, _unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __radd__(self, other: "Tensor") -> "Tensor":
        return self.__add__(other)

    def __mul__(self, scalar: float) -> "Tensor":
        out = Tensor(self.data * scalar, _children=(self,), _op="*scalar")

        def _backward() -> None:
            if self.requires_grad:
                _accumulate(self, out.grad * scalar)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def __rmul__(self, scalar: float) -> "Tensor":
        return self.__mul__(scalar)

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), _children=(self,), _op="relu")

        def _backward() -> None:
            if self.requires_grad:
                # Bool mask broadcasts as 0/1 in numpy arithmetic; the
                # explicit astype(float64) was an extra full-size buffer.
                _accumulate(self, out.grad * (self.data > 0))

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def log_softmax(self) -> "Tensor":
        """Numerically stable log-softmax along the last axis."""
        # Two N x N buffers total:
        #   shifted -> reused in place as out_data (log-softmax)
        #   softmax -> kept for the backward closure (avoids re-exp on backward)
        shifted = self.data - self.data.max(axis=-1, keepdims=True)
        softmax = np.exp(shifted)
        sum_exp = softmax.sum(axis=-1, keepdims=True)
        softmax /= sum_exp
        shifted -= np.log(sum_exp)                                # shifted is now out_data
        out = Tensor(shifted, _children=(self,), _op="log_softmax")

        def _backward() -> None:
            if self.requires_grad:
                # dL/dx = dL/d(lsm) - softmax * sum(dL/d(lsm))
                _accumulate(self, out.grad - softmax * out.grad.sum(axis=-1, keepdims=True))

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def nll_loss(self, targets: np.ndarray, mask: Optional[np.ndarray] = None) -> "Tensor":
        """
        Negative log-likelihood loss over masked nodes.

        Expects `self` to contain log-probabilities (output of log_softmax).
        `targets` is an integer array of class indices, shape (N,).
        `mask`    is a boolean array of shape (N,) selecting which nodes
                  contribute to the loss (train nodes only).
        """
        N = self.data.shape[0]
        if mask is None:
            mask = np.ones(N, dtype=bool)
        idx = np.where(mask)[0]
        loss_val = -self.data[idx, targets[idx]].mean()
        out = Tensor(np.array(loss_val), _children=(self,), _op="nll_loss")

        def _backward() -> None:
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx, targets[idx]] = -1.0 / idx.shape[0]
                _accumulate(self, grad * out.grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def backward(self) -> None:
        """
        Run reverse-mode AD from this (scalar) tensor.

        Builds a topological ordering of the computation graph, then
        calls each node's _backward() closure in reverse order.
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        topo: List["Tensor"] = []
        visited: Set[int] = set()

        def build_topo(node: "Tensor") -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

    def elu(self, alpha: float = 1.0) -> "Tensor":
        """
        Exponential Linear Unit (Clevert et al., 2016).

            f(x) = x                 if x > 0
            f(x) = alpha*(exp(x)-1)  otherwise

        Smooth at zero, mean activation closer to zero -> faster convergence.
        """
        # Build the output in a single buffer: copy the input, then rewrite
        # only the non-positive entries with alpha*(exp(x)-1). Avoids the
        # full-tensor `np.exp` and `np.where` allocations from before
        # (~3 N x N buffers down to ~2 plus a bool mask).
        positive = self.data > 0
        out_data = self.data.copy()
        if not positive.all():
            neg_x = self.data[~positive]
            out_data[~positive] = alpha * (np.exp(neg_x) - 1.0)
        out = Tensor(out_data, _children=(self,), _op="elu")

        def _backward() -> None:
            if self.requires_grad:
                # d/dx ELU = 1 (positives) or out + alpha (negatives).
                # Compute the gradient in place on a copy of out.grad to
                # avoid two extra full-tensor temporaries.
                grad = out.grad.copy()
                grad[~positive] *= (out_data[~positive] + alpha)
                _accumulate(self, grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def leaky_relu(self, negative_slope: float = 0.2) -> "Tensor":
        # Compute output with one temp buffer instead of three. We start
        # with a copy of self.data and rescale just the negative entries
        # in place, then derive the backward mask from the same predicate.
        positive = self.data > 0
        out_data = self.data.copy()
        if negative_slope != 1.0:
            out_data[~positive] *= negative_slope
        out = Tensor(out_data, _children=(self,), _op="leaky_relu")

        def _backward() -> None:
            if self.requires_grad:
                # Materialise the gradient mask only when backward actually
                # runs, so we don't keep a large bool buffer alive between
                # forward and backward when grads are not needed.
                grad = out.grad.copy()
                if negative_slope != 1.0:
                    grad[~positive] *= negative_slope
                _accumulate(self, grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def t(self) -> "Tensor":
        """Differentiable matrix transpose."""
        out = Tensor(self.data.T, _children=(self,), _op="t")

        def _backward() -> None:
            if self.requires_grad:
                _accumulate(self, out.grad.T)

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def T(self) -> "Tensor":
        """
        Differentiable transpose. Aliased to `.t()` so that using `.T` on a
        Tensor never silently severs the autograd graph (a previous version
        returned a fresh, gradient-less Tensor here, which made it easy to
        introduce stealth bugs like the GAT layer-1 detachment we hit before).
        """
        return self.t()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op='{self._op}', requires_grad={self.requires_grad})"


def _ensure_tensor(x: object) -> "Tensor":
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x, dtype=np.float64))


def _accumulate(t: "Tensor", grad: np.ndarray) -> None:
    if t.grad is None:
        t.grad = np.zeros_like(t.data)
    t.grad += grad


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum grad over axes that were broadcast so it matches shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def relu(x: Tensor) -> Tensor:
    return x.relu()


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    return x.elu(alpha)


def leaky_relu(x: Tensor, negative_slope: float = 0.2) -> Tensor:
    return x.leaky_relu(negative_slope)


def hstack(tensors: List["Tensor"]) -> "Tensor":
    """
    Concatenate Tensors along axis 1.  Backward gradient is split back
    along the same axis to each input.
    """
    sizes = [t.data.shape[1] for t in tensors]
    out = Tensor(
        np.hstack([t.data for t in tensors]),
        _children=tuple(tensors),
        _op="hstack",
    )

    def _backward() -> None:
        offset = 0
        for t, s in zip(tensors, sizes):
            if t.requires_grad:
                _accumulate(t, out.grad[:, offset:offset + s])
            offset += s

    out._backward = _backward
    out.requires_grad = any(t.requires_grad for t in tensors)
    return out


def log_softmax(x: Tensor) -> Tensor:
    return x.log_softmax()


def fixed_matmul(A: np.ndarray, H: Tensor) -> Tensor:
    """
    Multiply a fixed (non-learnable) numpy matrix A by Tensor H.

    Only H receives gradients:  dL/dH = A.T @ dL/dOut
    This is used in GCNLayer so that gradients flow from one layer's
    output back to the previous layer's weights.
    """
    out = Tensor(A @ H.data, _children=(H,), _op="fixed_mm")

    def _backward() -> None:
        if H.requires_grad:
            _accumulate(H, A.T @ out.grad)

    out._backward = _backward
    out.requires_grad = H.requires_grad
    return out


# Cache CSR conversion of fixed adjacency matrices keyed by id(A). For
# typical GNN training the same A_hat is passed to fixed_sparse_matmul
# every epoch, so we pay the conversion cost exactly once per run. Bound
# the cache so it can't grow unboundedly across exotic usages.
_FIXED_CSR_CACHE: "dict[int, Tuple[sp.csr_matrix, sp.csr_matrix, tuple]]" = {}
_FIXED_CSR_CACHE_LIMIT = 4


def _csr_for_fixed(A: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return (A_csr, A_T_csr), cached by id(A)."""
    key = id(A)
    cached = _FIXED_CSR_CACHE.get(key)
    if cached is not None and cached[2] == A.shape:
        return cached[0], cached[1]
    if sp.issparse(A):
        A_csr = A.tocsr()
    else:
        A_csr = sp.csr_matrix(A)
    A_T_csr = A_csr.T.tocsr()
    if len(_FIXED_CSR_CACHE) >= _FIXED_CSR_CACHE_LIMIT:
        _FIXED_CSR_CACHE.pop(next(iter(_FIXED_CSR_CACHE)))
    _FIXED_CSR_CACHE[key] = (A_csr, A_T_csr, A.shape)
    return A_csr, A_T_csr


def fixed_sparse_matmul(A: np.ndarray, H: Tensor) -> Tensor:
    """
    Like `fixed_matmul`, but routes the multiplication through scipy.sparse.

    Forward and backward both reduce from O(N^2 * F) to O(nnz(A) * F)
    operations, which on Cora (N=2708, nnz~13K, F=1433) is a ~550x
    reduction in flops. The CSR (and its transpose) are cached on id(A),
    so the conversion only happens on the first call.

    Math is identical to `fixed_matmul`; this is purely a faster
    backend for the same op.
    """
    A_csr, A_T_csr = _csr_for_fixed(A)
    out = Tensor(A_csr @ H.data, _children=(H,), _op="fixed_sparse_mm")

    def _backward() -> None:
        if H.requires_grad:
            _accumulate(H, A_T_csr @ out.grad)

    out._backward = _backward
    out.requires_grad = H.requires_grad
    return out


def nll_loss(
    log_probs: Tensor, targets: np.ndarray, mask: Optional[np.ndarray] = None
) -> Tensor:
    return log_probs.nll_loss(targets, mask)
