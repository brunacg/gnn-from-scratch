# GCN Postmortem -- The Aggregation Hot Path

The GCN model on Cora was correct from day one: it trained to ~78 %
test accuracy and matched the Kipf & Welling (2017) numbers. Unlike
GAT, it never had a silent gradient-flow bug. So why does it deserve a
postmortem?

Because it had a *performance* problem hiding in plain sight: the
neighbourhood aggregation `A_hat @ H` was a dense `(N, N) @ (N, F)`
matmul on a graph that's 99.8 % zeros. Fixing that cut trial time on
Cora from ~77 s to ~10.5 s -- a 7.3x speedup with zero accuracy
change. This file documents that change and a few smaller hygiene
fixes that landed alongside it.

## What GCN does

```
Input X (N, 1433)                    row-normalised bag-of-words
  ->  GCNLayer 1 (1433 -> 64, ReLU)  H' = ReLU( A_hat @ H @ W1 + b1 )
  ->  Dropout (training only)
  ->  GCNLayer 2 (64 -> 7,    log-softmax)
  ->  NLL loss on the 140 labelled training nodes
```

The defining op is the renormalised aggregation `A_hat @ H`, where
`A_hat = D^{-1/2} (A + I) D^{-1/2}`. For Cora, `A_hat` has shape
`(2708, 2708)` and ~13 K nonzeros (out of 7.3 M entries). The matmul
`A_hat @ H` does `N^2 * F` multiplications -- ~10 GFLOPs for layer 1
alone -- and ~99.8 % of those multiplications are by zero.

## Update 1: stop converting `A_hat` to a Python list of lists

The historical `_normalise_adjacency` ended with:

```python
return np.array((D_inv_sqrt @ A_tilde @ D_inv_sqrt).todense()).astype(np.float64)
```

`scipy.sparse.csr_matrix.todense()` returns a `numpy.matrix` object
(legacy 2-D `np.matrix`, not `ndarray`). Wrapping it in `np.array(...)`
forced a Python-level traversal over the matrix to copy it into an
ndarray. For Cora's 56 MiB adjacency that's not a hot path -- it runs
once per process -- but it's the kind of thing that catches you off
guard when you scale to a bigger graph.

Fix in `gnn/data/cora.py`:

```python
return (D_inv_sqrt @ A_tilde @ D_inv_sqrt).toarray().astype(np.float64)
```

`csr_matrix.toarray()` returns an `ndarray` directly via a single C
loop. Order of magnitude faster, same result.

## Update 2: drop the redundant float64 copy of X in the model

The original GCN forward did this on every train and eval pass:

```python
H = Tensor(X.astype(np.float64))
H = self.layer1(A_hat, H)
```

`X.astype(np.float64)` creates a fresh ~32 MiB buffer (Cora `X` is
2708 x 1433 float32). But `Tensor.__init__` already coerces to float64
internally, so the explicit `astype` was just paying the copy cost
twice -- once for the explicit `astype`, then implicitly via `Tensor`.

Fix in `gnn/models/gcn.py`:

```python
H = self.layer1(A_hat, Tensor(X))
```

Saves ~32 MiB of allocation per forward -- minor for a single epoch,
adds up over a long training run.

## Update 3: single-buffer dropout mask

The original dropout block was:

```python
keep_prob = 1.0 - self.dropout
drop_mask = (self._rng.random(H.shape) < keep_prob).astype(np.float64)
H = _differentiable_dropout(H, drop_mask / keep_prob)
```

Three temporaries:

1. `random(H.shape)` -- float64 buffer.
2. `< keep_prob` -- bool buffer.
3. `.astype(np.float64)` -- new float64 buffer.
4. `drop_mask / keep_prob` -- yet another float64 buffer.

The hidden activation `H` here has shape `(N, hidden_dim)` =
`(2708, 64)` = ~1.4 MB. Four buffers of that size = ~5.6 MB per
training forward, allocated and immediately discarded.

Fix in `gnn/models/gcn.py`:

```python
keep_prob = 1.0 - self.dropout
drop_mask = self._rng.random(H.shape)         # one float64 buffer
kept = drop_mask < keep_prob
drop_mask[kept]  = 1.0 / keep_prob            # in-place
drop_mask[~kept] = 0.0                        # in-place
H = _differentiable_dropout(H, drop_mask)     # already-scaled mask
```

One buffer, mutated in place. Saves ~75 % of the dropout allocation
churn. Same idiom is now used in GCN, GAT, and MLP.

## Update 4: sparse aggregation -- the big one

This is the change that delivered most of GCN's speedup. The dense
matmul `A_hat @ H` was costing ~10 GFLOPs per layer per forward on
Cora, with ~99.8 % of those operations being multiplications by zero.

### The slow path

`gnn/layers/gcn_layer.py` used to call `fixed_matmul(A_hat, H)`, which
is a thin autograd wrapper around `numpy`'s dense matmul:

```python
def fixed_matmul(A: np.ndarray, H: Tensor) -> Tensor:
    out = Tensor(A @ H.data, _children=(H,), _op="fixed_mm")
    def _backward():
        if H.requires_grad:
            _accumulate(H, A.T @ out.grad)
    out._backward = _backward
    out.requires_grad = H.requires_grad
    return out
```

Both `A @ H.data` (forward) and `A.T @ out.grad` (backward) are dense
N x N matmuls. For Cora that's `2 * (2708 * 2708 * F)` flops per layer
per epoch. With `F=1433` for layer 1 input, that's ~21 GFLOPs per
epoch per layer per direction.

### The fix

A new differentiable op `fixed_sparse_matmul(A, H)` in
`gnn/autograd/tensor.py`:

```python
def fixed_sparse_matmul(A: np.ndarray, H: Tensor) -> Tensor:
    A_csr, A_T_csr = _csr_for_fixed(A)            # cached on id(A)
    out = Tensor(A_csr @ H.data, _children=(H,), _op="fixed_sparse_mm")
    def _backward():
        if H.requires_grad:
            _accumulate(H, A_T_csr @ out.grad)
    out._backward = _backward
    out.requires_grad = H.requires_grad
    return out
```

Two key tricks:

1. **CSR conversion cached on `id(A)`**. The first call to
   `fixed_sparse_matmul(A_hat, ...)` builds `(A_csr, A_csr.T)` and
   stashes them in `_FIXED_CSR_CACHE`. Every subsequent call -- every
   layer, every epoch, train and eval -- reuses those exact same CSR
   matrices. The conversion runs once per training run.
2. **Transpose precomputed**. Backward needs `A.T @ out.grad`. Since
   `A_hat` is symmetric (renormalised adjacency), `A_csr.T == A_csr`
   structurally, but we still cache `A_T_csr` explicitly so the same
   path works for non-symmetric fixed adjacencies.

The forward and backward then both run through scipy.sparse's
optimised C routines for sparse-times-dense matmul, which is
approximately `O(nnz(A) * F)` rather than `O(N^2 * F)`. On Cora that's
a ~550x reduction in flops.

### What this saves on time

3-trial benchmark, 200 epochs, `patience=20`:

| Path | GCN trial time | GCN test acc |
|------|----------------|--------------|
| Dense `fixed_matmul`   | ~77 s   | 78.3 +/- 0.25 |
| Sparse `fixed_sparse_matmul` | **~10.5 s** | **78.30 +/- 0.24** |

7.3x speedup with bit-identical accuracy. The full multi-model
benchmark (MLP + GCN + GAT, 3 seeds, 200 epochs) now finishes in ~107
seconds total.

The speedup is "only" 7.3x rather than the theoretical 550x because
the per-epoch cost is not just the aggregation. We also pay for:

- The dense `(N, F_in) -> (N, F_out)` projection `AH @ W` (which is
  the right thing -- W is dense and learnable).
- The two `Tensor` allocations the autograd engine creates per op.
- AdamW updates over all parameters.
- Dropout, log-softmax, NLL loss, and validation forwards.

`A_hat @ H` was the single biggest line item, and it's gone.

### Correctness

The sparse path is mathematically identical to the dense one for the
same `(A, H)`. This is enforced by
`tests/test_autograd.py::TestFixedSparseMatmul`, which runs both
`fixed_matmul` and `fixed_sparse_matmul` on the same inputs and
asserts bit-equivalence in both forward and backward
(`atol=1e-12`).

## What stays dense

Two things are deliberately left as dense matmuls:

1. **The weight projections `(N, F_in) @ (F_in, F_out)`**. These are
   already dense-times-dense, with no structural sparsity to exploit.
   `numpy.matmul` is the right call.
2. **The public `A_hat` returned by `cora.py`**. We still hand back a
   dense `(N, N)` ndarray so the notebook visualisation code that
   indexes into `A_hat` directly (e.g. computing per-node degrees,
   plotting subgraphs) keeps working. The CSR conversion is opaque,
   internal to `fixed_sparse_matmul`.

This means `cora.py` does no extra work: it builds `A_hat` sparse,
toarray's it once at the end of `_normalise_adjacency`, and the GCN
layer converts it back to CSR on the first forward via the cache. The
double conversion is one-time-only.

## Files changed

- `gnn/autograd/tensor.py`:
  - Added `import scipy.sparse as sp`.
  - Added `_FIXED_CSR_CACHE` and `_csr_for_fixed(A)` helper.
  - Added `fixed_sparse_matmul` autograd op.
- `gnn/autograd/__init__.py`:
  - Re-exported `fixed_sparse_matmul`.
- `gnn/layers/gcn_layer.py`:
  - Switched the import from `fixed_matmul` to `fixed_sparse_matmul`.
  - `forward` now calls `fixed_sparse_matmul(A_hat, H)`.
- `gnn/models/gcn.py`:
  - Dropped redundant `Tensor(X.astype(np.float64))` -> `Tensor(X)`.
  - Dropout mask built in a single buffer instead of three.
- `gnn/data/cora.py`:
  - `_normalise_adjacency` ends with `.toarray().astype(np.float64)`
    instead of `np.array((...).todense()).astype(...)`.
- `tests/test_autograd.py`:
  - New `TestFixedSparseMatmul` class with two tests asserting
    forward + backward equivalence to `fixed_matmul` (`atol=1e-12`).

## What's still on the table

If we ever wanted to push GCN trial time below 10 s, the remaining
hot paths would be:

- The `(N, F_in) @ (F_in, F_out)` weight projections. These are dense
  matmuls. A float32 path through autograd would roughly halve their
  cost; the numerical risk is small for forward but Adam's `v_t`
  squared-gradient accumulator can be tricky in float32.
- The activation memory churn. Each layer's `H` is held live by the
  autograd graph until backward completes. A "checkpoint a forward to
  recompute on backward" trick could trade compute for memory, but on
  a 2-layer model it isn't worth the complexity.

These are not currently planned; the file is here for completeness.

## In one sentence

GCN was correct but slow because it did dense matmul against a 99.8 %
zero adjacency; routing the same op through `scipy.sparse` with a
once-per-run CSR cache cut trial time from ~77 s to ~10.5 s on Cora,
with no change in accuracy and a unit test that pins the math against
the dense reference.
