# MLP Postmortem -- Audit and Cleanup

The MLP baseline (`gnn/models/mlp.py`) is the smallest model in the
project: two linear layers with ReLU and dropout in between, no graph
structure. It exists so we can quantify how much of GCN/GAT accuracy
comes from the *features* alone (~52 % on Cora) versus the
*neighbourhood aggregation* on top of those features (a +26 pp jump).

This file is much shorter than `gat-fix.md` or `gcn-fix.md` because the
MLP never had a correctness bug -- it just accumulated a few small
hygiene problems while the GNN code was being optimised. They were all
audited and cleaned up in the same sweep.

## What MLP does

```
X (N, 1433)                      row-normalised bag-of-words
  ->  Linear (1433 -> 64)        + bias
  ->  ReLU
  ->  Dropout (p=0.5 by default, training only)
  ->  Linear (64   -> 7)         + bias
  ->  log-softmax
  ->  NLL loss on the 140 labelled training nodes
```

`A_hat` is never used. Each prediction is a function of one node's
features in isolation. That's deliberate: it isolates the contribution
of the graph structure when we compare against GCN/GAT later.

## Update 1: separate RNGs for weight init and dropout

The original `MLP.__init__` did this:

```python
self._rng = np.random.default_rng(seed)
# ... and then weight init also used self._rng ...
W1 = Tensor(self._rng.uniform(-limit, limit, (in_features, hidden_dim)))
```

Then during forward:

```python
drop_mask = (self._rng.random(H.shape) < keep_prob).astype(np.float64)
```

Both consumed from the same RNG stream. That meant changing the weight
init implementation (e.g. switching from one Glorot draw to two for
W1/W2) silently changed the dropout sampling sequence as well. The
opposite was also true: tweaking dropout shifted W2's initial
distribution.

Fix (`gnn/models/mlp.py`):

```python
self._rng = np.random.default_rng(seed)            # dropout stream
init_rng  = np.random.default_rng(seed)            # init stream

def glorot(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return init_rng.uniform(-limit, limit, (fan_in, fan_out))
```

Two independent generators seeded from the same `seed`. Now weight
distribution is decoupled from dropout sampling, which makes
benchmark numbers stable across small refactors.

## Update 2: drop the redundant float64 copy of X

The forward pass used to begin with:

```python
H = relu(Tensor(X.astype(np.float64)) @ self.W1 + self.b1)
```

Cora's `X` is float32 (~16 MiB). `X.astype(np.float64)` copies it to a
fresh ~32 MiB buffer every single forward, train and eval. But the
`Tensor` constructor already coerces to float64 internally if it's not
already, so the explicit `astype` is redundant -- it just costs a copy.

Same fix we applied to GCN and GAT:

```python
H = relu(Tensor(X) @ self.W1 + self.b1)
```

`Tensor.__init__` does the float64 coercion exactly once when an
explicit `astype` upstream isn't required.

## Update 3: single-buffer dropout mask

The original dropout block was:

```python
keep_prob = 1.0 - self.dropout
drop_mask = (self._rng.random(H.shape) < keep_prob).astype(np.float64)
H = _differentiable_dropout(H, drop_mask / keep_prob)
```

Three temporaries:

1. `self._rng.random(H.shape)` -- float64 buffer, shape `(N, hidden)`.
2. `... < keep_prob` -- bool buffer, same shape.
3. `.astype(np.float64)` -- new float64 buffer.
4. `drop_mask / keep_prob` -- yet another float64 buffer.

For Cora and `hidden=64` that's `2708 x 64 x 8 = ~1.4 MB` per buffer, or
~5.6 MB total per training forward. Trivial in absolute terms, but the
pattern was inconsistent with the single-buffer construction we use in
GCN and GAT. So MLP got the same treatment for symmetry:

```python
keep_prob = 1.0 - self.dropout
drop_mask = self._rng.random(H.shape)         # one float64 buffer
kept = drop_mask < keep_prob                  # bool view used twice below
drop_mask[kept]  = 1.0 / keep_prob            # in-place
drop_mask[~kept] = 0.0                        # in-place
H = _differentiable_dropout(H, drop_mask)     # already-scaled mask
```

One persistent buffer, reused. Saves about three quarters of the
dropout-mask allocation churn. Mostly a hygiene fix -- the perf delta
is below noise -- but it makes the three model files share the same
dropout idiom.

## Files changed

- `gnn/models/mlp.py`:
  - `__init__`: split `self._rng` into `(self._rng, init_rng)`.
  - `forward`: `Tensor(X.astype(np.float64))` -> `Tensor(X)`,
    single-buffer dropout mask construction.
- No tests changed -- all existing MLP tests still pass unmodified, and
  the existing 86-test suite plus the 2 new sparse-matmul tests cover
  the autograd ops MLP uses.

## What MLP does NOT need

For the record, none of the GNN-specific optimisations apply to MLP:

- No adjacency matrix, so no `fixed_sparse_matmul` and no edge-index
  cache.
- No multi-head anything.
- No leaky-relu / elu paths.
- No attention, no `last_alpha`.

The autograd-level wins -- `relu` backward bool mask, `log_softmax`
captured-softmax backward, differentiable `Tensor.T`, AdamW + grad-clip
+ early stopping in the training loop -- already help MLP transparently
because they live one layer below in `gnn/autograd/tensor.py` and
`gnn/train.py`.

## Benchmark

MLP doesn't move at all from the cleanup -- ~52 % test accuracy with
~4-5 s per trial on Cora. That's the floor we measure GCN (78 %) and
GAT (81 %) against, and it's the right floor: features-only
classification on row-normalised bag-of-words is genuinely capped near
this number. The remaining ~26 pp is the value of the graph.
