import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn.autograd.tensor import Tensor
from gnn.layers.gcn_layer import GCNLayer
from gnn.layers.gat_layer import (
    GATLayer,
    MultiHeadGATLayer,
    _edge_index,
    _edge_softmax,
    _gather_rows,
    _sparse_aggregate,
)


def _make_graph(N=10, F=8, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) > 0.7).astype(np.float64)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    # normalise
    A_tilde = A + np.eye(N)
    deg = A_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(deg ** -0.5)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    X = rng.standard_normal((N, F)).astype(np.float64)
    return A_hat, X, A


class TestGCNLayer:
    def test_output_shape(self):
        N, F_in, F_out = 15, 8, 6
        A_hat, X, _ = _make_graph(N, F_in)
        layer = GCNLayer(F_in, F_out, activation="relu", seed=0)
        H = layer(A_hat, Tensor(X))
        assert H.shape == (N, F_out)

    def test_log_softmax_activation(self):
        N, F = 10, 5
        A_hat, X, _ = _make_graph(N, F)
        layer = GCNLayer(F, 3, activation="log_softmax", seed=0)
        H = layer(A_hat, Tensor(X))
        probs = np.exp(H.data)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(N), atol=1e-6)

    def test_no_activation(self):
        N, F = 8, 4
        A_hat, X, _ = _make_graph(N, F)
        layer = GCNLayer(F, 4, activation="none", seed=0)
        H = layer(A_hat, Tensor(X))
        assert H.shape == (N, 4)

    def test_parameters(self):
        layer = GCNLayer(8, 4, bias=True)
        params = layer.parameters()
        assert len(params) == 2
        assert params[0].shape == (8, 4)
        assert params[1].shape == (4,)

    def test_parameters_no_bias(self):
        layer = GCNLayer(8, 4, bias=False)
        assert len(layer.parameters()) == 1

    def test_backward_runs(self):
        N, F = 12, 6
        A_hat, X, _ = _make_graph(N, F)
        layer = GCNLayer(F, 4, activation="relu", seed=1)
        H = layer(A_hat, Tensor(X))
        loss = Tensor(H.data.sum())
        H.grad = np.ones_like(H.data)
        # ensure _backward does not raise
        H._backward()

    def test_glorot_init_range(self):
        F_in, F_out = 100, 50
        layer = GCNLayer(F_in, F_out, bias=False, seed=42)
        limit = np.sqrt(6.0 / (F_in + F_out))
        assert layer.W.data.min() >= -limit - 1e-9
        assert layer.W.data.max() <=  limit + 1e-9


class TestGATLayer:
    def test_output_shape(self):
        N, F_in, F_out = 15, 8, 6
        A_hat, X, A = _make_graph(N, F_in)
        layer = GATLayer(F_in, F_out, activation="relu", seed=0)
        H = layer(A, Tensor(X))
        assert H.shape == (N, F_out)

    def test_log_softmax_activation(self):
        N, F = 10, 5
        A_hat, X, A = _make_graph(N, F)
        layer = GATLayer(F, 3, activation="log_softmax", seed=0)
        H = layer(A, Tensor(X))
        probs = np.exp(H.data)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(N), atol=1e-6)

    def test_attention_masked(self):
        """Attention weights for non-edges should be near zero."""
        N, F = 10, 6
        A_hat, X, A = _make_graph(N, F)
        layer = GATLayer(F, 4, seed=0)
        layer.forward(A, Tensor(X))
        # The alpha attribute is set during forward
        alpha = layer.last_alpha
        non_edges = (A == 0) & ~np.eye(N, dtype=bool)
        assert alpha[non_edges].max() < 1e-6

    def test_parameters(self):
        layer = GATLayer(8, 4)
        params = layer.parameters()
        assert len(params) == 3   # W, a_l, a_r
        assert params[0].shape == (8, 4)
        assert params[1].shape == (4, 1)
        assert params[2].shape == (4, 1)


class TestMultiHeadGATLayer:
    def test_concat_output_shape(self):
        N, F_in, F_per, K = 12, 6, 4, 3
        _, X, A = _make_graph(N, F_in)
        layer = MultiHeadGATLayer(F_in, F_per, n_heads=K, concat=True, seed=0)
        H = layer(A, Tensor(X))
        assert H.shape == (N, F_per * K)
        assert layer.out_features == F_per * K

    def test_mean_output_shape(self):
        N, F_in, F_per, K = 10, 5, 7, 4
        _, X, A = _make_graph(N, F_in)
        layer = MultiHeadGATLayer(F_in, F_per, n_heads=K, concat=False, seed=1)
        H = layer(A, Tensor(X))
        assert H.shape == (N, F_per)
        assert layer.out_features == F_per

    def test_parameters_count_scales_with_heads(self):
        F_in, F_per, K = 8, 4, 5
        layer = MultiHeadGATLayer(F_in, F_per, n_heads=K, concat=True, seed=0)
        # Per single-head GATLayer: W, a_l, a_r => 3 tensors. K heads => 3*K.
        assert len(layer.parameters()) == 3 * K

    def test_n_heads_one_matches_single_head(self):
        N, F_in, F_per = 9, 4, 3
        _, X, A = _make_graph(N, F_in)
        single = GATLayer(F_in, F_per, activation="none", seed=42)
        multi  = MultiHeadGATLayer(F_in, F_per, n_heads=1, concat=True, seed=42)
        # Force the single-head wrapper inside multi to use the same seed
        multi.heads[0] = single
        a = single(A, Tensor(X))
        b = multi(A, Tensor(X))
        np.testing.assert_allclose(a.data, b.data, atol=1e-9)


class TestEdgeIndex:
    def test_includes_self_loops(self):
        N = 6
        A = np.zeros((N, N))
        A[0, 1] = A[1, 0] = 1
        A[2, 3] = A[3, 2] = 1
        src, dst, indptr, n = _edge_index(A)
        assert n == N
        # Each node must have a self-loop in the edge list
        for i in range(N):
            mask = src == i
            assert i in dst[mask], f"missing self-loop for node {i}"

    def test_indptr_is_csr_consistent(self):
        N = 8
        rng = np.random.default_rng(0)
        A = (rng.random((N, N)) > 0.6).astype(np.float64)
        np.fill_diagonal(A, 0)
        src, dst, indptr, n = _edge_index(A)
        # indptr[i+1] - indptr[i] = out-degree of node i (incl. self-loop)
        for i in range(N):
            edges_in_row = src[indptr[i]:indptr[i+1]]
            assert (edges_in_row == i).all()

    def test_cache_returns_same_object(self):
        # Same id(A) -> same cached tuple
        N = 5
        A = np.eye(N)
        first  = _edge_index(A)
        second = _edge_index(A)
        assert first[0] is second[0]
        assert first[1] is second[1]


class TestGatherRows:
    def test_forward(self):
        t = Tensor(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        idx = np.array([2, 0, 0, 1], dtype=np.int32)
        out = _gather_rows(t, idx)
        np.testing.assert_array_equal(out.data, np.array([[5., 6.], [1., 2.], [1., 2.], [3., 4.]]))

    def test_backward_handles_duplicate_indices(self):
        """Plain `grad[idx] += g` would silently drop one of the duplicates."""
        t = Tensor(np.zeros((3, 2)), requires_grad=True)
        idx = np.array([0, 0, 1], dtype=np.int32)
        out = _gather_rows(t, idx)
        out.grad = np.array([[1., 1.], [2., 2.], [4., 4.]])
        out._backward()
        np.testing.assert_array_equal(t.grad, np.array([[3., 3.], [4., 4.], [0., 0.]]))


class TestEdgeSoftmax:
    def test_per_source_rows_sum_to_one(self):
        N = 5
        # Two-edge groups for nodes 0 and 1, no edges for the rest
        src = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        scores = Tensor(np.array([[1.0], [3.0], [2.0], [-1.0], [4.0]]))
        alpha = _edge_softmax(scores, src, N)
        # Per-source sums should be exactly 1
        sums = np.zeros(N)
        np.add.at(sums, src, alpha.data.reshape(-1))
        np.testing.assert_allclose(sums[:2], 1.0, atol=1e-9)

    def test_gradient_against_finite_differences(self):
        rng = np.random.default_rng(0)
        N = 4
        src = np.array([0, 0, 1, 1, 2, 3], dtype=np.int32)
        x = Tensor(rng.standard_normal((6, 1)), requires_grad=True)

        def forward_loss(data):
            t = Tensor(data, requires_grad=True)
            a = _edge_softmax(t, src, N)
            # Some scalar loss
            target = np.arange(6).reshape(6, 1).astype(np.float64) * 0.1
            loss = ((a.data - target) ** 2).sum()
            return loss, t, a

        # Analytical gradient
        loss, t, a = forward_loss(x.data)
        a.grad = 2 * (a.data - np.arange(6).reshape(6, 1).astype(np.float64) * 0.1)
        a._backward()
        analytical = t.grad.copy()

        # Finite differences
        eps = 1e-6
        numerical = np.zeros_like(x.data)
        for i in range(x.data.shape[0]):
            d = x.data.copy(); d[i, 0] += eps
            l_plus, *_  = forward_loss(d)
            d = x.data.copy(); d[i, 0] -= eps
            l_minus, *_ = forward_loss(d)
            numerical[i, 0] = (l_plus - l_minus) / (2 * eps)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)


class TestSparseAggregateMatchesDense:
    def test_forward_equals_dense_attention(self):
        """The full sparse pipeline should match a dense reference for the same inputs."""
        rng = np.random.default_rng(0)
        N, F_in, F_out = 8, 5, 4
        _, X, A = _make_graph(N, F_in, seed=0)
        layer = GATLayer(F_in, F_out, activation="none", attn_dropout=0.0,
                         negative_slope=0.2, seed=42)
        layer.eval()
        out = layer(A, Tensor(X))

        # Dense reference: build the full attention matrix the slow way.
        H = X
        H_W = H @ layer.W.data
        e_l = H_W @ layer.a_l.data        # (N, 1)
        e_r = H_W @ layer.a_r.data        # (N, 1)
        E = e_l + e_r.T
        E = np.where(E > 0, E, 0.2 * E)   # leaky_relu
        adj = (A != 0)
        np.fill_diagonal(adj, True)
        E[~adj] = -1e30
        E -= E.max(axis=1, keepdims=True)
        exp_E = np.exp(E)
        exp_E[~adj] = 0.0
        alpha = exp_E / exp_E.sum(axis=1, keepdims=True)
        expected = alpha @ H_W
        np.testing.assert_allclose(out.data, expected, atol=1e-9)
