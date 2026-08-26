import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn.data.cora import (
    _normalise_adjacency,
    _build_masks,
    _parse_cites,
    _parse_content,
    _row_normalise_features,
)
import scipy.sparse as sp


def _simple_adj(N=8, density=0.4, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) > (1 - density)).astype(np.float32)
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    return sp.csr_matrix(A)


class TestNormaliseAdjacency:
    def test_eigenvalues_in_range(self):
        adj = _simple_adj()
        A_hat = _normalise_adjacency(adj)
        # All entries non-negative (A is non-negative, D^{-1/2} has positive diagonal)
        assert (A_hat >= -1e-9).all()
        # Diagonal entries are 1/(degree+1) -- in (0,1]
        diag = np.diag(A_hat)
        assert (diag > 0).all()
        assert (diag <= 1.0 + 1e-9).all()

    def test_symmetry(self):
        adj = _simple_adj()
        A_hat = _normalise_adjacency(adj)
        np.testing.assert_allclose(A_hat, A_hat.T, atol=1e-9)

    def test_output_dtype(self):
        adj = _simple_adj()
        A_hat = _normalise_adjacency(adj)
        assert A_hat.dtype == np.float64

    def test_shape(self):
        N = 12
        adj = _simple_adj(N)
        A_hat = _normalise_adjacency(adj)
        assert A_hat.shape == (N, N)

    def test_isolated_node_handled(self):
        # Node 0 has no edges -- degree is 0 before self-loop, 1 after
        A = sp.csr_matrix(np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=np.float32))
        A_hat = _normalise_adjacency(A)
        assert np.isfinite(A_hat).all()


class TestBuildMasks:
    def setup_method(self):
        np.random.seed(0)
        # 7 classes, need at least 140 train + 500 val + 1000 test = 1640 nodes
        self.N = 2000
        self.labels = np.tile(np.arange(7), self.N // 7 + 1)[:self.N].astype(np.int64)

    def test_train_size(self):
        train, val, test = _build_masks(self.labels, self.N)
        assert train.sum() == 7 * 20

    def test_val_size(self):
        train, val, test = _build_masks(self.labels, self.N)
        assert val.sum() == 500

    def test_test_size(self):
        train, val, test = _build_masks(self.labels, self.N)
        assert test.sum() == 1000

    def test_no_overlap(self):
        train, val, test = _build_masks(self.labels, self.N)
        assert not (train & val).any()
        assert not (train & test).any()
        assert not (val   & test).any()

    def test_train_balanced(self):
        train, _, _ = _build_masks(self.labels, self.N)
        for c in range(7):
            assert (self.labels[train] == c).sum() == 20


class TestRowNormaliseFeatures:
    def test_rows_sum_to_one(self):
        X = np.array([[1.0, 1.0, 2.0], [0.0, 4.0, 0.0]], dtype=np.float32)
        Xn = _row_normalise_features(X)
        np.testing.assert_allclose(Xn.sum(axis=1), np.ones(2), atol=1e-6)

    def test_zero_row_left_unchanged(self):
        # A node with all-zero features must not produce NaN/inf.
        X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
        Xn = _row_normalise_features(X)
        assert np.isfinite(Xn).all()
        np.testing.assert_array_equal(Xn[0], np.zeros(3))

    def test_dtype_preserved(self):
        X = np.ones((4, 3), dtype=np.float32)
        Xn = _row_normalise_features(X)
        assert Xn.dtype == np.float32
