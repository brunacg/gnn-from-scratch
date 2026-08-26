import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn.autograd.tensor import (
    Tensor,
    relu,
    log_softmax,
    nll_loss,
    elu,
    hstack,
    fixed_matmul,
    fixed_sparse_matmul,
)


def _numerical_grad(f, x, eps=1e-5):
    """Finite-difference gradient of scalar f w.r.t. numpy array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        grad[idx] = (fp - fm) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


class TestMatmul:
    def setup_method(self):
        np.random.seed(0)
        self.A = np.random.randn(4, 5) * 0.3
        self.B = np.random.randn(5, 3) * 0.3

    def test_forward(self):
        A = Tensor(self.A.copy())
        B = Tensor(self.B.copy())
        C = A @ B
        np.testing.assert_allclose(C.data, self.A @ self.B)

    def test_grad_A(self):
        B_data = self.B.copy()

        def f(a):
            return (a @ B_data).sum()

        A = Tensor(self.A.copy(), requires_grad=True)
        B = Tensor(B_data)
        C = A @ B
        C.grad = np.ones_like(C.data)
        C._backward()

        expected = _numerical_grad(f, self.A.copy())
        np.testing.assert_allclose(A.grad, expected, atol=1e-6)

    def test_grad_B(self):
        A_data = self.A.copy()

        def f(b):
            return (A_data @ b).sum()

        A = Tensor(A_data)
        B = Tensor(self.B.copy(), requires_grad=True)
        C = A @ B
        C.grad = np.ones_like(C.data)
        C._backward()

        expected = _numerical_grad(f, self.B.copy())
        np.testing.assert_allclose(B.grad, expected, atol=1e-6)


class TestAdd:
    def setup_method(self):
        np.random.seed(1)
        self.A = np.random.randn(3, 4) * 0.3
        self.B = np.random.randn(4,) * 0.3  # bias broadcast

    def test_forward(self):
        A = Tensor(self.A.copy())
        B = Tensor(self.B.copy())
        C = A + B
        np.testing.assert_allclose(C.data, self.A + self.B)

    def test_grad_broadcast(self):
        A_data = self.A.copy()
        B_data = self.B.copy()

        def f_a(a):
            return (a + B_data).sum()

        def f_b(b):
            return (A_data + b).sum()

        A = Tensor(A_data, requires_grad=True)
        B = Tensor(B_data, requires_grad=True)
        C = A + B
        C.grad = np.ones_like(C.data)
        C._backward()

        np.testing.assert_allclose(A.grad, _numerical_grad(f_a, A_data.copy()), atol=1e-6)
        np.testing.assert_allclose(B.grad, _numerical_grad(f_b, B_data.copy()), atol=1e-6)


class TestRelu:
    def setup_method(self):
        np.random.seed(2)
        self.X = np.random.randn(5, 6) * 0.5

    def test_forward(self):
        X = Tensor(self.X.copy())
        Y = X.relu()
        np.testing.assert_allclose(Y.data, np.maximum(0, self.X))

    def test_grad(self):
        X_data = self.X.copy()

        def f(x):
            return np.maximum(0, x).sum()

        X = Tensor(X_data, requires_grad=True)
        Y = X.relu()
        Y.grad = np.ones_like(Y.data)
        Y._backward()

        np.testing.assert_allclose(X.grad, _numerical_grad(f, X_data.copy()), atol=1e-6)


class TestLogSoftmax:
    def setup_method(self):
        np.random.seed(3)
        self.X = np.random.randn(4, 7) * 0.5

    def test_forward_sums_to_one(self):
        X = Tensor(self.X.copy())
        lsm = X.log_softmax()
        probs = np.exp(lsm.data)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-6)

    def test_grad(self):
        X_data = self.X.copy()

        def f(x):
            shifted = x - x.max(axis=1, keepdims=True)
            lsm = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            return lsm.sum()

        X = Tensor(X_data, requires_grad=True)
        lsm = X.log_softmax()
        lsm.grad = np.ones_like(lsm.data)
        lsm._backward()

        np.testing.assert_allclose(X.grad, _numerical_grad(f, X_data.copy()), atol=1e-5)


class TestNllLoss:
    def setup_method(self):
        np.random.seed(4)
        self.logits = np.random.randn(6, 7) * 0.3
        self.targets = np.array([0, 3, 1, 6, 2, 4])
        self.mask = np.ones(6, dtype=bool)

    def _forward_np(self, logits, targets, mask):
        shifted = logits - logits.max(axis=1, keepdims=True)
        lsm = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return -lsm[mask, targets[mask]].mean()

    def test_forward(self):
        X = Tensor(self.logits.copy())
        lsm = X.log_softmax()
        loss = lsm.nll_loss(self.targets, self.mask)
        expected = self._forward_np(self.logits, self.targets, self.mask)
        np.testing.assert_allclose(float(loss.data), expected, atol=1e-6)

    def test_grad(self):
        logits_data = self.logits.copy()
        targets = self.targets
        mask = self.mask

        def f(x):
            return self._forward_np(x, targets, mask)

        X = Tensor(logits_data, requires_grad=True)
        lsm = X.log_softmax()
        loss = lsm.nll_loss(targets, mask)
        loss.backward()

        np.testing.assert_allclose(X.grad, _numerical_grad(f, logits_data.copy()), atol=1e-5)

    def test_masked_loss(self):
        mask = np.array([True, False, True, False, True, False])
        X = Tensor(self.logits.copy(), requires_grad=True)
        lsm = X.log_softmax()
        loss = lsm.nll_loss(self.targets, mask)
        loss.backward()
        # Gradients should be zero for masked-out nodes
        assert np.all(X.grad[~mask] == 0)


class TestElu:
    def setup_method(self):
        np.random.seed(7)
        self.X = np.random.randn(5, 6) * 0.7

    def test_forward(self):
        X = Tensor(self.X.copy())
        Y = elu(X, alpha=1.0)
        expected = np.where(self.X > 0, self.X, np.exp(self.X) - 1.0)
        np.testing.assert_allclose(Y.data, expected, atol=1e-9)

    def test_grad(self):
        X_data = self.X.copy()

        def f(x):
            return np.where(x > 0, x, 1.5 * (np.exp(x) - 1.0)).sum()

        X = Tensor(X_data, requires_grad=True)
        Y = elu(X, alpha=1.5)
        Y.grad = np.ones_like(Y.data)
        Y._backward()

        np.testing.assert_allclose(X.grad, _numerical_grad(f, X_data.copy()), atol=1e-5)


class TestHstack:
    def test_forward(self):
        A = np.arange(6, dtype=np.float64).reshape(3, 2)
        B = np.arange(9, dtype=np.float64).reshape(3, 3)
        T_A, T_B = Tensor(A), Tensor(B)
        out = hstack([T_A, T_B])
        np.testing.assert_array_equal(out.data, np.hstack([A, B]))
        assert out.shape == (3, 5)

    def test_grad_splits_back(self):
        A_data = np.random.RandomState(0).randn(4, 2)
        B_data = np.random.RandomState(1).randn(4, 3)
        T_A = Tensor(A_data.copy(), requires_grad=True)
        T_B = Tensor(B_data.copy(), requires_grad=True)
        out = hstack([T_A, T_B])
        out.grad = np.arange(20, dtype=np.float64).reshape(4, 5)
        out._backward()
        np.testing.assert_array_equal(T_A.grad, out.grad[:, :2])
        np.testing.assert_array_equal(T_B.grad, out.grad[:, 2:])


class TestTransposeProperty:
    """Tensor.T must be differentiable -- a non-differentiable .T silently
    detaches the autograd graph, which is the same shape of bug we hit
    in the GAT layer-1 detachment regression."""

    def test_T_is_differentiable(self):
        X_data = np.random.RandomState(3).randn(4, 3)
        X = Tensor(X_data.copy(), requires_grad=True)
        Y = X.T  # property access
        loss = (Y.data ** 0).sum()  # only triggers backward, not the value
        Y.grad = np.ones_like(Y.data)
        Y._backward()
        assert X.grad is not None
        np.testing.assert_array_equal(X.grad, np.ones_like(X_data))


class TestFixedSparseMatmul:
    """fixed_sparse_matmul must be numerically identical to fixed_matmul."""

    def _make_sparse_adj(self, N, density=0.1, seed=0):
        rng = np.random.default_rng(seed)
        A = (rng.random((N, N)) < density).astype(np.float64)
        np.fill_diagonal(A, 1.0)
        # Symmetrise -- the GCN normalisation produces a symmetric A_hat
        A = np.maximum(A, A.T)
        return A

    def test_forward_matches_dense(self):
        rng = np.random.default_rng(0)
        N, F = 12, 5
        A = self._make_sparse_adj(N, density=0.15, seed=1)
        H = Tensor(rng.standard_normal((N, F)))
        out_dense  = fixed_matmul(A, H)
        out_sparse = fixed_sparse_matmul(A, H)
        np.testing.assert_allclose(out_sparse.data, out_dense.data, atol=1e-12)

    def test_backward_matches_dense(self):
        rng = np.random.default_rng(0)
        N, F = 12, 5
        A = self._make_sparse_adj(N, density=0.15, seed=2)
        # Two parallel runs with separate H Tensors so gradients don't mix.
        H_dense  = Tensor(rng.standard_normal((N, F)).copy(), requires_grad=True)
        H_sparse = Tensor(H_dense.data.copy(),                requires_grad=True)
        upstream = rng.standard_normal((N, F))

        out_dense = fixed_matmul(A, H_dense)
        out_dense.grad = upstream.copy()
        out_dense._backward()

        out_sparse = fixed_sparse_matmul(A, H_sparse)
        out_sparse.grad = upstream.copy()
        out_sparse._backward()

        np.testing.assert_allclose(H_sparse.grad, H_dense.grad, atol=1e-12)


class TestBackwardChain:
    """Full two-layer MLP gradient check."""

    def test_two_layer_mlp(self):
        np.random.seed(42)
        N, d_in, d_h, d_out = 8, 6, 4, 3
        X_data  = np.random.randn(N, d_in) * 0.3
        W1_data = np.random.randn(d_in, d_h) * 0.1
        W2_data = np.random.randn(d_h, d_out) * 0.1
        targets = np.random.randint(0, d_out, N)
        mask    = np.ones(N, dtype=bool)

        def f_np(w1, w2):
            H = np.maximum(0, X_data @ w1)
            Z = H @ w2
            s = Z - Z.max(axis=1, keepdims=True)
            lsm = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
            return -lsm[mask, targets[mask]].mean()

        W1 = Tensor(W1_data.copy(), requires_grad=True)
        W2 = Tensor(W2_data.copy(), requires_grad=True)
        H  = (Tensor(X_data) @ W1).relu()
        lsm = (H @ W2).log_softmax()
        loss = lsm.nll_loss(targets, mask)
        loss.backward()

        expected_W1 = _numerical_grad(lambda w1: f_np(w1, W2_data), W1_data.copy())
        expected_W2 = _numerical_grad(lambda w2: f_np(W1_data, w2), W2_data.copy())

        np.testing.assert_allclose(W1.grad, expected_W1, atol=1e-5)
        np.testing.assert_allclose(W2.grad, expected_W2, atol=1e-5)
