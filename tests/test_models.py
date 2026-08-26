import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn.models.gcn import GCN
from gnn.models.gat import GAT
from gnn.models.mlp import MLP
from gnn.train import Adam, accuracy, train, evaluate


def _make_problem(N=40, F=20, n_classes=7, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) > 0.6).astype(np.float64)
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    A_tilde = A + np.eye(N)
    deg = A_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(deg ** -0.5)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    X = rng.standard_normal((N, F))
    y = rng.integers(0, n_classes, N)
    mask = np.zeros(N, dtype=bool); mask[:14] = True
    val_mask = np.zeros(N, dtype=bool); val_mask[14:24] = True
    return A_hat, A, X, y, mask, val_mask


class TestGCN:
    def setup_method(self):
        self.A_hat, self.A, self.X, self.y, self.mask, self.val = _make_problem()

    def test_output_shape(self):
        model = GCN(in_features=20, hidden_dim=16, n_classes=7, seed=1)
        lp, _ = model(self.A_hat, self.X)
        assert lp.shape == (40, 7)

    def test_loss_computed(self):
        model = GCN(in_features=20, hidden_dim=16, n_classes=7, seed=1)
        _, loss = model(self.A_hat, self.X, self.y, self.mask)
        assert loss is not None
        assert np.isfinite(float(loss.data))

    def test_backward_populates_grads(self):
        model = GCN(in_features=20, hidden_dim=16, n_classes=7, seed=1)
        _, loss = model(self.A_hat, self.X, self.y, self.mask)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_training_decreases_loss(self):
        model = GCN(in_features=20, hidden_dim=16, n_classes=7, seed=1)
        opt = Adam(model.parameters(), lr=0.05)
        losses = []
        for _ in range(20):
            model.train(); opt.zero_grad()
            _, l = model(self.A_hat, self.X, self.y, self.mask)
            l.backward(); opt.step()
            losses.append(float(l.data))
        assert losses[-1] < losses[0]

    def test_eval_mode_no_dropout(self):
        model = GCN(in_features=20, hidden_dim=16, n_classes=7, dropout=0.9, seed=1)
        model.eval()
        lp1, _ = model(self.A_hat, self.X)
        lp2, _ = model(self.A_hat, self.X)
        np.testing.assert_allclose(lp1.data, lp2.data)

    def test_n_parameters(self):
        model = GCN(in_features=20, hidden_dim=16, n_classes=7)
        # W1: 20x16, b1: 16, W2: 16x7, b2: 7 = 320+16+112+7 = 455
        assert model.n_parameters() == 455


class TestGAT:
    def setup_method(self):
        self.A_hat, self.A, self.X, self.y, self.mask, self.val = _make_problem()

    def test_output_shape(self):
        model = GAT(in_features=20, hidden_dim=16, n_classes=7, dropout=0.0, seed=1)
        lp, _ = model(self.A, self.X)
        assert lp.shape == (40, 7)

    def test_training_decreases_loss(self):
        # Disable dropout so the test exercises gradient flow, not regularisation noise.
        model = GAT(in_features=20, hidden_dim=16, n_classes=7, dropout=0.0, seed=1)
        opt = Adam(model.parameters(), lr=0.05)
        losses = []
        for _ in range(20):
            model.train(); opt.zero_grad()
            _, l = model(self.A, self.X, self.y, self.mask)
            l.backward(); opt.step()
            losses.append(float(l.data))
        assert losses[-1] < losses[0]

    def test_layer1_gradient_flow(self):
        """
        Regression test for the GAT bug where a stray Tensor() wrapper detached
        the computational graph between layer 2 and layer 1, leaving every
        layer-1 parameter with grad == 0 (training collapsed to ~random acc).
        """
        model = GAT(in_features=20, hidden_dim=16, n_classes=7, dropout=0.0, seed=1)
        _, loss = model(self.A, self.X, self.y, self.mask)
        loss.backward()
        for i, p in enumerate(model.layer1.parameters()):
            assert p.grad is not None, f"layer1 param {i} grad is None"
            assert np.linalg.norm(p.grad) > 0.0, (
                f"layer1 param {i} (shape={p.data.shape}) has zero gradient -- "
                "the autograd graph is detached between layers."
            )


class TestMLP:
    def setup_method(self):
        self.A_hat, self.A, self.X, self.y, self.mask, self.val = _make_problem()

    def test_output_shape(self):
        model = MLP(in_features=20, hidden_dim=16, n_classes=7, seed=1)
        lp, _ = model(self.X)
        assert lp.shape == (40, 7)

    def test_training_decreases_loss(self):
        model = MLP(in_features=20, hidden_dim=16, n_classes=7, seed=1)
        opt = Adam(model.parameters(), lr=0.05)
        losses = []
        for _ in range(20):
            model.train(); opt.zero_grad()
            _, l = model(self.X, self.y, self.mask)
            l.backward(); opt.step()
            losses.append(float(l.data))
        assert losses[-1] < losses[0]


class TestAccuracy:
    def test_perfect_predictions(self):
        from gnn.autograd.tensor import Tensor
        N, C = 10, 4
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
        logits = np.zeros((N, C))
        logits[np.arange(N), y] = 10.0
        lp = Tensor(logits)
        mask = np.ones(N, dtype=bool)
        assert accuracy(lp, y, mask) == 1.0

    def test_all_wrong(self):
        from gnn.autograd.tensor import Tensor
        N, C = 4, 3
        y = np.array([0, 0, 0, 0])
        logits = np.zeros((N, C))
        logits[:, 1] = 10.0  # always predict class 1
        lp = Tensor(logits)
        mask = np.ones(N, dtype=bool)
        assert accuracy(lp, y, mask) == 0.0
