"""Tests: each checks an equation or a claim from Du et al. (2022).

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest

from experiment_iris import load_iris, split_and_scale
from mlp import MLP
from optimizers import METHODS, line_search, train
from perceptron import accuracy, lms_train, perceptron_train, pocket_train

X_LOGIC = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


@pytest.fixture(scope="module")
def iris():
    X, labels = load_iris()
    return split_and_scale(X, labels, np.random.default_rng(0))


# ---- Section 3: single-layer perceptron ----

@pytest.mark.parametrize("y", [[0, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 1, 0]])
def test_perceptron_learns_separable_gates(y):
    # AND, OR, NAND and "x1 AND NOT x2" are all linearly separable.
    w, th, ok, _, _ = perceptron_train(X_LOGIC, np.array(y), rng=0)
    assert ok and accuracy(w, th, X_LOGIC, np.array(y)) == 1.0


def test_perceptron_never_converges_on_xor():
    # Section 3.2: on inseparable data the algorithm "cannot stop the iteration".
    _, _, ok, _, errs = perceptron_train(X_LOGIC, np.array([0, 1, 1, 0]), max_epochs=500, rng=0)
    assert not ok and min(errs) > 0


def test_perceptron_convergence_theorem_on_random_separable_data():
    # Perceptron convergence theorem: separable data -> zero errors in finite time.
    for seed in range(10):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(100, 5))
        w_true = rng.normal(size=5)
        margin = X @ w_true
        keep = np.abs(margin) > 0.3                  # leave a gap between the classes
        X, y = X[keep], (margin[keep] > 0).astype(int)
        _, _, ok, _, _ = perceptron_train(X, y, max_epochs=1000, rng=seed)
        assert ok


def test_pocket_gets_best_line_for_xor():
    _, _, acc = pocket_train(X_LOGIC, np.array([0, 1, 1, 0]), steps=500, rng=0)
    assert acc == 0.75


@pytest.mark.parametrize("normalized, lr", [(False, 0.02), (True, 0.1)])
def test_lms_reaches_least_squares_solution(normalized, lr):
    # LMS (Eq. 19 with update 18, or alpha-LMS Eq. 21) minimizes the MSE, so it
    # should end near the exact least-squares weights (small lr -> close).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.3 + rng.normal(0, 0.1, 50)
    w, th, hist = lms_train(X, y, lr=lr, epochs=300, normalized=normalized, rng=0)
    exact = np.linalg.lstsq(np.c_[X, np.ones(50)], y, rcond=None)[0]
    assert np.allclose(np.r_[w, th], exact, atol=0.05)
    assert hist[-1] < hist[0]


# ---- Section 4: MLP and backpropagation ----

@pytest.mark.parametrize("hidden, output", [("logistic", "linear"), ("tanh", "logistic"), ("relu", "linear")])
def test_backprop_gradient_matches_finite_differences(hidden, output):
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))
    net = MLP([3, 4, 3, 2], hidden, output, rng=1)
    net.masks[1][0] = 0                              # include a dead node, too
    w = net.get_params()
    g = net.gradient(X, Y, decay=0.01)
    fd = np.zeros_like(w)
    for i in range(len(w)):
        step = np.zeros_like(w)
        step[i] = 1e-6
        net.set_params(w + step); up = net.loss(X, Y, 0.01)
        net.set_params(w - step); down = net.loss(X, Y, 0.01)
        fd[i] = (up - down) / 2e-6
    assert np.allclose(g, fd, atol=1e-6)


def test_jacobian_matches_finite_differences_and_gradient():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(5, 4)), rng.normal(size=(5, 3))
    net = MLP([4, 4, 3], rng=2)
    w = net.get_params()
    e, J = net.jacobian(X, Y)
    Jfd = np.zeros_like(J)
    for i in range(len(w)):
        step = np.zeros_like(w)
        step[i] = 1e-6
        net.set_params(w + step); up = (net.predict(X) - Y).ravel()
        net.set_params(w - step); down = (net.predict(X) - Y).ravel()
        Jfd[:, i] = (up - down) / 2e-6
    net.set_params(w)
    assert np.allclose(J, Jfd, atol=1e-6)
    assert np.allclose(J.T @ e / len(X), net.gradient(X, Y))   # Eq. (45): g = J^T e


def test_params_round_trip():
    net = MLP([4, 5, 3], rng=0)
    w = net.get_params()
    assert len(w) == net.n_params == 4 * 5 + 5 + 5 * 3 + 3
    net.set_params(w * 2)
    assert np.allclose(net.get_params(), w * 2)


def xor_successes(method, seeds=20):
    """How many of `seeds` random starts end with XOR solved."""
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    ok = 0
    for seed in range(seeds):
        net = MLP([2, 3, 1], hidden="logistic", output="logistic", rng=seed)
        train(net, X_LOGIC, Y, method=method, epochs=10000, goal=1e-3, lr=2.0)
        ok += np.array_equal(net.predict(X_LOGIC).ravel() > 0.5, [False, True, True, False])
    return ok


@pytest.mark.parametrize("method", ["gd", "gdm", "lm", "bfgs", "rprop", "scg", "cgp"])
def test_mlp_solves_xor(method):
    # Section 4.1: a hidden layer overcomes the linear-separability limit.
    # Not every random start succeeds (local minima), but most do.
    assert xor_successes(method) >= 10


def test_second_order_gets_trapped_more_often_than_bp():
    # Section 8: "second-order methods become trapped in a local minimum more
    # frequently than the BP algorithm". On XOR: BP solves 19/20 starts, BFGS 12/20.
    assert xor_successes("bfgs") < xor_successes("gd")


def test_momentum_speeds_up_bp():
    # Section 4.5: momentum improves convergence.
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    epochs = {}
    for method in ["gd", "gdm"]:
        net = MLP([2, 3, 1], hidden="logistic", output="logistic", rng=3)
        epochs[method] = len(train(net, X_LOGIC, Y, method=method, epochs=10000, goal=1e-3, lr=2.0))
    assert epochs["gdm"] < epochs["gd"] / 3


def test_line_search_finds_a_decrease():
    f = lambda v: float(np.sum((v - 3) ** 2))
    w = np.zeros(2)
    g = 2 * (w - 3)
    a, fa = line_search(f, w, -g, f(w), g @ -g, 1.0)
    assert a > 0 and fa < f(w)


# ---- Sections 7-8 and 12.1: every training algorithm on Iris ----

@pytest.mark.parametrize("method", METHODS)
def test_every_method_learns_iris(iris, method):
    Xtr, Ytr, ltr, Xte, lte = iris
    net = MLP([4, 4, 3], rng=0)
    hist = train(net, Xtr, Ytr, method=method, epochs=300, goal=1e-3, rng=0)
    assert hist[-1] < hist[0] / 5                          # training error fell a lot
    assert np.mean(net.predict(Xte).argmax(1) == lte) >= 0.9


def test_second_order_methods_stop_first(iris):
    # Table 1 / Section 12.2: LM and BFGS need far fewer epochs than RProp.
    Xtr, Ytr, *_ = iris
    epochs = {}
    for method in ["lm", "bfgs", "rprop"]:
        runs = []
        for seed in range(5):
            net = MLP([4, 4, 3], rng=seed)
            runs.append(len(train(net, Xtr, Ytr, method=method, epochs=1000, goal=1e-3)) - 1)
        epochs[method] = np.mean(runs)
    assert epochs["lm"] < epochs["rprop"] / 2
    assert epochs["bfgs"] < epochs["rprop"] / 2


# ---- Section 5: generalization ----

def test_weight_decay_shrinks_weights():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (20, 1))
    Y = np.sin(np.pi * X) + rng.normal(0, 0.25, X.shape)
    norms = []
    for decay in [0.0, 1e-3]:
        net = MLP([1, 20, 1], hidden="tanh", rng=0)
        train(net, X, Y, method="lm", epochs=200, goal=0, decay=decay)
        norms.append(np.linalg.norm(net.get_params()))
    assert norms[1] < norms[0] / 3


def test_early_stopping_keeps_best_validation_weights():
    rng = np.random.default_rng(0)
    X, Xv = rng.uniform(-1, 1, (20, 1)), rng.uniform(-1, 1, (20, 1))
    Y = np.sin(np.pi * X) + rng.normal(0, 0.25, X.shape)
    Yv = np.sin(np.pi * Xv) + rng.normal(0, 0.25, Xv.shape)
    net = MLP([1, 20, 1], hidden="tanh", rng=0)
    log = []
    train(net, X, Y, method="lm", epochs=200, goal=0, val=(Xv, Yv), val_log=log)
    final_val = np.mean((net.predict(Xv) - Yv) ** 2)
    assert np.isclose(final_val, min(log))                  # restored the best epoch
    assert min(log) < log[-1]                               # and it wasn't the last one


# ---- Section 10: fault tolerance ----

def test_open_node_fault_zeroes_the_node():
    net = MLP([4, 5, 3], rng=0)
    net.masks[0][2] = 0
    hidden = net.forward(np.ones((1, 4)))[1]
    assert hidden[0, 2] == 0 and np.all(hidden[0, [0, 1, 3, 4]] != 0)


def test_fault_injection_training_repairs_masks_and_helps(iris):
    Xtr, Ytr, ltr, Xte, lte = iris

    def worst_single_fault(net):
        worst = 1.0
        for j in range(net.sizes[1]):
            net.masks[0][:] = 1
            net.masks[0][j] = 0
            worst = min(worst, np.mean(net.predict(Xte).argmax(1) == lte))
        net.masks[0][:] = 1
        return worst

    scores = {}
    for rate in [0.0, 0.2]:
        worst = []
        for seed in range(3):
            net = MLP([4, 10, 3], rng=seed)
            train(net, Xtr, Ytr, method="sgd", epochs=200, goal=0, lr=0.02, rng=seed, fault_rate=rate)
            assert all(np.all(m == 1) for m in net.masks)    # no node left broken
            worst.append(worst_single_fault(net))
        scores[rate] = np.mean(worst)
    assert scores[0.2] > scores[0.0]
