"""Tests: each checks a statement of Hochreiter & Schmidhuber (1997). A few seconds in total.

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest
import torch

from lstm import LSTM1997, LSTM1997Torch, VanillaRNN, df, error_scaling, g, h, max_sigmoid_factor
from tasks import (adding_problem, embedded_reber, multiplication, noise_free_2a, reber_is_valid, temporal_order)

rng = np.random.default_rng(0)


# ---- Appendix A.1 ----

def test_squashing_function_ranges():
    x = np.linspace(-50, 50, 1001)
    assert g(x).min() == pytest.approx(-2) and g(x).max() == pytest.approx(2)
    assert h(x).min() == pytest.approx(-1) and h(x).max() == pytest.approx(1)
    assert df(x).max() == pytest.approx(0.25)


# ---- Section 3.1: vanishing error ----

def test_logistic_errors_vanish_when_weights_are_below_4():
    for w in (1.0, 2.0, 3.9):
        assert max_sigmoid_factor(w) < 1.0                       # 0.25 |w| < 1
    assert error_scaling([max_sigmoid_factor(3.9)] * 100) < 0.1  # even the BEST case at every step: 0.975^100
    assert error_scaling([max_sigmoid_factor(4.5)] * 100) > 1e5  # |w| > 4 can make it BLOW UP instead


def test_error_vanishes_exponentially_with_the_time_lag():
    nets = rng.normal(size=200)
    factors = df(nets) * 1.0                                      # w = 1, real derivative values
    e10, e100 = error_scaling(factors[:10]), error_scaling(factors[:100])
    assert e10 < 0.25 ** 10 * 1.0001 and e100 < 1e-60


def test_constant_error_carrousel_has_factor_one():
    # f linear and w_jj = 1.0  =>  f'(net) * w = 1 at every step, whatever q is
    assert error_scaling(np.ones(1000)) == 1.0


# ---- the CEC really stores information ----

def test_closed_input_gate_keeps_the_state_constant_for_1000_steps():
    net = LSTM1997Torch(1, 1, 1, 1, dtype=torch.float64)
    with torch.no_grad():
        net.W_in.zero_(); net.W_in[0, 0] = 50.0; net.W_in[0, -1] = -25.0     # gate opens only when x = 1
        net.W_c.zero_(); net.W_c[0, -1] = 2.0                                  # cell input g(2) > 0
    xs = torch.zeros(1000, 1, 1, dtype=torch.float64)
    xs[0, 0, 0] = 1.0                                                          # one 'write' at t = 0
    _, states = net(xs, return_states=True)
    stored = states[0, 0, 0].item()
    assert stored > 1.0
    assert torch.allclose(states[:, 0, 0], torch.full((1000,), stored, dtype=torch.float64), atol=1e-9)


def test_error_flows_back_through_the_cec_unchanged():
    net = LSTM1997Torch(1, 1, 1, 1, dtype=torch.float64)
    xs = torch.randn(500, 1, 1, dtype=torch.float64)
    s0 = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
    _, states = net(xs, return_states=True, s0=s0)
    # d s(500) / d s(0): with truncation, s(t) = s(t-1) + (terms that don't depend on s), so exactly 1
    grad, = torch.autograd.grad(states[-1].sum(), s0)
    assert grad.item() == pytest.approx(1.0)


# ---- Appendix A.1: the truncated learning rule ----

def make_pair(seed=0):
    np_net = LSTM1997(n_in=3, n_blocks=2, block_size=2, n_out=2, init=0.5, seed=seed)
    t_net = LSTM1997Torch(3, 2, 2, 2, truncate=True, init=0.5, dtype=torch.float64, seed=seed)
    return np_net, t_net


def test_numpy_and_torch_forward_agree():
    np_net, t_net = make_pair()
    xs = rng.normal(size=(12, 3))
    ys_np = np.array([st["y"] for st in np_net.forward(xs)])
    ys_t = t_net(torch.tensor(xs)[:, None, :]).detach().numpy()[:, 0]
    assert np.allclose(ys_np, ys_t)


def test_paper_truncated_gradient_equals_autograd_with_detached_recurrence():
    np_net, t_net = make_pair(seed=3)
    T = 15
    xs = rng.normal(size=(T, 3))
    targets = rng.random((T, 2))
    mask = np.zeros(T, bool); mask[[4, 9, 14]] = True                         # errors only at some steps
    loss_np, grads_np = np_net.truncated_gradient(xs, targets, mask)
    out = t_net(torch.tensor(xs)[:, None, :])[:, 0]
    m = torch.tensor(mask)
    loss_t = 0.5 * ((out[m] - torch.tensor(targets)[m]) ** 2).sum()
    loss_t.backward()
    assert loss_np == pytest.approx(loss_t.item())
    for g_np, p in zip(grads_np, (t_net.W_in, t_net.W_out, t_net.W_c, t_net.W_k)):
        assert np.allclose(g_np, p.grad.numpy(), atol=1e-10)


def test_truncated_gradient_differs_from_full_bptt():
    xs = torch.tensor(rng.normal(size=(10, 1, 3)))
    grads = []
    for trunc in (True, False):
        net = LSTM1997Torch(3, 2, 2, 2, truncate=trunc, init=0.5, dtype=torch.float64, seed=1)
        net(xs)[-1].sum().backward()
        grads.append(net.W_c.grad.clone())
    assert not torch.allclose(grads[0], grads[1])


def test_sgd_with_the_paper_rule_reduces_the_error():
    net = LSTM1997(n_in=2, n_blocks=1, block_size=2, n_out=1, init=0.2, seed=0)
    xs = rng.normal(size=(8, 2))
    targets = np.full((8, 1), 0.9)
    mask = np.ones(8, bool)
    first = net.sgd_step(xs, targets, mask, 0.5)
    for _ in range(50):
        last = net.sgd_step(xs, targets, mask, 0.5)
    assert last < first


def test_vanilla_rnn_runs():
    assert VanillaRNN(3, 5, 2)(torch.randn(7, 4, 3)).shape == (7, 4, 2)


# ---- the tasks ----

def test_embedded_reber_strings_are_valid_and_targets_are_legal():
    r = np.random.default_rng(1)
    for _ in range(200):
        s, X, Y = embedded_reber(r)
        assert reber_is_valid(s)
        idx = {c: i for i, c in enumerate("BTSXPVE")}
        for t in range(len(s) - 1):
            assert Y[t, idx[s[t + 1]]] == 1                                     # the true next symbol is legal
        assert Y[-2].sum() == 1 and Y[-2, idx[s[1]]] == 1                        # the long dependency


def test_task_2a_needs_the_first_symbol():
    X, Y = noise_free_2a(p=10, rng=np.random.default_rng(0))
    assert X.shape == (10, 11) and Y.shape == (10, 11)
    assert np.argmax(Y[-1]) == np.argmax(X[0])                                   # last target = first input


def test_adding_problem_structure():
    r = np.random.default_rng(2)
    for _ in range(100):
        x, target = adding_problem(100, r)
        assert 100 <= len(x) <= 110
        marked = np.where(x[:, 1] == 1.0)[0]
        assert len(marked) == 2 and marked.min() < 10 and marked.max() < 49
        assert target == pytest.approx(0.5 + x[marked, 0].sum() / 4)
        assert 0.0 <= target <= 1.0


def test_multiplication_target():
    x, target = multiplication(100, np.random.default_rng(3))
    marked = np.where(x[:, 1] == 1.0)[0]
    assert target == pytest.approx(np.prod(x[marked, 0]))


def test_temporal_order_class():
    r = np.random.default_rng(4)
    for _ in range(50):
        X, cls, (t1, t2) = temporal_order(r)
        s1, s2 = np.argmax(X[t1]), np.argmax(X[t2])
        assert {6, 7} >= {s1, s2} and cls == 2 * (s1 - 6) + (s2 - 6)
        assert 10 <= t1 <= 20 and 50 <= t2 <= 60 and 100 <= len(X) <= 110
