"""Tests for Sutskever's thesis (2013): Chapter 3 (RTRBM) and Chapter 4 (HF with structural damping).
A few seconds in total.

Run with:  python3 -m pytest -q
"""

import itertools

import numpy as np
import pytest
import torch

from hf_rnn import RNN, HFStructural, loss_fn
from rtrbm import RTRBM, rbm_grad_exact, rbm_log_prob_exact, sigmoid
from tasks import (PROBLEMS, addition, bouncing_balls, error_rate, memorization, random_permutation, temporal_order,
                   temporal_order_3bit, xor)

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Chapter 3: RTRBM
# ---------------------------------------------------------------------------

def test_exact_rbm_probabilities_sum_to_one():
    W, bv, bh = rng.normal(size=(2, 3)), rng.normal(size=3), rng.normal(size=2)
    total = sum(np.exp(rbm_log_prob_exact(W, bv, bh, np.array(s, float))) for s in itertools.product([0, 1], repeat=3))
    assert total == pytest.approx(1.0)


def test_exact_rbm_gradient_matches_finite_differences():
    W, bv, bh = rng.normal(size=(2, 3)), rng.normal(size=3), rng.normal(size=2)
    v = np.array([1.0, 0.0, 1.0])
    dW, dbv, dbh = rbm_grad_exact(W, bv, bh, v)
    e = 1e-6
    W2 = W.copy(); W2[1, 2] += e
    assert dW[1, 2] == pytest.approx((rbm_log_prob_exact(W2, bv, bh, v) - rbm_log_prob_exact(W, bv, bh, v)) / e, rel=1e-4)
    bh2 = bh.copy(); bh2[0] += e
    assert dbh[0] == pytest.approx((rbm_log_prob_exact(W, bv, bh2, v) - rbm_log_prob_exact(W, bv, bh, v)) / e, rel=1e-4)


def test_inference_is_the_deterministic_recursion():
    m = RTRBM(4, 3, seed=1)
    vs = (rng.random((5, 4)) < 0.5).astype(float)
    rs = m.infer(vs)
    assert np.allclose(rs[0], m.r0)
    assert np.allclose(rs[3], sigmoid(m.W @ vs[2] + m.bh + m.Wp @ rs[2]))


def torch_log_prob(m, vs):
    """The exact RTRBM log-likelihood in torch, for autograd (tiny models)."""
    P = {k: torch.tensor(v, requires_grad=True) for k, v in zip(["W", "Wp", "bv", "bh", "r0"], m.params())}
    states = torch.tensor(list(itertools.product([0.0, 1.0], repeat=m.V)), dtype=torch.float64)
    r, total = P["r0"], 0.0
    for v in torch.tensor(vs):
        b = P["bh"] + P["Wp"] @ r
        negF = lambda s: s @ P["bv"] + torch.nn.functional.softplus(s @ P["W"].T + b).sum(-1)
        total = total + negF(v) - torch.logsumexp(negF(states), 0)
        r = torch.sigmoid(P["W"] @ v + P["bh"] + P["Wp"] @ r)
    total.backward()
    return total.item(), [P[k].grad.numpy() for k in ["W", "Wp", "bv", "bh", "r0"]]


def test_bptt_with_exact_rbm_gradients_is_the_exact_gradient():
    # Section 3.10: 'the gradient of the RTRBM would be computed exactly if CD were replaced with
    # the derivatives of the RBM's log probability'
    m = RTRBM(4, 3, init_std=0.5, seed=2)
    m.W = rng.normal(0, 0.5, (3, 4))
    vs = (rng.random((6, 4)) < 0.5).astype(float)
    lp, grads_torch = torch_log_prob(m, vs)
    assert m.log_prob_exact(vs) == pytest.approx(lp)
    for ours, theirs in zip(m.gradient(vs, exact=True), grads_torch):
        assert np.allclose(ours, theirs, atol=1e-8)


def test_cd_gradient_is_a_noisy_estimate_of_the_exact_one():
    m = RTRBM(4, 3, init_std=0.3, seed=3)
    vs = (rng.random((3, 4)) < 0.5).astype(float)
    exact = m.gradient(vs, exact=True)[0]
    cd = np.mean([m.gradient(vs, k=20)[0] for _ in range(400)], axis=0)
    assert np.corrcoef(exact.ravel(), cd.ravel())[0, 1] > 0.9


def test_sampling_shapes():
    m = RTRBM(9, 4)
    assert m.sample(5, gibbs=3).shape == (5, 9)


# ---------------------------------------------------------------------------
# Chapter 4: problems
# ---------------------------------------------------------------------------

def test_addition_problem_matches_section_4_4_1():
    X, Y, M, kind = addition(100, 50, np.random.default_rng(1))
    assert kind == "mse" and M[-1].sum() == 50 and M[:-1].sum() == 0
    for n in range(50):
        marks = torch.nonzero(X[:, n, 1]).ravel()
        assert len(marks) == 2
        assert Y[-1, n, 0].item() == pytest.approx(X[marks, n, 0].sum().item() / 2, abs=1e-6)


def test_xor_targets():
    X, Y, M, _ = xor(50, 30, np.random.default_rng(2))
    for n in range(30):
        marks = torch.nonzero(X[:, n, 1]).ravel()
        assert Y[-1, n, 0].item() == float(int(X[marks[0], n, 0].item()) ^ int(X[marks[1], n, 0].item()))


def test_temporal_order_classes():
    for fn, n_cls in ((temporal_order, 4), (temporal_order_3bit, 8)):
        X, Y, M, kind = fn(100, 200, np.random.default_rng(3))
        assert kind == "ce" and set(Y[-1].tolist()) == set(range(n_cls))
        specials = (X[:, :, :2].sum(-1) > 0).sum(0)
        assert (specials == (2 if n_cls == 4 else 3)).all()


def test_random_permutation_first_equals_last():
    X, Y, M, _ = random_permutation(30, 20, np.random.default_rng(4))
    first, last = X[0].argmax(-1), X[-1].argmax(-1)
    assert torch.equal(first, last) and (first < 2).all()
    assert torch.equal(Y[-2], last)                                      # the predictable target


def test_memorization_reproduces_the_bits():
    X, Y, M, _ = memorization(20, 10, np.random.default_rng(5))
    assert torch.equal(Y[-5:], X[:5].argmax(-1))
    assert (X[20 + 5 - 1].argmax(-1) == 3).all()                         # the trigger, at step T + 5
    assert (X[5:20 + 5 - 1].argmax(-1) == 2).all()                       # blanks in between


def test_bouncing_balls():
    v = bouncing_balls(40, res=15, rng=np.random.default_rng(6))
    assert v.shape == (40, 225) and v.min() >= 0 and v.max() <= 1
    assert np.abs(np.diff(v, axis=0)).mean() > 0                          # the balls move


# ---------------------------------------------------------------------------
# Chapter 4: HF with structural damping
# ---------------------------------------------------------------------------

def test_sparse_init_15_connections():
    m = RNN(2, 40, 1)
    assert ((m.W_hh != 0).sum(1) == 15).all() and ((m.W_hv != 0).sum(1) == 2).all()


def test_curvature_product_matches_explicit_matrices():
    torch.manual_seed(0)
    m = RNN(2, 3, 1, sparse=False).double()
    data = tuple(t.double() if t.is_floating_point() else t for t in addition(6, 4, np.random.default_rng(7))[:3]) + ("mse",)
    hf = HFStructural(m, "mse", lam=0.7, mu=0.3)
    names = [n for n, _ in m.named_parameters()]
    shapes = [p.shape for p in m.parameters()]
    flat0 = torch.cat([p.detach().reshape(-1) for p in m.parameters()])

    def outs(flat):
        prm, i = {}, 0
        for n, s in zip(names, shapes):
            k = int(np.prod(s)); prm[n] = flat[i:i + k].view(s); i += k
        o, x = torch.func.functional_call(m, prm, (data[0],))
        return o, x

    Jo = torch.autograd.functional.jacobian(lambda f: outs(f)[0], flat0).reshape(-1, len(flat0))
    Jx = torch.autograd.functional.jacobian(lambda f: outs(f)[1], flat0).reshape(-1, len(flat0))
    o, x = outs(flat0)
    mask = data[2][..., None].expand_as(o).reshape(-1)
    Gf = Jo.T @ torch.diag(mask / data[2].sum()) @ Jo
    N = x.shape[1]
    GS = Jx.T @ torch.diag((1 - torch.tanh(x) ** 2).reshape(-1) / N) @ Jx
    v = torch.randn(len(flat0), dtype=torch.float64)
    expected = (Gf + 0.7 * 0.3 * GS) @ v
    assert torch.allclose(hf.curvature_product(data[:3], v), expected, atol=1e-8)
    assert torch.allclose(hf.curvature_product(data[:3], v, structural=False), Gf @ v, atol=1e-8)


def test_hf_learns_a_short_memorization_task():
    torch.manual_seed(1)
    r = np.random.default_rng(8)
    m = RNN(4, 20, 4)
    hf = HFStructural(m, "ce", lam=0.1, mu=1 / 30, max_cg=50)
    data = memorization(5, 32, r)[:3]
    first = hf.objective(data).item()
    for _ in range(8):
        hf.step(data, data)
    assert hf.objective(data).item() < 0.6 * first


def test_every_problem_generates():
    for name, fn in PROBLEMS.items():
        X, Y, M, kind = fn(30, 4, np.random.default_rng(9))
        m = RNN(X.shape[-1], 8, 1 if kind == "mse" else int(Y.max()) + 1)
        assert 0.0 <= error_rate(m, (X, Y, M, kind)) <= 1.0, name
