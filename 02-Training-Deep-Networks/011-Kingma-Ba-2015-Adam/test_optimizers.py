"""Tests: each checks a statement of Kingma & Ba (2015). All tiny (about a second in total).

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest
import torch

from optimizers import AdaDelta, AdaGrad, AdaMax, Adam, RMSProp, SGDNesterov, TemporalAverage


def run(opt, grads_seq, theta0):
    p = [np.array(theta0, float)]
    path = [p[0].copy()]
    for g in grads_seq:
        p = opt.step(p, [np.array(g, float)])
        path.append(p[0].copy())
    return np.array(path)


# ---- Algorithm 1 ----

def test_adam_matches_pytorch():
    # Our Algorithm 1 must give the same iterates as torch.optim.Adam.
    torch.manual_seed(0)
    w = torch.randn(5, requires_grad=True)
    A = torch.randn(5, 5); A = A @ A.T + torch.eye(5)
    torch_opt = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    ours, p = Adam(alpha=0.01), [w.detach().clone()]
    for _ in range(50):
        g = (A @ p[0]).clone()
        p = ours.step(p, [g])
        torch_opt.zero_grad(); (0.5 * w @ A @ w).backward(); torch_opt.step()
    assert torch.allclose(p[0], w.detach(), atol=1e-5)


def test_first_step_has_size_alpha():
    # With bias correction, step 1 is exactly alpha * g/|g| = alpha * sign(g) (Section 2.1).
    path = run(Adam(alpha=0.1), [[5.0, -0.001, 300.0]], [0, 0, 0])
    assert np.allclose(path[1], [-0.1, 0.1, -0.1], atol=1e-6)


def test_steps_are_bounded_by_alpha():
    # Section 2.1: |step| <~ alpha in common cases; <= alpha (1-beta1)/sqrt(1-beta2) in the worst case.
    rng = np.random.default_rng(0)
    grads = rng.normal(0.3, 1.0, size=(500, 10))
    path = run(Adam(alpha=0.01), grads, np.zeros(10))
    steps = np.abs(np.diff(path, axis=0))
    assert steps.max() <= 0.01 * (1 - 0.9) / np.sqrt(1 - 0.999) + 1e-9
    assert np.median(steps) < 0.01


def test_invariant_to_gradient_scale():
    # Section 2.1: multiplying all gradients by c doesn't change the steps (eps aside).
    rng = np.random.default_rng(1)
    grads = rng.normal(size=(100, 4))
    a = run(Adam(alpha=0.01, eps=0), grads, np.zeros(4))
    b = run(Adam(alpha=0.01, eps=0), grads * 1000.0, np.zeros(4))
    assert np.allclose(a, b, atol=1e-10)


def test_bias_correction_removes_the_zero_start_bias():
    # Section 3: with a constant gradient, v_t = g^2 (1 - beta2^t), so v_hat_t = g^2 exactly.
    opt = Adam(alpha=0.001, beta2=0.999)
    p = [np.zeros(1)]
    for t in range(1, 6):
        p = opt.step(p, [np.array([2.0])])
        v_hat = opt.v[0] / (1 - opt.beta2 ** t)
        assert v_hat[0] == pytest.approx(4.0)
        assert opt.v[0][0] == pytest.approx(4.0 * (1 - 0.999 ** t))    # the uncorrected one is tiny


def test_without_bias_correction_early_steps_explode():
    # Section 6.4: no bias correction + beta2 close to 1 -> huge early steps (up to ~1/sqrt(1-beta2)).
    g = [[1.0]] * 3
    with_bc = np.abs(np.diff(run(Adam(alpha=0.01, beta1=0.0, beta2=0.9999), g, [0]), axis=0)).max()
    no_bc = np.abs(np.diff(run(Adam(alpha=0.01, beta1=0.0, beta2=0.9999, bias_correction=False), g, [0]), axis=0)).max()
    assert with_bc == pytest.approx(0.01, rel=1e-3)
    assert no_bc > 50 * with_bc


# ---- Section 5: AdaGrad as a special case ----

def test_adagrad_is_a_limit_of_adam():
    # beta1 = 0, beta2 -> 1, alpha_t = alpha / sqrt(t)  =>  AdaGrad.
    rng = np.random.default_rng(2)
    grads = rng.normal(size=(20, 3))
    a = run(Adam(alpha=0.1, beta1=0.0, beta2=1 - 1e-9, eps=0, decay_sqrt_t=True), grads, np.zeros(3))
    b = run(AdaGrad(alpha=0.1, eps=0), grads, np.zeros(3))
    assert np.allclose(a, b, atol=1e-5)


# ---- Section 7.1: AdaMax ----

def test_adamax_u_is_the_limit_of_the_lp_norm():
    # Eqs. (8)-(11): u_t = lim_{p->inf} ((1-beta2^p) sum beta2^{p(t-i)} |g_i|^p)^{1/p} = max_i beta2^{t-i} |g_i|
    rng = np.random.default_rng(3)
    g = np.abs(rng.normal(size=8))
    beta2, t = 0.9, len(g)
    u = 0.0
    for gi in g:
        u = max(beta2 * u, gi)                                           # Eq. (12)
    p = 400
    lp = ((1 - beta2 ** p) * sum(beta2 ** (p * (t - i)) * g[i - 1] ** p for i in range(1, t + 1))) ** (1 / p)
    assert u == pytest.approx(max(beta2 ** (t - i) * g[i - 1] for i in range(1, t + 1)))
    assert lp == pytest.approx(u, rel=0.01)


def test_adamax_steps_never_exceed_alpha():
    rng = np.random.default_rng(4)
    grads = rng.standard_cauchy(size=(300, 5))                              # wild gradients
    path = run(AdaMax(alpha=0.002), grads, np.zeros(5))
    assert np.abs(np.diff(path, axis=0)).max() <= 0.002 + 1e-9


# ---- all optimizers make progress on a quadratic ----

@pytest.mark.parametrize("opt, steps", [(Adam(0.05), 300), (AdaMax(0.05), 300), (AdaGrad(0.5), 300),
                                        (RMSProp(0.01), 300), (SGDNesterov(0.05), 300),
                                        (AdaDelta(), 20000)])   # AdaDelta starts with tiny steps (~sqrt(eps))
def test_every_optimizer_descends(opt, steps):
    A = np.diag([1.0, 10.0])
    p = [np.array([3.0, -2.0])]
    f0 = 0.5 * p[0] @ A @ p[0]
    for _ in range(steps):
        p = opt.step(p, [A @ p[0]])
    assert 0.5 * p[0] @ A @ p[0] < f0 / 10


def test_optimizers_work_on_torch_tensors():
    p = [torch.tensor([1.0, -1.0])]
    for Opt in (Adam, AdaMax, AdaGrad, RMSProp, AdaDelta, SGDNesterov):
        out = Opt().step(p, [torch.tensor([0.5, -0.5])])
        assert isinstance(out[0], torch.Tensor)


# ---- Section 7.2: temporal averaging ----

def test_temporal_average_is_unbiased_for_a_constant():
    avg = TemporalAverage(beta=0.999)
    for _ in range(3):
        out = avg.update([np.array([7.0])])
    assert out[0][0] == pytest.approx(7.0)
