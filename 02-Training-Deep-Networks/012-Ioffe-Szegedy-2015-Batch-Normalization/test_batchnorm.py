"""Tests: each checks a statement of Ioffe & Szegedy (2015). All tiny (a second or two).

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest
import torch

from batchnorm import (MLP, BatchNorm, bn_backward, bn_backward_compact, bn_conv_backward, bn_conv_forward,
                       bn_forward, fuse_for_inference, normalize_outside_gradient, population_stats)

rng = np.random.default_rng(0)


# ---- Algorithm 1 ----

def test_output_has_zero_mean_unit_variance_per_feature():
    x = rng.normal(5.0, 3.0, size=(64, 7)) * np.arange(1, 8)       # every feature has its own scale
    y, _ = bn_forward(x, np.ones(7), np.zeros(7), eps=0)
    assert np.allclose(y.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(y.var(axis=0), 1, atol=1e-10)


def test_gamma_beta_can_recover_the_identity():
    # Section 3: gamma = sqrt(Var), beta = E recovers the original activations.
    x = rng.normal(2.0, 4.0, size=(32, 3))
    y, _ = bn_forward(x, np.sqrt(x.var(axis=0)), x.mean(axis=0), eps=0)
    assert np.allclose(y, x)


def numerical_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h; fp = f()
        x[idx] = old - h; fm = f()
        x[idx] = old
        g[idx] = (fp - fm) / (2 * h)
    return g


def test_backward_matches_finite_differences():
    x = rng.normal(size=(6, 4)); gamma = rng.normal(size=4); beta = rng.normal(size=4)
    w = rng.normal(size=(6, 4))                                     # loss = sum(w * y)
    y, cache = bn_forward(x, gamma, beta)
    dx, dgamma, dbeta = bn_backward(w, cache)
    loss = lambda: (w * bn_forward(x, gamma, beta)[0]).sum()
    assert np.allclose(dx, numerical_grad(loss, x), atol=1e-5)
    assert np.allclose(dgamma, numerical_grad(loss, gamma), atol=1e-5)
    assert np.allclose(dbeta, numerical_grad(loss, beta), atol=1e-5)


def test_backward_matches_torch_and_compact_form():
    x = rng.normal(size=(16, 5)); gamma = rng.normal(size=5); beta = rng.normal(size=5); dy = rng.normal(size=(16, 5))
    _, cache = bn_forward(x, gamma, beta)
    dx, dg, db = bn_backward(dy, cache)
    dx2, dg2, db2 = bn_backward_compact(dy, cache)
    xt = torch.tensor(x, requires_grad=True)
    yt = torch.nn.functional.batch_norm(xt, None, None, torch.tensor(gamma), torch.tensor(beta), training=True, eps=1e-5)
    yt.backward(torch.tensor(dy))
    assert np.allclose(dx, xt.grad.numpy(), atol=1e-8)
    assert np.allclose(dx, dx2, atol=1e-10) and np.allclose(dg, dg2) and np.allclose(db, db2)


def test_gradient_has_no_mean_and_no_component_along_x_hat():
    # What the backward pass does: the gradient reaching x can't shift or rescale the batch.
    x = rng.normal(size=(20, 3)); dy = rng.normal(size=(20, 3))
    _, cache = bn_forward(x, np.ones(3), np.zeros(3), eps=0)      # exact with eps = 0
    dx, _, _ = bn_backward(dy, cache)
    x_hat = cache[3]
    assert np.allclose(dx.sum(axis=0), 0, atol=1e-10)
    assert np.allclose((dx * x_hat).sum(axis=0), 0, atol=1e-10)


# ---- Section 3.3: scale invariance ----

def test_scaling_the_weights_changes_nothing_but_the_weight_gradient():
    # BN((aW)u) = BN(Wu); dBN/du unchanged; dBN/d(aW) = (1/a) dBN/dW.
    u = rng.normal(size=(32, 10)); W = rng.normal(size=(10, 4)); dy = rng.normal(size=(32, 4)); a = 7.0
    y1, c1 = bn_forward(u @ W, np.ones(4), np.zeros(4), eps=0)
    y2, c2 = bn_forward(u @ (a * W), np.ones(4), np.zeros(4), eps=0)
    assert np.allclose(y1, y2)
    dz1, _, _ = bn_backward(dy, c1)
    dz2, _, _ = bn_backward(dy, c2)
    du1, du2 = dz1 @ W.T, dz2 @ (a * W).T
    dW1, dW2 = u.T @ dz1, u.T @ dz2
    assert np.allclose(du1, du2)
    assert np.allclose(dW2, dW1 / a)


# ---- Algorithm 2: inference ----

def test_population_variance_is_unbiased():
    batches = [rng.normal(0, 2.0, size=(4, 3)) for _ in range(20000)]   # tiny batches: bias is m/(m-1) = 4/3
    mean, var = population_stats(batches)
    assert np.allclose(mean, 0, atol=0.03)
    assert np.allclose(var, 4.0, rtol=0.03)
    assert np.mean([b.var(axis=0) for b in batches]) == pytest.approx(3.0, rel=0.03)   # the biased one


def test_inference_is_a_single_linear_map():
    gamma, beta, mean, var = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3), rng.uniform(0.5, 2, 3)
    x = rng.normal(size=(5, 3))
    a, c = fuse_for_inference(gamma, beta, mean, var)
    assert np.allclose(a * x + c, gamma * (x - mean) / np.sqrt(var + 1e-5) + beta)


# ---- Section 3.2: convolutional BN ----

def test_conv_bn_normalizes_each_feature_map_over_batch_and_locations():
    x = rng.normal(size=(4, 3, 5, 5)) * np.array([1, 10, 100])[None, :, None, None] + 3
    y, cache = bn_conv_forward(x, np.ones(3), np.zeros(3), eps=0)
    assert np.allclose(y.mean(axis=(0, 2, 3)), 0, atol=1e-10)
    assert np.allclose(y.var(axis=(0, 2, 3)), 1, atol=1e-10)
    dy = rng.normal(size=x.shape)
    _, cache = bn_conv_forward(x, np.ones(3), np.zeros(3), eps=1e-5)   # torch needs eps > 0
    dx, dg, db = bn_conv_backward(dy, cache)
    xt = torch.tensor(x, requires_grad=True)
    torch.nn.functional.batch_norm(xt, None, None, torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
                                   training=True, eps=1e-5).backward(torch.tensor(dy))
    assert np.allclose(dx, xt.grad.numpy(), atol=1e-8)


# ---- Section 2 ----

def test_normalizing_outside_the_gradient_makes_b_drift_forever():
    u = rng.normal(size=50); target = rng.normal(1.0, 1.0, size=50)        # target mean != 0
    b_out, loss_out = normalize_outside_gradient(u, target, inside=False)
    b_in, loss_in = normalize_outside_gradient(u, target, inside=True)
    assert np.allclose(loss_out, loss_out[0])                              # the loss never improves...
    assert abs(b_out[-1]) > 10 * abs(b_out[0]) and abs(b_out[-1]) > 5      # ...while b grows without limit
    assert np.all(b_in == 0)                                               # with the right gradient, b doesn't move


# ---- the PyTorch layer ----

def test_torch_layer_train_and_eval_modes():
    torch.manual_seed(0)
    bn = BatchNorm(3, momentum=1.0)                                      # momentum 1: running stats = last batch
    x = torch.randn(64, 3) * 5 + 2
    y = bn(x)
    assert torch.allclose(y.mean(0), torch.zeros(3), atol=1e-5)
    bn.eval()
    y_eval = bn(x)                                                        # uses running (unbiased) stats
    assert torch.allclose(y_eval.std(0, unbiased=False), torch.ones(3) * np.sqrt(63 / 64), atol=1e-3)   # divided by the unbiased var
    assert torch.allclose(bn(x[:1]), y_eval[:1])                          # eval: one example alone gives the same output


def test_train_mode_output_depends_on_the_other_examples():
    # Section 3.4: in training, an example's output depends on its batch-mates (the regularizing noise).
    torch.manual_seed(0)
    bn = BatchNorm(2)
    x = torch.randn(8, 2)
    y1 = bn(x)[0]
    x2 = x.clone(); x2[1:] = torch.randn(7, 2)
    assert not torch.allclose(y1, bn(x2)[0])


def test_mlp_forward_shapes():
    for bn in (False, True):
        net = MLP(bn=bn)
        out, pre = net(torch.randn(60, 784), return_preact=True)
        assert out.shape == (60, 10) and len(pre) == 3 and pre[0].shape == (60, 100)
    assert MLP(bn=True).linears[0].bias is None                           # BN's beta replaces the bias
