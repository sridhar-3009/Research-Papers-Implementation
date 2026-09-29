"""Tests: each checks an equation or claim of Glorot & Bengio (2010).

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest
import torch

from glorot import (CLASSES, DeepNet, gradient_stats, jacobian_singular_values, normalized_init,
                    shapeset, standard_init)


@pytest.fixture(scope="module")
def data():
    X, y = shapeset(300, rng=7)
    return torch.tensor(X), torch.tensor(y)


# ---- the two initializations ----

def test_standard_init_variance():
    # Eq. (1): U[-1/sqrt(n), 1/sqrt(n)] has variance 1/(3n), so n Var[W] = 1/3 (Eq. 15)
    W = standard_init(1000, 1000, torch.Generator().manual_seed(0))
    assert 1000 * W.var().item() == pytest.approx(1 / 3, rel=0.02)


def test_normalized_init_variance():
    # Eq. (16) gives Var[W] = 2 / (n_in + n_out) (Eq. 12)
    W = normalized_init(784, 1000, torch.Generator().manual_seed(0))
    assert W.var().item() == pytest.approx(2 / (784 + 1000), rel=0.02)
    assert W.abs().max().item() <= np.sqrt(6 / 1784)


# ---- Section 4.2: gradients at initialization ----

def test_standard_init_shrinks_activations_and_backprop_gradients(data):
    X, y = data
    act, back, wgrad = gradient_stats(DeepNet(1024, 9, act="tanh", init="standard"), X, y)
    assert all(a > b for a, b in zip(act, act[1:]))               # Fig. 6 top: activations shrink upward
    assert all(a < b for a, b in zip(back[:5], back[1:5]))        # Fig. 7 top: gradients shrink downward
    assert back[4] / back[0] > 5


def test_normalized_init_keeps_gradients_level(data):
    X, y = data
    act, back, wgrad = gradient_stats(DeepNet(1024, 9, act="tanh", init="normalized"), X, y)
    assert max(back[:5]) / min(back[:5]) < 1.5                    # Fig. 7 bottom
    assert max(act) / min(act) < 1.5                              # Fig. 6 bottom


def test_weight_gradients_level_even_with_standard_init(data):
    # Fig. 8's "surprise", explained by Eq. (14)
    X, y = data
    _, _, wgrad = gradient_stats(DeepNet(1024, 9, act="tanh", init="standard"), X, y)
    assert max(wgrad[:5]) / min(wgrad[:5]) < 1.3


def test_jacobian_singular_values_match_the_paper(data):
    # Section 4.2.2: "around 0.8 [normalized] whereas with the standard initialization, it drops to 0.5"
    X, _ = data
    assert np.mean(jacobian_singular_values(DeepNet(1024, 9, act="tanh", init="normalized"), X)) == pytest.approx(0.8, abs=0.05)
    assert np.mean(jacobian_singular_values(DeepNet(1024, 9, act="tanh", init="standard"), X)) == pytest.approx(0.5, abs=0.05)


# ---- the network ----

def test_network_shapes_and_biases():
    net = DeepNet(784, 10, hidden=1000, depth=5)
    assert [tuple(W.shape) for W in net.W] == [(784, 1000)] + [(1000, 1000)] * 4 + [(1000, 10)]
    assert all(float(b.detach().abs().sum()) == 0 for b in net.b)           # biases start at 0 (Section 2.3)


# ---- Shapeset-3x2 ----

def test_shapeset_has_nine_balanced_classes():
    X, y = shapeset(3000, rng=0)
    assert len(CLASSES) == 9 and X.shape == (3000, 1024)
    counts = np.bincount(y, minlength=9)
    assert counts.min() > 250                                      # roughly balanced
    assert 0 <= X.min() and X.max() <= 1


def test_shapeset_two_object_images_have_more_ink():
    X, y = shapeset(2000, rng=1)
    one = X[y < 3].astype(bool).sum(1).mean()
    two = X[y >= 3].astype(bool).sum(1).mean()
    assert two > one * 1.3
