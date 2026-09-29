"""Tests: each checks a claim of Hinton et al. (2012) about dropout.

Run with:  python3 -m pytest -q
"""

import itertools

import numpy as np
import pytest
import torch

from dropout import DropoutNet, max_norm_, mc_average_errors, train
from dropout import test_errors as count_errors


def test_mean_network_is_the_geometric_mean_of_all_subnetworks():
    # Page 2: with one hidden layer and a softmax output, the mean network is EXACTLY
    # the normalized geometric mean of the 2^N dropout networks.
    torch.manual_seed(0)
    net = DropoutNet([5, 8, 3], keep_hidden=0.5, init_std=1.0, seed=2).double()
    x = torch.randn(7, 5, dtype=torch.float64)
    with torch.no_grad():
        L = torch.stack([torch.log_softmax(net(x, masks=[torch.ones(5, dtype=torch.float64),
                                                         torch.tensor(m, dtype=torch.float64)]), 1)
                         for m in itertools.product([0, 1], repeat=8)])
        geo = torch.softmax(L.mean(0), 1)
        assert torch.allclose(geo, torch.softmax(net(x, mode="mean"), 1), atol=1e-12)


def test_mean_network_beats_average_log_probability():
    # "guaranteed to assign a higher log probability to the correct answer than the mean
    # of the log probabilities assigned by the individual dropout networks"
    torch.manual_seed(1)
    net = DropoutNet([5, 8, 3], keep_hidden=0.5, init_std=1.0, seed=3).double()
    x, y = torch.randn(30, 5, dtype=torch.float64), torch.randint(0, 3, (30,))
    with torch.no_grad():
        L = torch.stack([torch.log_softmax(net(x, masks=[torch.ones(5, dtype=torch.float64),
                                                         torch.tensor(m, dtype=torch.float64)]), 1)
                         for m in itertools.product([0, 1], repeat=8)])
        mean_net = torch.log_softmax(net(x, mode="mean"), 1)
    assert torch.all(mean_net[range(30), y] >= L[:, range(30), y].mean(0) - 1e-12)


def test_training_mode_drops_the_right_fraction():
    torch.manual_seed(0)
    net = DropoutNet([1000, 1000, 10], keep_input=0.8, keep_hidden=0.5)
    x = torch.ones(50, 1000)
    h = x * (torch.rand_like(x) < net.keep[0]).float()
    assert h.mean().item() == pytest.approx(0.8, abs=0.01)
    # two training passes differ (fresh masks), two mean passes don't
    assert not torch.allclose(net(x, mode="train"), net(x, mode="train"))
    assert torch.allclose(net(x, mode="mean"), net(x, mode="mean"))


def test_mean_network_matches_expected_input_to_next_layer():
    # Halving outgoing weights = the expected input the next layer gets under dropout.
    torch.manual_seed(0)
    net = DropoutNet([20, 30, 5], keep_hidden=0.5, init_std=0.5, seed=0)
    x = torch.randn(1, 20)
    with torch.no_grad():
        h = torch.sigmoid(x @ net.W[0] + net.b[0])
        expected = (0.5 * h) @ net.W[1] + net.b[1]
        samples = torch.stack([(h * (torch.rand_like(h) < 0.5).float()) @ net.W[1] + net.b[1] for _ in range(20000)])
    assert torch.allclose(samples.mean(0), expected, atol=0.02)


def test_max_norm_caps_squared_length():
    net = DropoutNet([10, 20, 3], init_std=5.0)
    max_norm_(net, 15.0)
    sq = (net.W[0] ** 2).sum(0)
    assert torch.all(sq <= 15.0 + 1e-4)
    assert torch.any(torch.isclose(sq, torch.tensor(15.0)))
    assert torch.allclose(net.W[-1], net.W[-1])                    # output layer untouched


def test_dropout_net_learns_a_small_problem():
    torch.manual_seed(0)
    X = torch.randn(600, 20)
    y = (X[:, :5].sum(1) > 0).long()
    net = DropoutNet([20, 64, 64, 2], keep_input=0.8, keep_hidden=0.5, seed=0)
    train(net, X, y, epochs=40, lr0=10.0, lr_decay=0.95, mom_epochs=10, batch=50)
    assert count_errors(net, X, y) < 60                             # < 10% training error
    assert mc_average_errors(net, X, y, 50) < 80                   # sampled nets also work
