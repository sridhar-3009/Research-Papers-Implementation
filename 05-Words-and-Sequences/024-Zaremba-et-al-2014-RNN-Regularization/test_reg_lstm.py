"""Tests for Zaremba, Sutskever & Vinyals (2014). About a second.

Run with:  python3 -m pytest -q
"""

import math

import pytest
import torch

from reg_lstm import (DeepLSTM, LSTMCell, batchify, count_params, dropouts_on_path, ensemble_perplexity,
                      run_epoch)

torch.manual_seed(0)


def test_cell_matches_torch_lstm():
    n = 5
    ours = LSTMCell(n, n)
    ref = torch.nn.LSTMCell(n, n)
    with torch.no_grad():
        W = ours.T.weight                                            # (4n, 2n), gates in our order i, f, o, g
        i, f, o, g = W.chunk(4, 0)
        bi, bf, bo, bg = ours.T.bias.chunk(4)
        ref.weight_ih.copy_(torch.cat([i, f, g, o])[:, :n])          # torch's order is i, f, g, o
        ref.weight_hh.copy_(torch.cat([i, f, g, o])[:, n:])
        ref.bias_ih.copy_(torch.cat([bi, bf, bg, bo])); ref.bias_hh.zero_()
    x, h, c = torch.randn(3, n), torch.randn(3, n), torch.randn(3, n)
    h1, c1 = ours(x, h, c)
    h2, c2 = ref(x, (h, c))
    assert torch.allclose(h1, h2, atol=1e-6) and torch.allclose(c1, c2, atol=1e-6)


def test_model_sizes_of_section_4_1():
    sizes = {name: count_params(DeepLSTM(10000, n, 2)) for name, n in (("small", 200), ("medium", 650), ("large", 1500))}
    assert round(sizes["medium"] / 1e6) == 20                        # the well-known "medium (20M)"
    assert round(sizes["large"] / 1e6) == 66                         # and "large (66M)"
    assert sizes["small"] < 5e6


def test_nonrecurrent_mode_drops_only_vertical_connections():
    m = DeepLSTM(50, 16, 2, dropout=0.5, mode="nonrecurrent")
    m.train(); m.record = True
    tokens = torch.randint(0, 50, (6, 4))
    m(tokens, m.init_state(4))
    for l, vertical, recurrent, h_prev in m.trace:
        assert torch.equal(recurrent, h_prev)                        # recurrent input passes untouched
    zeros = sum((v == 0).float().mean().item() for _, v, _, _ in m.trace) / len(m.trace)
    assert zeros == pytest.approx(0.5, abs=0.1)                      # about half of the vertical input dropped


def test_naive_mode_also_drops_recurrent_connections():
    m = DeepLSTM(50, 16, 2, dropout=0.5, mode="naive")
    m.train(); m.record = True
    m(torch.randint(0, 50, (6, 4)), m.init_state(4))
    later = [(r, h) for _, _, r, h in m.trace[4:]]                    # after the first step (h0 = 0)
    assert any(not torch.equal(r, h) for r, h in later)


def test_no_dropout_at_test_time():
    m = DeepLSTM(50, 16, 2, dropout=0.65)
    m.eval()
    tokens = torch.randint(0, 50, (5, 2))
    a, _ = m(tokens, m.init_state(2))
    b, _ = m(tokens, m.init_state(2))
    assert torch.equal(a, b)


def test_information_is_corrupted_L_plus_1_times_whatever_the_lag():
    for k in (1, 10, 1000):
        assert dropouts_on_path(L=2, k=k, mode="nonrecurrent") == 3
    assert dropouts_on_path(2, 1000, "naive") == 1003


def test_batchify_makes_parallel_streams():
    data = batchify(torch.arange(100), 4)
    assert data.shape == (25, 4)
    assert data[:, 1].tolist() == list(range(25, 50))                 # each column is one contiguous stream


def test_untrained_model_perplexity_is_about_the_vocab_size():
    m = DeepLSTM(40, 8, 1, init=0.0)
    data = batchify(torch.randint(0, 40, (2000,)), 4)
    assert run_epoch(m, data) == pytest.approx(40, rel=1e-3)          # uniform predictions: perplexity = V


def test_training_reduces_perplexity_on_a_repeating_stream():
    m = DeepLSTM(5, 16, 1, dropout=0.0, init=0.1)
    data = batchify(torch.tensor([0, 1, 2, 3, 4] * 200), 4)
    first = run_epoch(m, data, steps=10, lr=1.0)
    for _ in range(5):
        last = run_epoch(m, data, steps=10, lr=1.0)
    assert last < 1.5 < first


def test_ensemble_of_one_equals_the_model():
    m = DeepLSTM(20, 8, 1)
    data = batchify(torch.randint(0, 20, (400,)), 2)
    assert ensemble_perplexity([m], data, steps=10) == pytest.approx(run_epoch(m, data, steps=10), rel=1e-5)
