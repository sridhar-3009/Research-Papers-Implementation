"""Tests for Cho et al. (2014). About a second.

Run with:  python3 -m pytest -q
"""

import math

import pytest
import torch

from encdec import EOS, PAD, EncoderDecoder, GRUCell, bleu, loglinear, orthogonal_

torch.manual_seed(0)


def test_gru_equations_5_to_8():
    cell = GRUCell(3, 4)
    x, h = torch.randn(2, 3), torch.randn(2, 4)
    Wx, Wz, Wr = (x @ W.T for W in cell.W.weight.chunk(3, 0))
    Uz, Ur = (h @ U.T for U in cell.U_zr.weight.chunk(2, 0))
    z, r = torch.sigmoid(Wz + Uz), torch.sigmoid(Wr + Ur)
    expected = z * h + (1 - z) * torch.tanh(Wx + (r * h) @ cell.U.weight.T)
    assert torch.allclose(cell(x, h), expected, atol=1e-6)


def test_update_gate_near_one_copies_the_state():
    cell = GRUCell(3, 4)
    with torch.no_grad():
        cell.W.weight[4:8].zero_(); cell.U_zr.weight[:4].zero_()
        cell.W.weight[4:8, 0] = 100.0                                   # W_z x large -> z ~ 1
    h = torch.randn(1, 4)
    assert torch.allclose(cell(torch.tensor([[1.0, 0.0, 0.0]]), h), h, atol=1e-4)


def test_reset_gate_near_zero_ignores_the_past():
    cell = GRUCell(3, 4)
    with torch.no_grad():
        cell.U_zr.weight.zero_(); cell.W.weight[4:].zero_()
        cell.W.weight[8:, 0] = -100.0                                   # r ~ 0 ; z = 0.5
    x = torch.tensor([[1.0, 0.0, 0.0]])
    h1, h2 = torch.randn(1, 4), torch.randn(1, 4)
    h_tilde = torch.tanh(cell.W(x).chunk(3, -1)[0])
    assert torch.allclose(cell(x, h1) - 0.5 * h1, 0.5 * h_tilde, atol=1e-4)  # h~ doesn't depend on h1
    assert torch.allclose(cell(x, h2) - 0.5 * h2, 0.5 * h_tilde, atol=1e-4)


def test_orthogonal_init():
    W = torch.empty(6, 6)
    orthogonal_(W)
    assert torch.allclose(W @ W.T, torch.eye(6), atol=1e-5)


def tiny():
    return EncoderDecoder(src_vocab=12, tgt_vocab=10, n=16, emb=8, maxout=6, rank=4)


def test_encoder_ignores_padding():
    m = tiny()
    a = torch.tensor([[5], [6], [7]])
    b = torch.tensor([[5], [6], [7], [PAD], [PAD]])
    assert torch.allclose(m.encode(a), m.encode(b))


def test_log_prob_sums_the_target_tokens_only():
    m = tiny()
    src = torch.tensor([[3, 4], [5, 6], [7, PAD]])
    tgt = torch.tensor([[4, 5], [6, EOS], [EOS, PAD]])
    lp = m.log_prob(src, tgt)
    assert lp.shape == (2,) and (lp < 0).all()
    one = m.log_prob(src[:2, 1:], tgt[:2, 1:])                          # the second pair alone, unpadded
    assert lp[1].item() == pytest.approx(one.item(), abs=1e-5)


def test_first_word_distribution_sums_to_one():
    m = tiny()
    src = torch.tensor([[3], [4]])
    probs = [math.exp(m.log_prob(src, torch.tensor([[y]])).item()) for y in range(1, 10)]   # log_prob masks PAD (0)
    p_pad = torch.softmax(m(src, torch.tensor([[1]]))[0, 0], -1)[0].item()                # so add P(PAD) directly
    assert sum(probs) + p_pad == pytest.approx(1.0, abs=1e-5)


def test_maxout_halves_the_output_layer():
    m = tiny()
    assert m.O_h.out_features == 12 and m.G_r.in_features == 6


def test_training_learns_to_copy():
    torch.manual_seed(1)
    m = EncoderDecoder(8, 8, n=32, emb=16, maxout=16, rank=8)
    opt = torch.optim.Adam(m.parameters(), 1e-2)        # (the paper used AdaDelta; it starts slowly from std-0.01 init)
    src = torch.randint(3, 8, (4, 64))
    tgt = torch.cat([src, torch.full((1, 64), EOS)])
    first = -m.log_prob(src, tgt).mean().item()
    for _ in range(150):
        loss = -m.log_prob(src, tgt).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.7 * first


def test_generation_stops_at_eos():
    m = tiny()
    out = m.generate(torch.tensor([[3], [4]]), max_len=5)
    assert out.shape[0] <= 5


def test_bleu():
    ref = "the cat is on the mat".split()
    assert bleu([ref], [ref]) == pytest.approx(100.0)
    assert bleu([["the"] * 7], [ref]) == 0.0                            # no 2-gram matches
    # a shorter but correct hypothesis is punished by the brevity penalty only
    hyp = "the cat is on the".split()
    p = (5 / 5 * 4 / 4 * 3 / 3 * 2 / 2) ** 0.25
    assert bleu([hyp], [ref]) == pytest.approx(100 * math.exp(1 - 6 / 5) * p)


def test_loglinear_is_a_weighted_sum():
    assert loglinear([-2.0, -1.0, 3.0], [0.5, 1.0, 0.1]) == pytest.approx(-1.7)
