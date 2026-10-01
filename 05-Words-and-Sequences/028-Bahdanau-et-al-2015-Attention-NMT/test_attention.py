"""Tests for Bahdanau, Cho & Bengio (2015). A couple of seconds.

Run with:  python3 -m pytest -q
"""

import pytest
import torch

from attention import EOS, PAD, GRU, RNNsearch, beam_search, paper_init_

torch.manual_seed(0)


def tiny(mode="search"):
    return paper_init_(RNNsearch(15, 12, m=8, n=10, l=6, n_align=7, mode=mode))


def test_gru_update_convention():
    g = GRU(4, 5)
    e, s = torch.randn(2, 4), torch.randn(2, 5)
    we, wz, wr = g.W(e).chunk(3, -1)
    uz, ur = g.U_zr(s).chunk(2, -1)
    z, r = torch.sigmoid(wz + uz), torch.sigmoid(wr + ur)
    assert torch.allclose(g(e, s), (1 - z) * s + z * torch.tanh(we + g.U(r * s)), atol=1e-6)


def test_annotations_concatenate_forward_and_backward_states():
    m = tiny()
    src = torch.randint(3, 15, (6, 2))
    ann, mask, s0 = m.encode(src)
    assert ann.shape == (6, 2, 20) and s0.shape == (2, 10)
    src2 = src.clone(); src2[-1] = (src2[-1] + 1 - 3) % 12 + 3                 # change the LAST word
    ann2, _, _ = m.encode(src2)
    assert torch.allclose(ann[0, :, :10], ann2[0, :, :10])                     # forward half of h_1: unchanged
    assert not torch.allclose(ann[0, :, 10:], ann2[0, :, 10:])                 # backward half of h_1: sees the future


def test_attention_weights_are_a_distribution_over_real_words():
    m = tiny()
    with torch.no_grad():
        m.v_a.weight.normal_()                                                 # make attention non-uniform
    src = torch.tensor([[3, 4], [5, 6], [7, PAD], [8, PAD]])
    ann, mask, s0 = m.encode(src)
    c, alpha = m.attend(s0, ann, m.U_a(ann), mask)
    assert torch.allclose(alpha.sum(0), torch.ones(2))
    assert torch.all(alpha[2:, 1] == 0)                                        # no attention on padding
    assert torch.allclose(c, (alpha[..., None] * ann).sum(0))                  # Eq. (5): an expected annotation


def test_zero_v_a_gives_uniform_attention_at_the_start():
    m = tiny()                                                                  # paper init: v_a = 0
    src = torch.randint(3, 15, (5, 3))
    ann, mask, s0 = m.encode(src)
    _, alpha = m.attend(s0, ann, m.U_a(ann), mask)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.2))


def test_precomputing_U_a_h_changes_nothing():
    m = tiny()
    with torch.no_grad():
        m.v_a.weight.normal_()
    src = torch.randint(3, 15, (5, 2))
    ann, mask, s0 = m.encode(src)
    direct = m.v_a(torch.tanh(m.W_a(s0)[None] + torch.stack([m.U_a(ann[j]) for j in range(5)])))[..., 0]
    _, alpha = m.attend(s0, ann, m.U_a(ann), mask)
    assert torch.allclose(alpha, torch.softmax(direct, 0), atol=1e-6)


def test_encdec_baseline_uses_one_fixed_context():
    m = tiny("encdec")
    src = torch.randint(3, 15, (4, 2))
    ann, mask, _ = m.encode(src)
    c = m.fixed_context(ann, mask)
    assert c.shape == (2, 20)
    lp, alpha = m(src, torch.randint(3, 12, (3, 2)), return_alpha=True)
    assert alpha is None and lp.shape == (3, 2, 12)


def test_padding_does_not_change_the_translation_probability():
    m = tiny()
    with torch.no_grad():
        m.v_a.weight.normal_()
    tgt = torch.tensor([[4], [5], [EOS]])
    a = m.log_prob(torch.tensor([[3], [5], [7]]), tgt)
    b = m.log_prob(torch.tensor([[3], [5], [7], [PAD], [PAD]]), tgt)
    assert torch.allclose(a, b, atol=1e-5)


def test_paper_initialization():
    m = tiny()
    U = m.dec.U.weight
    assert torch.allclose(U @ U.T, torch.eye(10), atol=1e-5)                    # orthogonal recurrent matrix
    assert m.W_a.weight.std() < 0.005 and m.v_a.weight.abs().sum() == 0


def test_attention_model_learns_to_reverse_and_aligns_sharply():
    # PyTorch's default init here: with the paper's tiny init (all weights ~0.01, v_a = 0) a short run barely
    # moves (loss 7.3 after 400 steps, attention still nearly uniform); the paper trained for 5 days.
    torch.manual_seed(1)
    m = RNNsearch(10, 10, m=16, n=32, l=16, n_align=16)
    opt = torch.optim.Adam(m.parameters(), 5e-3)
    for _ in range(400):
        src = torch.randint(3, 10, (6, 32))
        tgt = torch.cat([src.flip(0), torch.full((1, 32), EOS)])
        loss = -m.log_prob(src, tgt).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    src = torch.randint(3, 10, (6, 1))
    tgt_in = torch.cat([torch.full((1, 1), 1), src.flip(0)])
    _, alpha = m(src, tgt_in, return_alpha=True)
    peaks = alpha[:6, :, 0].argmax(-1).tolist()                                # where each output word looks
    assert peaks == [5, 4, 3, 2, 1, 0]                                         # the anti-diagonal: reversal
    best = beam_search(m, src, beam=3, max_len=8)[0][1]
    assert best[:6] == src.flip(0)[:, 0].tolist()
