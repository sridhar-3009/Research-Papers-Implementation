"""Tests for the Transformer (Vaswani et al. 2017). A few seconds.

Run with:  python3 -m pytest -q
"""

import collections
import math

import torch

from transformer import (BOS, EOS, PAD, MultiHeadAttention, Transformer, apply_bpe, attention, average_checkpoints,
                         beam_search, causal_mask, count_params, label_smoothed_loss, learn_bpe, length_penalty,
                         noam_lr, sinusoidal_positions)

torch.manual_seed(0)


def test_scaled_dot_product_attention_and_masking():
    Q, K, V = torch.randn(3, 4), torch.randn(5, 4), torch.randn(5, 2)
    out, w = attention(Q, K, V)
    manual = torch.softmax(Q @ K.T / 2.0, -1)                                       # sqrt(d_k) = 2
    assert torch.allclose(w, manual) and torch.allclose(out, manual @ V)
    mask = torch.tensor([[True, True, False, False, True]] * 3)
    _, w = attention(Q, K, V, mask)
    assert torch.all(w[:, 2:4] == 0) and torch.allclose(w.sum(-1), torch.ones(3))


def test_why_divide_by_sqrt_dk():
    q, k = torch.randn(20000, 512), torch.randn(20000, 512)
    dots = (q * k).sum(-1)
    assert abs(dots.var().item() / 512 - 1) < 0.05                                 # footnote 4: variance d_k
    assert abs((dots / math.sqrt(512)).var().item() - 1) < 0.05                    # scaled: variance 1


def test_multi_head_with_one_head_is_projected_attention():
    mha = MultiHeadAttention(8, h=1)
    x = torch.randn(2, 5, 8)
    out, _ = attention(mha.W_Q(x), mha.W_K(x), mha.W_V(x))
    assert torch.allclose(mha(x, x, x), mha.W_O(out), atol=1e-6)
    assert MultiHeadAttention(8, h=4)(x, x, x).shape == (2, 5, 8)


def test_sinusoidal_positions_formula_and_relative_offsets():
    pe = sinusoidal_positions(100, 16)
    assert abs(pe[7, 4] - math.sin(7 / 10000 ** (4 / 16))) < 1e-6 and abs(pe[7, 5] - math.cos(7 / 10000 ** (4 / 16))) < 1e-6
    k = 5                                                                           # PE(pos + k) = M_k PE(pos)
    w = 1 / torch.pow(10000.0, torch.arange(0, 16, 2) / 16) * k
    M = torch.zeros(16, 16)
    for j in range(8):
        c, s = math.cos(w[j]), math.sin(w[j])
        M[2 * j:2 * j + 2, 2 * j:2 * j + 2] = torch.tensor([[c, s], [-s, c]])
    assert torch.allclose(pe[k:] , pe[:-k] @ M.T, atol=1e-5)
    d = (pe[10] @ pe[10 + k]).item(), (pe[40] @ pe[40 + k]).item()               # depends only on the offset
    assert abs(d[0] - d[1]) < 1e-4


def test_decoder_cannot_see_future_tokens_and_padding_is_ignored():
    m = Transformer(20, 20, N=2, d_model=16, d_ff=32, h=2, dropout=0.0).eval()
    src, tgt = torch.randint(3, 20, (1, 6)), torch.randint(3, 20, (1, 7))
    a = m(src, tgt)
    tgt2 = tgt.clone(); tgt2[0, 4] = (tgt2[0, 4] + 1 - 3) % 17 + 3
    b = m(src, tgt2)
    assert torch.allclose(a[0, :4], b[0, :4], atol=1e-5) and not torch.allclose(a[0, 4:], b[0, 4:])
    padded = torch.cat([src, torch.full((1, 3), PAD)], 1)
    assert torch.allclose(m.encode(src)[0], m.encode(padded)[0][:, :6], atol=1e-5)
    assert torch.equal(causal_mask(3), torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool))


def test_model_sizes_match_table_3():
    with torch.device("meta"):
        base, big = Transformer(37000, 37000), Transformer(37000, 37000, d_model=1024, d_ff=4096, h=16)
    assert abs(count_params(base) / 1e6 - 65) < 3 and abs(count_params(big) / 1e6 - 213) < 3


def test_noam_schedule():
    assert abs(noam_lr(4000) - 512 ** -0.5 * 4000 ** -0.5) < 1e-12                 # the peak
    assert noam_lr(2000) < noam_lr(4000) > noam_lr(8000)
    assert abs(noam_lr(16000) / noam_lr(4000) - 0.5) < 1e-9                        # 1/sqrt(step) decay


def test_label_smoothing():
    logits, target = torch.randn(1, 3, 5), torch.tensor([[1, 4, PAD]])
    lp = torch.log_softmax(logits, -1)[0, :2]
    want = (0.9 * -lp[[0, 1], [1, 4]] + 0.1 * -lp.mean(-1)).mean()
    assert torch.allclose(label_smoothed_loss(logits, target), want)
    K, eps = 12, 0.1                                                                # best achievable loss = entropy
    q = torch.full((K,), eps / K); q[1] += 1 - eps                                  # (index 0 is PAD)
    best = label_smoothed_loss(torch.log(q)[None, None], torch.tensor([[1]]))
    assert abs(best.item() - (-(q * q.log()).sum()).item()) < 1e-5


def test_checkpoint_averaging_and_length_penalty():
    a, b = {"w": torch.tensor([1.0, 3.0])}, {"w": torch.tensor([3.0, 5.0])}
    assert torch.equal(average_checkpoints([a, b])["w"], torch.tensor([2.0, 4.0]))
    assert length_penalty(1) == 1.0 and length_penalty(10, 0.6) > length_penalty(5, 0.6)


def test_bpe_merges_frequent_pairs():
    counts = collections.Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
    merges = learn_bpe(counts, 10)
    assert merges[0] in {("e", "s"), ("s", "t"), ("t", "</w>")}                      # a most frequent pair (9) first
    assert apply_bpe("newest", merges)[-1].endswith("</w>")
    assert "".join(apply_bpe("lowest", merges)).replace("</w>", "") == "lowest"     # unseen word still covered


def test_transformer_learns_to_reverse():
    torch.manual_seed(0)
    V = 10
    m = Transformer(V, V, N=1, d_model=32, d_ff=64, h=4, dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), 2e-3, betas=(0.9, 0.98), eps=1e-9)
    for _ in range(300):
        src = torch.randint(3, V, (32, 6))
        tgt = torch.cat([torch.full((32, 1), BOS), src.flip(1), torch.full((32, 1), EOS)], 1)
        loss = label_smoothed_loss(m(src, tgt[:, :-1]), tgt[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    src = torch.randint(3, V, (20, 6), generator=torch.Generator().manual_seed(1))
    assert sum(beam_search(m, src[i:i + 1], beam=2) == src[i].flip(0).tolist() for i in range(20)) >= 16
