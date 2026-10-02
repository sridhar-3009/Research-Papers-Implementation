"""Tests for BERT (Devlin et al. 2019). A few seconds.

Run with:  python3 -m pytest -q
"""

import random

import torch
import torch.nn.functional as F

from bert import (CLS, IGNORE, MASK, PAD, SEP, SPECIALS, Bert, Classifier, MultipleChoice, SpanQA, collate,
                  count_params, feature_based, mask_tokens, nsp_pairs, optimizer, pack_pair, warmup_linear,
                  wordpiece_tokenize)

torch.manual_seed(0)


def test_wordpiece_greedy_longest_match():
    vocab = {"un", "##aff", "##able", "##a", "play", "##ing", "##in", "##g"}
    assert wordpiece_tokenize("unaffable", vocab) == ["un", "##aff", "##able"]
    assert wordpiece_tokenize("playing", vocab) == ["play", "##ing"]              # longest match first
    assert wordpiece_tokenize("xyz", vocab) == ["[UNK]"]


def test_input_packing():
    ids, seg = pack_pair([10, 11, 12], [20, 21])
    assert ids == [CLS, 10, 11, 12, SEP, 20, 21, SEP] and seg == [0, 0, 0, 0, 0, 1, 1, 1]
    ids, _ = pack_pair(list(range(10, 20)), list(range(30, 33)), max_len=10)
    assert len(ids) == 10                                                          # the longer one was cut


def test_masking_rates():
    ids = torch.randint(len(SPECIALS), 1000, (400, 128))
    ids[:, 0] = CLS
    inp, lab = mask_tokens(ids, 1000, generator=torch.Generator().manual_seed(0))
    chosen = lab != IGNORE
    assert abs(chosen.float().mean().item() - 0.15) < 0.01 and not chosen[:, 0].any()   # never the [CLS]
    to_mask = (inp == MASK)[chosen].float().mean().item()
    kept = (inp == ids)[chosen].float().mean().item()
    assert abs(to_mask - 0.8) < 0.02 and abs(kept - 0.1) < 0.02                    # (random draws can hit the same id)
    assert torch.equal(inp[~chosen], ids[~chosen])


def test_next_sentence_pairs():
    docs = [[[d, s] for s in range(5)] for d in range(20)]
    pairs = nsp_pairs(docs, 2000, random.Random(0))
    assert abs(sum(lab for _, _, lab in pairs) / 2000 - 0.5) < 0.04
    for a, b, lab in pairs:
        if lab == 0:
            assert b[0] == a[0] and b[1] == a[1] + 1                               # the true next sentence
        else:
            assert b[0] != a[0]                                                    # from another document


def test_model_sizes():
    with torch.device("meta"):
        assert abs(count_params(Bert(30522)) / 1e6 - 110) < 2
        assert abs(count_params(Bert(30522, H=1024, L=24, A=16)) / 1e6 - 340) < 7


def test_bidirectional_vs_left_to_right_context():
    for causal in (False, True):
        m = Bert(30, H=16, L=2, A=2, max_len=16, causal=causal).eval()
        ids, seg, mask = collate([pack_pair([10, 11, 12, 13])])
        T1, _ = m(ids, seg, mask)
        ids2 = ids.clone(); ids2[0, 4] = 20                                        # change a LATER token
        T2, _ = m(ids2, seg, mask)
        assert torch.allclose(T1[0, 1], T2[0, 1], atol=1e-5) == causal             # earlier token sees it iff bidirectional


def test_padding_is_ignored():
    m = Bert(30, H=16, L=2, A=2, max_len=16).eval()
    a = collate([pack_pair([10, 11, 12])])
    b = collate([pack_pair([10, 11, 12]), pack_pair([10, 11, 12, 13, 14, 15])])
    assert torch.allclose(m(*a)[0][0], m(*b)[0][0, :5], atol=1e-5)


def test_pretraining_loss_and_tied_output():
    m = Bert(30, H=16, L=1, A=2, max_len=16)
    ids, seg, mask = collate([pack_pair([10, 11, 12], [13, 14])] * 2)
    inp, lab = mask_tokens(ids, 30, rate=0.5, generator=torch.Generator().manual_seed(1))
    total, mlm, nsp = m.pretrain_loss(inp, seg, mask, lab, torch.tensor([0, 1]))
    assert torch.allclose(total, mlm + nsp)
    T, _ = m(inp, seg, mask)
    assert torch.allclose(m.mlm_logits(T), m.mlm_ln(F.gelu(m.mlm_dense(T))) @ m.tok.weight.T + m.mlm_bias)


def test_fine_tuning_heads():
    m = Bert(30, H=16, L=1, A=2, max_len=16).eval()
    ids, seg, mask = collate([pack_pair([10, 11], [12, 13, 14])] * 3)
    assert Classifier(m, 3)(ids, seg, mask).shape == (3, 3)
    s, e = SpanQA(m)(ids, seg, mask)
    assert s.shape == ids.shape
    start, end = torch.tensor([0.0, 1, 5, 0, 0, 0]), torch.tensor([0.0, 0, 0, 4, 1, 0])
    assert SpanQA.best_span(start, end) == (2, 3)                                  # max S_i + E_j with j >= i
    assert SpanQA.best_span(start, end, null_threshold=0.0) == (2, 3)
    start[0] = end[0] = 6.0                                                        # s_null = 12 > 9
    assert SpanQA.best_span(start, end, null_threshold=0.0) == (0, 0)              # SQuAD 2.0: no answer
    mc = MultipleChoice(m)
    assert mc(ids[None].expand(2, -1, -1), seg[None].expand(2, -1, -1), mask[None].expand(2, -1, -1)).shape == (2, 3)
    m4 = Bert(30, H=16, L=4, A=2, max_len=16).eval()
    assert feature_based(m4, ids, seg, mask).shape == (3, ids.shape[1], 4 * 16)   # top four layers concatenated


def test_schedule_and_optimizer_groups():
    assert warmup_linear(500, 1000, 10000) == 0.5 and warmup_linear(1000, 1000, 10000) == 1.0
    assert abs(warmup_linear(5500, 1000, 10000) - 0.5) < 1e-9
    opt = optimizer(Bert(30, H=16, L=1, A=2, max_len=16))
    assert [g["weight_decay"] for g in opt.param_groups] == [0.01, 0.0]


def test_masked_lm_uses_right_context():
    """Markers are fixed by the token to their RIGHT: a bidirectional MLM recovers them, a left-to-right LM can't."""
    torch.manual_seed(0)
    V = 25

    def seqs(N, g=None):
        x = torch.randint(5, 15, (N, 6), generator=g)
        s = torch.stack([x + 10, x], 2).reshape(N, 12)
        return torch.cat([torch.full((N, 1), CLS), s, torch.full((N, 1), SEP)], 1)

    m = Bert(V, H=48, L=2, A=4, max_len=32, dropout=0.0)
    opt = optimizer(m, 3e-3)
    for _ in range(250):
        ids = seqs(64)
        inp, lab = mask_tokens(ids, V)
        T, _ = m(inp, torch.zeros_like(ids), ids != PAD)
        loss = F.cross_entropy(m.mlm_logits(T).reshape(-1, V), lab.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    ids = seqs(300, torch.Generator().manual_seed(1))
    inp = ids.clone(); pos = torch.arange(1, 13, 2); inp[:, pos] = MASK
    with torch.no_grad():
        pred = m.mlm_logits(m(inp, torch.zeros_like(ids), inp != PAD)[0]).argmax(-1)
    assert (pred[:, pos] == ids[:, pos]).float().mean() > 0.9
