"""Tests for Show and Tell (Vinyals et al. 2015). About a second.

Run with:  python3 -m pytest -q
"""

import random

import torch

import toy
from nic import (BOS, EOS, NIC, LSTMCell, beam_search, bleu, cider_d, human_bleu, nearest_words, normalize_for_annotation,
                 novelty, pad_captions, ranks, recall_report, sample)

torch.manual_seed(0)


def tiny(**kw):
    return NIC(len(toy.WORDS), toy.FEAT, d=12, **kw).eval()


def test_lstm_is_eqs_4_to_8_with_m_equal_o_times_c():
    cell = LSTMCell(3, 4)
    x, m, c = torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 4)
    i, f, o, g = (cell.Wx(x) + cell.Wm(m)).chunk(4, -1)
    c2 = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    m2, c3 = cell(x, (m, c))
    assert torch.allclose(c3, c2) and torch.allclose(m2, torch.sigmoid(o) * c2)            # Eq. (8): no tanh
    cell.cell_tanh = True
    assert torch.allclose(cell(x, (m, c))[0], torch.sigmoid(o) * torch.tanh(c2))


def test_image_is_fed_once_at_t_minus_1():
    m = tiny()
    f = toy.features([toy.ALL[0]])
    state, v = m.start(f)
    lp1, _ = m.step(torch.tensor([BOS]), state, v)
    lp2, _ = m.step(torch.tensor([BOS]), state, torch.randn_like(v))                    # v is NOT used after t = -1
    assert torch.allclose(lp1, lp2)
    m2 = tiny(image_every_step=True)                                                        # the ablation: fed every step
    state, v = m2.start(f)
    assert not torch.allclose(m2.step(torch.tensor([BOS]), state, v)[0], m2.step(torch.tensor([BOS]), state, -v)[0])


def test_loss_scores_words_1_to_N_and_ignores_padding():
    m = tiny()
    f = toy.features(toy.ALL[:2])
    caps = pad_captions([[3, 8], [3, 9, 12, 14]])
    lp = m.log_prob(f, caps)
    alone = m.log_prob(f[:1], pad_captions([[3, 8]]))
    assert torch.allclose(lp[0], alone[0], atol=1e-6)
    state, v = m.start(f[:1])                                                               # by hand: p_1(S_1) p_2(S_2) p_3(EOS)
    total = 0
    for prev, nxt in zip([BOS, 3, 8], [3, 8, EOS]):
        out, state = m.step(torch.tensor([prev]), state, v)
        total += out[0, nxt]
    assert torch.allclose(alone[0], total, atol=1e-6)


def test_beam_of_one_is_greedy_and_nbest_is_sorted_and_distinct():
    m = tiny()
    f = toy.features([toy.ALL[3]])[0]
    best = beam_search(m, f, beam=1, max_len=6)[0][1]
    state, v = m.start(f[None])
    w, greedy = torch.tensor([BOS]), []
    for _ in range(6):
        lp, state = m.step(w, state, v)
        w = lp.argmax(-1)
        if w.item() == EOS:
            break
        greedy.append(w.item())
    assert best == greedy
    nb = beam_search(m, f, beam=8, max_len=6)
    scores = [s for s, _ in nb]
    assert scores == sorted(scores, reverse=True) and len({tuple(t) for _, t in nb}) == len(nb)


def test_sampling_is_reproducible_and_never_emits_eos():
    m = tiny()
    f = toy.features(toy.ALL[:4])
    a = sample(m, f, generator=torch.Generator().manual_seed(1))
    b = sample(m, f, generator=torch.Generator().manual_seed(1))
    assert a == b and all(EOS not in s and len(s) <= 20 for s in a)


def test_ranking_recall_and_median_rank():
    S = torch.tensor([[5., 1., 0., 4.],                 # image 0: its captions are 0 and 1
                      [0., 2., 3., 1.]])                # image 1: its captions are 2 and 3
    ann, search = ranks(S, [0, 0, 1, 1])
    assert ann == [1, 1]                                # best true caption of each image ranked first
    assert search == [1, 2, 1, 2]                       # captions 1 and 3 prefer the wrong image
    r = recall_report(search, ks=(1, 2))
    assert r["R@1"] == 50 and r["R@2"] == 100


def test_normalization_changes_annotation_but_not_search():
    S = torch.randn(4, 6)
    N = normalize_for_annotation(S)
    assert torch.equal(S.argsort(0), N.argsort(0))      # per caption, a column shift: image search unchanged
    assert torch.allclose(torch.logsumexp(N, 0), torch.full((6,), torch.log(torch.tensor(4.))))


def test_bleu_and_human_bleu():
    refs = [[[3, 4, 5, 6], [3, 4, 7, 6]]]
    assert abs(bleu([[3, 4, 5, 6]], refs) - 100) < 1e-9
    assert bleu([[3, 3, 3, 3]], refs, n=1) == 25        # clipped: 'a' appears once in the references
    assert bleu([[9, 9]], refs, n=1) == 0
    same = [[[3, 4, 5, 6]] * 3]
    assert abs(human_bleu(same) - 100) < 1e-9


def test_cider_d_prefers_the_right_caption():
    refs = [[toy.encode(c) for c in toy.captions("big", "red", "circle")],
            [toy.encode(c) for c in toy.captions("small", "blue", "square")]]
    good = cider_d([toy.encode("a big red circle"), toy.encode("a small blue square")], refs)
    swapped = cider_d([toy.encode("a small blue square"), toy.encode("a big red circle")], refs)
    assert good > 3 * swapped


def test_novelty():
    train = {(3, 4), (3, 5)}
    assert novelty([[3, 4], [3, 6], [3, 5], [7]], train) == 0.5


def test_nearest_words_skip_special_tokens():
    m = tiny()
    nb = nearest_words(m, toy.STOI["red"], k=4)
    assert toy.STOI["red"] not in nb and not set(nb) & {0, 1, 2}


def test_nic_learns_to_caption_unseen_combinations():
    torch.manual_seed(0)
    rng = random.Random(0)
    train, held = toy.split()
    m = NIC(len(toy.WORDS), toy.FEAT, d=48)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(250):
        kinds, caps = toy.batch(train, 64, rng)
        loss = -m.log_prob(toy.features(kinds), pad_captions(caps)).sum() / sum(len(c) + 1 for c in caps)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    for kind in held:                                   # never seen this (size, colour, shape) together
        best = toy.decode(beam_search(m, toy.features([kind], noise=0.0)[0], beam=5)[0][1])
        assert best in toy.captions(*kind), (kind, best)
