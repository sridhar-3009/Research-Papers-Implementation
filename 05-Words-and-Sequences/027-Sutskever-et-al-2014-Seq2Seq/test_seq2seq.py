"""Tests for Sutskever, Vinyals & Le (2014). About a second.

Run with:  python3 -m pytest -q
"""

import itertools
import random

import pytest
import torch
import torch.nn as nn

from seq2seq import (BOS, EOS, Seq2Seq, beam_search, clip_grad_, count_params, length_batches, lr_schedule,
                     paper_init_, rescore_nbest, reverse_source, time_lags)

torch.manual_seed(0)


def test_the_paper_model_has_384M_parameters():
    with torch.device("meta"):
        m = Seq2Seq(160_000, 80_000, emb=1000, hidden=1000, layers=4)
    assert round(count_params(m) / 1e6) == 384                                  # '384M parameters'
    lstm = sum(p.numel() for n, p in m.named_parameters() if "encoder" in n)
    assert round(lstm / 1e6) == 32                                              # '32M for the encoder LSTM'


def test_sentence_representation_is_8000_numbers():
    m = Seq2Seq(20, 20, emb=8, hidden=1000, layers=4)
    h, c = m.encode(torch.randint(3, 20, (5, 1)))
    assert h.numel() + c.numel() == 8000


def test_paper_init_range():
    m = paper_init_(Seq2Seq(10, 10, 8, 8, 2))
    assert all(p.abs().max() <= 0.08 for p in m.parameters())


def test_reversal_keeps_the_average_lag_but_shrinks_the_minimal_lag():
    n = 20
    normal, rev = time_lags(n, False), time_lags(n, True)
    assert sum(normal) / n == sum(rev) / n == n                                  # 'the average distance ... is unchanged'
    assert min(normal) == n and min(rev) == 1                                    # 'minimal time lag is greatly reduced'
    assert reverse_source([1, 2, 3]) == [3, 2, 1]


def test_lr_schedule():
    assert lr_schedule(0) == lr_schedule(4.9) == 0.7
    assert lr_schedule(5.0) == pytest.approx(0.35) and lr_schedule(5.5) == pytest.approx(0.175)
    assert lr_schedule(7.4) == pytest.approx(0.7 / 2 ** 5)


def test_clipping_scales_the_gradient_to_norm_5():
    m = Seq2Seq(10, 10, 4, 4, 1)
    for p in m.parameters():
        p.grad = torch.ones_like(p) * 10
    s = clip_grad_(m)
    total = torch.cat([p.grad.reshape(-1) for p in m.parameters()]).norm().item()
    assert s > 5 and total == pytest.approx(5.0, rel=1e-4)


def test_length_batches_group_similar_lengths():
    pairs = [([0] * random.Random(i).randint(1, 50), [0]) for i in range(1000)]
    for b in length_batches(pairs, 128, random.Random(0)):
        lens = [len(pairs[i][0]) for i in b]
        assert max(lens) - min(lens) <= 10


class Table(nn.Module):
    """A fake 'model' whose next-word probabilities depend only on the previous word (for beam tests)."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def encode(self, src, lengths=None):
        return torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)

    def step(self, y_prev, state):
        h, c = state
        return torch.log_softmax(self.logits[y_prev], -1), (h.expand(1, len(y_prev), 1), c.expand(1, len(y_prev), 1))


def exhaustive_best(logits, max_len=3):
    lp = torch.log_softmax(logits, -1)
    best = (-1e9, None)
    for L in range(1, max_len + 1):
        for seq in itertools.product(range(3, logits.shape[0]), repeat=L - 1):
            toks = list(seq) + [EOS]
            s, prev = 0.0, BOS
            for w in toks:
                s += lp[prev, w].item(); prev = w
            best = max(best, (s, toks), key=lambda x: x[0])
    return best


def test_greedy_can_miss_the_best_sequence_but_a_wide_beam_finds_it():
    # BOS -> 3 is likely, but 3 is then followed by a near-uniform mess; BOS -> 4 -> EOS is the best.
    V = 6
    logits = torch.full((V, V), -10.0)
    logits[BOS, 3], logits[BOS, 4] = 1.0, 0.8
    logits[3, 3:] = 0.0; logits[3, EOS] = 0.0                                   # after 3: uniform
    logits[4, EOS] = 5.0
    for w in range(3, V):
        logits[w, EOS] = max(logits[w, EOS].item(), 0.0)
    model = Table(logits)
    greedy = beam_search([model], torch.zeros(1, 1, dtype=torch.long), beam=1, max_len=3)[0]
    wide = beam_search([model], torch.zeros(1, 1, dtype=torch.long), beam=20, max_len=3)[0]
    best = exhaustive_best(logits)
    assert wide[1] == best[1] and wide[0] == pytest.approx(best[0], abs=1e-5)
    assert greedy[0] < wide[0]


def test_ensemble_of_identical_models_equals_one_model():
    torch.manual_seed(3)
    m = Seq2Seq(12, 12, 8, 16, 2)
    src = torch.randint(3, 12, (4, 1))
    a = beam_search([m], src, beam=3, max_len=6)
    b = beam_search([m, m], src, beam=3, max_len=6)
    assert [x[1] for x in a] == [x[1] for x in b]


def test_beam_results_are_sorted_and_scores_are_log_probs():
    torch.manual_seed(4)
    m = Seq2Seq(12, 12, 8, 16, 2)
    src = torch.randint(3, 12, (4, 1))
    res = beam_search([m], src, beam=4, max_len=5)
    scores = [s for s, _ in res]
    assert scores == sorted(scores, reverse=True) and all(s <= 0 for s in scores)
    s, toks = res[0]
    if toks and toks[-1] == EOS:                                                 # recompute log p(y|x) by teacher forcing
        tgt_in = torch.tensor([BOS] + toks[:-1])[:, None]
        lp = torch.log_softmax(m(src, tgt_in), -1)[:, 0].gather(-1, torch.tensor(toks)[:, None]).sum().item()
        assert lp == pytest.approx(s, abs=1e-4)


def test_rescoring_is_an_even_average():
    assert rescore_nbest([(-2.0, [5]), (-4.0, [6])], [-6.0, -1.0]) == [-4.0, -2.5]
