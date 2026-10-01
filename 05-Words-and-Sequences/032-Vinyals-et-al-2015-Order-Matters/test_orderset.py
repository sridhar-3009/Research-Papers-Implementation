"""Tests for Order Matters (Vinyals, Bengio & Kudlur 2016). A few seconds.

Run with:  python3 -m pytest -q
"""

import collections
import itertools
import random

import torch

from orderset import (ReadProcessWrite, SetLM, best_order, bfs_delinearize, bfs_linearize, dfs_linearize, eq9_step,
                      order_log_prob, reverse_words, sample_order, sort_batch, star_log_prob, star_model, star_sample,
                      star_tokens, three_word_reversal)

torch.manual_seed(0)
FIG2 = ("S", [("NP", [("DT", [])]), ("VP", [("VBZ", []), ("NP", [("DT", []), ("NN", [])])]), (".", [])])


def test_read_process_is_permutation_invariant_but_the_lstm_encoder_is_not():
    m = ReadProcessWrite(d=16, steps=3)
    X = torch.rand(3, 7, 1)
    P = torch.randperm(7)
    assert torch.allclose(m.encode(X)[1], m.encode(X[:, P])[1], atol=1e-6)
    lstm = ReadProcessWrite(d=16, encoder="lstm")
    assert not torch.allclose(lstm.encode(X)[1], lstm.encode(X[:, P])[1], atol=1e-4)


def test_process_attention_is_a_distribution_and_reads_a_convex_combination():
    m = ReadProcessWrite(d=8, steps=2)
    M = torch.randn(2, 5, 8)
    q_star, attn = m.process(M)
    for a in attn:
        assert torch.allclose(a.sum(-1), torch.ones(2)) and (a >= 0).all()
    r = q_star[:, 8:]
    assert torch.allclose(r, (attn[-1][..., None] * M).sum(1), atol=1e-6)     # Eq. (6)


def test_zero_process_steps_leave_the_writer_blind():
    m = ReadProcessWrite(d=8, steps=0)
    assert torch.allclose(m.encode(torch.rand(1, 4, 1))[1], m.encode(torch.rand(1, 4, 1))[1])


def test_log_prob_is_the_chain_rule_and_greedy_returns_a_permutation():
    m = ReadProcessWrite(d=8, steps=1, mask_repeats=True)
    X = torch.rand(2, 4, 1)
    order = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
    lp = m.log_prob(X, order)
    assert lp.shape == (2,) and (lp <= 0).all()
    assert all(sorted(row.tolist()) == [0, 1, 2, 3] for row in m.greedy(X))


def test_figure_2_linearizations():
    assert " ".join(dfs_linearize(FIG2)) == "S NP DT !DT !NP VP VBZ !VBZ NP DT !DT NN !NN !NP !VP . !. !S"
    assert " ".join(bfs_linearize(FIG2)) == "S LEV NP VP . LEV DT PAR VBZ NP LEV PAR PAR DT NN DONE"


def test_breadth_first_linearization_is_invertible():
    rng = random.Random(0)

    def tree(depth):
        kids = [] if depth > 3 or rng.random() < 0.3 else [tree(depth + 1) for _ in range(rng.randint(1, 3))]
        return (rng.choice("ABCDE"), kids)

    for _ in range(50):
        t = tree(0)
        assert bfs_delinearize(bfs_linearize(t)) == t


def test_word_order_transforms():
    s = "This is a sentence .".split()
    assert three_word_reversal(s) == ["a", "is", "This", "<pad>", ".", "sentence"]
    assert reverse_words(s) == [".", "sentence", "a", "is", "This"]


def test_star_model_is_a_normalized_distribution():
    g = torch.Generator().manual_seed(0)
    model = star_model(3, K=3, peaky=2.0, generator=g)
    allv = torch.tensor(list(itertools.product(range(3), repeat=3)))
    assert abs(star_log_prob(model, allv).exp().sum().item() - 1) < 1e-5
    vals = star_sample(model, 4, g)
    assert torch.equal(star_tokens(vals, head_first=False)[:, -1], vals[:, 0])  # head moved last, tokens j*K + v


def test_eq9_max_is_the_exact_best_order():
    lm = SetLM(12, d=8)
    items = torch.tensor([[1, 5, 9], [2, 4, 7]])
    order, scores = best_order(lm, items)
    for row in range(2):
        brute = max(itertools.permutations(range(3)),
                    key=lambda p: order_log_prob(lm, items[row:row + 1], torch.tensor([p])).item())
        assert order[row].tolist() == list(brute)


def test_ancestral_sampling_matches_its_stated_probability():
    torch.manual_seed(0)
    lm = SetLM(6, d=8)
    items = torch.tensor([[0, 2, 4]]).expand(4000, -1)
    order, logq = sample_order(lm, items, torch.Generator().manual_seed(1))
    counts = collections.Counter(tuple(o.tolist()) for o in order)
    q = {tuple(o.tolist()): lq.exp().item() for o, lq in zip(order, logq)}
    for perm, c in counts.items():
        assert abs(c / 4000 - q[perm]) < 0.03
    assert abs(sum(q.values()) - 1) < 1e-4


def test_eq9_training_steps_use_valid_orders_and_reduce_the_loss():
    torch.manual_seed(0)
    lm = SetLM(20, d=16)
    opt = torch.optim.Adam(lm.parameters(), 1e-2)
    items = torch.tensor([[1, 6, 11, 16]] * 16)
    first = None
    for mode in ("given", "uniform", "max", "sample") * 10:
        loss, order = eq9_step(lm, opt, items, mode)
        assert all(sorted(r.tolist()) == [0, 1, 2, 3] for r in order)
        first = first or loss
    assert loss < first


def test_read_process_write_learns_to_sort():
    torch.manual_seed(0)
    m = ReadProcessWrite(d=64, steps=1, glimpses=1)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(450):
        X, o = sort_batch(64, 5)
        loss = -m.log_prob(X, o).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2); opt.step()
    X, o = sort_batch(300, 5, torch.Generator().manual_seed(5))
    assert (m.greedy(X) == o).all(1).float().mean() > 0.6
