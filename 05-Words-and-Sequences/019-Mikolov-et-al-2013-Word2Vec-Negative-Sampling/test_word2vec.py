"""Tests: each checks a statement or formula of Mikolov et al. (2013). About a second in total.

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest

from word2vec import (SkipGram, Vocab, analogy, huffman_codes, keep_probability, merge_phrases, nearest,
                      noise_distribution, sigmoid, skipgram_pairs)

rng = np.random.default_rng(0)


def random_model(W=12, d=6, seed=0):
    r = np.random.default_rng(seed)
    counts = r.integers(1, 1000, W).astype(float)
    m = SkipGram(W, d, counts, seed=seed)
    m.v_in = r.normal(size=(W, d)); m.v_out = r.normal(size=(W, d)); m.v_node = r.normal(size=(W - 1, d))
    return m, counts


# ---- Eq. (2) and Eq. (3): both define probability distributions ----

def test_full_softmax_sums_to_one():
    m, _ = random_model()
    assert sum(np.exp(m.softmax_log_prob(3, w)) for w in range(12)) == pytest.approx(1.0)


def test_hierarchical_softmax_sums_to_one():
    # "It can be verified that sum_w p(w | w_I) = 1"
    m, _ = random_model(W=37)
    for w_in in (0, 5, 20):
        assert sum(np.exp(m.hs_log_prob(w_in, w)) for w in range(37)) == pytest.approx(1.0)


def test_huffman_tree_gives_short_codes_to_frequent_words():
    counts = np.array([1000, 500, 200, 100, 50, 20, 10, 5, 2, 1], dtype=float)
    paths, codes = huffman_codes(counts)
    lengths = [len(c) for c in codes]
    assert lengths[0] <= lengths[-1] and lengths == sorted(lengths)
    strings = ["".join(map(str, c)) for c in codes]
    assert not any(a != b and b.startswith(a) for a in strings for b in strings)     # prefix-free
    avg = np.dot(lengths, counts) / counts.sum()
    assert avg <= np.log2(len(counts)) + 1                                            # "no greater than log W"


# ---- the gradients written by hand ----

def numerical_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        old = x.flat[i]
        x.flat[i] = old + h; fp = f()
        x.flat[i] = old - h; fm = f()
        x.flat[i] = old
        g.flat[i] = (fp - fm) / (2 * h)
    return g


def test_negative_sampling_step_is_gradient_descent_on_eq_4():
    m, _ = random_model()
    w_in, w_out, negs, lr = 2, 7, np.array([1, 4, 9]), 1e-4
    gin = numerical_grad(lambda: m.neg_loss(w_in, w_out, negs), m.v_in)
    gout = numerical_grad(lambda: m.neg_loss(w_in, w_out, negs), m.v_out)
    before_in, before_out = m.v_in.copy(), m.v_out.copy()
    m.neg_step(w_in, w_out, negs, lr)
    assert np.allclose(m.v_in - before_in, -lr * gin, atol=1e-9)
    assert np.allclose(m.v_out - before_out, -lr * gout, atol=1e-9)


def test_hierarchical_softmax_step_is_gradient_descent():
    m, _ = random_model(W=20)
    w_in, w_out, lr = 3, 11, 1e-4
    gin = numerical_grad(lambda: -m.hs_log_prob(w_in, w_out), m.v_in)
    gnode = numerical_grad(lambda: -m.hs_log_prob(w_in, w_out), m.v_node)
    b_in, b_node = m.v_in.copy(), m.v_node.copy()
    m.hs_step(w_in, w_out, lr)
    assert np.allclose(m.v_in - b_in, -lr * gin, atol=1e-9)
    assert np.allclose(m.v_node - b_node, -lr * gnode, atol=1e-9)


def test_batched_negative_sampling_equals_single_steps():
    a, _ = random_model(seed=1)
    b, _ = random_model(seed=1)
    a.neg_step(2, 5, np.array([1, 3]), 0.1)
    b.neg_step_batch(np.array([2]), np.array([5]), np.array([[1, 3]]), 0.1)
    assert np.allclose(a.v_in, b.v_in) and np.allclose(a.v_out, b.v_out)


def test_batched_nce_equals_single_nce_step():
    a, counts = random_model(seed=2)
    b, _ = random_model(seed=2)
    P = noise_distribution(counts)
    a.nce_step(2, 5, np.array([1, 3]), P, 2, 0.1)
    b.neg_step_batch(np.array([2]), np.array([5]), np.array([[1, 3]]), 0.1, shift=np.log(2 * P))
    assert np.allclose(a.v_in, b.v_in) and np.allclose(a.v_out, b.v_out)


# ---- subsampling and noise ----

def test_subsampling_formula():
    counts = np.array([5e4, 1.0, 1e6 - 5e4 - 1.0])
    keep = keep_probability(counts, t=1e-5)
    assert keep[0] == pytest.approx(np.sqrt(1e-5 / 0.05))          # 'the' at 5% frequency: kept ~1.4%
    assert keep[1] == 1.0                                          # rarer than t: always kept
    f = counts / counts.sum()
    assert np.all(np.diff(keep[np.argsort(f)]) <= 0)               # more frequent -> discarded more


def test_three_quarter_power_flattens_the_unigram():
    counts = np.array([100.0, 1.0])
    p = noise_distribution(counts)
    assert p[0] / p[1] == pytest.approx(100 ** 0.75)               # 31.6 instead of 100


def test_skipgram_pairs_fixed_window():
    c, o = skipgram_pairs(np.arange(4), c=1, dynamic=False)
    assert sorted(zip(c.tolist(), o.tolist())) == [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]


def test_dynamic_window_prefers_near_words():
    ids = np.arange(5000)
    c, o = skipgram_pairs(ids, c=5, rng=np.random.default_rng(0))
    dist = np.abs(c - o)
    counts = np.bincount(dist)[1:]
    assert np.all(np.diff(counts) < 0)                             # distance 1 most common, 5 least


# ---- phrases ----

def test_phrase_detection_eq_6():
    r = np.random.default_rng(1)
    filler = [f"w{i}" for i in range(50)]
    tokens = []
    for _ in range(400):
        tokens += list(r.choice(filler, 5))
        if r.random() < 0.3:
            tokens += ["new", "york"]
    # "new york": (~120 - 2) / (120 * 120) ~ 0.008; a lucky random pair: (4 - 2) / (40 * 40) ~ 0.0013
    merged = merge_phrases(tokens, threshold=4e-3, delta=2)
    assert "new_york" in merged
    assert not any(t.startswith("w") and "_" in t for t in merged)     # random pairs are not phrases


# ---- the analogy test ----

def test_analogy_with_linear_structure():
    words = ["man", "woman", "king", "queen", "apple"]
    gender, royal = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    V = np.array([np.zeros(3) + 0.1, gender + 0.1, royal + 0.1, gender + royal + 0.1, np.array([0, 0, 1.0])])
    assert words[analogy(V, 0, 1, 2)[0]] == "queen"                 # man : woman :: king : ?
    assert words[nearest(V, V[2], 2, exclude=(2,))[0]] == "queen"


# ---- learning works on a toy corpus ----

def test_negative_sampling_learns_co_occurrence():
    # two "topics": words 0-4 always appear together, words 5-9 always appear together
    r = np.random.default_rng(0)
    corpus = np.concatenate([r.integers(0, 5, 10) if r.random() < 0.5 else r.integers(5, 10, 10) for _ in range(300)])
    counts = np.bincount(corpus, minlength=10).astype(float)
    m = SkipGram(10, 8, seed=0)
    P = noise_distribution(counts)
    c, o = skipgram_pairs(corpus, c=2, rng=r)
    for _ in range(3):
        for s in range(0, len(c), 64):
            negs = r.choice(10, size=(len(c[s:s + 64]), 3), p=P)
            m.neg_step_batch(c[s:s + 64], o[s:s + 64], negs, 0.05)
    V = m.v_in / np.linalg.norm(m.v_in, axis=1, keepdims=True)
    S = V @ V.T
    within = (S[:5, :5].sum() - 5 + S[5:, 5:].sum() - 5) / 40
    across = S[:5, 5:].mean()
    assert within > across + 0.5


def test_vocab_min_count():
    v = Vocab(["a"] * 6 + ["b"] * 5 + ["c"] * 4, min_count=5)
    assert v.words == ["a", "b"] and v.encode(["a", "c", "b"]).tolist() == [0, 1]
