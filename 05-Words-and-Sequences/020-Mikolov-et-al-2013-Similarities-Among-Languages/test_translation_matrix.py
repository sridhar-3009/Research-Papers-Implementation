"""Tests for Mikolov, Le & Sutskever (2013). About a second in total.

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest

from translation_matrix import (CBOW, cbow_windows, combined_scores, confidence, cooccurrence_vectors,
                                coverage_and_precision, edit_distance, edit_similarity, fit_least_squares,
                                fit_translation_matrix, precision_at_k, translate)

rng = np.random.default_rng(0)


def two_languages(n=300, d1=20, d2=12, noise=0.05, seed=0):
    """'Language 2' is a linear map of 'language 1' plus noise (Figure 1's idea)."""
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d1))
    A = r.normal(size=(d2, d1)) / np.sqrt(d1)
    Z = X @ A.T + noise * r.normal(size=(n, d2))
    return X, Z, A


def test_sgd_solves_eq_3():
    X, Z, _ = two_languages()
    W_sgd = fit_translation_matrix(X[:200], Z[:200], lr=0.002, epochs=60)
    W_ls = fit_least_squares(X[:200], Z[:200])
    assert np.allclose(W_sgd, W_ls, atol=0.02)


def test_learned_matrix_translates_unseen_words():
    X, Z, _ = two_languages()
    W = fit_least_squares(X[:200], Z[:200])                       # learn from 200 "dictionary" pairs
    test = np.arange(200, 300)                                    # translate 100 words never seen in training
    assert precision_at_k(W, X[test], test, Z, 1) > 0.95
    assert translate(W, X[250], Z, 5)[0] == 250


def test_source_and_target_can_have_different_dimensions():
    X, Z, _ = two_languages(d1=40, d2=10)                         # "800-d English to 200-d Spanish"
    W = fit_least_squares(X[:200], Z[:200])
    assert W.shape == (10, 40)


def test_confidence_filter_trades_coverage_for_precision():
    X, Z, _ = two_languages(noise=0.6)
    W = fit_least_squares(X[:200], Z[:200])
    test = np.arange(200, 300)
    cov0, p0 = coverage_and_precision(W, X[test], test, Z, threshold=-1.0)
    cov1, p1 = coverage_and_precision(W, X[test], test, Z, threshold=np.quantile(
        [confidence(W, x, Z) for x in X[test]], 0.7))
    assert cov0 == 1.0 and cov1 < 0.4
    assert p1 >= p0                                               # Table 3: less coverage, more precision


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3 and edit_distance("same", "same") == 0
    assert edit_similarity("animal", "animal") == 1.0
    assert edit_similarity("information", "informacion") > edit_similarity("dog", "perro")


def test_combined_scores():
    tm, ed = np.array([0.9, 0.2]), np.array([0.1, 0.8])
    assert np.allclose(combined_scores(tm, ed, 0.5), [0.5, 0.5])
    assert combined_scores(tm, ed, 0.8).argmax() == 0


def test_cooccurrence_vectors_are_unit_length_counts():
    ids = np.array([0, 1, 2, 1, 0, 3, 1])
    M = cooccurrence_vectors(ids, dict_word_ids=[1, 3], window=1)
    assert M.shape == (4, 2)
    assert np.allclose(np.linalg.norm(M[[0, 2]], axis=1), 1.0)
    raw0 = np.array([2.0, 1.0])                                    # word 0 sees word 1 twice, word 3 once
    assert np.allclose(M[0], np.log1p(raw0) / np.linalg.norm(np.log1p(raw0)))


def test_cbow_windows():
    ctx, center = cbow_windows(np.array([5, 6, 7]), c=1)
    assert ctx.tolist() == [[-1, 6], [5, 7], [6, -1]] and center.tolist() == [5, 6, 7]


def test_cbow_gradient_is_correct():
    m = CBOW(6, 4, seed=0)
    m.v_in = rng.normal(size=(6, 4)); m.v_out = rng.normal(size=(6, 4))
    ctx, center, negs = np.array([[1, 2, -1]]), np.array([3]), np.array([[0, 5]])

    def loss():
        h = m.v_in[1] + m.v_in[2]
        sig = lambda x: 1 / (1 + np.exp(-x))
        return -(np.log(sig(m.v_out[3] @ h)) + np.log(sig(-(m.v_out[[0, 5]] @ h))).sum())

    g = np.zeros_like(m.v_in)
    for i in range(m.v_in.size):
        old = m.v_in.flat[i]
        m.v_in.flat[i] = old + 1e-6; fp = loss()
        m.v_in.flat[i] = old - 1e-6; fm = loss()
        m.v_in.flat[i] = old
        g.flat[i] = (fp - fm) / 2e-6
    before = m.v_in.copy()
    m.step(ctx, center, negs, 1e-4)
    assert np.allclose(m.v_in - before, -1e-4 * g, atol=1e-9)
