"""Mikolov, Le & Sutskever (2013), "Exploiting Similarities among Languages for Machine Translation".

  CBOW                      Section 2 / Figure 2: predict the middle word from the SUM of its context vectors
                            (negative sampling, gradients by hand; the paper used CBOW for speed)
  fit_translation_matrix    Eq. (3): min_W sum ||W x_i - z_i||^2, by SGD as in the paper
  fit_least_squares         the same problem solved exactly (a reference)
  translate, precision_at_k map with W, then the cosine-nearest target words; P@1, P@5
  confidence                Section 6.1: max_i cos(W x, z_i); skip translations below a threshold
  edit_distance             baseline 1 (Section 5.2): spelling similarity
  cooccurrence_vectors      baseline 2 (Section 5.2): counts of dictionary words in a window,
                            rescaled by corpus size, log, L2-normalized
  combined_scores           "ED + TM": a weighted mix of both similarity scores
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# CBOW with negative sampling (Figure 2, left)
# ---------------------------------------------------------------------------

class CBOW:
    """h = sum of the input vectors of the 2c context words; predict the middle word from h with
    negative sampling: maximize log sigma(v'_w . h) + sum_k log sigma(-v'_{n_k} . h)."""

    def __init__(self, vocab_size, dim, seed=0):
        rng = np.random.default_rng(seed)
        self.v_in = (rng.random((vocab_size, dim)) - 0.5) / dim
        self.v_out = np.zeros((vocab_size, dim))

    def step(self, context, center, negatives, lr):
        """context: (B, 2c) ids (use -1 for padding at sentence edges), center: (B,), negatives: (B, k)."""
        mask = (context >= 0)[:, :, None]
        h = (self.v_in[np.maximum(context, 0)] * mask).sum(1)                   # (B, d)
        targets = np.concatenate([center[:, None], negatives], 1)               # (B, 1 + k)
        u = self.v_out[targets]
        s = sigmoid(np.einsum("bkd,bd->bk", u, h))
        labels = np.zeros_like(s); labels[:, 0] = 1
        g = labels - s
        grad_h = np.einsum("bk,bkd->bd", g, u)                                   # d logL / d h
        np.add.at(self.v_out, targets.ravel(), lr * (g[:, :, None] * h[:, None, :]).reshape(-1, h.shape[1]))
        # h is a sum, so every context word receives the same gradient
        ctx = np.maximum(context, 0)
        upd = np.broadcast_to(lr * grad_h[:, None, :], (len(ctx), ctx.shape[1], h.shape[1])) * mask
        np.add.at(self.v_in, ctx.ravel(), upd.reshape(-1, h.shape[1]))
        return -(np.log(s[:, 0] + 1e-12) + np.log(1 - s[:, 1:] + 1e-12).sum(1)).mean()


def cbow_windows(ids, c):
    """For every position: the 2c surrounding ids (-1 past the ends) and the middle id."""
    n = len(ids)
    padded = np.concatenate([np.full(c, -1), ids, np.full(c, -1)])
    offsets = [o for o in range(-c, c + 1) if o != 0]
    context = np.stack([padded[c + o: c + o + n] for o in offsets], 1)
    return context, ids


# ---------------------------------------------------------------------------
# Section 4: the translation matrix
# ---------------------------------------------------------------------------

def fit_translation_matrix(X, Z, lr=0.01, epochs=100, seed=0):
    """Eq. (3): minimize sum_i ||W x_i - z_i||^2 'with stochastic gradient descent'.
    X: (n, d1) source vectors, Z: (n, d2) target vectors -> W: (d2, d1).
    Gradient of one term: 2 (W x_i - z_i) x_i^T."""
    rng = np.random.default_rng(seed)
    W = np.zeros((Z.shape[1], X.shape[1]))
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            W -= lr * 2 * np.outer(W @ X[i] - Z[i], X[i])
    return W


def fit_least_squares(X, Z):
    """The exact minimizer of Eq. (3): W^T = argmin ||X W^T - Z||, i.e. W = (X^+ Z)^T."""
    return np.linalg.lstsq(X, Z, rcond=None)[0].T


def translate(W, x, target_vectors, k=5):
    """z = W x, then the k target words with the highest cosine similarity to z."""
    sims = unit(target_vectors) @ unit(W @ x)
    return np.argsort(-sims)[:k]


def precision_at_k(W, X_test, gold, target_vectors, k):
    """Fraction of test words whose correct translation is among the top k candidates.
    gold[i] = id of the correct target word (exact match only, synonyms count as wrong)."""
    sims = unit(X_test @ W.T) @ unit(target_vectors).T                       # (n_test, V_target)
    topk = np.argsort(-sims, axis=1)[:, :k]
    return float(np.mean([g in row for g, row in zip(gold, topk)]))


def confidence(W, x, target_vectors):
    """Section 6.1: max_i cos(W x, z_i). If W x lands far from every real word, don't trust it."""
    return float((unit(target_vectors) @ unit(W @ x)).max())


def coverage_and_precision(W, X_test, gold, target_vectors, threshold, k=1):
    """Table 3: keep only translations with confidence >= threshold; report (coverage, P@k)."""
    conf = np.array([confidence(W, x, target_vectors) for x in X_test])
    keep = conf >= threshold
    if not keep.any():
        return 0.0, float("nan")
    return float(keep.mean()), precision_at_k(W, X_test[keep], np.asarray(gold)[keep], target_vectors, k)


# ---------------------------------------------------------------------------
# Section 5.2: baselines
# ---------------------------------------------------------------------------

def edit_distance(a, b):
    """Levenshtein distance: the fewest insertions, deletions and substitutions turning a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(a, b):
    """1 - distance / longer length: 1 for identical spellings, ~0 for unrelated ones."""
    return 1.0 - edit_distance(a, b) / max(len(a), len(b), 1)


def cooccurrence_vectors(ids, dict_word_ids, window=10, size_ratio=1.0):
    """Baseline 2: for every word, count how often each DICTIONARY word appears within `window`
    positions. Then (as in the paper) divide by the corpus-size ratio, take the log and L2-normalize.
    Returns a (vocab, len(dict_word_ids)) matrix. (log(1 + count) is used so zero counts stay finite.)"""
    V = int(ids.max()) + 1
    col = {w: j for j, w in enumerate(dict_word_ids)}
    M = np.zeros((V, len(dict_word_ids)))
    for off in range(1, window + 1):
        for a, b in ((ids[:-off], ids[off:]), (ids[off:], ids[:-off])):
            cols = np.array([col.get(int(w), -1) for w in b])
            ok = cols >= 0
            np.add.at(M, (a[ok], cols[ok]), 1.0)
    return unit(np.log1p(M / size_ratio))


def combined_scores(tm_sims, ed_sims, weight=0.5):
    """ED + TM: a weighted combination of the two similarity scores for every candidate word."""
    return weight * tm_sims + (1 - weight) * ed_sims
