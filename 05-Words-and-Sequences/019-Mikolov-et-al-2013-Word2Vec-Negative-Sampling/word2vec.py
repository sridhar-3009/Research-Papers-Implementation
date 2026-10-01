"""Word2Vec - Mikolov, Sutskever, Chen, Corrado & Dean (2013), "Distributed Representations of Words
and Phrases and their Compositionality". Written from scratch in NumPy, gradients by hand.

  Vocab                 word <-> id, counts, min_count
  keep_probability      Eq. (5): subsampling of frequent words, P(discard) = 1 - sqrt(t / f(w))
  noise_distribution    U(w)^(3/4) / Z, the noise for negative sampling / NCE
  skipgram_pairs        (center, context) pairs with a window of up to c words (Eq. 1)
  SkipGram              input vectors v_w and output vectors v'_w, with three output layers:
      .softmax_log_prob     Eq. (2): the full softmax (cost ~ W)
      .hs_log_prob, .hs_step  Eq. (3): hierarchical softmax over a Huffman tree (cost ~ log W)
      .neg_loss, .neg_step  Eq. (4): negative sampling (cost ~ k)
      .nce_step             noise contrastive estimation (for Table 1)
  huffman_codes         the binary Huffman tree: frequent words get short codes
  phrase_scores, merge_phrases   Eq. (6): find phrases like "new_york"
  analogy, nearest      the analogical reasoning test: vec(b) - vec(a) + vec(c)
"""

import heapq
from collections import Counter

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# Vocabulary, subsampling, noise
# ---------------------------------------------------------------------------

class Vocab:
    """'We discarded from the vocabulary all words that occurred less than 5 times.'
    Words are sorted by count (most frequent first)."""

    def __init__(self, tokens, min_count=5):
        counts = Counter(tokens)
        self.words = [w for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if c >= min_count]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.counts = np.array([counts[w] for w in self.words], dtype=np.float64)

    def __len__(self):
        return len(self.words)

    def encode(self, tokens):
        return np.array([self.index[t] for t in tokens if t in self.index], dtype=np.int64)


def keep_probability(counts, t=1e-5):
    """Eq. (5): word w_i is DISCARDED with probability 1 - sqrt(t / f(w_i)), f = relative frequency.
    So it is kept with probability min(1, sqrt(t / f)). Words rarer than t are always kept;
    'the' (f ~ 0.05) is kept only ~1.4% of the time when t = 1e-5.
    (The released C code uses a slightly different formula: (sqrt(f/t) + 1) * t / f.)"""
    f = counts / counts.sum()
    return np.minimum(1.0, np.sqrt(t / f))


def subsample(ids, counts, t=1e-5, rng=None):
    rng = rng or np.random.default_rng()
    return ids[rng.random(len(ids)) < keep_probability(counts, t)[ids]]


def noise_distribution(counts, power=0.75):
    """P_n(w) = U(w)^(3/4) / Z: 'outperformed significantly the unigram and the uniform distributions'.
    Raising to 3/4 flattens the distribution: rare words get sampled a bit more often."""
    p = counts ** power
    return p / p.sum()


def skipgram_pairs(ids, c=5, rng=None, dynamic=True):
    """All (center, context) pairs with |offset| <= window, for Eq. (1). With dynamic=True the window
    for each center word is drawn uniformly from 1..c ('c can be a function of the center word'), as in
    the released code: near words are used more often than far ones."""
    rng = rng or np.random.default_rng()
    centers, contexts = [], []
    n = len(ids)
    windows = rng.integers(1, c + 1, n) if dynamic else np.full(n, c)
    for j in range(1, c + 1):
        use = windows >= j
        for sign in (-1, 1):
            idx = np.arange(n)
            other = idx + sign * j
            ok = use & (other >= 0) & (other < n)
            centers.append(ids[idx[ok]]); contexts.append(ids[other[ok]])
    return np.concatenate(centers), np.concatenate(contexts)


# ---------------------------------------------------------------------------
# Hierarchical softmax: the Huffman tree
# ---------------------------------------------------------------------------

def huffman_codes(counts):
    """Build a binary Huffman tree over the W words. Returns, for every word, its path:
    (inner-node ids from the root, and the branch taken at each: 0 or 1).
    The W-1 inner nodes are numbered 0..W-2; each has its own output vector v'_n."""
    W = len(counts)
    heap = [(float(c), i) for i, c in enumerate(counts)]          # leaves are 0..W-1
    heapq.heapify(heap)
    parent, branch = {}, {}
    next_id = W
    while len(heap) > 1:
        c1, a = heapq.heappop(heap)
        c2, b = heapq.heappop(heap)
        parent[a], branch[a] = next_id, 0
        parent[b], branch[b] = next_id, 1
        heapq.heappush(heap, (c1 + c2, next_id))
        next_id += 1
    root = next_id - 1
    paths, codes = [], []
    for w in range(W):
        nodes, bits, x = [], [], w
        while x != root:
            nodes.append(parent[x] - W); bits.append(branch[x])
            x = parent[x]
        paths.append(np.array(nodes[::-1], dtype=np.int64))      # from the root down
        codes.append(np.array(bits[::-1], dtype=np.int64))
    return paths, codes


# ---------------------------------------------------------------------------
# The Skip-gram model
# ---------------------------------------------------------------------------

class SkipGram:
    """v_in[w]  : the 'input' vector v_w   (this is the word vector we keep)
       v_out[w] : the 'output' vector v'_w (softmax / NEG / NCE)
       v_node[n]: one vector per inner node of the Huffman tree (hierarchical softmax)"""

    def __init__(self, vocab_size, dim=100, counts=None, seed=0):
        rng = np.random.default_rng(seed)
        self.v_in = (rng.random((vocab_size, dim)) - 0.5) / dim    # as in the C code: small uniform
        self.v_out = np.zeros((vocab_size, dim))                    # output vectors start at zero
        self.v_node = np.zeros((max(vocab_size - 1, 1), dim))
        if counts is not None:
            self.paths, self.codes = huffman_codes(counts)

    # ---- Eq. (2): the full softmax ----
    def softmax_log_prob(self, w_in, w_out):
        """log p(w_O | w_I) = v'_O . v_I - log sum_w exp(v'_w . v_I). Cost proportional to W."""
        s = self.v_out @ self.v_in[w_in]
        m = s.max()
        return s[w_out] - (m + np.log(np.exp(s - m).sum()))

    # ---- Eq. (3): hierarchical softmax ----
    def hs_log_prob(self, w_in, w_out):
        """log p(w | w_I) = sum over the path of log sigma([[branch]] * v'_n . v_I), with
        [[branch]] = +1 for one child and -1 for the other. Only ~log2(W) terms."""
        nodes, bits = self.paths[w_out], self.codes[w_out]
        sign = 1 - 2 * bits                                        # bit 0 -> +1, bit 1 -> -1
        return np.log(sigmoid(sign * (self.v_node[nodes] @ self.v_in[w_in]))).sum()

    def hs_step(self, w_in, w_out, lr):
        """One SGD step on -log p(w_O | w_I) (hierarchical softmax). Returns the loss."""
        nodes, bits = self.paths[w_out], self.codes[w_out]
        sign = 1 - 2 * bits
        v = self.v_in[w_in].copy()
        u = self.v_node[nodes]
        s = sigmoid(sign * (u @ v))
        g = (1 - s) * sign                                          # d log sigma(sign*x)/dx
        self.v_in[w_in] += lr * (g @ u)
        self.v_node[nodes] += lr * g[:, None] * v[None, :]
        return -np.log(s).sum()

    # ---- Eq. (4): negative sampling ----
    def neg_loss(self, w_in, w_out, negatives):
        """-[ log sigma(v'_O . v_I) + sum_i log sigma(-v'_{n_i} . v_I) ]"""
        v = self.v_in[w_in]
        return -(np.log(sigmoid(self.v_out[w_out] @ v)) + np.log(sigmoid(-(self.v_out[negatives] @ v))).sum())

    def neg_step(self, w_in, w_out, negatives, lr):
        """One SGD step on the NEG objective, gradients written out:
           positive word:  d/dv' = (1 - sigma(v'.v)) v       pull v'_O towards v_I
           negative words: d/dv' = -sigma(v'.v) v            push v'_n away from v_I
           input vector:   d/dv  = (1 - sigma) v'_O - sum sigma_i v'_{n_i}"""
        v = self.v_in[w_in].copy()
        targets = np.concatenate([[w_out], negatives])
        labels = np.zeros(len(targets)); labels[0] = 1.0
        u = self.v_out[targets]
        s = sigmoid(u @ v)
        g = labels - s                                              # gradient of the log-likelihood w.r.t. u.v
        self.v_in[w_in] += lr * (g @ u)
        np.add.at(self.v_out, targets, lr * g[:, None] * v[None, :])   # a negative may repeat
        return -(np.log(s[0]) + np.log(1 - s[1:]).sum())

    def neg_step_batch(self, w_in, w_out, negatives, lr, shift=None):
        """The same update for a batch of B pairs at once (negatives: B x k). Used for training.
        shift: optional per-word logit offset; shift = log(k P_n(w)) turns NEG into NCE."""
        v = self.v_in[w_in]                                         # (B, d)
        targets = np.concatenate([w_out[:, None], negatives], 1)   # (B, 1 + k)
        u = self.v_out[targets]                                     # (B, 1 + k, d)
        logits = np.einsum("bkd,bd->bk", u, v)
        s = sigmoid(logits if shift is None else logits - shift[targets])
        labels = np.zeros_like(s); labels[:, 0] = 1.0
        g = labels - s
        np.add.at(self.v_in, w_in, lr * np.einsum("bk,bkd->bd", g, u))
        np.add.at(self.v_out, targets.ravel(), lr * (g[:, :, None] * v[:, None, :]).reshape(-1, v.shape[1]))
        return -(np.log(s[:, 0] + 1e-12).sum() + np.log(1 - s[:, 1:] + 1e-12).sum()) / len(w_in)

    # ---- NCE (Gutmann & Hyvarinen; Mnih & Teh), for comparison in Table 1 ----
    def nce_step(self, w_in, w_out, negatives, noise_probs, k, lr):
        """Like NEG, but the logit is shifted by log(k P_n(w)): NCE uses the noise PROBABILITIES,
        NEG only the samples. That shift makes NCE approximate the softmax's log-probabilities."""
        v = self.v_in[w_in].copy()
        targets = np.concatenate([[w_out], negatives])
        labels = np.zeros(len(targets)); labels[0] = 1.0
        u = self.v_out[targets]
        s = sigmoid(u @ v - np.log(k * noise_probs[targets]))
        g = labels - s
        self.v_in[w_in] += lr * (g @ u)
        np.add.at(self.v_out, targets, lr * g[:, None] * v[None, :])
        return -(np.log(s[0]) + np.log(1 - s[1:]).sum())


# ---------------------------------------------------------------------------
# Section 4: phrases
# ---------------------------------------------------------------------------

def phrase_scores(tokens, delta=5):
    """Eq. (6): score(a, b) = (count(a b) - delta) / (count(a) * count(b)).
    High when a and b appear together much more often than their counts suggest; delta stops
    rare word pairs from scoring high by accident."""
    uni = Counter(tokens)
    bi = Counter(zip(tokens[:-1], tokens[1:]))
    return {(a, b): (c - delta) / (uni[a] * uni[b]) for (a, b), c in bi.items()}


def merge_phrases(tokens, threshold, delta=5):
    """Replace every bigram scoring above threshold by one token 'a_b'. Run 2-4 passes with a
    decreasing threshold to build longer phrases ('new_york_times')."""
    scores = phrase_scores(tokens, delta)
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and scores.get((tokens[i], tokens[i + 1]), -1) > threshold:
            out.append(tokens[i] + "_" + tokens[i + 1]); i += 2
        else:
            out.append(tokens[i]); i += 1
    return out


# ---------------------------------------------------------------------------
# The analogy test
# ---------------------------------------------------------------------------

def normalized(vectors):
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


def nearest(vectors, query, k=5, exclude=()):
    """Indices of the k vectors with the highest cosine similarity to `query`."""
    V = normalized(vectors)
    sims = V @ (query / (np.linalg.norm(query) + 1e-12))
    sims[list(exclude)] = -np.inf
    return np.argsort(-sims)[:k]


def analogy(vectors, a, b, c, k=1):
    """'a is to b as c is to ?': the word closest (cosine) to vec(b) - vec(a) + vec(c),
    'discarding the input words from the search'."""
    V = normalized(vectors)
    return nearest(vectors, V[b] - V[a] + V[c], k, exclude=(a, b, c))
