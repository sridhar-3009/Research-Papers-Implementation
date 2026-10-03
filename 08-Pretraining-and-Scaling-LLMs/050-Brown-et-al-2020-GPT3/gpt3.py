"""Language Models are Few-Shot Learners (GPT-3; Brown et al., NeurIPS 2020).

  In-context learning: the task is given INSIDE the prompt, with no weight updates:
    zero-shot = task description only; one-shot = description + 1 demonstration; few-shot = K demonstrations
    (typically 10-100, as many as fit in n_ctx = 2048 tokens) (Section 2, Figure 2.1).
  Models (Table 2.1): 8 sizes from 125M to 175B, all trained on 300B tokens; GPT-2's architecture (pre-LN, modified
    init, byte-level BPE) with alternating dense and locally banded sparse attention layers; d_ff = 4 d_model.
  Data (Table 2.2): filtered Common Crawl 60%, WebText2 22%, Books1 8%, Books2 8%, Wikipedia 3%, sampled by quality
    rather than size (Wikipedia seen 3.4 times, Common Crawl 0.44 times).
  Common Crawl processing (Appendix A): a logistic-regression quality classifier, documents kept when
    np.random.pareto(alpha = 9) > 1 - score; fuzzy de-duplication with MinHash LSH (10 hashes).
  Evaluation (Section 2.4): multiple choice by per-token-normalised likelihood, or (ARC, OpenBookQA, RACE) by
    P(completion | context) / P(completion | 'Answer: '); free-form by beam search (width 4, alpha 0.6).
  Contamination (Section 4): an example is 'dirty' if any of its N-grams (N = 13 or smaller for short examples)
    also appears in the training data.
  Compute: C ~ 6 N D FLOPs (Appendix D); 175B x 300B tokens ~ 3.14e23 FLOPs ~ 3,640 petaflop/s-days.
"""

import hashlib
import math
import random
import re

import torch

# ---------------------------------------------------------------------------------------------------- models
TABLE_2_1 = {                      # name: (n_params, n_layers, d_model, n_heads, d_head, batch tokens, lr)
    "Small": (125e6, 12, 768, 12, 64, 0.5e6, 6.0e-4),
    "Medium": (350e6, 24, 1024, 16, 64, 0.5e6, 3.0e-4),
    "Large": (760e6, 24, 1536, 16, 96, 0.5e6, 2.5e-4),
    "XL": (1.3e9, 24, 2048, 24, 128, 1e6, 2.0e-4),
    "2.7B": (2.7e9, 32, 2560, 32, 80, 1e6, 1.6e-4),
    "6.7B": (6.7e9, 32, 4096, 32, 128, 2e6, 1.2e-4),
    "13B": (13.0e9, 40, 5140, 40, 128, 2e6, 1.0e-4),
    "175B": (175.0e9, 96, 12288, 96, 128, 3.2e6, 0.6e-4),
}


def param_estimate(n_layers, d_model, vocab=50257, n_ctx=2048):
    """12 L d^2 for the blocks (attention 4 d^2 + MLP 8 d^2) + token and position embeddings (tied output)."""
    return 12 * n_layers * d_model ** 2 + (vocab + n_ctx) * d_model


def training_flops(n_params, n_tokens):
    """Forward + backward ~ 6 FLOPs per parameter per token (Appendix D)."""
    return 6 * n_params * n_tokens


PFS_DAY = 1e15 * 86400                                                             # one petaflop/s for a day


# ---------------------------------------------------------------------------------------------------- attention
def layer_mask(layer, T, window=4):
    """Alternating dense / locally banded sparse layers (Section 2.1): even layers causal-dense, odd layers attend only
    to the previous `window` positions (and themselves). Returns a (T, T) bool 'may attend' matrix."""
    i = torch.arange(T)
    causal = i[None, :] <= i[:, None]
    if layer % 2 == 0:
        return causal
    return causal & (i[:, None] - i[None, :] < window)


# ---------------------------------------------------------------------------------------------------- data mixture
TABLE_2_2 = {"Common Crawl (filtered)": (410e9, 0.60), "WebText2": (19e9, 0.22), "Books1": (12e9, 0.08),
             "Books2": (55e9, 0.08), "Wikipedia": (3e9, 0.03)}


def epochs_elapsed(mix=TABLE_2_2, total_tokens=300e9):
    """How many passes over each dataset after `total_tokens`: weight x total / dataset size."""
    return {k: w * total_tokens / n for k, (n, w) in mix.items()}


def sample_source(rng, mix=TABLE_2_2):
    names, weights = zip(*[(k, w) for k, (_, w) in mix.items()])
    return rng.choices(names, weights=weights)[0]


# ---------------------------------------------------------------------------------------------------- filtering
def pareto_keep(score, rng, alpha=9.0):
    """Appendix A: keep a Common Crawl document if np.random.pareto(alpha) > 1 - score. High-quality-looking
    documents are almost always kept; low-scoring ones occasionally survive (keeps diversity)."""
    u = 1.0 - rng.random()
    return (u ** (-1.0 / alpha) - 1.0) > 1.0 - score                             # Pareto (Lomax) sample via inverse CDF


def shingles(text, k=5):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def minhash(text, n_hashes=10, k=5):
    """MinHash signature: for each of n hash functions, the minimum hash over the document's shingles. The chance
    that two signatures agree in one slot equals the Jaccard similarity of their shingle sets."""
    sh = shingles(text, k)
    return [min(int.from_bytes(hashlib.md5(f"{i}|{s}".encode()).digest()[:8], "little") for s in sh)
            for i in range(n_hashes)]


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def estimated_jaccard(sig_a, sig_b):
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


def dedup(docs, threshold=0.5, n_hashes=10):
    """Keep a document unless its MinHash-estimated Jaccard similarity to an already-kept one is >= threshold."""
    kept, sigs = [], []
    for d in docs:
        s = minhash(d, n_hashes)
        if all(estimated_jaccard(s, t) < threshold for t in sigs):
            kept.append(d); sigs.append(s)
    return kept


def ngram_set(text, n):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def is_dirty(example, train_ngrams, n=13):
    """Section 4: an evaluation example is 'dirty' if any of its n-grams occurs in the training data."""
    return bool(ngram_set(example, n) & train_ngrams)


# ---------------------------------------------------------------------------------------------------- prompting
def build_prompt(description, demos, query, k, sep="\n\n", arrow=" => "):
    """Zero-shot: k = 0 (description + query); one-shot: k = 1; few-shot: k demonstrations (Figure 2.1)."""
    parts = [description] if description else []
    parts += [f"{x}{arrow}{y}" for x, y in demos[:k]]
    parts.append(f"{query}{arrow}".rstrip())
    return sep.join(parts)


def choose(logprob_fn, context, completions, mode="per_token", answer_context="Answer:"):
    """Section 2.4's multiple-choice scoring. logprob_fn(context, completion) -> (sum log-prob, n_tokens).
      'sum':            total log P(completion | context)
      'per_token':      divided by the completion's token count (length-normalised; the default)
      'unconditional':  log P(completion | context) - log P(completion | answer_context)  (ARC, OBQA, RACE)"""
    scores = []
    for c in completions:
        lp, n = logprob_fn(context, c)
        if mode == "per_token":
            lp = lp / n
        elif mode == "unconditional":
            lp = lp - logprob_fn(answer_context, c)[0]
        scores.append(lp)
    return max(range(len(completions)), key=lambda i: scores[i])


# ---------------------------------------------------------------------------------------------------- synthetic tasks
def arithmetic_example(rng, digits=2, op="+"):
    """Section 3.9.1: 'Q: What is 48 plus 76? A: 124'."""
    a, b = rng.randrange(10 ** (digits - 1), 10 ** digits), rng.randrange(10 ** (digits - 1), 10 ** digits)
    words = {"+": "plus", "-": "minus", "*": "times"}
    ans = {"+": a + b, "-": a - b, "*": a * b}[op]
    return f"Q: What is {a} {words[op]} {b}?", f"A: {ans}"


def scramble(word, kind, rng):
    """Section 3.9.2: CL (cycle letters), A1 (anagram all but first/last), A2 (all but first/last two),
    RI (random insertion of a symbol between letters), RW (reversed word)."""
    if kind == "CL":
        k = rng.randrange(1, len(word))
        return word[k:] + word[:k]
    if kind in ("A1", "A2"):
        e = 1 if kind == "A1" else 2
        mid = list(word[e:-e]); rng.shuffle(mid)
        return word[:e] + "".join(mid) + word[-e:]
    if kind == "RI":
        return "".join(c + (rng.choice(" .,!?-") if i < len(word) - 1 else "") for i, c in enumerate(word))
    if kind == "RW":
        return word[::-1]
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------------- in-context toy
def dirichlet_sequences(n, T, V, alpha, generator=None):
    """Each sequence has its OWN random token distribution p ~ Dirichlet(alpha); a model can only predict well by
    estimating p from the context it has seen so far: learning within the context."""
    p = torch.distributions.Dirichlet(torch.full((V,), alpha)).sample((n,))
    return torch.multinomial(p, T, replacement=True, generator=generator)


def bayes_optimal_nll(x, V, alpha):
    """The best possible predictor: the Dirichlet-multinomial posterior predictive (count + alpha) / (t + V alpha)."""
    oh = torch.nn.functional.one_hot(x, V).float()
    before = oh.cumsum(1) - oh
    pred = (before + alpha) / (torch.arange(x.shape[1])[None, :, None] + V * alpha)
    return -pred.gather(2, x[..., None]).squeeze(-1).log()
