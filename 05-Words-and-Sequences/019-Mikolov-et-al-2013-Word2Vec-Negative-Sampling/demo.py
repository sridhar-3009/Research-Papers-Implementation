"""A light tour of Word2Vec (Mikolov et al. 2013). A tiny made-up corpus; about a second.

    python3 demo.py
"""

import numpy as np

from word2vec import (SkipGram, Vocab, analogy, huffman_codes, keep_probability, merge_phrases, nearest,
                      noise_distribution, phrase_scores, skipgram_pairs)

rng = np.random.default_rng(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


# A tiny made-up world: countries, their capitals and currencies, plus filler words.
countries = ["france", "germany", "italy", "spain", "japan", "china"]
capitals = ["paris", "berlin", "rome", "madrid", "tokyo", "beijing"]
currencies = ["euro", "euro", "euro", "euro", "yen", "yuan"]
sentences = []
for _ in range(3000):
    i = rng.integers(6)
    template = rng.integers(4)
    if template == 0:
        s = ["the", "capital", "of", countries[i], "is", capitals[i]]
    elif template == 1:
        s = [capitals[i], "is", "a", "city", "in", countries[i]]
    elif template == 2:
        s = ["in", countries[i], "people", "pay", "in", currencies[i]]
    else:
        s = ["the", "new", "york", "times", "wrote", "about", countries[i]]
    sentences += s
tokens = sentences

line("1. Subsampling frequent words (Eq. 5) and the 3/4-power noise")
vocab = Vocab(tokens, min_count=5)
keep = keep_probability(vocab.counts, t=1e-3)
P = noise_distribution(vocab.counts)
for w in ("the", "in", "is", "paris", "yen"):
    i = vocab.index[w]
    f = vocab.counts[i] / vocab.counts.sum()
    print(f"{w:7s} frequency {f:.4f}   kept with prob {keep[i]:.2f}   noise prob {P[i]:.4f} (unigram {f:.4f})")
print("-> very frequent words are mostly thrown away; the 3/4 power gives rare words more noise samples.")

line("2. The Huffman tree for hierarchical softmax")
paths, codes = huffman_codes(vocab.counts)
for w in ("the", "in", "paris", "yen"):
    print(f"{w:7s} code {''.join(map(str, codes[vocab.index[w]])):12s} ({len(codes[vocab.index[w]])} sigmoid decisions instead of a {len(vocab)}-way softmax)")
print("-> frequent words sit near the root: short codes, cheap updates.")

line("3. Phrases (Eq. 6) on a corpus of random words with 'new york' mixed in")
filler = [f"word{i}" for i in range(50)]
text = []
for _ in range(400):
    text += list(rng.choice(filler, 5))
    if rng.random() < 0.3:
        text += ["new", "york"]
scores = phrase_scores(text, delta=2)
best = sorted(scores.items(), key=lambda kv: -kv[1])[:4]
for (a, b), sc in best:
    print(f"score({a}, {b}) = {sc:.5f}")
merged = merge_phrases(text, threshold=4e-3, delta=2)
print("tokens merged at threshold 0.004:", sorted({t for t in merged if "_" in t}))
print(f"-> 'new york' scores {best[0][1] / best[1][1]:.1f}x higher than the best pair of random words that met by chance.")

line("4. Train skip-gram with negative sampling (one small run)")
ids = vocab.encode(tokens)
m = SkipGram(len(vocab), 20, seed=0)
c, o = skipgram_pairs(ids, c=3, rng=rng)
order = rng.permutation(len(c))
c, o = c[order], o[order]
for epoch in range(3):
    loss = 0.0
    for s in range(0, len(c), 256):
        negs = rng.choice(len(vocab), size=(len(c[s:s + 256]), 5), p=P)
        loss += m.neg_step_batch(c[s:s + 256], o[s:s + 256], negs, 0.05) * len(c[s:s + 256])
    print(f"epoch {epoch + 1}: average NEG loss {loss / len(c):.3f}")
V = m.v_in
w = vocab.index
for q in ("paris", "japan", "euro"):
    print(f"nearest to {q:6s}: {[vocab.words[i] for i in nearest(V, V[w[q]], 3, exclude=(w[q],))]}")
right = 0
for a, b, cc, d in (("france", "paris", "japan", "tokyo"), ("germany", "berlin", "china", "beijing"),
                    ("paris", "france", "rome", "italy")):
    got = vocab.words[analogy(V, w[a], w[b], w[cc])[0]]
    right += got == d
    print(f"{a} : {b} :: {cc} : {got}   (correct: {d})")
print(f"-> similar words get similar vectors (the neighbours above are right), but {right}/3 analogies")
print("   are correct here: analogies need the SAME offset for every country->capital pair, which")
print("   only emerges from huge, varied text (the paper used ~1 billion words for 61% accuracy).")
