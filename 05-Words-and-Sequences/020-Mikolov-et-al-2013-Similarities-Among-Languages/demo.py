"""A light tour of Mikolov, Le & Sutskever (2013). Synthetic "languages"; about a second.

    python3 demo.py
"""

import numpy as np

from translation_matrix import (coverage_and_precision, edit_similarity, fit_least_squares, fit_translation_matrix,
                                precision_at_k, translate, unit)

rng = np.random.default_rng(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Two 'languages' that share the same geometry (Figure 1's idea)")
print("English concepts get random 50-d vectors. 'Spanish' vectors are a ROTATION + SCALING of them")
print("into 30-d, plus noise: the same concepts, arranged the same way, in a different coordinate system.")
n, d_en, d_es = 2000, 50, 30
X = rng.normal(size=(n, d_en))
A = rng.normal(size=(d_es, d_en)) / np.sqrt(d_en)
Z = X @ A.T + 0.3 * rng.normal(size=(n, d_es))
print(f"cosine(english word 7, spanish word 7) directly: {unit(X[7])[:d_es] @ unit(Z[7]):+.2f}   (meaningless: different spaces)")

line("2. Learn W from a small dictionary, translate words never seen (Section 4)")
train, test = np.arange(500), np.arange(500, 1500)
W = fit_translation_matrix(X[train], Z[train], lr=0.002, epochs=30)
print(f"W is {W.shape[0]}x{W.shape[1]}, learned by SGD on {len(train)} word pairs")
for k in (1, 5):
    print(f"precision@{k} on {len(test)} unseen words: {100 * precision_at_k(W, X[test], test, Z, k):.1f}%")
print(f"translation candidates for english word 900: {translate(W, X[900], Z, 5).tolist()} (correct: 900)")
W_ls = fit_least_squares(X[train], Z[train])
print(f"SGD solution vs exact least squares: max difference {np.abs(W - W_ls).max():.4f}")

line("3. How many dictionary pairs are needed?")
for m in (20, 40, 60, 100, 300):
    Wm = fit_least_squares(X[:m], Z[:m])
    print(f"{m:4d} pairs -> P@1 {100 * precision_at_k(Wm, X[test], test, Z, 1):5.1f}%")
print(f"-> W has {d_es}x{d_en} = {d_es * d_en} numbers; each pair gives {d_es} equations, so ~{d_en} pairs")
print("   are the bare minimum. Real languages are not exactly linear, so the paper used 5,000.")

line("4. Confidence: trust translations that land near a real word (Section 6.1, Table 3)")
Zn = X @ A.T + 1.2 * rng.normal(size=(n, d_es))                 # a much noisier "language"
Wn = fit_least_squares(X[train], Zn[train])
for th in (-1.0, 0.6, 0.7, 0.8):
    cov, p1 = coverage_and_precision(Wn, X[test], test, Zn, th, 1)
    print(f"threshold {th:+.1f}: translate {100 * cov:5.1f}% of words, P@1 {100 * p1:5.1f}%")
print("-> fewer translations, but more of them right: the paper's 92.5% -> 17% coverage, 53% -> 78% P@1.")

line("5. The edit-distance baseline (Section 5.2)")
for a, b in (("information", "informacion"), ("university", "universidad"), ("dog", "perro"), ("house", "casa")):
    print(f"{a:12s} ~ {b:12s} spelling similarity {edit_similarity(a, b):.2f}")
print("-> helps for related spellings (English/Spanish), useless for 'dog'/'perro' and for English/Czech;")
print("   the translation matrix works from MEANING (context), so it covers both cases.")
