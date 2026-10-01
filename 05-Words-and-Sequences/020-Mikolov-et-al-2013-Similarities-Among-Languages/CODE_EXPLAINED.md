# The code, explained simply

How the code in this folder implements Mikolov, Le & Sutskever (2013).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `translation_matrix.py` | CBOW (negative sampling, hand-written gradients), the translation matrix (SGD and exact), translation + P@k, confidence, edit distance, co-occurrence baseline, ED + TM |
| `experiments.py` | Table 2, vector sizes, Table 3, Figure 1 on Europarl + MUSE dictionaries (heavy, not run here) |
| `demo.py` | A light tour on synthetic "languages" (about a second) |
| `test_translation_matrix.py` | 9 quick tests (under a second) |

**Run it** (from `05-Words-and-Sequences/020-Mikolov-et-al-2013-Similarities-Among-Languages`):
```
python3 -m pytest -q             # < 1 second
python3 demo.py                  # ~1 second
python3 experiments.py --quick   # downloads ~190 MB, 3M words per language: ~15-30 minutes
```

---

## 2. `translation_matrix.py`

### `CBOW` and `cbow_windows`
- `cbow_windows(ids, c)` gives, for each position, its 2c neighbours (−1 beyond the ends) and the middle word.
- `CBOW.step(context, center, negatives, lr)`:
  ```python
  h = sum of v_in[context words]                    # padding (-1) is masked out
  s = sigmoid(v_out[[center, negatives]] @ h)       # real word vs noise words
  g = labels - s
  grad_h = g @ v_out[targets]
  v_out[targets] += lr * g * h
  v_in[each context word] += lr * grad_h            # h is a SUM, so each gets the full gradient
  ```

### The translation matrix
- `fit_translation_matrix(X, Z, lr, epochs)`: SGD on Eq. 3. The gradient of one term is 2·(W·x_i − z_i)·x_iᵀ. W starts at 0.
- `fit_least_squares(X, Z)`: the exact answer via `np.linalg.lstsq` (a reference for the tests and demo).
- `translate(W, x, Z, k)`: the k target words with the highest cosine to W·x.
- `precision_at_k(W, X_test, gold, Z, k)`: maps all test words at once and checks whether the gold id is in the top k.
- `confidence` and `coverage_and_precision`: Table 3.

### Baselines
- `edit_distance(a, b)`: Levenshtein, computed with dynamic programming one row at a time. `edit_similarity` = 1 − distance / longer length.
- `cooccurrence_vectors(ids, dict_word_ids, window, size_ratio)`: for every word, counts of each dictionary word within the window (both directions, vectorized per offset with `np.add.at`), divided by the corpus-size ratio, then log1p and L2-normalized.
- `combined_scores(tm, ed, weight)`: the weighted sum used for "ED + TM".

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Table 2 in both directions (EN→ES, ES→EN): the four methods' P@1 / P@5, with the top-20K frequent target words as candidates |
| `e2` | the source/target vector-size grid (100–400-d English, 100–200-d Spanish) |
| `e3` | Table 3: thresholds 0.0 / 0.5 / 0.6 / 0.7 |
| `e4` | Figure 1: PCA of numbers and animals in both languages |

- **Preprocessing follows Section 5.1:** tokens with capitals are dropped, digits become `<num>`, punctuation is removed, and the minimum count is 5.
- **CBOW** uses window 5, 5 negatives, subsampling 1e-4, and a learning rate decaying from 0.05.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_sgd_solves_eq_3` | SGD ≈ exact least squares |
| `test_learned_matrix_translates_unseen_words` | P@1 > 95% on unseen words of a linear "language" |
| `test_source_and_target_can_have_different_dimensions` | W is d₂ × d₁ |
| `test_confidence_filter_trades_coverage_for_precision` | Table 3's trade-off |
| `test_edit_distance` | Levenshtein; related spellings score higher |
| `test_combined_scores` | ED + TM |
| `test_cooccurrence_vectors_are_unit_length_counts` | the count baseline, exactly |
| `test_cbow_windows` | the context windows |
| `test_cbow_gradient_is_correct` | the hand-written CBOW update = −lr × numerical gradient |

---

## 5. Try it yourself

1. Constrain W to be **orthogonal** (Procrustes: W = UVᵀ from the SVD of ZᵀX, with equal dimensions). Compare P@1 with plain least squares in `e1`.
2. Train W in the **reverse** direction and translate back. How often do you return to the starting word?
3. Use the confidence score to **find suspicious entries** in the MUSE training dictionary.
4. Replace CBOW with Paper 019's Skip-gram and compare rare-word translation (ranks 5K–20K).
