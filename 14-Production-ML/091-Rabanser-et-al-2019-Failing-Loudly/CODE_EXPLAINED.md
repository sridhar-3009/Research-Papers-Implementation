# The code, explained simply

How the code in this folder implements the shift-detection pipeline of *Failing Loudly* (Rabanser et al. 2019).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `shiftdetect.py` | the digits data, a numpy MLP, 8 dimensionality-reduction methods, 4 statistical tests, 10 shift types, detection rates, most-anomalous samples and the malignancy check |
| `experiments.py` | the full grid (E1), per-shift tables (E2), latent size (E3), the malignancy estimate (E4), optional MNIST (E5) |
| `demo.py` | false positives, the Table 1a and 1b analogues, and malignancy (~20 seconds) |
| `test_shiftdetect.py` | 4 quick tests (~1 second) |

**Run it** (from `14-Production-ML/091-Rabanser-et-al-2019-Failing-Loudly`; needs scikit-learn and scipy):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `shiftdetect.py`

### Data and models
| Name | What it does |
|---|---|
| `load_data(seed)` | scikit-learn's digits scaled to [0, 1]; 900 train / 450 source (validation) / 447 target pool |
| `MLP(d_in, hidden, d_out)` | one ReLU hidden layer, softmax output, Adam training (`fit`); `input_grad` gives ∂loss/∂input for FGSM |
| `Reducers(Xtr, ytr, K)` | fits everything on the training split: PCA (SVD), the SRP matrix of eq. (1) with v = √D, the UAE (random ReLU encoder), the TAE (tanh autoencoder trained with Adam on reconstruction error), and the label classifier. `transform(X, method)` returns the representation |

### Tests
| Function | What it does |
|---|---|
| `ks_bonferroni(A, B)` | `scipy.stats.ks_2samp` per dimension, returning min(p) · K (capped at 1), so "p < α" is the Bonferroni rule |
| `mmd2_unbiased(K, m)` | eq. (3) from a joint kernel matrix (diagonals excluded) |
| `mmd_test(A, B, perms)` | RBF kernel exp(−d² / σ) with σ = median squared distance; permutation p-value (count + 1)/(perms + 1) |
| `chi2_test(a, b)` | 2 × C contingency table of predicted classes (empty columns dropped), via `scipy.stats.chi2_contingency` |
| `domain_classifier_test(A, B)` | half/half split, MLP 64-32-2, `scipy.stats.binomtest(correct, N, 0.5)` |
| `METHODS`, `detect(...)` | the 14 (method, test) pairs, and one p-value for a source/target pair |

### Shifts and experiments
| Name | What it does |
|---|---|
| `GN`, `IMG` | our noise σ for s/m/l gn; (rotation°, translation fraction, zoom) for s/m/l img, as in the paper |
| `image_transform` | random rotation, shift and zoom with `scipy.ndimage`, then centre-cropped to 8×8 |
| `apply_shift(X, y, shift, delta, red)` | none / s,m,l_gn / s,m,l_img / ko (drop a fraction δ of class 0) / adv (FGSM, step 0.15, against the label classifier) / m_img+ko / oz+m_img |
| `detection_rate(...)` | share of `reps` random draws of n source and n target samples with p < α |
| `most_anomalous(Xs, Xt, yt, red, top)` | the domain classifier's top-scored target samples, and the label classifier's accuracy on them |

`REPORTED` holds Table 1a's numbers, the Table 1b groups, the main findings and the MNIST-split result, checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | 14 methods × 10 shifts × δ ∈ {0.1, 0.5, 1} × n ∈ {10 … 400} × 5 splits × 20 draws: the full Table 1a analogue (long) |
| `e2` | BBSDs-KS and UAE-MMD per shift and per δ |
| `e3` | K ∈ {4, 8, 16, 32} for PCA / SRP / UAE / TAE |
| `e4` | correlation between true target accuracy and top-k anomalous accuracy across shifts |
| `e5` | opt-in MNIST run via `fetch_openml` (downloads ~55 MB) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_tests_are_correct` | KS and MMD accept same-distribution samples and reject shifted ones; MMD² = 0 for identical points; χ² on equal counts gives p ≈ 1 |
| `test_srp_matrix_follows_eq1` | SRP entries are only 0 and ±√(v/K), with sparsity near 1 − 1/√D |
| `test_classifier_and_shifts` | classifier > 93% accurate; knock-out removes class 0; only-zero leaves only 0s; FGSM drops accuracy below 70% |
| `test_detects_large_shift_not_none` | BBSDs detects m_img in ≥ 80% of draws and fires on no-shift in ≤ 20%; the top anomalous samples are misclassified |

---

## 5. Try it yourself

1. Remove the always-zero border pixels before NoRed, and see whether raw pixels still beat BBSDs.
2. Replace Bonferroni with the less conservative Simes or Benjamini–Hochberg procedure. Do false positives stay below 5%?
3. Create a harmless covariate shift (e.g. thicker strokes via dilation). Does the malignancy check call it benign?
4. Run a sliding-window detector over a stream where the shift starts halfway. How many samples after the change until it fires?
5. Use the domain classifier's scores as importance weights to estimate target accuracy without labels.
