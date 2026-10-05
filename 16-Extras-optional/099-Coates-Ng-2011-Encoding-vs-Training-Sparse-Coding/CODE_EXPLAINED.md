# The code, explained simply

How the code in this folder reproduces Coates & Ng's "encoding vs training" study (ICML 2011) in numpy.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `encoding.py` | patch extraction, per-patch normalisation, ZCA whitening, FISTA for the lasso, batched OMP, the soft threshold, five dictionary-training methods, the pooling pipeline, logistic regression with few labels |
| `experiments.py` | the table with hyperparameter grids, dictionary size, labels per class, timing, optional CIFAR-10 (E1–E5) |
| `demo.py` | the 5 × 5 dictionary-by-encoder table with its conclusions (~11 seconds) |
| `test_encoding.py` | 4 quick tests (~0.7 seconds) |

**Run it** (from `16-Extras-optional/099-Coates-Ng-2011-Encoding-vs-Training-Sparse-Coding`; needs scikit-learn):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `encoding.py`

### Preprocessing
| Name | What it does |
|---|---|
| `load_digits_images` | 1,797 images of 8 × 8 pixels in [0, 1] |
| `extract_patches(images, p)` | every p × p patch with stride 1 → (images, positions, p²), plus the grid size |
| `normalize_patches` | subtract each patch's mean, divide by √(variance + 0.1) |
| `fit_zca` | covariance eigen-decomposition; whitening matrix U diag(1/√(λ + ε)) Uᵀ |

### Encoders
| Name | What it does |
|---|---|
| `lasso_fista(X, D, lam)` | batched FISTA: a gradient step on ‖Ds − x‖² with step 1/L (L = 2‖D‖²), shrinkage by λ/L, Nesterov momentum |
| `omp(X, D, k)` | batched OMP: at each step add the atom most correlated with the residual (never repeating one), refit the selected coefficients by solving the small normal equations for all patches at once |
| `encode(X, D, method, param)` | 'SC', 'OMP' or 'T', always polarity-split |

### Training
`train_dictionary(X, d, method)`:
- **R:** random Gaussian columns;
- **RP:** random patches;
- **OMP / SC:** start from random patches, then alternate codes (OMP-k or FISTA) with the least-squares update D = Xᵀ S (SᵀS)⁻¹, re-seeding unused atoms and renormalising columns.

### Pipeline and evaluation
| Name | What it does |
|---|---|
| `Pipeline(images)` | patches → normalise → whiten (fitted on a 20,000-patch sample, also kept for dictionary training) |
| `Pipeline.features(D, method, param)` | encode every patch, then average over 4 overlapping quadrants of the 5 × 5 patch grid (rows/columns 0–2 and 2–4) |
| `logistic_regression`, `few_label_accuracy` | standardise, multinomial logistic regression; per_class labels per class, test on the rest, mean over random draws |
| `TRAINERS`, `NATURAL`, `ENCODERS` | the configurations in the table (natural encoders: T with α = 0 for R and RP, OMP-k for OMP-k, SC with λ = 1 for SC) |

`REPORTED` holds Tables 1–3 and the Caltech numbers, checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | best accuracy per (dictionary, encoder) over λ, k and α grids, 5 label draws |
| `e2` | dictionary size 16 to 256 with OMP-1 + T |
| `e3` | labels per class 5 to 100: raw pixels vs random patches + T |
| `e4` | wall-clock time of each training and encoding method on 10,000 patches |
| `e5` | CIFAR-10 (greyscale, 6 × 6 patches) if `cifar-10-batches-py/` is present |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_lasso_fista_matches_kkt_conditions` | FISTA's solution satisfies the lasso optimality conditions (gradient = −λ·sign on the support, \|gradient\| ≤ λ off it) |
| `test_omp_recovers_sparse_signal_and_omp1_is_gain_shape_vq` | OMP-2 exactly recovers a 2-sparse signal; OMP-1 keeps the single most correlated atom with coefficient Dⱼᵀx |
| `test_soft_threshold_and_polarity_split` | the worked example from EXPLAINED.md |
| `test_whitening_and_dictionaries_are_unit_norm` | whitened patches have about unit variance; every training method returns unit-norm atoms |

---

## 5. Try it yourself

1. Replace the soft threshold's fixed α with a learned per-atom bias (that's a ReLU layer) and train it with the classifier.
2. Add K-means with hard assignment (1-of-K) as an encoder and compare with OMP-1.
3. Grow the dictionary to 512 atoms with OMP-1 + T (E2). Does it keep improving, as in the paper?
4. Remove the whitening step. Which encoders suffer most?
5. Use max pooling instead of average pooling over the quadrants.
