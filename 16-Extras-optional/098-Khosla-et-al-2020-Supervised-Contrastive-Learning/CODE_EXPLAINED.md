# The code, explained simply

How the code in this folder implements supervised contrastive learning (Khosla et al. 2020) in numpy.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `supcon.py` | the three contrastive losses with vectorised gradients, normalisation backprop, the hard-positive gradient study, digits data with augmentations and corruptions, an encoder + projection MLP with manual backprop and Adam, contrastive training, linear probes, the cross-entropy baseline |
| `experiments.py` | seeds, temperature, batch size, robustness, learning-rate sensitivity (E1–E5) |
| `demo.py` | gradient checks, hard mining, the four training methods, the corruption table (~13 seconds) |
| `test_supcon.py` | 4 quick tests (~1 second) |

**Run it** (from `16-Extras-optional/098-Khosla-et-al-2020-Supervised-Contrastive-Learning`; needs scikit-learn and scipy):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `supcon.py`

### The losses
`contrastive_loss(Z, labels, tau, kind)` works on 2N normalised rows Z:
- **similarities:** S = Z Zᵀ / τ, with the diagonal masked out; log-sum-exp per row gives the denominators, and `soft` = P_ia;
- **positives:** `pos[i, a]` = same label, a ≠ i;
- **`kind="out"` (Eq. 2):** loss −mean_p (S_ip − lse_i); gradient dL/dS_ia = P_ia − 1[a ∈ P(i)]/|P(i)|;
- **`kind="in"` (Eq. 3):** loss −(lse over positives − log|P| − lse_i); gradient P_ia − X_ia, where X is the softmax over the positives only;
- **`kind="self"`:** Eq. 1, by passing image ids as labels (each anchor then has one positive);
- **averaging:** the loss is averaged over anchors that have positives;
- **back to Z:** because S is symmetric in Z, dL/dZ = (G + Gᵀ) Z / τ.

### Helpers
| Name | What it does |
|---|---|
| `normalize`, `normalize_backward` | z = u/‖u‖ and du = (dz − z(z·dz))/‖u‖ |
| `hard_positive_gradient(cosines)` | one anchor, one positive at a set cosine, 20 random negatives; the norm of the gradient on the positive's un-normalised vector |
| `load_digits_split` | scikit-learn digits scaled to [0, 1]; 1,300 / 497 split |
| `augment` | random sub-pixel shift (±1), intensity × U(0.8, 1.2), noise σ = 0.1 |
| `corrupt(X, kind, level)` | test-time noise, Gaussian blur, or a diagonal shift |

### Networks and training
| Name | What it does |
|---|---|
| `init_params` | encoder W1 (64→128), W2 (128→64); projection P1 (64→64), P2 (64→32); classifier head Wc (64→10) |
| `encode` | ReLU MLP, then normalise (r); `project`: ReLU MLP, then normalise (z) |
| `train_contrastive(X, y, kind, tau)` | two augmented views per image, manual backprop through both normalisations and both MLPs, Adam |
| `linear_probe`, `probe_accuracy` | softmax regression on the frozen r (logits scaled ×10, since r is unit-length) |
| `train_cross_entropy`, `ce_accuracy` | the same encoder and normalised r, plus a linear head with fixed scale 10, trained end to end on one augmented view |

`REPORTED` holds Tables 1–3, the ImageNet-C mCE numbers and the other findings, checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `run` | trains one method and returns the model and test accuracy |
| `e1` | CE / L_out / L_in / self × 5 seeds × {100, 200} epochs |
| `e2` | τ ∈ {0.05 … 1.0} for L_out and L_in |
| `e3` | batch ∈ {32 … 256} for CE and L_out |
| `e4` | corruption table averaged over 5 seeds |
| `e5` | accuracy spread over 4 learning rates for CE and L_out |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_gradients_match_finite_differences_and_jensen` | gradients of L_out and L_in are correct; L_in ≤ L_out |
| `test_single_positive_reduces_to_simclr_form` | with one positive per anchor, Eq. 2 = Eq. 3 |
| `test_normalize_backward_and_hard_positive` | the normalisation Jacobian is correct; a hard positive's gradient is > 20× an easy one's |
| `test_supcon_training_learns` | the loss decreases, and a linear probe after short training beats 80% |

---

## 5. Try it yourself

1. Add a memory bank (a queue of past embeddings with labels) to get many more negatives and positives per anchor with small batches.
2. Use stronger augmentations (random erasing, rotation) and see whether SupCon's shift robustness improves.
3. Train with label noise (flip 20% of labels): does SupCon degrade more or less than cross-entropy?
4. Fine-tune the whole encoder after the probe instead of freezing it.
5. Count positives per anchor (batch size / classes) and plot L_out vs L_in accuracy as it grows.
