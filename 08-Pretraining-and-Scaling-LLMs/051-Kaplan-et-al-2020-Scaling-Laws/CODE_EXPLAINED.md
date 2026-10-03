# The code, explained simply

How the code in this folder implements the scaling laws (Kaplan et al. 2020).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `scaling.py` | the paper's fitted constants and every law (L(N), L(D), L(C_min), L(N, D), L(N, S), B_crit, S_min, C_min), the overfitting penalty and data rule, the early-stopping bound, parameter / FLOP counts, the allocation exponents, Table 5, the numerical compute frontier, power-law and Eq. 1.5 fitting, the "why power laws" spectrum toy |
| `experiments.py` | measures the laws on real small transformers trained on text: L(N), shape, L(N, D), the frontier, the critical batch (heavy, not run here) |
| `demo.py` | five tables built from the laws (instant) |
| `test_scaling.py` | 9 quick tests (~1.5 seconds) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/051-Kaplan-et-al-2020-Scaling-Laws`):
```
python3 -m pytest -q             # ~1.5 seconds
python3 demo.py                  # instant
python3 experiments.py --quick
```

---

## 2. `scaling.py`

### The laws
- **Constants:** `A_N, N_C`, `A_D, D_C`, `A_CMIN, C_CMIN`, `A_B, B_STAR`, plus the `ND` (Table 2) and `NS` (Table 3) fits.
- **Functions:** `L_of_N`, `L_of_D`, `L_of_Cmin`, `L_of_ND`, `L_of_NS`, `B_crit`, `S_min`, `C_min` implement Eqs. 1.1–1.6 and 5.4–5.5 directly.
- **Overfitting and early stopping:** `overfit_penalty` (Eq. 4.3), `tokens_to_avoid_overfitting` (Eq. 4.4), `S_stop_lower_bound` (Eq. 5.7).

### Sizes and compute
| Function | What it computes |
|---|---|
| `non_embedding_params` | 12 n d² |
| `forward_flops_per_token` | 2N + 2 n n_ctx d_attn |
| `training_compute` | 6ND |

### The compute frontier
- **`allocation_exponents`** (Eq. 1.8) and **`table5`** (the empirical recipe).
- **`loss_at_compute(N, C)`:** the loss reached by model N with C_min PF-days.
  - It needs S_min = E/B_crit(L), which depends on L itself.
  - The function L − L(N, S_min(L)) increases with L, so bisection finds the unique solution.
- **`compute_optimal_N(C)`** searches a log grid of N for the lowest loss.

### Fitting and the toy
- **`fit_power_law(x, y)`:** a log–log linear fit giving (α, x_c).
- **`fit_L_ND(N, D, L)`:** fits Eq. 1.5's four parameters with Adam on squared log error (in log-parameter space).
- **`spectrum_toy`, `spectrum_loss_vs_N`, `spectrum_loss_vs_D`:** power-law importance λ_k = k^(−α):
  - the capacity-limited error is the tail sum;
  - the data-limited error comes from ridge regression on samples.

---

## 3. `experiments.py`

**Setup:**
- **data:** OpenWebText (if `datasets` is installed) or Gutenberg, with byte-level BPE from paper 049;
- **models:** paper 049's `GPT2`;
- **`run(...)`:** trains one model with warm-up + cosine and AdamW, records a validation curve, and can stop at a target loss.

| Function | Reproduces |
|---|---|
| `e1` | 8 sizes; fits α_N with non-embedding N vs with all parameters (residual spread) |
| `e2` | fixed N, depth from 2 to 24, 1 or 4 heads |
| `e3` | sizes × data fractions; the Eq. 1.5 fit and the implied D ∝ N^(α_N/α_D) |
| `e4` | the lowest loss reachable for each compute budget across sizes; fits N_opt ∝ C^p |
| `e5` | steps-to-target at several batch sizes; grid fit of S_min and E_min in Eq. 5.1, giving B_crit |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_doubling_parameters_multiplies_loss_by_0_95` | 2^(−α_N) = 0.949 |
| `test_L_ND_reduces_to_the_one_variable_laws` | the limits; less data means more loss; the penalty is 0 at infinite data |
| `test_overfitting_rule_keeps_penalty_near_noise_level` | Eq. 4.4 gives 0.5–5% penalties |
| `test_critical_batch_and_step_compute_tradeoff` | the B_crit formula; doubling per 13%; S = 2·S_min and C = 2·C_min at B_crit |
| `test_sizes_and_compute` | 12 n d²; Eq. 2.2; 6ND |
| `test_allocation_exponents_eq_1_8` | α_C ≈ 0.050; the N / B / S exponents; Table 5 at C = 1 |
| `test_numerical_frontier_matches_the_paper` | slope 0.6–0.75; loss = (1 + α_N/α_S) × converged; agreement with Eq. 1.3 |
| `test_power_law_fits` | exact recovery of a synthetic power law and of Eq. 1.5's exponents |
| `test_spectrum_toy_gives_a_power_law` | the exponent near α − 1 |

---

## 5. Try it yourself

1. Change α_S in `NS` from 0.76 to 0.5 and rerun the frontier. How do N_opt's exponent and the 10% gap change?
2. Use `fit_power_law` on the first half and the second half of `L_of_N` values: a true power law gives the same α everywhere.
3. In the spectrum toy, set α = 2.0. What exponents do you get for N and D?
4. Use `overfit_penalty` to find the largest model that 10¹⁰ tokens can support with under 1% penalty.
