# The code, explained simply

How the code in this folder implements Chinchilla's analysis (Hoffmann et al. 2022).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `chinchilla.py` | the fitted parametric loss, the closed-form and numeric frontiers, Kaplan's recipe for comparison, Table 3, Appendix F FLOP and parameter counts, the Huber + L-BFGS fit (Approach 3), the envelope (Approach 1) and IsoFLOP (Approach 2) analyses, and synthetic runs generated from a known law |
| `experiments.py` | the three approaches on real small transformers, cosine cycle length, and Chinchilla-style vs Kaplan-style allocation (heavy, not run here) |
| `demo.py` | loss decomposition, the frontier vs Kaplan, Table 3, the three approaches, FLOPs, exponent sensitivity (~0.5 seconds) |
| `test_chinchilla.py` | 7 quick tests (~1 second) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/052-Hoffmann-et-al-2022-Chinchilla`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~0.5 seconds
python3 experiments.py --quick
```

---

## 2. `chinchilla.py`

### The law and the frontier
| Name | What it is |
|---|---|
| `E, A, B, ALPHA, BETA` | Eq. 10's values |
| `loss(N, D, p)` | E + A/N^α + B/D^β |
| `frontier(C, p)` | Eq. 4: G = (αA/βB)^(1/(α+β)), N = G(C/6)^a, D = (C/6)^b/G |
| `frontier_numeric` | brute-force minimisation over a log grid of N |
| `kaplan_N_opt` | Kaplan's 1.3·10⁹·C^0.73 (C in PF-days) |

### Counting
- **`TABLE_3`** holds the paper's Approach 1 projections.
- **`flops_appendix_f`:** the forward FLOPs, then × 3 for forward + backward.
  - Embeddings: 2·seq·vocab·d.
  - Attention: QKV projections, the logits, the softmax (3·heads·seq²), the reductions, the output projection.
  - Dense: 4·seq·d·ffw.
  - Logits: 2·seq·d·vocab.
- **`params_count`** includes the embedding matrix, as the paper does.

### Fitting
- **`huber`, `fit_parametric`:** minimise the Huber loss (δ = 10⁻³) of the log-sum-exp prediction against log L.
  - **parameters:** (a, b, e, α, β);
  - **optimiser:** L-BFGS with strong-Wolfe line search, from a grid of starts;
  - **output:** returns (E, A, B, α, β) from the best start.
- **`approach1_envelope(curves)`:** at each FLOP level, interpolate every curve in log-FLOPs, pick the lowest, then fit log N_opt against log C.
- **`approach2_isoflop(isoflop)`:** a quadratic fit in log N per budget; the vertex is N_opt.
- **`synthetic_curves`, `synthetic_isoflop`:** runs generated from the law, optionally with multiplicative noise, used to check that each approach recovers the planted exponent.

---

## 3. `experiments.py`

**Training:** paper 049's `GPT2` and byte-level BPE; parameters counted including embeddings. `train` uses:
- AdamW (β₂ = 0.95, weight decay 0.1);
- a short warm-up;
- a cosine decaying 10× over `cycle_mult` × the run length.

| Function | Reproduces |
|---|---|
| `e1` | IsoFLOP profiles at 5 budgets (Approach 2) |
| `e2` | sizes × several training lengths; the envelope (Approach 1) |
| `e3` | the Approach 3 fit on all saved runs; tokens per parameter on the frontier |
| `e4` | cosine cycle 1×–5× the run (Figure A1) |
| `e5` | equal FLOPs: the ~20 tokens-per-parameter model vs the Kaplan-sized model |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_closed_form_frontier_matches_numeric_minimisation` | Eq. 4 = brute force; 6ND = C; the slope equals a |
| `test_frontier_balances_the_two_terms` | αA/N^α = βB/D^β at the optimum |
| `test_table_3_is_consistent_with_6ND_and_about_20_tokens_per_parameter` | every row |
| `test_chinchilla_vs_gopher_under_the_fitted_law` | lower loss at the same budget; Kaplan picks > 10× bigger |
| `test_rounding_sensitivity_of_the_published_exponents` | 32B vs 40B |
| `test_appendix_f_flops_close_to_6ND_for_large_models` | ratio ≈ 1 at large scale; larger for small models |
| `test_approaches_1_2_3_recover_the_planted_law` | all three recover a ≈ 0.452 and the law's parameters |

---

## 5. Try it yourself

1. Set α = β in `frontier`. What exponents and tokens per parameter do you get?
2. Add 2% noise to `synthetic_isoflop` and rerun Approaches 2 and 3. Which is more robust?
3. Use `loss` to compute what LLaMA-7B (7B parameters, 1T tokens) would score vs a compute-optimal model of the same FLOPs.
4. Change δ in `fit_parametric` to 0.1, add a few outlier runs to the data, and see how the fit moves.
