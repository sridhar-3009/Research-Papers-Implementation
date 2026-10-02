# The code, explained simply

How the code in this folder implements the VAE (Kingma & Welling 2014).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `vae.py` | Gaussian log-densities, the closed-form KL, the reparameterization, the Gaussian/Bernoulli MLPs, the VAE (both SGVB estimators, MAP objective, sampling, the 2-D manifold, an importance-sampled log-likelihood check), the score-function vs reparameterized gradient estimators, wake-sleep losses, HMC, and Appendix D's marginal likelihood estimator |
| `experiments.py` | Figures 2–5 (MNIST, Frey Face; AEVB vs wake-sleep vs Monte Carlo EM) and an estimator-variance study (heavy, not run here) |
| `demo.py` | Eq. 1 on an exact model, gradient variance, estimator A vs B, AEVB vs wake-sleep on an MNIST slice, Nz = 20 vs 2 (~2.5 seconds) |
| `test_vae.py` | 9 quick tests (~1 second) |

**Run it** (from `07-Generative-Models/039-Kingma-Welling-2014-VAE`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~2.5 seconds
python3 experiments.py --quick   # ~30-60 minutes
```

---

## 2. `vae.py`

### Densities
| Function | What it computes |
|---|---|
| `log_normal_diag(x, mu, logvar)` | log N(x; μ, diag e^logvar), summed over the last axis |
| `log_standard_normal` | the same with μ = 0 and unit variance |
| `kl_to_standard_normal(mu, logvar)` | Appendix B's −½ Σ(1 + log σ² − μ² − σ²) |
| `reparameterize(mu, logvar, eps)` | z = μ + e^{logvar/2} · ε (Eq. 4) |

### Networks
- **`GaussianMLP`** (App. C.2): tanh hidden layer, then the heads μ and log σ². `sigmoid_mean=True` gives the Frey Face decoder.
- **`BernoulliMLP`** (App. C.1): returns logits. The loss uses `binary_cross_entropy_with_logits` for numerical stability.

### `VAE(x_dim, z_dim, hidden, decoder, init_std=0.01)`
- **Init:** every parameter starts at N(0, 0.01) (Section 5).
- **`log_px_given_z`:** works with an extra leading sample dimension (L samples per image).
- **`elbo(x, L, estimator)`:** `"B"` is Eq. 7/10 (analytic KL); `"A"` is Eq. 6 (everything sampled).
- **`map_objective(x, N)`:** the mean bound − ½|θ|²/N. This is the paper's weight decay, seen as a N(0, I) prior on the weights.
- **`sample`, `decode_mean`, `manifold(n)`:** ancestral sampling, and Figure 4's grid through the inverse normal CDF.
- **`importance_log_likelihood(x, K)`:** a modern estimator (not from this paper) used as a cross-check.

### Gradient estimators
- **`score_function_grad`:** per-sample f(z) · ∇ log q, computed analytically for a 1-D Gaussian.
- **`reparam_grad`:** per-sample gradients via autograd, giving each sample its own copy of (μ, log σ).

### Baselines and evaluation
| Function | What it does |
|---|---|
| `wake_sleep_losses(model, x)` | wake loss (decoder only: z from q is detached) and sleep loss (encoder only: dreamed x from the prior and decoder) |
| `hmc(log_prob, z, n_steps, leapfrog, step)` | batched Hamiltonian Monte Carlo with Metropolis acceptance; returns samples and the acceptance rate |
| `log_joint(model, x)` | z ↦ log p(z) + log p(x\|z) |
| `marginal_likelihood_appendix_d(model, x, ...)` | HMC posterior samples; fits a full-covariance Gaussian q(z); 1/p(x) ≈ mean q(z)/p(x, z) on fresh samples |

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figure 2 (MNIST): AEVB vs wake-sleep, Nz ∈ {3, 5, 10, 20, 200}, curves saved as figures |
| `e2` | Figure 2 (Frey Face, downloaded from Roweis' page), Gaussian decoder |
| `e3` | Figure 3: Appendix D marginal likelihood for AEVB, wake-sleep and `train_mcem` (HMC Monte Carlo EM), N_train = 1000 / 50000 |
| `e4` | Figure 4: 2-D manifolds |
| `e5` | Figure 5: samples for Nz = 2, 5, 10, 20 |
| `e6` | variance of the encoder gradient for estimators A vs B, M ∈ {1, 10, 100}, L ∈ {1, 10} |

`pick_lr` chooses Adagrad's step size from {0.01, 0.02, 0.1} by the training bound after a few iterations, as the paper did.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_eq1_log_likelihood_is_bound_plus_kl_to_posterior` | Eq. 1 exactly, on a linear-Gaussian model |
| `test_closed_form_kl_appendix_b` | our KL = PyTorch's KL; 0 when q is the prior |
| `test_reparameterization_gives_the_right_distribution_and_gradients` | z has mean μ and variance σ²; the gradients of E[z²] are right |
| `test_reparameterized_gradient_has_far_lower_variance_than_score_function` | both unbiased; > 5× variance gap |
| `test_estimators_a_and_b_agree_and_bound_is_below_likelihood` | E[L_A] = E[L_B]; the importance-sampled log p(x) ≥ the bound |
| `test_sizes_and_shapes` | 815,824 parameters for 784-500-20; Frey samples in (0, 1); manifold grid shape |
| `test_wake_sleep_losses_touch_the_right_parameters` | wake → decoder only, sleep → encoder only |
| `test_hmc_and_appendix_d_estimator_on_an_exact_model` | HMC posterior moments; Appendix D log p(x) within 0.05 |
| `test_aevb_learns` | the bound rises from below −13 to above −3 on four binary prototypes (best possible: −ln 4 = −1.39) |

---

## 5. Try it yourself

1. In `demo.py` section 4, train 40 epochs instead of 8. Does wake-sleep catch up?
2. Set the KL weight to 0 (`elbo` without the KL) and decode `vae.sample(16)`. What do random samples look like now?
3. Multiply the KL by β = 4 (the later "β-VAE"). How many dimensions of the Nz = 20 model stay active?
4. Replace `score_function_grad`'s f(z) with f(z) − f(μ) (a "baseline"). How much does its variance drop?
