# The code, explained simply

How the code in this folder implements Inverse Autoregressive Flow (Kingma et al. 2016).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `iaf.py` | MADE, the IAF step (gated / affine / location-only, with its sequential inverse), the IAF posterior (Algorithm 1), linear IAF, planar flow, free bits, the discretized logistic likelihood, and a small MLP VAE with any of these posteriors |
| `experiments.py` | Table 1 (MNIST), Table 3, CIFAR-10 posteriors with free bits, Figure 7's λ study, Figure 1 plots, synthesis speed (heavy, not run here) |
| `demo.py` | masks, the correlated-Gaussian gap, the banana posterior, Figure 1's toy, forward vs inverse cost (~5 seconds) |
| `test_iaf.py` | 9 quick tests (~1.3 seconds) |

**Run it** (from `07-Generative-Models/040-Kingma-et-al-2016-Inverse-Autoregressive-Flow`):
```
python3 -m pytest -q             # ~1.3 seconds
python3 demo.py                  # ~5 seconds
python3 experiments.py --quick
```

---

## 2. `iaf.py`

### MADE
- **`MaskedLinear`** multiplies its weight by a fixed 0/1 mask.
- **`MADE(D, hidden, n_hidden_layers, context, n_out)`:**
  - **degrees:** inputs 1…D; hidden units cycle through 1…D−1;
  - **hidden mask:** `deg_h ≥ deg_in`; **output mask:** `deg_out > deg_h` (strict);
  - **context h** is added unmasked to the first hidden layer (or to the outputs when there are no hidden layers);
  - **returns** `n_out` tensors of shape (…, D): here m and s.

### Flow steps
- **`IAFStep(D, hidden, n_hidden_layers, context, mode, forget_bias=1.5)`:**

  | mode | update |
  |---|---|
  | `gated` | σ = sigmoid(s + 1.5), z' = σz + (1−σ)m (Eqs. 13–14) |
  | `affine` | z' = m + e^s z (Eq. 10) |
  | `location` | z' = z + m (Table 3) |

  - It returns (z', Σ log σ).
  - The output layer starts at zero, so the step begins as z' = 0.82 z.
  - **`inverse(y)`** fills z one coordinate at a time. It is exact but uses D passes.
- **`IAFPosterior(D, T, ...)`** is Algorithm 1: z₀ = μ + σε with log q₀ = log N(z₀), then T steps, flipping the order between steps and subtracting each log det.

### Other posteriors
- **`LinearIAF(D, context)`:** the encoder's context produces a unit lower-triangular L; z = L(μ + σε); log q unchanged (Appendix A).
- **`PlanarFlow(D, K)`:** K steps of z + û tanh(w·z + b), with û constrained so the step stays invertible, and log|1 + ûᵀψ| as the log det.

### Objectives
| Function | What it does |
|---|---|
| `free_bits_kl(kl_per_group, λ)` | Σ_j max(λ, batch-mean KL_j) (Eq. 15) |
| `discretized_logistic_log_prob(x, μ, log s)` | the mass of each pixel's 1/256 bin under a logistic, with the edge bins taking the tails |

### `FlowVAE`
- An MLP encoder outputs (μ, log σ², h). The posterior is `diag` / `linear` / `iaf` / `planar`, and the decoder is Bernoulli.
- **`log_weights`** gives log p(x, z) − log q(z|x) for n samples.
- **`elbo`** is their mean.
- **`log_likelihood`** is the importance-sampled log p(x) (Table 1 uses 128 samples).

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Table 1: `ConvVAE` (strided conv/ELU encoder, transposed-conv decoder, 32 latents) on dynamically binarized MNIST; posteriors diagonal / IAF (2×320, 2×1920, 4×1920, 8×1920); VLB and 128-sample log p(x) over seeds |
| `e2` | Table 3: location-only vs location+scale |
| `e3` | CIFAR-10 with a 1024-dim latent, discretized logistic decoder, free bits; posteriors diagonal / linear IAF / IAF (1 or 2 steps) in bits/dim |
| `e4` | free bits λ ∈ {0, 0.125, 0.5, 2}: bits/dim, total KL, number of active latent dimensions, KL history |
| `e5` | Figure 1 scatter plots (prior, diagonal posteriors, IAF posteriors) |
| `e6` | seconds per image: one decoder pass vs pixel-by-pixel sampling from a 784-dim MADE |

- **Training:** Adamax (as in the paper's code), learning-rate decay, gradient clipping.
- **Free bits** are applied per latent dimension for the diagonal posterior. Flows give only a total log q, so there it acts on the total.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_made_is_strictly_autoregressive` | the Jacobian's upper triangle (incl. the diagonal) is 0 and the strictly-lower part is used, for 0/1/2 hidden layers |
| `test_iaf_step_jacobian_is_triangular_with_sigma_on_the_diagonal` | autograd log det = Σ log σ for all three modes; location-only gives 0 |
| `test_iaf_step_inverse_is_sequential_and_exact` | inverse(step(z)) = z |
| `test_algorithm_1_density_integrates_to_one` | ∫ q = 1 by importance sampling (2 steps with a flip) |
| `test_algorithm_1_log_q_matches_inversion` | Algorithm 1's log q = base log-density at the inverted point − log det |
| `test_linear_iaf_gives_full_covariance` | sample covariance = L diag(σ²) Lᵀ; log q = the exact full-covariance Gaussian density |
| `test_iaf_closes_the_gap_a_diagonal_gaussian_cannot` | diagonal q stops at −½ log(1−ρ²) = 0.830; linear IAF reaches < 0.02 |
| `test_free_bits_and_discretized_logistic` | the free-bits value and gradients; logistic bins sum to 1 |
| `test_flow_vae_bound_and_planar_flow` | importance-sampled log p(x) ≥ bound for all four posteriors; planar log det matches autograd |

---

## 5. Try it yourself

1. In `demo.py` section 3, give the planar flow K = 64 steps (about the same parameter count as IAF T = 1). Does it catch up?
2. Turn off the order reversal (`IAFPosterior(..., reverse=False)`) for the banana. Does T = 2 still beat T = 1?
3. Set `forget_bias=-2` in `IAFStep` and train the Figure 1 toy. Is training less stable?
4. Fit a **bimodal** target (a mixture of two Gaussians) with IAF T = 4. Can a flow from one Gaussian split into two modes?
