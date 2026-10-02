# The code, explained simply

How the code in this folder implements Wasserstein GAN (Arjovsky et al. 2017).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `wgan.py` | TV / KL / JS / Wasserstein (from 1-D samples, from 1-D histograms, and the primal and dual linear programs), Example 1's closed forms, the critic, weight clipping, Lipschitz bounds, the critic and generator objectives, Algorithm 1 |
| `experiments.py` | Figures 3–7 and two ablations on CIFAR-10 instead of LSUN (heavy, not run here) |
| `demo.py` | Example 1, primal = dual, estimates and gradients at fixed θ, WGAN learning Example 1, the loss as a progress meter, clipping (~4 seconds) |
| `test_wgan.py` | 7 quick tests (~1.7 seconds) |

**Run it** (from `07-Generative-Models/044-Arjovsky-et-al-2017-Wasserstein-GAN`):
```
python3 -m pytest -q             # ~1.7 seconds
python3 demo.py                  # ~4 seconds
python3 experiments.py --quick
```

---

## 2. `wgan.py`

### Distances
- **`tv`, `kl`, `js`:** for discrete distributions.
  - `kl` returns ∞ when q = 0 somewhere that p > 0.
  - `js` uses the ½ weights, so its maximum is log 2.
- **Wasserstein, three ways:**

  | Function | Method |
  |---|---|
  | `wasserstein_1d_samples(a, b)` | sort both samples and average the absolute differences (exact for equal sizes) |
  | `wasserstein_1d_hist(p, q, x)` | Σ \|F_p − F_q\| · Δx |
  | `wasserstein_lp(p, q, x)` | Eq. 1 as a linear program over the n × n plan γ (row sums p, column sums q, cost \|x_i − x_j\|), solved with scipy's HiGHS |
  | `wasserstein_dual_lp(p, q, x)` | Eq. 2 as a linear program: maximise Σ f(p − q) subject to \|f_i − f_j\| ≤ \|x_i − x_j\| |

- **`example1_distances(θ)`:** the closed forms of Example 1.

### The critic
- **`Critic`:** a ReLU MLP that returns a real number. There is no sigmoid.
- **`clip_weights(module, c)`:** Algorithm 1, line 7.
- **Lipschitz checks:**
  - `lipschitz_bound` is the product of the spectral norms (an upper bound);
  - `max_grad_norm` is the largest gradient seen on the given points (a lower estimate).

### Objectives and Algorithm 1
- **`critic_objective(f, real, fake)`** = E_r[f] − E_g[f]. The critic maximises it.
- **`generator_loss(f, fake)`** = −E[f(G(z))] (Theorem 3).
- **`wgan_step(G, f, opt_g, opt_f, sample_real, sample_z, m, n_critic, clip)`:** n_critic ascent steps on the critic, each followed by clipping, then one generator step. It returns the last estimate.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figure 3: WGAN curves for MLP G + DCGAN critic, DCGAN G + DCGAN critic, and MLPs with a high learning rate; sample grids saved every `save_every` iterations |
| `e2` | Figure 4: GAN JS lower-bound curves for the same generators |
| `e3` | Figures 5–7: DCGAN / DCGAN without batch norm and with constant filters / 4×512 MLP generators, WGAN vs GAN; a mode-collapse proxy (mean pairwise sample distance) |
| `e4` | Adam (β₁ = 0.5) vs RMSProp on the critic; the cosine between Adam's step and the ascent gradient |
| `e5` | c ∈ {0.001, 0.01, 0.1} |

**Settings:**
- RMSProp, learning rate 5·10⁻⁵, c = 0.01, m = 64, n_critic = 5.
- Following the authors' released code, the critic gets 100 steps for the first 25 generator iterations and every 500th one.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_example_1_parallel_lines` | the closed forms; the straight-across coupling costs exactly \|θ\| |
| `test_divergences_on_disjoint_supports` | JS = log 2, TV = 1, KL = ∞, while W = 2 |
| `test_primal_equals_dual_kantorovich_rubinstein` | LP primal = LP dual = the CDF formula; the plan's marginals; the optimal f is 1-Lipschitz |
| `test_samples_w1_matches_shifted_gaussians` | W1 between N(0, 1) and N(1.5, 1) is 1.5 |
| `test_theorem_3_gradient_with_a_linear_critic` | ∇θ W = 1 through the generator loss; the estimate ≈ θ |
| `test_weight_clipping_bounds_the_lipschitz_constant` | weights within ±c; the bound shrinks; gradients ≤ the bound |
| `test_wgan_learns_example_1` | θ goes from 1 to below 0.1 in 200 iterations |

---

## 5. Try it yourself

1. In demo section 4, swap `wgan_step` for a standard GAN step with an optimal discriminator and plain SGD on θ. Does θ move?
2. Replace weight clipping with a gradient penalty: add λ(‖∇f(x̂)‖ − 1)² at random interpolates x̂ (WGAN-GP). Is the section 5 estimate less noisy?
3. Use Adam with β₁ = 0.9 for the critic in section 5. Do you see the instability the paper describes?
4. Compute W between two 2-D point clouds with `wasserstein_lp` (stack the points as `x`). How does it compare with the distance between their means?
