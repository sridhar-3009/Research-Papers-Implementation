# The code, explained simply

How the code in this folder implements Generative Adversarial Nets (Goodfellow et al. 2014).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `gan.py` | maxout, the generator and discriminator MLPs, both generator losses, Algorithm 1, the theory helpers (D*, V, KL, JSD, C(G)), the Parzen-window estimator with σ selection, nearest neighbours, interpolation |
| `experiments.py` | Table 1 on MNIST, saturating vs non-saturating, k = 1 vs 5, CIFAR-10 samples, Figures 2–3 (heavy, not run here) |
| `demo.py` | the theory on a grid, 1-D Figure 1, saturation, the Helvetica scenario, Parzen pitfalls (~5 seconds) |
| `test_gan.py` | 7 quick tests (~2.4 seconds) |

**Run it** (from `07-Generative-Models/042-Goodfellow-et-al-2014-GAN`):
```
python3 -m pytest -q             # ~2.4 seconds
python3 demo.py                  # ~5 seconds
python3 experiments.py --quick
```

---

## 2. `gan.py`

### Networks
- **`Maxout(n_in, n_out, pieces)`:** one Linear to n_out × pieces outputs, then the max over the pieces.
- **`Generator(z_dim, hidden, x_dim, out)`:**
  - ReLU MLP with a sigmoid output (or a linear one for toy data);
  - **`sample_z`** draws uniform noise on [−1, 1].
- **`Discriminator(x_dim, hidden, pieces, dropout, input_dropout)`:** maxout MLP with dropout. It returns a **logit**, so D(x) = sigmoid(logit).

### Losses and Algorithm 1
- **`d_loss`:**
  - −[log D(real) + log(1 − D(fake))];
  - computed as `logsigmoid(l)` and `logsigmoid(−l)` for stability. That uses log(1 − σ(l)) = log σ(−l).
- **`g_loss(saturating)`:**
  - `True`: log(1 − D(G(z))), the minimax loss;
  - `False`: −log D(G(z)), non-saturating.
- **`train_step`:** k discriminator steps on fresh minibatches (fakes detached), then one generator step.

### Theory and evaluation helpers
| Function | What it computes |
|---|---|
| `optimal_discriminator`, `value`, `kl`, `jsd`, `virtual_criterion` | Eqs. 2–6 on discrete distributions or densities on a grid (with spacing `dx`) |
| `parzen_log_likelihood(samples, x, σ)` | the logsumexp of the Gaussian kernels, in chunks |
| `select_sigma` | σ by validation log-likelihood |
| `nearest_neighbours`, `interpolate` | Figures 2 and 3 |

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Table 1: the paper's MNIST nets, SGD with momentum ramping 0.5 → 0.7 and a decaying learning rate (as in the released code), Parzen score from 10k samples with σ on 10k validation images |
| `e2` | Parzen curves for minimax vs non-saturating generator losses |
| `e3` | k = 1 vs k = 5 at an equal number of discriminator updates |
| `e4` | CIFAR-10 sample grids from an FC model (8000-unit G, 1600-maxout D) and a conv/deconv model |
| `e5` | MNIST samples with their nearest training image (Figure 2), and z-interpolations (Figure 3) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_proposition_1_optimal_discriminator` | the argmax is a/(a+b); no random D beats D* |
| `test_theorem_1_criterion_is_minus_log4_plus_2jsd` | Eq. 6 on random distributions; −log 4 at equality; JSD = log 2 for disjoint supports |
| `test_saturating_vs_non_saturating_gradient` | −0.01 vs −0.99 at D = 0.01 |
| `test_losses_match_definitions` | both losses equal their formulas computed with probabilities |
| `test_maxout_and_shapes` | maxout; noise range; sigmoid output range |
| `test_parzen_window_estimator` | a hand-computed value; σ selection picks a sensible width; close to the exact log-likelihood |
| `test_algorithm_1_learns_a_1d_gaussian` | G reaches mean ≈ 3, std ≈ 0.5; D near ½ on the data |

---

## 5. Try it yourself

1. In demo section 2, try plain SGD with momentum (the paper's optimiser). Does it still converge? Does it oscillate more?
2. In demo section 4, use k = 5 discriminator steps per generator step. Are more modes covered?
3. Train the 1-D GAN with `saturating=True` from a start at −6 for 2000 steps. How far does G get?
4. Replace the uniform noise with Gaussian noise. Does it matter for the 1-D task?
