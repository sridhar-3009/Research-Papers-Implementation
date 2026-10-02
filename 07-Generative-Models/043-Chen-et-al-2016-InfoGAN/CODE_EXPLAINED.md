# The code, explained simply

How the code in this folder implements InfoGAN (Chen et al. 2016).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `infogan.py` | the latent-code specification, Q's output split, the L_I terms, the MNIST networks of Table 1, small MLP networks, the InfoGAN training step, cluster accuracy, exact discrete mutual information, Lemma 5.1 |
| `experiments.py` | Figure 1, Figure 2 (+ c₁ as a classifier), SVHN, CelebA, a λ sweep (heavy, not run here) |
| `demo.py` | the bound on an exact channel, Figure 1 in miniature, disentangling a 2-D toy, exaggerated codes (~7.5 seconds) |
| `test_infogan.py` | 8 quick tests (~2 seconds) |

**Run it** (from `07-Generative-Models/043-Chen-et-al-2016-InfoGAN`):
```
python3 -m pytest -q             # ~2 seconds
python3 demo.py                  # ~7.5 seconds
python3 experiments.py --quick
```

---

## 2. `infogan.py`

### Codes
- **`LatentSpec(z_dim, cats, n_cont, noise)`:**
  - **`sample(n)`** returns (z, categorical indices, continuous values);
  - **`pack`** concatenates z, the one-hot categorical codes and the continuous codes into the generator input;
  - **`entropy()`** is H(c): Σ log K + n_cont · log 2.
- **`q_out_dim`, `split_q`:** Q's head outputs the logits for each categorical code, then the means, then the log-stds of the continuous codes.
- **`mi_terms`:**
  - **categorical part:** −cross-entropy;
  - **continuous part:** Σ log N(c; μ, e^{2·log σ}).

  **`mi_lower_bound`** adds H(c) (Eq. 5).

### Networks
| Class | What it is |
|---|---|
| `MNISTGenerator` | Table 1's G (sigmoid output) |
| `MNISTDiscriminatorQ` | Table 1's shared body with a D logit head and a Q head |
| `MLPGenerator`, `MLPDiscriminatorQ` | small versions for 2-D toys; D and Q share the body |

### Training
**`infogan_step(G, DQ, opt_g, opt_d, real, spec, lam_cat, lam_cont, info)`:**
1. **D/Q update:** GAN discriminator loss − λ · (L_I terms) on fakes that are detached from G. Q learns to read codes.
2. **G update:**
   - `info=True`: non-saturating GAN loss − λ · L_I. G learns to write codes.
   - `info=False`: the paper's baseline, a plain GAN whose Q is still trained (so L_I can be measured) but which G ignores.

It returns the losses and the current L_I estimate.

### Evaluation and theory helpers
| Function | What it does |
|---|---|
| `cluster_accuracy` | best one-to-one matching between code values and labels (Hungarian algorithm, from scipy) |
| `mutual_information_discrete` | exact I from a joint table |
| `lemma_5_1_sides` | both sides of Lemma 5.1 for discrete variables |

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figure 1: L_I curves over 1000 iterations, InfoGAN vs GAN, c ~ Cat(10) |
| `e2` | Figure 2: c₁ / c₂ / c₃ manipulation grids (codes up to ±2) for InfoGAN and c₁ for a GAN; c₁ read by Q as an unsupervised MNIST classifier |
| `e3` | SVHN (Table 2 networks; 4×10 categorical + 4 continuous codes) |
| `e4` | CelebA 32×32 (Table 3; 10 × 10-way codes): one grid per code |
| `e5` | λ_cont ∈ {0, …, 1}: how much one unit of c₂ changes the image, and D's real/fake accuracy |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_latent_spec_matches_mnist_setup` | 74-dim input; one-hot codes; ranges; H(c); Q output size 14 |
| `test_lemma_5_1` | both sides are equal |
| `test_variational_bound_on_mutual_information` | L_I ≤ I with equality at the posterior; I = H(c) for a code that is fully revealed |
| `test_l_i_extremes_for_categorical_codes` | a perfect Q gives log K; a uniform Q gives 0 |
| `test_continuous_q_is_a_gaussian_log_likelihood` | matches `torch.distributions.Normal` |
| `test_mnist_networks_match_table_1` | output shapes |
| `test_cluster_accuracy_is_permutation_invariant` | relabelled clusters still score 1.0 |
| `test_infogan_uses_the_code_and_gan_does_not` | after 400 steps: L_I > 1.2 for InfoGAN, < 0.7 for a GAN (max ln 4 = 1.386) |

---

## 5. Try it yourself

1. In the demo, use a 3-way categorical code on the 4-cluster data. What does each code value cover?
2. Use 8 categories for 4 clusters. Does InfoGAN split clusters, or leave codes unused?
3. Set `lam_cont=1.0`. Does the continuous code take over and hurt sample quality?
4. Give G a continuous code whose prior is N(0, 1) instead of U(−1, 1) (and change `entropy()` accordingly).
