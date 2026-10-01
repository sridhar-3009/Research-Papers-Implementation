# Srivastava et al. (2014), explained from scratch

**Paper:** *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*
**Authors:** Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
**Published in:** Journal of Machine Learning Research 15 (2014), pp. 1929–1958
**Page numbers** below are the journal's (1929–1958).

This is the **full version** of Paper 009. Read 009 first: it proves the "mean network = geometric mean" fact. This guide covers the extras:
- the exact connection to ridge regression, derived line by line;
- Gaussian dropout;
- what the experiments say about *when* dropout helps.

---

## 0. The whole idea in one line

> **Train a large network while randomly dropping units, so it becomes an ensemble of exponentially many thinned networks with shared weights. At test time, scale the weights to average them all. This is one of the most effective and general regularizers ever found.**

---

## 1. Motivation (pages 1929–1931)

- **The problem:**
  - Big networks overfit.
  - The best cure, **averaging many different models**, is expensive: each model must be trained, and all of them must be run at test time.
- **Dropout's answer:** approximately average 2ⁿ thinned networks at the cost of training **one**.
- **The sex analogy:**
  - A gene must work with *random* partner genes from another parent, so it can't depend on a fixed set of collaborators. This favours robust, individually useful genes.
  - Dropout imposes the same pressure on hidden units.

---

## 2. The model (Section 4, pages 1933–1934)

### 2.1 Equations
**A standard layer:**
```
z = W·y + b,     y_next = f(z)
```
**A dropout layer:**
```
r_j ~ Bernoulli(p)      independently for every unit j and every training case
ỹ = r ⊙ y               thinned outputs (⊙ = elementwise product)
z = W·ỹ + b,  y_next = f(z)
```
- **p is the probability of KEEPING a unit.** The paper uses 0.5 for hidden layers and 0.8 for inputs.
- **Test time:** W_test = p·W. This matches the expected input, since E[r_j y_j] = p·y_j.

**Our demo:**
- y = (2, 0.5, 1, 3, 0, 1.5);
- three random masks give thinned outputs such as (2, 0, 1, 3, 0, 0);
- at test time, p·y = (1, 0.25, 0.5, 1.5, 0, 0.75), the average over all masks.

### 2.2 "Inverted dropout" (the modern convention)
- **The trick:** divide by p **during training** (ỹ = r ⊙ y / p) instead of multiplying by p at test time.
- **The expectations are identical,** and the test-time network is just the plain network. PyTorch's `nn.Dropout` does this.

---

## 3. Training (Section 5, pages 1934–1935)

- **SGD as usual,** but each training case gets its own thinned network. Dropped units contribute **zero gradient** for that case.
- **Max-norm:**
  - keep each hidden unit's incoming weight vector inside a ball, ‖w‖ ≤ c (typically c = 3–4);
  - after any update that leaves the ball, rescale: w ← c·w/‖w‖.
- **The winning combination:** max-norm + a **large, decaying learning rate** + **high momentum**.
  - The ball stops the weights from blowing up even with huge steps.
  - Dropout's noise and the large steps explore weight space.
  - The decay lets training settle at the end.
- **Fine-tuning pretrained nets:** scale the pretrained weights **up by 1/p** first, so the expected input under dropout matches what they were trained with. Use a smaller learning rate.

---

## 4. Results across domains (Section 6, pages 1935–1943)

| Dataset | Domain | Best without dropout | With dropout |
|---|---|---|---|
| **MNIST** | digits | 1.60% | **0.95%** (8192-unit layers, max-norm) |
| SVHN | house numbers | 3.95% | **2.55%** |
| CIFAR-10 | photos | 14.98% | **12.61%** |
| CIFAR-100 | photos | 43.48% | **37.20%** |
| ImageNet (ILSVRC-2012) | photos | 26.2% (top-5) | **16.4%** |
| TIMIT | speech | 23.4% | **21.8%** |
| Reuters | text | 31.05% | 29.62% |

### MNIST in detail (Table 2)
| Network | Error |
|---|---|
| standard net | 1.60% |
| dropout, logistic | 1.35% |
| dropout, ReLU | 1.25% |
| dropout + max-norm, ReLU | **1.06%** |

- **A net with 65 million weights on 60,000 images doesn't overfit with dropout,** and doesn't even need early stopping.
- **Figure 4:** the same hyperparameters work across many architectures.
- **Bayesian neural nets** beat dropout slightly on a tiny genetics dataset, but they're far slower.

### Table 9: dropout vs other regularizers (MNIST, 784-1024-1024-2048-10, ReLU)
| Method | Error |
|---|---|
| L2 | 1.62% |
| L2 + L1 | 1.60% |
| L2 + KL-sparsity | 1.55% |
| max-norm | 1.35% |
| dropout + L2 | 1.25% |
| **dropout + max-norm** | **1.05%** |

---

## 5. Why and when it works (Section 7, pages 1943–1947)

### 5.1 Features (Figure 7)
A 256-unit ReLU autoencoder on MNIST:
- **without** dropout, its features are co-adapted noise;
- **with** dropout, they are clean edges, strokes and spots.

### 5.2 Sparsity for free (Figure 8)
- **Dropout makes the hidden activations sparse** without any sparsity penalty: the mean activation falls from ~2.0 to ~0.7, and most activations sit near 0.
- **The paper's reading:**
  - A unit must be useful even when its partners are missing, so it learns to fire for a few distinctive patterns.
  - Firing weakly for everything would make it an unreliable helper.

### 5.3 The keep-probability p (Figure 9)
- **Fixed width n** (784-2048×3-10):
  - very small p **underfits**, because too little of the network survives each step;
  - the error is flat for 0.4 ≤ p ≤ 0.8 and rises again near p = 1 (no dropout).
- **Fixed p·n** (widen the layers as p drops, so the *expected number of active units* stays the same):
  - small p hurts far less (p = 0.1: 2.7% → 1.7% error);
  - p ≈ 0.6 is best, and 0.5 is close.
- **Lesson:** with dropout, make the network **wider**, about n/p units.

### 5.4 Dataset size (Figure 10)
- **100–500 examples:** no help; the network memorizes anyway.
- **Medium size:** big gains.
- **Very large data:** smaller gains, because overfitting was less of a problem to begin with.

Regularizers help most where variance (overfitting) is the main error.

### 5.5 Monte-Carlo averaging vs weight scaling (Figure 11)
- **The comparison:** averaging k sampled thinned nets (k forward passes) vs the one-pass weight-scaling rule. Weight scaling equals the MC average at about **k = 50**, and MC is only slightly better beyond that.
- **Why the two differ at all:** weight scaling is exact only for one softmax layer (Paper 009). With non-linear hidden layers, E[f(x)] ≠ f(E[x]) in general.

---

## 6. Dropout RBMs (Section 8, pages 1947–1949)

- **How:** drop hidden units of a Restricted Boltzmann Machine during CD-1 training.
- **Effect:** coarser features, fewer dead units, sparser codes.
- (Not implemented here; RBMs are a separate topic, see Paper 023's RTRBM.)

---

## 7. Marginalizing dropout: the exact link to ridge regression (Section 9, pages 1949–1950)

### 7.1 The derivation
**The setup:** linear regression, prediction ŷ = Σ_j r_j x_j w_j, with r_j ~ Bernoulli(p) dropping input features.

**One data row first.** Use E[A²] = (E[A])² + Var(A):
```
E[(y − Σ_j r_j x_j w_j)²]  =  (y − p Σ_j x_j w_j)²  +  Var(Σ_j r_j x_j w_j)
```
The r_j are independent with Var(r_j) = p(1 − p), so:
```
Var(Σ_j r_j x_j w_j) = Σ_j x_j² w_j² · p(1 − p)
```
**Summing over all rows of X:**
```
E ‖y − (R ⊙ X) w‖²  =  ‖y − pXw‖²  +  p(1 − p) Σ_j (Σ_rows x_j²) w_j²
                    =  ‖y − pXw‖²  +  p(1 − p) ‖Γ w‖²,      Γ = diag(XᵀX)^½
```
**This is ridge regression** (squared error + a penalty on the squared weights), where each weight's penalty is scaled by the energy of its input column.

**With the effective weights w̃ = p·w (what the test-time network uses):**
```
minimize  ‖y − X w̃‖²  +  ((1 − p)/p) · ‖Γ w̃‖²
```
The penalty strength is λ = (1 − p)/p:
- p = 0.8 gives λ = 0.25;
- p = 0.5 gives λ = 1;
- p = 0.2 gives λ = 4.

**Less keeping means stronger shrinkage.**

### 7.2 What the Γ scaling really does (our analysis)
- **Take uncorrelated input columns,** so XᵀX = diag(a_j). Each weight is then solved separately:
  ```
  minimize  a_j (w̃_j − w*_j)² + λ a_j w̃_j²   ⇒   w̃_j = w*_j / (1 + λ) = p · w*_j
  ```
  (w* = the ordinary least-squares weight.)
- **Every weight shrinks by the same factor p, whatever its input's spread.** The variance-scaled penalty exactly cancels the variance in the fit term. So dropout's shrinkage is **scale-invariant**: rescaling a feature (e.g. metres → millimetres) doesn't change how strongly it's regularized.
- **Plain ridge (λ‖w‖²) is different:** it gives w̃_j = a_j w*_j/(a_j + λ), which squeezes **low-spread** features hardest.
- **Our numbers** (true weights 1, 1, 1; input spreads 1.0, 3.0, 0.3):

  | method | effective weights |
  |---|---|
  | dropout ridge, p = 0.5, 200,000 samples | **0.497, 0.500, 0.498** (uniform ✔) |
  | dropout ridge, p = 0.5, 200 samples | 0.472, 0.495, 0.356 |
  | plain ridge, λ = 50, 200 samples | 0.806, 0.964, **0.291** |

  - **With 200 samples,** the small column correlates by chance with the big one. The penalty only sees the diagonal of XᵀX, so the correlation shifts that column. With enough data it disappears.
  - **Plain ridge** really does single out the low-spread column.

### 7.3 Logistic regression and deep nets
- **No exact closed form exists.** An approximate Gaussian marginalization (Wang & Manning, 2013) works for logistic regression, but not well for deep nets.
- **Later theory** (Wager et al., 2013) showed that dropout in generalized linear models acts like an **adaptive L2 penalty** weighted by the curvature of the loss.

---

## 8. Gaussian dropout (Section 10, pages 1950–1951)

- **The idea:** multiply each unit by **r ~ N(1, σ²)** instead of a 0/1 mask.
- **Matching the moments of inverted Bernoulli dropout** (r/p, where r ~ Bernoulli(p)):
  ```
  E[r/p] = p/p = 1
  Var[r/p] = p(1 − p)/p² = (1 − p)/p
  ```
  So use **σ² = (1 − p)/p**:
  - p = 0.5 gives σ² = 1;
  - p = 0.8 gives σ² = 0.25.
- **The mean is 1,** so there's **no test-time scaling**.
- **Results:** as good or slightly better (MNIST: Bernoulli 1.08% vs Gaussian **0.95%**; CIFAR-10: 12.6% vs 12.5%).
- **Our demo:** one sample of r = (0.74, 2.07, 1.42, 1.84, 0.17, 0.21). That's noisy, but centred on 1.

---

## 9. Practical guide (Section 11 and Appendix A, pages 1951–1954)

| Setting | Advice | Why |
|---|---|---|
| training time | expect **2–3×** longer | gradients are noisy; each weight trains in only part of the network |
| width | if n units is right without dropout, use ≥ **n/p** | about n units remain active (section 5.3) |
| learning rate | **10–100×** larger than usual | dropout's noise averages out; max-norm keeps it safe |
| momentum | **0.95–0.99** | smooths the noisy gradients |
| max-norm c | **3–4** | |
| p | 0.5 hidden, 0.8 real-valued inputs | inputs carry unique information, so drop fewer |

---

## 10. What our code found

**Scale note:** at your request, the MNIST experiments were **not run** on this laptop. `experiments.py` reproduces Table 9, Figures 7–11 and Table 10 (≈ 1 hour on a GPU). Compare with the paper's numbers above.

**Checked (small computations):**
- **Dropout in linear regression = ridge regression (Section 9.1).**
  - SGD with **actual random dropout masks** lands on the closed-form ridge weights: [0.84, −2.01, 0.01, 1.01, −0.02] vs [0.88, −2.01, 0.02, 1.01, −0.06].
  - On the dropout objective, both beat ordinary least squares (1,610 vs 1,624).
- **Uniform shrinkage:** with uncorrelated data, every effective weight → p·w* (table in section 7.2).
- **Tests also check:**
  - Bernoulli test-time scaling = the expected training input;
  - Gaussian dropout has mean 1 and variance (1 − p)/p;
  - max-norm projection;
  - the ridge solution has zero gradient;
  - smaller p shrinks harder.

**A correction to our own demo:**
- An earlier version of `demo.py` printed "inputs with big spread are squeezed hardest". Its own numbers showed the opposite.
- **The real story** (section 7.2): dropout shrinks every weight equally; the leftover differences come from chance correlations in small samples.
- The demo now says this.

---

## 11. Check yourself

1. Write the dropout layer. What is p? What do you do at test time, and with inverted dropout?
2. Derive E‖y − (R ⊙ X)w‖² = ‖y − pXw‖² + p(1 − p)‖Γw‖² for one row.
3. With w̃ = pw, what is the penalty strength? Compute it for p = 0.5 and p = 0.8.
4. Show that with uncorrelated inputs, dropout shrinks every weight by exactly p. How is plain ridge different?
5. Why does Gaussian dropout use σ² = (1 − p)/p, and why does it need no test-time scaling?
6. Explain Figure 9: why is "p·n fixed" better than "n fixed" for small p?
7. Why does dropout not help with 100 training examples?
8. Why are MC averaging and weight scaling not exactly equal in deep nets?
