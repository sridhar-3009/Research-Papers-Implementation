# Kingma & Ba (2015), explained simply

**Paper:** *Adam: A Method for Stochastic Optimization*
**Authors:** Diederik P. Kingma, Jimmy Lei Ba
**Published at:** ICLR 2015 (arXiv:1412.6980)

Read Paper 008 (momentum) first. Adam is momentum plus a per-weight learning rate.

---

## The big idea in one line

> **Keep a running average of the gradient (m, "which way") and of the squared gradient (v, "how big"). Step by m / √v. Every weight then moves by about α per step, whatever its gradient's scale, and the zero-start bias of the averages is divided out.**

---

## 1. The problem (Section 1)

Plain SGD uses **one learning rate for every weight**. That's bad when:
- different weights have gradients of very different sizes (sparse features, different layers);
- the gradients are **noisy** (minibatches, dropout);
- the objective **changes over time**, as in RNNs and online learning.

Two earlier fixes each solve half of this:

| Method | Good at | Weakness |
|---|---|---|
| **AdaGrad** (Duchi 2011): divide by √(sum of all past g²) | sparse gradients | the sum only grows, so learning slowly stops |
| **RMSProp** (Hinton 2012): divide by √(running average of g²) | non-stationary problems | no bias correction, which is unstable early on (Section 6.4) |

Adam = RMSProp's running average + momentum + **bias correction**.

---

## 2. The algorithm (Algorithm 1, page 2)

For each step t = 1, 2, … with gradient g_t:

```
m_t = β1·m_{t-1} + (1 − β1)·g_t          running mean of g         ("1st moment")
v_t = β2·v_{t-1} + (1 − β2)·g_t²         running mean of g²        ("2nd raw moment")
m̂_t = m_t / (1 − β1^t)                   bias correction
v̂_t = v_t / (1 − β2^t)
θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)
```
Everything is **element-wise**: each weight has its own m and v.

**Default settings:** α = 0.001, β1 = 0.9, β2 = 0.999, ε = 10⁻⁸. These defaults work surprisingly often.

- β1 = 0.9 means m averages roughly the last 10 gradients.
- β2 = 0.999 means v averages roughly the last 1000.

---

## 3. Why the step is "about α" (Section 2.1)

The effective step is **Δ = α · m̂/√v̂**.

- **The ratio m̂/√v̂ is like a signal-to-noise ratio.**
  - If the gradient keeps pointing the same way, m̂ ≈ √v̂, so |Δ| ≈ α.
  - If it flips sign randomly, m̂ ≈ 0 while √v̂ stays large, so the steps become tiny. Near a minimum the SNR drops, and Adam automatically **anneals**.
- **Bounded:** |Δ| ≤ α·(1 − β1)/√(1 − β2) in the worst case (a sudden huge gradient after many zeros), and |Δ| ≤ α in common cases.
  - So **α sets a trust region**. You can choose α from how far the weights should ever move (e.g. "weights are ~0.1 in size, so α = 0.001 is safe").
- **Scale-invariant:** multiply every gradient by c, and m̂ and √v̂ both scale by c, so the step is unchanged.
- **The first step** is exactly α·sign(g).

---

## 4. Bias correction (Section 3)

m_0 = v_0 = 0, so the early averages are **pulled toward 0**.

With a constant gradient g:
```
v_t = (1 − β2)(g² + β2 g² + … + β2^{t−1} g²) = g²·(1 − β2^t)
```
So E[v_t] = E[g²]·(1 − β2^t), plus a small term if g drifts. Dividing by (1 − β2^t) removes the bias exactly.

**Why it matters:** with β2 = 0.999, v_1 = 0.001·g², so √v_1 is ~0.03|g|. Without correction, the first steps would be **~30× too big** (and ~100× for β2 = 0.9999). β2 near 1 is exactly what you want for sparse gradients, so the correction is needed right where Adam should shine. That is what Figure 4 shows.

---

## 5. Convergence (Section 4)

- Framework: **online convex optimization**. At each step a new convex loss f_t arrives. **Regret** = Σ_t [f_t(θ_t) − f_t(θ*)], the total extra loss compared with the best fixed θ.
- **Theorem 4.1:** with α_t = α/√t and β1 decaying (β1,t = β1·λ^{t−1}), Adam's regret is **O(√T)**, so the average regret R(T)/T → 0. That's the best known bound for this setting, and like AdaGrad it is much better when gradients are sparse.
- **Caveat (not in the paper):** Reddi et al. (2018, "On the Convergence of Adam and Beyond", AMSGrad) found an error in this proof and built convex problems where Adam fails to converge. In practice Adam still works very well, but the theorem as stated is not correct.

---

## 6. Related methods (Section 5)

| Method | Relation to Adam |
|---|---|
| **RMSProp** | Adam without bias correction (and with momentum applied to the rescaled gradient rather than to g) |
| **AdaGrad** | Adam with β1 = 0, β2 → 1 and α_t = α/√t: then v̂_t = (1/t)Σg², and α/√t · g/√((1/t)Σg²) = α·g/√Σg². **Exactly AdaGrad.** |
| **Natural gradient / Fisher** | v̂ is a diagonal approximation to the Fisher information. Adam uses its **square root**, which is more conservative than the natural gradient. |

---

## 7. Experiments (Section 6)

All use the same models and initialization across optimizers; hyperparameters are chosen by a dense grid search.

| Figure | Problem | Result |
|---|---|---|
| **1 (left)** | MNIST logistic regression, L2, α/√t decay, minibatch 128 | Adam ≈ SGD + Nesterov, both faster than AdaGrad |
| **1 (right)** | IMDB bag-of-words (10,000 features, sparse), 50% dropout on inputs | Adam ≈ AdaGrad, both ≫ SGD Nesterov; RMSProp is also good. Sparse features favor adaptive methods |
| **2** | MNIST MLP, 2×1000 ReLU, dropout | Adam fastest of AdaGrad, RMSProp, SGD Nesterov, AdaDelta. Also faster than the quasi-Newton SFO (which can't handle dropout noise) |
| **3** | CIFAR-10 CNN: c64-c64-c128-1000 (5×5 conv, 3×3 max-pool stride 2), 45 epochs | Adam ≈ SGD Nesterov, both much better than AdaGrad. **Early on AdaGrad is fast, then stalls:** v fills up, and in CNNs the gradient scale matters less |
| **4** | VAE (500 softplus hidden, 50-d latent), bias correction on/off, β1 ∈ {0, 0.9}, β2 ∈ {0.99, 0.999, 0.9999}, log10 α ∈ [−5, −1] | With β2 close to 1, **no bias correction** is unstable, especially early in training. Adam with correction is as good as or better than RMSProp everywhere |

---

## 8. Extensions (Section 7)

### 7.1 AdaMax
Generalize v from the L2 norm to Lp: v_t = β2^p·v_{t−1} + (1 − β2^p)|g_t|^p. As p → ∞, this becomes simple:
```
u_t = max(β2·u_{t−1}, |g_t|)                   no bias correction needed
θ_t = θ_{t−1} − (α / (1 − β1^t)) · m_t / u_t
```
- |step| ≤ α **always**.
- Defaults: α = 0.002, β1 = 0.9, β2 = 0.999.

### 7.2 Temporal averaging
The last iterate is noisy, so evaluate on an **exponential moving average of the parameters**, θ̄_t = β2·θ̄_{t−1} + (1 − β2)·θ_t, bias-corrected in the same way (θ̂ = θ̄/(1 − β2^t)). This is the ancestor of the "EMA weights" used in today's image models and LLM training.

---

## 9. What our code found

**Scale note:** at your request, the neural-network experiments (Figures 1–4) were **not run** on this laptop. `experiments.py` reproduces each figure (from minutes to hours per figure). IMDB is replaced by 20 Newsgroups (also a sparse 10,000-word bag-of-words) because it ships with scikit-learn.

**Checked (tests and demo, under a second):**
- Our Algorithm 1 gives **the same iterates as `torch.optim.Adam`**.
- The first step is exactly α·sign(g), for gradients of 0.001, 1 or 1000.
- Scaling every gradient by 1000 leaves the path unchanged (with ε = 0).
- Steps respect the α(1 − β1)/√(1 − β2) bound; the median step is < α.
- Bias correction: with a constant gradient 2, v̂_t = 4 exactly from t = 1, while raw v_1 = 0.004.
- Without it, with β1 = 0 and β2 = 0.9999, the first step is **1.0 instead of 0.01** (100× too big).
- Adam with β1 = 0, β2 → 1 and α/√t reproduces **AdaGrad** to 10⁻⁵.
- AdaMax's u_t equals the limit of the Lp norm (checked at p = 400), and AdaMax never steps more than α, even with Cauchy (heavy-tailed) gradients.
- **The demo's badly scaled quadratic** (curvatures 1 and 100):
  - SGD Nesterov needs a hand-picked learning rate: 0.005 works, while 0.03 **diverges to 10²⁵²**. Adam and AdaMax run safely with α = 0.1.
  - AdaDelta barely moves in 200 steps, because its steps start at ~√ε.
- **The demo's noisy gradients:** Adam's momentum puts it ~5× closer to the minimum than RMSProp without momentum. The α/√t schedule was slower here: it shrinks the steps before arriving.

---

## 10. Check yourself

1. Write Algorithm 1 from memory. What do m and v estimate?
2. Why is the first step exactly α·sign(g)?
3. Show that v_t = g²(1 − β2^t) for a constant gradient. Why is bias correction most important when β2 is close to 1?
4. Why does Adam take small steps near a minimum, even with a constant α?
5. Derive AdaGrad from Adam (β1 = 0, β2 → 1, α_t = α/√t).
6. Why does the AdaMax update never exceed α?
7. In Figure 3, why does AdaGrad start fast on the CNN and then stall?
