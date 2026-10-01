# Kingma & Ba (2015), explained from scratch

**Paper:** *Adam: A Method for Stochastic Optimization*
**Authors:** Diederik P. Kingma, Jimmy Lei Ba
**Published at:** ICLR 2015 (arXiv:1412.6980)

Read Paper 008 (momentum) first: Adam is momentum plus a per-weight learning rate. This guide builds Adam piece by piece from one simple tool, the **exponential moving average**, and works through every formula with numbers.

---

## 0. The whole idea in one line

> **Keep a running average of the gradient (m: "which way") and of the squared gradient (v: "how big"). Step by m/√v. Every weight then moves by about α per step whatever its gradient's scale, and the zero-start bias of the averages is divided out.**

Adam is the default optimizer of modern deep learning: most Transformers and LLMs are trained with it (or its variant AdamW).

---

## 1. The problem (Section 1)

**Plain SGD uses one learning rate for every weight.** That is bad when:
- **gradient sizes differ wildly between weights:** rare (sparse) features, different layers (Paper 006 showed lower layers have smaller curvature);
- **the gradients are noisy** (minibatches, dropout);
- **the objective changes over time** (RNNs, online learning).

**Two earlier methods each fix half of this:**

| Method | Rule | Good at | Weakness |
|---|---|---|---|
| **AdaGrad** (Duchi 2011) | divide by √(Σ all past g²) | sparse gradients | the sum only grows, so steps shrink toward 0 and learning stops |
| **RMSProp** (Hinton 2012) | divide by √(running average of g²) | changing problems | no bias correction, unstable early (Section 6.4) |

**Adam = RMSProp's running average + momentum + bias correction.**

---

## 2. The tool: the exponential moving average (EMA)

```
a_t = β · a_{t−1} + (1 − β) · x_t          (a_0 = 0)
```
**Unrolled:**
```
a_t = (1 − β) [ x_t + β x_{t−1} + β² x_{t−2} + … + β^{t−1} x_1 ]
```
- **What it is:** an average in which old values fade geometrically.
- **How far back it remembers:** the weights (1 − β)βᵏ sum to about 1 when t is large, and the "effective window" is about **1/(1 − β)** steps:
  - β = 0.9 averages roughly the last **10** values;
  - β = 0.999 averages roughly the last **1,000**.

### 2.1 Bias from starting at zero
- **The problem:** early on, the weights don't yet sum to 1. They sum to:
  ```
  (1 − β)(1 + β + … + β^{t−1}) = (1 − β) · (1 − βᵗ)/(1 − β) = 1 − βᵗ           (geometric series)
  ```
- **So if the inputs are all the same value x,** then a_t = (1 − βᵗ)·x: **too small by the factor (1 − βᵗ)**.
- **The fix:** divide by it:
  ```
  â_t = a_t / (1 − βᵗ)
  ```
  This gives exactly x from step 1.
- **The size of the bias** depends on β: with β = 0.999, a_1 = 0.001x. That is a 1000× underestimate at the first step, and still 37% too small after 1,000 steps (0.999¹⁰⁰⁰ ≈ 0.37).

---

## 3. The algorithm (Algorithm 1, page 2)

For t = 1, 2, … with gradient g_t (everything is **per weight**):
```
m_t = β₁ m_{t−1} + (1 − β₁) g_t          EMA of g    ("first moment":  which way, on average)
v_t = β₂ v_{t−1} + (1 − β₂) g_t²         EMA of g²   ("second raw moment": how big, typically)
m̂_t = m_t / (1 − β₁ᵗ)                    bias corrections (section 2.1)
v̂_t = v_t / (1 − β₂ᵗ)
θ_t = θ_{t−1} − α · m̂_t / (√v̂_t + ε)
```
**The defaults** (α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸) work surprisingly often:
- m remembers about 10 gradients;
- v remembers about 1,000.

### 3.1 Worked example: a constant gradient g = 2, with β₁ = 0.9 and β₂ = 0.999 (our demo)
| t | m_t | m̂_t | v_t | v̂_t | step α·m̂/√v̂ |
|---|---|---|---|---|---|
| 1 | 0.2000 | 2.0000 | 0.004000 | 4.0000 | α·2/2 = **α** |
| 2 | 0.3800 | 2.0000 | 0.007996 | 4.0000 | **α** |
| 3 | 0.5420 | 2.0000 | 0.011988 | 4.0000 | **α** |

- **Without correction,** the step at t = 1 would be α·0.2/√0.004 = α·0.2/0.0632 = **3.2α**.
- **With β₂ = 0.9999 and β₁ = 0,** it's α·2/√(0.0004) = α·2/0.02 = **100α**. The demo shows exactly this: 1.0 instead of 0.01.

---

## 4. Why the step is "about α" (Section 2.1)

The step is Δ = α · m̂/√v̂.

### 4.1 The ratio is like a signal-to-noise ratio
- **If the gradient keeps the same sign and size,** m̂ ≈ g and √v̂ ≈ |g|, so |Δ| ≈ **α**.
- **If it flips randomly (pure noise),** m̂ ≈ 0 while √v̂ ≈ the noise level, so the step is tiny.
- **This gives automatic annealing:** near a minimum the gradient is mostly noise, so the SNR falls and Adam takes small steps even with a constant α.

### 4.2 A bound on every step
- **The worst case:** a huge gradient g arrives after a long run of zeros. Then m ≈ (1 − β₁)g and v ≈ (1 − β₂)g², so:
  ```
  |Δ| ≈ α (1 − β₁) |g| / (√(1 − β₂) |g|) = α (1 − β₁)/√(1 − β₂)
  ```
- **With the defaults** that's α·0.1/0.0316 ≈ **3.16α**, and in common cases |Δ| ≤ α.
- **So α acts as a trust region:** you choose it from how far the weights should ever move in one step.

### 4.3 Scale invariance
- **The setup:** multiply every gradient by c > 0. Then m̂ scales by c, and √v̂ by c.
- **The step is unchanged** (ignoring ε).
- **The consequences:**
  - the first step is exactly **α·sign(g)**;
  - in the demo it is −0.1000 for gradients of 0.001, 1 and 1000 alike.

### 4.4 The geometry: a diagonal preconditioner
- **The rescaling:** dividing by √v̂ gives each coordinate its own learning rate α/√v̂_i.
- **The effect:** coordinates with consistently big gradients (steep directions) get smaller rates, and flat ones get bigger rates. That roughly evens out Paper 006's eigenvalue spread, though only along the coordinate axes.
- **The demo:** f = ½(x² + 100y²):
  - SGD+Nesterov needs a hand-picked rate (0.005 works, 0.03 **diverges to 10²⁵²**);
  - Adam and AdaMax run safely with α = 0.1.

---

## 5. Bias correction, and why it matters (Section 3)

- **The statement:** for slowly changing gradients, E[v_t] = E[g²]·(1 − β₂ᵗ) + a small term (section 2.1), so dividing by (1 − β₂ᵗ) removes the bias.
- **Why it matters most for sparse problems:**
  - Sparse gradients (mostly zeros, with occasional values) **need β₂ close to 1**, so that v remembers the rare non-zero gradients.
  - But β₂ close to 1 makes the start-up bias largest and longest-lasting. Without correction the early steps are 30–100× too big.
  - **Figure 4** (a VAE): with β₂ = 0.999 or 0.9999, training without correction is unstable early on.
  - RMSProp has this problem.

---

## 6. Convergence (Section 4)

- **The framework: online convex optimization.**
  - At each step t a new convex loss f_t arrives.
  - **Regret** R(T) = Σ_t [f_t(θ_t) − f_t(θ*)] is the total extra loss compared with the best fixed parameter θ* in hindsight.
  - If R(T)/T → 0, the algorithm does as well on average as the best fixed answer.
- **Theorem 4.1:**
  - the conditions are α_t = α/√t and β₁ decaying (β₁,t = β₁λ^{t−1});
  - under them Adam's regret is **O(√T)**, so R(T)/T = O(1/√T) → 0;
  - this is the best possible rate for this setting, and like AdaGrad it is much better when gradients are sparse.
- **A caveat** (later work, not in the paper):
  - Reddi et al. (2018, "On the Convergence of Adam and Beyond") found an error in this proof and built simple convex problems where Adam does **not** converge. The problem is that v̂ can shrink, so the effective learning rate can *increase*.
  - Their fix, **AMSGrad**, uses the running maximum of v̂.
  - In practice Adam still works very well.

---

## 7. Related methods (Section 5)

| Method | Relation to Adam |
|---|---|
| **RMSProp** | Adam without bias correction (and with momentum applied to the rescaled gradient, if used) |
| **AdaGrad** | set β₁ = 0, β₂ → 1, α_t = α/√t. Then v̂_t → (1/t)Σg², and the step is (α/√t)·g/√((1/t)Σg²) = **α·g/√(Σg²)**: exactly AdaGrad. Our code reproduces this to 10⁻⁵ |
| **natural gradient** | v̂ is a diagonal estimate of the Fisher information (the curvature of the log-likelihood). The natural gradient divides by F; Adam divides by **√F**, which is more conservative |

---

## 8. Experiments (Section 6)

All optimizers share the same models and initialization; hyperparameters come from dense grid searches.

| Figure | Problem | Result |
|---|---|---|
| **1 (left)** | MNIST logistic regression, L2, α/√t decay, minibatch 128 | Adam ≈ SGD+Nesterov, both faster than AdaGrad |
| **1 (right)** | IMDB bag of words (10,000 sparse features), 50% input dropout | Adam ≈ AdaGrad ≫ SGD+Nesterov. **Sparse features favour per-weight rates** |
| **2** | MNIST MLP 2×1000 ReLU, dropout | Adam fastest (vs AdaGrad, RMSProp, SGD+Nesterov, AdaDelta); also beats the quasi-Newton SFO, which can't handle dropout noise |
| **3** | CIFAR-10 CNN c64-c64-c128-1000, 45 epochs | Adam ≈ SGD+Nesterov ≫ AdaGrad. **AdaGrad starts fast, then stalls:** its Σg² keeps growing, so its steps keep shrinking |
| **4** | VAE, bias correction on/off, β₁ ∈ {0, 0.9}, β₂ ∈ {0.99, 0.999, 0.9999} | without correction, β₂ near 1 is unstable; Adam with correction ≥ RMSProp everywhere |

---

## 9. Extensions (Section 7)

### 9.1 AdaMax
- **Generalize** v from squares (the L2 norm) to p-th powers (the Lp norm):
  ```
  v_t = β₂ᵖ v_{t−1} + (1 − β₂ᵖ)|g_t|ᵖ,      step ∝ m / v_t^{1/p}
  ```
- **As p → ∞,** the p-th root of a weighted sum of p-th powers tends to the **largest** term. (For big p, the biggest number dominates a sum of powers: (aᵖ + bᵖ)^{1/p} → max(a, b).) So:
  ```
  u_t = max(β₂ · u_{t−1}, |g_t|)                 no bias correction needed (max isn't pulled toward 0)
  θ_t = θ_{t−1} − (α / (1 − β₁ᵗ)) · m_t / u_t
  ```
- **A hard bound:** u_t ≥ |every recent gradient|, so **|step| ≤ α always**. Our test confirms this even for heavy-tailed (Cauchy) gradients.
- **Defaults:** α = 0.002, β₁ = 0.9, β₂ = 0.999.

### 9.2 Temporal averaging
- **The problem:** the last iterate is noisy.
- **The fix:** evaluate an EMA of the **parameters**, θ̄_t = β₂θ̄_{t−1} + (1 − β₂)θ_t, bias-corrected as in section 2.1.
- **Legacy:** this is the ancestor of the "EMA weights" used to train today's image generators and LLMs.

---

## 10. What our code found

**Scale note:** at your request, the neural-network experiments (Figures 1–4) were **not run** on this laptop. `experiments.py` reproduces each figure. IMDB is replaced by 20 Newsgroups, also a sparse 10,000-word bag of words, because it ships with scikit-learn.

**Checked (tests and demo, under a second):**
- **Our Algorithm 1 = `torch.optim.Adam`,** iterate for iterate.
- **First step = exactly α·sign(g)** for gradients 0.001, 1 and 1000.
- **Scaling all gradients by 1000 leaves the path unchanged** (ε = 0).
- **Steps respect the α(1 − β₁)/√(1 − β₂) bound;** the median step is < α.
- **Bias correction:** with constant gradient 2, v̂_t = 4 exactly from t = 1 (raw v_1 = 0.004). Without correction, the first step is **1.0 instead of 0.01** (β₁ = 0, β₂ = 0.9999).
- **Adam → AdaGrad** in the limit, to 10⁻⁵.
- **AdaMax:** its u_t equals the Lp limit (checked at p = 400), and its steps never exceed α.
- **Badly scaled quadratic** (curvatures 1 and 100, 200 steps; final loss):

  | optimizer | final loss |
  |---|---|
  | AdaGrad | 2.5e-14 |
  | AdaMax | 1.1e-8 |
  | Adam | 1.6e-7 |
  | SGD+Nesterov (hand-tuned lr) | 1.3e-9 |
  | RMSProp (no momentum) | 2.7e-2: it jitters at a distance ~α |
  | AdaDelta | 69: it hardly moves, since its first steps are ~√ε |

- **Noisy gradients** (noise std 5), mean |x| over the last 500 steps:

  | optimizer | mean \|x\| |
  |---|---|
  | RMSProp | 0.306 |
  | **Adam** | **0.062**: momentum averages the noise, ~5× closer |
  | Adam with α/√t | 0.861: the steps shrink before arriving. Decay schedules need tuning too |

---

## 11. Check yourself

1. Unroll a_t = βa_{t−1} + (1 − β)x_t. Why is its effective window ≈ 1/(1 − β)?
2. Show that the weights sum to 1 − βᵗ, and hence why we divide by it.
3. Write Algorithm 1. What do m and v estimate?
4. Compute the first Adam step for g = 5. (It's α, in the direction of the gradient's sign.)
5. Derive the worst-case step α(1 − β₁)/√(1 − β₂). What is it with the defaults?
6. Why does Adam take small steps near a minimum even with constant α?
7. Derive AdaGrad from Adam.
8. Why does AdaMax's step never exceed α?
9. In Figure 3, why does AdaGrad start fast on the CNN and then stall?
