# Efficient BackProp (1998), explained from scratch

**Paper:** *Efficient BackProp*
**Authors:** Yann LeCun, Léon Bottou, Genevieve B. Orr, Klaus-Robert Müller
**Published in:** *Neural Networks: Tricks of the Trade*, Springer, 1998 (a book chapter, 44 pages)
**Page numbers** below are the PDF's pages (1–44).

Read Paper 004 (backprop) first. This guide derives *why* each trick works, using small examples. The only new math tool is the "curvature" of the error, explained from scratch in section 4.

---

## 0. The whole idea in one line

> **Backprop works in theory, but in practice it crawls or gets stuck unless the data and network are prepared carefully. This chapter lists the tricks, and uses the curvature of the error surface (the Hessian) to explain why each one works.**

**Most of these tricks are still standard today:**
- normalize the inputs;
- use zero-centred activations;
- initialize by fan-in;
- use stochastic gradient descent;
- keep the learning rate below 2/curvature.

---

## 1. Learning and generalization (pages 1–3)

- **Training** minimizes the average error on the training set, E_train = mean of ½(D − output)².
- What we really want is low error on **new** data.
- **Bias–variance:**
  - early in training the network is far from the target (**high bias**);
  - trained too long, it fits this dataset's noise (**high variance**, "overtraining").
- This chapter is about **minimizing E quickly and reliably**. Generalization tricks are a separate topic.

---

## 2. Backprop in matrix form (pages 3–4, Eqs. 1–10)

The network is a stack of **modules** X_n = F_n(W_n, X_{n−1}). For a standard layer:
```
Y_n = W_n X_{n−1}               weighted sums                       (2)
X_n = F(Y_n)                    elementwise sigmoid                 (3)
∂E/∂Y_n = F′(Y_n) ⊙ ∂E/∂X_n     back through the sigmoid            (7)
∂E/∂W_n = ∂E/∂Y_n · X_{n−1}ᵀ    gradient for the weights            (8)
∂E/∂X_{n−1} = W_nᵀ ∂E/∂Y_n      error sent down a layer             (9)
W ← W − η ∂E/∂W                 gradient descent                    (10)
```
These are Paper 004's Eqs. (4)–(7) for a whole layer at once (⊙ = elementwise product).

---

## 3. The practical tricks (pages 4–13)

### 3.1 Stochastic vs batch learning (Section 4.1)
- **Batch** (Eq. 10) averages the gradient over all P examples, then takes one step.
- **Stochastic** (Eq. 11) steps after **each** example.

**Why stochastic wins on real data: redundancy.**
- Suppose your 1,000 examples are secretly 10 copies of 100. Batch computes the same gradient 10 times to make one step; stochastic makes 10 steps in the same time.
- Real datasets are full of near-duplicates, so stochastic learning is often **orders of magnitude** faster.
- **Our test:** on redundant data, after 5 epochs, stochastic MSE is **0.033** vs batch **0.827**.

| | stochastic | batch |
|---|---|---|
| speed on redundant data | **much faster** | slow |
| solution quality | noise can hop out of a poor basin | stays where it starts |
| changing data | tracks it | averages it away |
| convergence | keeps fluctuating near the minimum (anneal η or use mini-batches) | clean; allows second-order methods |

### 3.2 Shuffle (Section 4.2)
- Networks learn most from **surprising** examples, so successive examples should come from **different classes**.
- You can show high-error examples more often ("emphasizing"), but outliers then get too much weight.

### 3.3 Normalize the inputs (Section 4.3)

**(1) Subtract the mean.** Here is the cleanest argument:
- **The setup:** for a unit with error signal δ, every incoming weight's gradient is **δ · x_i** (Eq. 8).
- **What all-positive inputs do:** if every x_i > 0, then **every** gradient component has the **same sign** as δ. All weights into the unit must increase together or decrease together.
- **Why that's slow:** to raise w₁ while lowering w₂, the weights must **zig-zag** over many steps.
- **The fix:** zero-mean inputs remove this constraint.

**(2) Scale to equal variance:** so that every weight learns at a similar speed. Section 4 makes this precise.

**(3) Decorrelate** (PCA, the "KL expansion"): then each weight can be optimized on its own.

### 3.4 The sigmoid (Section 4.4)
- **Symmetric sigmoids** (tanh) beat the logistic (0 to 1). Their outputs average near 0, so they apply trick 3.3(1) to **every layer's** inputs, not just the first.
- **Recommended:** **f(x) = 1.7159 · tanh(2x/3)**.
  - **Where 1.7159 comes from:** we want f(1) = 1, so a·tanh(2/3) = 1 and **a = 1/tanh(2/3) = 1/0.5828 = 1.7159** ✔.
  - The "2/3" puts the point of **maximum curvature** (where the curve bends most) near x = ±1.
  - If the input has variance ~1, the output has variance ~1 too, so signals neither shrink nor blow up from layer to layer.
- **Optionally add a small linear term** (tanh(x) + ax), so the slope is never exactly 0.

### 3.5 Target values (Section 4.5)
- **Never aim at the asymptotes** (±1.7159). To reach them, the network must push the weighted sums toward ±∞:
  - the weights grow without bound;
  - the units saturate, so f′ ≈ 0 and learning stalls;
  - the outputs become 0/1-like, so they no longer signal confidence.
- **Use targets ±1**, where f bends most, so the outputs stay in a responsive region.

### 3.6 Initialize the weights (Section 4.6, Eqs. 14–16): the variance calculation
A unit computes y = Σ_{i=1..m} w_i x_i, where m is the **fan-in**. Assume:
- the inputs x_i are independent with mean 0 and variance 1 (they were normalized);
- the weights w_i are independent with mean 0 and variance σ².

Then:
```
Var(y) = Σ_i Var(w_i x_i) = Σ_i E[w_i²] E[x_i²] = m · σ² · 1 = m σ²
```
- **We want Var(y) ≈ 1**, which puts the sigmoid in its useful region (not saturated, not flat). That gives **σ = m^(−1/2)**.
- **Too big** → saturated units → vanishing gradients. **Too small** → near the flat origin, tiny signals.
- **Our check:** 400 unit-variance inputs with this rule give sums with standard deviation **1.00**.
- This is the ancestor of Xavier/Glorot init (Paper 007), which also balances the *backward* pass, and He init (for ReLU).

### 3.7 Learning rates (Section 4.7)
- Ideally each weight has its own rate, so that all weights converge together.
- **Lower layers usually need larger rates:** their curvature is smaller (section 6).
- **Momentum** Δw(t+1) = −η ∂E/∂w + μ Δw(t) helps in long, narrow valleys.
- **Adaptive rates** (Eqs. 17–19): shrink η as you approach the minimum.

### 3.8 RBF units (Section 4.8)
- **What they are:** Gaussian bumps exp(−‖x − c‖²/2σ²) instead of sigmoids.
- **Trade-off:** they respond only near a centre c, so they are good in low dimensions and poor in high dimensions, where you'd need exponentially many bumps.

---

## 4. Why the tricks work: curvature (pages 13–21)

### 4.1 One weight (Eqs. 20–26, Figure 6)
- **The setup:** near a minimum w*, any smooth error looks like a parabola:
  ```
  E(w) ≈ E(w*) + ½ h (w − w*)²          h = E″ = the curvature
  ```
- Its slope is E′(w) = h (w − w*). One gradient step gives:
  ```
  w_new − w* = (w − w*) − η h (w − w*) = (1 − η h)(w − w*)
  ```
- **Every step multiplies the distance to the minimum by (1 − ηh).** So:

| η | factor 1 − ηh | behaviour |
|---|---|---|
| η < 1/h | between 0 and 1 | creeps toward the minimum |
| **η = 1/h** | **0** | **lands exactly in one step** (η_opt) |
| 1/h < η < 2/h | between −1 and 0 | overshoots, oscillates, converges |
| η = 2/h | −1 | bounces forever |
| **η > 2/h** | below −1 | **diverges**: each bounce grows |

- **Worked example:** h = 4, starting 1 unit from the minimum:
  - η = 0.1 gives factor 0.6: distances 1, 0.6, 0.36, …;
  - η = 0.25 gives factor 0: done in one step;
  - η = 0.4 gives factor −0.6: distances 1, −0.6, 0.36, …;
  - η = 0.6 gives factor −1.4: distances 1, −1.4, 1.96, …, a blow-up.

### 4.2 Many weights: the Hessian and its eigenvalues (Eqs. 27–39)
- **The Hessian:** with many weights, the curvature is a matrix H, with H_ij = ∂²E/∂w_i∂w_j.
- **Its directions:** H has **eigenvectors** (special directions) and **eigenvalues** λ (the curvature along each).
- **Why they matter:** along eigenvector k, gradient descent behaves exactly like the 1-D case with h = λ_k. One shared η must work for **all** directions at once:
  ```
  stability:  η < 2 / λ_max                 (Eq. 38)
  best single rate: η ≈ 1 / λ_max            (Eq. 39)
  then the slowest direction shrinks by (1 − λ_min/λ_max) per step
  ```
- **Condition number:** the number of steps needed grows like **λ_max / λ_min**.
- **The picture:** a long, narrow "taco-shell" valley. A step small enough for the steep walls barely moves you along the flat floor.

### 4.3 The linear unit: H is the input covariance (Eqs. 28–29)
- For a linear unit, E = ½ ⟨(d − wᵀx)²⟩ (averaged over the data). Differentiating twice gives:
  ```
  ∂E/∂w = −⟨(d − wᵀx) x⟩       ∂²E/∂w∂wᵀ = ⟨x xᵀ⟩ = H
  ```
- The Hessian is the matrix of input second moments. **Now every input trick has an exact reason:**
  - **Non-zero mean:** ⟨xxᵀ⟩ = covariance + mean·meanᵀ. The term mean·meanᵀ adds **one huge eigenvalue** ≈ ‖mean‖², which kills the condition number.
    - **Our check:** shifting the inputs by +3 raises the condition number from **33 to 2,096**, and learning at the best safe rate needs **5,888 epochs instead of 4**.
  - **Unequal variances** spread the eigenvalues apart.
  - **Correlations** tilt the valley away from the weight axes, so per-weight learning rates can't fix it.

**The paper's example (Figures 9–12):**
- two Gaussian classes, a linear net with 2 weights and a bias;
- input covariance eigenvalues 0.84 and 0.036;
- the paper claims η_max = 2/0.84 = 2.38, and shows η = 1.5 converging and η = 2.5 diverging.

**Section 9 explains why the true limit is 2.0.**

---

## 5. Classical second-order methods (pages 21–24)

| Method | Idea | Cost | Verdict for big nets |
|---|---|---|---|
| **Newton** (Eq. 40) | Δw = −H⁻¹ ∇E: rescales every direction by its own curvature, so every eigen-direction gets factor 0 | N×N storage, O(N³) to invert | impractical; it goes *uphill* where H has negative eigenvalues |
| **Conjugate gradient** (Eqs. 41–43) | directions that don't undo each other; at most N steps on a quadratic | O(N) | batch only (needs line searches); good for small problems |
| **Quasi-Newton / BFGS** (Eqs. 44–45) | builds H⁻¹ from successive gradient differences | O(N²) | small nets only |
| **Gauss–Newton / LM** (Eqs. 46–47) | H ≈ JᵀJ (+λI): always positive (semi-)definite | O(N³) | squared error only; small nets |

The chapter's verdict: *"Classical second-order methods are impractical in almost all useful cases."* (Paper 005 used them on a 35-weight net, where they do work.)

---

## 6. Cheap Hessian information (pages 24–27)

- **Finite differences** (Section 7.1): one column of H ≈ [∇E(w + δe_k) − ∇E(w)]/δ.
- **The Gauss–Newton approximation** (Section 7.2, Eq. 51):
  - For E = ½ Σ (output − target)², the second derivative contains a term multiplied by the **error**.
  - Near a good fit the error is small, so drop that term: H ≈ Σ JᵀJ, which is always ≥ 0.
- **The diagonal Hessian by "backprop with squares"** (Section 7.4, Eqs. 54–56). This is the Gauss–Newton diagonal, propagated like backprop:
  ```
  ∂²E/∂y² ≈ ∂²E/∂o² · f′(y)²          (squared slope instead of slope)
  ∂²E/∂w_ki² = ∂²E/∂y_k² · x_i²       (squared input instead of input)
  ∂²E/∂x_i²  = Σ_k ∂²E/∂y_k² · w_ki²  (squared weights instead of weights)
  ```
  It costs about one backprop pass. (Optimal Brain Damage uses the same quantity to prune weights.)
- **Hessian × vector without the Hessian** (Eq. 59):
  ```
  Hψ ≈ [∇E(w + αψ) − ∇E(w)] / α
  ```
  This is just two gradient computations. **Why it works:** the gradient's change along ψ is, to first order, H times ψ.

---

## 7. What a real network's Hessian looks like (pages 27–29)

- **The spectrum** (Figures 19–20): a few small eigenvalues, many medium ones, and **a few huge ones** ("big killers"). Those few set the safe η and slow everything else.
- **Their causes:** non-zero-mean inputs or activations, curvature differences between layers, and correlated variables.
- **Layers differ** (Figure 21): curvature is smaller in **lower** layers, so they learn slowly while the upper layers oscillate. The remedy is larger rates for lower layers.

---

## 8. Second-order methods that do work for big nets (pages 29–35)

### 8.1 Stochastic diagonal Levenberg–Marquardt (Section 9.1, Eqs. 61–62)
```
η_ki = η / (⟨∂²E/∂w_ki²⟩ + μ)
```
- **The idea:** each weight's rate is divided by **its own curvature** (a diagonal Newton step). ⟨·⟩ is a running average of section 6's diagonal, re-estimated every few epochs on a subset.
- **Why μ:** it stops the rate exploding where the curvature is ≈ 0.
- **Payoff:** the extra cost is negligible. The paper's rule of thumb is **~3× faster** than well-tuned SGD.

### 8.2 Setting η automatically from λ_max (Section 9.2, Eqs. 60, 63–64)
**The power method:** repeat ψ ← Hψ / ‖ψ‖, computing Hψ with Eq. 59.

**Why it converges to the largest eigenvalue:**
1. Write ψ as a sum of eigenvectors, ψ = Σ c_k v_k.
2. Multiplying by H scales each component by its λ_k.
3. After t multiplications, the largest one dominates: (λ_max)ᵗ outgrows the rest.
4. So ψ lines up with v_max, and ‖Hψ‖/‖ψ‖ → λ_max.

**Then:**
- **An on-line version** (Eq. 64) averages over single examples. It reaches the right order of magnitude within 100 examples (Figure 24).
- **Set η ≈ 1/λ_max.** Figure 25 claims this "predicted optimal rate" is close to the best rate found by trial.

---

## 9. The recipe (page 35, Section 10)

1. **Shuffle** the examples.
2. **Center** each input.
3. **Scale** each input to unit variance.
4. **Decorrelate** if possible.
5. Use **1.7159·tanh(2x/3)**.
6. Use targets **±1**.
7. Initialize with standard deviation **m^(−1/2)**.
8. **Large, redundant classification data:** use **SGD** or **stochastic diagonal LM**.
9. **Small data or regression:** use **conjugate gradient**.

---

## 10. What our reproduction found

See [results.md](results.md) and [figures/](figures/). Every number comes from running the code.

### ✅ Confirmed
- **Backprop, the diagonal Hessian and the power method are exact:** they match finite differences, the Gauss–Newton diagonal, and the true λ_max (4.267 vs 4.267).
- **The sigmoid:** f(±1) = ±1, and its curvature peaks at x ≈ 1.
- **Fan-in init:** standard deviation **1.00** for 400 inputs.
- **The paper's data:** covariance eigenvalues 0.84 and 0.036 (ours: 0.805 and 0.030).
- **Centring matters:** a shift of +3 takes the condition number from 33 to 2,096 and the epochs from 4 to 5,888.
- **Stochastic ≫ batch on redundant data:** MSE 0.033 vs 0.827 after 5 epochs.
- **Tricks on real digits** (accuracy after 3 epochs, best η per row, 3 seeds):

  | configuration | accuracy |
  |---|---|
  | full recipe | **93.2%** |
  | inputs not normalized | 75.2% |
  | logistic sigmoid (0/1) | 78.5% |
  | weights too big (±3) | 66.2% |

- **The Hessian:**
  - lower-layer curvature 0.20 < upper-layer curvature 0.58, as in Figure 21;
  - raw inputs give "big killers" (1st/11th eigenvalue = **17×** vs 1.8× when normalized).
- **Stochastic diagonal LM** reaches SGD's 6-epoch error in ~3 epochs, **~2× faster** (the paper says ~3×).

### ⚠️ Different from the paper
1. **The paper's learning-rate limit for its own example is wrong.**
   - It uses 2/λ_max = 2/0.84 = **2.38**, but the network has a **bias**: an input that is always 1.
   - By section 4.3, that input adds its own second moment ⟨1·1⟩ = 1 to H, so there's an eigenvalue of **~1.00**.
   - The true limit is **2/1.00 = 2.0**.
   - Our demo: η = 1.9 converges (MSE 0.0221), and **η = 2.1 diverges** (MSE 149), though the paper's bound says it's safe.
2. **Two warnings didn't show up in short, shallow runs:** targets at the asymptotes (95.3%) and tiny initial weights (93.7%) both did as well as the recipe. Their failure modes (saturation, flat regions) appear in deeper nets and longer training. Paper 004's 5-layer net did stall with small weights.
3. **1/λ_max is only a ballpark:**
   - **Batch:** it's safe, but 2–3× larger did better within 50 updates, because the curvature shrinks as training proceeds.
   - **Stochastic:** it was **~10× too large** (MSE 0.35 at 0.1× vs 4.1 at 1×). Single-example gradients are far noisier than their average.
   - We couldn't reproduce Figure 25's "ratio 1 is optimal enough" for stochastic learning.
4. **The on-line eigenvalue estimate (Eq. 64) is noisy:** 17.6 after 100 examples vs a true 10.3; the average of the last 500 was 9.2. The paper also had to tune its averaging rate γ.

**Data note:** the paper's digit sets aren't available, so we used scikit-learn's 8×8 digits (1,797 images).

---

## 11. Check yourself

1. Show that one gradient step on ½h(w − w*)² multiplies the distance to w* by (1 − ηh). What η gives one-step convergence? Above which η does it diverge?
2. With h = 10, what are η_opt and the divergence threshold?
3. Why do all-positive inputs force the weights into a unit to move in the same direction?
4. Derive σ = m^(−1/2) for the initial weights.
5. Show that 1.7159 = 1/tanh(2/3). Why is ±1 a good target?
6. For a linear unit, why is H the input second-moment matrix? What does a non-zero mean do to its eigenvalues?
7. How does the power method find λ_max using only gradients?
8. Why did the paper's example diverge at η = 2.1?
