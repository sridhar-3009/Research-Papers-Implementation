# Efficient BackProp (1998), explained simply

**Paper:** *Efficient BackProp*
**Authors:** Yann LeCun, Léon Bottou, Genevieve B. Orr, Klaus-Robert Müller
**Published in:** *Neural Networks: Tricks of the Trade*, Springer, 1998 (a book chapter, 44 pages)
**Page numbers** below are the PDF's pages (1–44).

---

## The big idea in one line

> **Backprop (Paper 004) works in theory, but in practice it is slow or gets stuck unless you prepare the data and the network carefully. This paper lists the tricks, and uses the curvature of the error surface (the Hessian) to explain why each one works.**

---

## How it connects to the earlier papers

- **Paper 004** gave backprop but trained slowly. Our reproduction hit **vanishing gradients** and **saturated sigmoids**.
- **This paper** explains those problems and gives the standard fixes. Most of them are still used today: normalize inputs, use a zero-centred activation, initialize by fan-in, use SGD.
- **Paper 005** surveyed many of the same ideas (Sections 7.1–7.4 there).

---

## 1. Learning and generalization (pages 1–3)

- **Training** = minimize the average error on the training set, E_train (the mean squared error, ½(D − output)²).
- What really matters is performance on **new** data (the test set).
- **Bias–variance:**
  - Early in training the network is far from the target: **high bias**.
  - Trained too long, it learns the noise in this particular dataset: **high variance** (**overtraining**).
- This paper is about **minimizing E well and fast**. Generalization tricks (early stopping, regularization) are a separate topic.

---

## 2. Backprop in matrix form (pages 3–4, Eqs. 1–10)

The network is a stack of **modules**, each computing X_n = F_n(W_n, X_{n−1}). Backprop runs the chain rule through them in reverse. For the usual layers:
```
Y_n = W_n X_{n−1}             weighted sums                  (2)
X_n = F(Y_n)                  sigmoid                        (3)
∂E/∂Y_n = F'(Y_n) ∂E/∂X_n     through the sigmoid            (7)
∂E/∂W_n = X_{n−1} ∂E/∂Y_n     gradient for the weights       (8)
∂E/∂X_{n−1} = W_nᵀ ∂E/∂Y_n    error sent down a layer        (9)
W(t) = W(t−1) − η ∂E/∂W       gradient descent               (10)
```
These are exactly Paper 004's Eqs. (4)–(7), written with matrices.

---

## 3. The practical tricks (pages 4–13)

### 3.1 Stochastic vs batch learning (Section 4.1)

| | **Stochastic** (update after each example, Eq. 11) | **Batch** (update after all examples, Eq. 10) |
|---|---|---|
| Speed | **Much faster**, especially on large, **redundant** data | Slow: recomputes the same information over and over |
| Solutions | Noise can jump out of a poor basin into a better one | Stays in the basin where it started |
| Changing data | Can **track** changes over time | Averages them away |
| Convergence | Keeps **fluctuating** near the minimum (anneal η, or use mini-batches) | Well understood; allows second-order methods |

**Example:** 1,000 examples that are secretly 10 copies of 100. Batch computes the same gradient 10 times per update. Stochastic makes 10 updates in the same time.

### 3.2 Shuffle the examples (Section 4.2)
- Networks learn most from **unexpected** examples.
- Make consecutive examples come from **different classes**.
- Optionally show high-error examples more often ("emphasizing"), but beware of outliers.

### 3.3 Normalize the inputs (Section 4.3)
1. **Subtract the mean.** If every input is positive, all the weights into a unit must move in the **same** direction at each step, so the weights **zig-zag**.
2. **Scale to the same variance** (about 1), so all weights learn at similar speeds.
3. **Decorrelate** (PCA, "KL expansion") if you can, so each weight can be solved on its own.

### 3.4 The sigmoid (Section 4.4)
- **Symmetric** sigmoids (tanh) beat the logistic (0 to 1): their outputs average near 0, which is trick 3.3 applied to every layer.
- **The recommended sigmoid:** `f(x) = 1.7159 · tanh(2x/3)`. It gives f(±1) = ±1, its curvature peaks at ±1, and unit-variance input gives about unit-variance output.
- Optionally add a small linear term (tanh(x) + ax) to avoid flat spots.

### 3.5 Target values (Section 4.5)
- **Don't** put targets at the sigmoid's asymptotes (±1.7159). The network pushes weights to infinity chasing them, units saturate, learning stalls, and the outputs stop signalling confidence.
- **Use ±1**, where the recommended sigmoid bends the most.

### 3.6 Initialize the weights (Section 4.6, Eqs. 14–16)
- **Too big:** units saturate and gradients vanish.
- **Too small:** gradients are tiny, and near 0 the error surface is flat.
- **Rule:** random, mean 0, **standard deviation m^(−1/2)**, where m = fan-in (the number of inputs to the unit). With normalized inputs, each unit's weighted sum then has variance about 1.

This is the ancestor of **Xavier/Glorot** (Paper 007) and **He** initialization.

### 3.7 Learning rates (Section 4.7)
- Ideally, **each weight gets its own rate**, so all weights converge at about the same speed.
- Lower layers usually need **larger** rates, because their second derivatives are smaller.
- **Momentum** (Δw(t+1) = −η ∂E/∂w + μ Δw(t)) helps on elongated error surfaces.
- **Adaptive rates** (Eqs. 17–19): shrink η as you approach the minimum.

### 3.8 RBF units (Section 4.8)
Gaussian "bumps" instead of sigmoids. They're local, so they're good in low dimensions and poor in high dimensions.

---

## 4. Why the tricks work: the Hessian (pages 13–21)

### 4.1 One dimension (Eqs. 20–26, Figure 6)
Near a minimum, E looks like a parabola with curvature E'' (the second derivative).
- **η_opt = 1 / E''** reaches the minimum in **one step**.
- η < η_opt: slow. η_opt < η < 2η_opt: oscillates, but converges. **η > 2η_opt: diverges.**

### 4.2 Many dimensions (Eqs. 27–39)
- The curvature is a matrix, the **Hessian** H (all second derivatives, Eq. 27). Its **eigenvalues** λ are the curvatures along its main directions.
- With one learning rate for all weights:
  - **Diverges if η > 2 / λ_max** (Eq. 38).
  - The best choice is **η = 1 / λ_max** (Eq. 39).
  - Convergence time grows with the **condition number** λ_max / λ_min.
- **The picture:** a long narrow valley (a "taco shell"). A step size small enough for the steep walls barely moves you along the flat floor.

### 4.3 Example: a linear unit (LMS, Eqs. 28–29)
For a linear unit, **H = the covariance of the inputs** (Eq. 29). So:
- **a non-zero input mean creates one huge eigenvalue**, which is why we subtract the mean (Section 5.3);
- **unequal variances** spread the eigenvalues, which is why we scale;
- **correlations** rotate the eigenvectors away from the weight axes, which is why we decorrelate.

Now every trick in Section 3.3 has a reason: each one **reduces the eigenvalue spread**.

**Paper's example (Figures 9–12):** two Gaussian classes, a linear net with 2 weights and a bias, covariance eigenvalues 0.84 and 0.036. The paper says η_max = 2/0.84 = 2.38, and shows η = 1.5 converging and η = 2.5 diverging.

---

## 5. Classical second-order methods (pages 21–24)

| Method | Idea | Cost | Verdict for big nets |
|---|---|---|---|
| **Newton** (Eq. 40) | Δw = −H⁻¹ ∂E/∂w. One step on a quadratic; unaffected by input scaling or rotation | stores and inverts N×N, O(N³) | impractical; diverges if H isn't positive definite |
| **Conjugate gradient** (Eqs. 41–43) | Directions that don't undo earlier progress | O(N), no Hessian | **batch only** (needs line searches); good for small regression problems |
| **Quasi-Newton / BFGS** (Eqs. 44–45) | Builds up H⁻¹ from gradients | O(N²) memory | small nets only |
| **Gauss–Newton / Levenberg–Marquardt** (Eqs. 46–47) | H ≈ JᵀJ (+ λI) | O(N³) | squared error only; small nets |

**Conclusion:** *"Classical second-order methods are impractical in almost all useful cases."* (Paper 005's Iris experiment used exactly these methods, on a tiny 35-weight net, where they do work.)

---

## 6. Getting Hessian information cheaply (pages 24–27)

- **Finite differences** (Section 7.1): one row of H = (gradient at w + δ − gradient at w) / δ.
- **Gauss–Newton approximation** (Section 7.2, Eq. 51): H ≈ Σ Jᵀ J, which is always positive semi-definite.
- **Diagonal Hessian by backprop** (Section 7.4, Eqs. 54–56): same as backprop, but with **squared** derivatives and **squared** weights:
  ```
  ∂²E/∂y² = ∂²E/∂o² · f'(y)²
  ∂²E/∂w_ki² = ∂²E/∂y_k² · x_i²
  ∂²E/∂x_i² = Σ_k ∂²E/∂y_k² · w_ki²
  ```
  It costs about the same as one backprop pass. (Also used in "Optimal Brain Damage" pruning.)
- **Hessian × vector without the Hessian** (Eq. 59): Hψ ≈ [∇E(w + αψ) − ∇E(w)] / α. That's just two gradients.

---

## 7. What the Hessian of a real network looks like (pages 27–29)

- **Eigenvalue spectrum** (Figures 19–20): a few small eigenvalues, many medium ones, and a **few huge ones** ("big killers"). The huge ones set the safe learning rate, so they slow everything down.
- **Causes:** non-zero-mean inputs or unit outputs, large differences in curvature between layers, and correlated variables.
- **Layers differ** (Figure 21): the curvature is **smaller in lower layers**, so lower layers learn slowly and upper layers fast (even oscillating). Hence: give lower layers **larger** learning rates.

---

## 8. Second-order methods that do work for big networks (pages 29–35)

### 8.1 Stochastic diagonal Levenberg–Marquardt (Section 9.1, Eqs. 61–62)
Each weight gets its own learning rate from its own curvature:
```
η_ki = η / (⟨∂²E/∂w_ki²⟩ + μ)
```
- ⟨·⟩ is a running average of the diagonal Hessian (Section 6 above), re-estimated every few epochs on a subset.
- μ stops the rate exploding where the curvature is near 0.
- The extra cost is negligible. Rule of thumb: **about 3× faster** than well-tuned SGD.

### 8.2 The largest eigenvalue, to set η automatically (Section 9.2, Eqs. 60, 63–64)
- **Power method:** repeat ψ ← Hψ / ‖ψ‖, using Eq. 59 for Hψ. Then ‖ψ‖ → λ_max.
- **On-line version** (Eq. 64): a running average over single examples. It gets the right order of magnitude in under 100 examples (Figure 24).
- Then set **η = 1/λ_max**. Figure 25 shows this "predicted optimal rate" is close to the best rate found by trial.

---

## 9. The recipe (page 35, Section 10)

1. **Shuffle** the examples.
2. **Center** each input (subtract the mean).
3. **Scale** each input to standard deviation 1.
4. **Decorrelate** the inputs if possible.
5. Use the sigmoid **1.7159 tanh(2x/3)**.
6. Use targets **+1 and −1**.
7. **Initialize** with standard deviation m^(−1/2).
8. **Large, redundant classification data:** use **stochastic gradient** (carefully tuned) or **stochastic diagonal LM**.
9. **Small data or regression:** use **conjugate gradient**.

---

## 10. What our reproduction found

See [results.md](results.md) and [figures/](figures/). Every number comes from running the code.

### ✅ Confirmed
- **Backprop, the diagonal Hessian and the power method are exact:** they match finite differences, the Gauss–Newton diagonal, and the true λ_max (4.267 vs 4.267).
- **The recommended sigmoid:** f(±1) = ±1, and its curvature peaks at x ≈ 1.
- **LeCun initialization:** 400 unit-variance inputs → weighted sums with standard deviation 1.00.
- **The paper's data:** covariance eigenvalues 0.84 and 0.036 (we get 0.805 and 0.030).
- **Non-centred inputs are poison:** shifting the inputs by +3 raises the condition number from **33 to 2,096**, and learning at the best safe rate needs **5,888 epochs instead of 4**.
- **Stochastic ≫ batch on redundant data:** MSE after 5 epochs was **0.033 vs 0.827**.
- **Tricks on real digits** (test accuracy after 3 epochs, best learning rate per row, 3 seeds):

  | configuration | accuracy |
  |---|---|
  | full recipe | **93.2%** |
  | inputs not normalized | 75.2% |
  | logistic sigmoid (0/1) | 78.5% |
  | initial weights too big (±3) | 66.2% |

- **The Hessian:**
  - lower-layer curvature < upper-layer curvature (0.20 vs 0.58), matching Figure 21;
  - raw inputs create a few "big killer" eigenvalues: 1st/11th = **17×**, vs 1.8× with normalized inputs.
- **Stochastic diagonal LM** (tuned) beats the best plain SGD: it reaches SGD's 6-epoch error in about **3 epochs, about 2× faster** (the paper says ~3×).

### ⚠️ Different from the paper
1. **The paper's learning-rate limit for its own example is wrong.**
   - It uses 2/λ_max = 2/0.84 = **2.38**, but the network has a **bias**. The bias acts as an always-1 input, which adds a Hessian eigenvalue of **1.00** (Eq. 29 applies to it too).
   - So the real limit is **2.00**. We checked: **η = 2.1 and 2.3 diverge**, although the paper's bound says they're safe.
2. **Two warnings didn't show up in our short runs:**
   - targets at the asymptotes (95.3%);
   - tiny initial weights (93.7%).

   Both did as well as the full recipe. The paper's reasons (saturation, flat regions) are real, but they bite in deeper nets and longer training; a 1-hidden-layer net trained for 3 epochs is too easy to show them. (Paper 004's 5-layer net did stall with small weights.)
3. **The "predicted optimal learning rate" 1/λ_max is only a ballpark:**
   - **Batch:** 1/λ_max is safe and reasonable, but 2–3× larger did better within 50 updates. The curvature measured at the start shrinks as training moves on.
   - **Stochastic:** 1/λ_max was **about 10× too large**: at 0.1× the error was 0.35, at 1× it was 4.1. Single-example gradients are much noisier than the average, so SGD needs a smaller rate.
   - The paper's Figure 25 claims ratio 1 is "optimal enough" in both its networks. We couldn't reproduce that for stochastic learning.
4. **The on-line eigenvalue estimate (Eq. 64) is noisy.** After 100 examples it read 17.6 against a true 10.3; the average of the last 500 was 9.2. The paper also had to tune the averaging rate γ to calm it down.

**Data note:** the paper used its own handwritten-digit sets, which aren't available. We used scikit-learn's 8×8 digits (1,797 images).

---

## 11. Check yourself

1. Why is stochastic learning faster on redundant data?
2. Why does a non-zero input mean slow learning? Explain it with eigenvalues.
3. What's special about 1.7159·tanh(2x/3) and the targets ±1?
4. What standard deviation should initial weights have, and why?
5. What happens with η = 1.5/λ_max? With 2.5/λ_max?
6. How do you get the Hessian's largest eigenvalue without computing the Hessian?
7. Why did the paper's example diverge at η = 2.1 in our test?
