# Du et al. (2022), explained from scratch

**Paper:** *Perceptron: Learning, Generalization, Model Selection, Fault Tolerance, and Role in the Deep Learning Era*
**Authors:** Ke-Lin Du, Chi-Sing Leung, Wai Ho Mow, M. N. S. Swamy
**Published in:** Mathematics 2022, 10, 4730 ([doi:10.3390/math10244730](https://doi.org/10.3390/math10244730)), open access (CC BY 4.0)
**Page numbers** below are the paper's own ("x of 46").

This is a **review (survey)**: it summarizes hundreds of papers rather than proposing one method. "Implementing it" means building its main ideas and reproducing its one experiment (Section 12.1). This guide explains every idea from the ground up, with the math behind each one and small worked examples.

---

## 0. The whole idea in one line

> **One neuron can only draw a straight line. Stack neurons in layers, train them with backpropagation, and they can learn almost anything. The hard part is training them *well*: fast, without overfitting, and robustly. This survey reviews 70 years of ideas for that.**

---

## 1. History (pages 1–2)

| Year | Milestone |
|---|---|
| 1943 | McCulloch & Pitts: the neuron as a threshold unit (Paper 001) |
| 1958 | Rosenblatt: the perceptron, a neuron that **learns** (Paper 002) |
| 1960 | Widrow & Hoff: **Adaline** trained with **LMS** |
| 1969 | Minsky & Papert: a single perceptron can't do XOR; interest collapses (Paper 003) |
| 1986 | Rumelhart, Hinton & Williams: **backpropagation** trains multi-layer nets (Paper 004) |
| 1989 | LeCun: convolutional networks (Paper 013) |

**Supervised learning:** you show the network inputs **and the right answers**. It compares its output with the answer and uses the **error** to adjust its weights.

---

## 2. The single neuron (pages 2–4)

### 2.1 The equations (Eqs. 1–6)
```
net = w₁x₁ + w₂x₂ + … + wₙxₙ + θ        weighted sum plus a bias θ
o   = φ(net)                             an activation function
```
| φ | Formula | Range | Slope φ′ | Used for |
|---|---|---|---|---|
| hard limiter | 1 if net ≥ 0 else 0 | {0, 1} | 0 (undefined at 0) | the original perceptron |
| logistic | 1/(1 + e^(−net)) | (0, 1) | o(1 − o) | hidden layers |
| tanh | (e^net − e^(−net))/(e^net + e^(−net)) | (−1, 1) | 1 − o² | hidden layers |
| linear | net | all reals | 1 | regression outputs |

**Why smoothness matters:** gradient-based learning asks "if I nudge this weight, how does the output change?" That is the slope φ′. The hard limiter's slope is 0 almost everywhere, which gives no signal. Smooth φ's give useful slopes.

### 2.2 Linear separability (Eq. 7)
- The neuron's decision boundary is **w·x + θ = 0**: a straight line in 2-D, a flat hyperplane in more dimensions.
- **AND and OR** can be split by one line. **XOR** cannot. (See Paper 003 for the 4-line proof.)

### 2.3 The goal: mean squared error (Eq. 10)
```
MSE = (1/N) Σ_p ‖d_p − o_p‖²            average squared distance between target d and output o
```

---

## 3. Learning with one layer (pages 4–7)

### 3.1 The perceptron rule (Eqs. 15–18)
For each example (x, d), with d ∈ {0, 1}:
```
o = step(w·x)            (x includes a constant 1, so the bias is just another weight)
w ← w + η (d − o) x
```
- **Correct answer:** d − o = 0, so nothing changes.
- **Should be 1, said 0:** w moves **toward** x, so next time w·x is bigger.
- **Should be 0, said 1:** w moves **away** from x.

**Worked step:**
- w = (0, 0, −1) (the last entry is the bias), example x = (1, 1, 1) with d = 1, and η = 1.
- w·x = −1, so the output is 0. That's wrong.
- w ← (0, 0, −1) + 1·(1 − 0)·(1, 1, 1) = **(1, 1, 0)**.
- Now w·x = 2 ≥ 0, so the output is 1. ✔

**Convergence theorem:**
- **Statement:** if a separating line exists (with margin γ, and all inputs no longer than R), the rule makes **at most (R/γ)² mistakes**.
- **Why, in short:**
  - each mistake raises w's alignment with the true separator by at least γ;
  - each mistake raises ‖w‖² by at most R²;
  - after k mistakes the alignment is ≥ kγ, while ‖w‖ ≤ √k·R;
  - an alignment can't exceed the length, so kγ ≤ √k R, giving k ≤ (R/γ)².
- **If no line exists (XOR),** the rule cycles forever. Our demo: AND converges in 7 epochs, OR in 5, and XOR still makes 4 mistakes per epoch after 100.

**Fixes for inseparable data:**
- **Pocket algorithm:** keep the best weights seen so far "in your pocket". On XOR it reaches 3/4, the best any line can do.
- **Thermal perceptron:** shrink the updates over time.

### 3.2 LMS / Adaline (Eqs. 19–21): gradient descent on a straight line
The only change is that the error uses the **raw sum**, not the 0/1 output:
```
E = ½ (d − net)²,      net = w·x
∂E/∂w = −(d − net) · x                     (chain rule: ∂net/∂w = x)
w ← w − η ∂E/∂w = w + η (d − net) x
```
- **Why it's better behaved:** it really is **gradient descent** on a bowl-shaped (quadratic) error. So it settles at the **least-squares** solution even when the classes overlap.
- **Our demo** (OR with ±1 targets): LMS reaches w = (0.982, 0.975, −0.499), while the exact least-squares answer is (1, 1, −0.5). ✔
- This rule (Widrow–Hoff, 1960) is the core of adaptive filters, such as the echo cancellers in phones.

---

## 4. Multilayer perceptrons and backpropagation (pages 7–11)

### 4.1 Layers (Eqs. 22–25)
```
hidden:  h = φ(W₁ x + b₁)
output:  o = φ_out(W₂ h + b₂)
```
The hidden layer **re-describes** the input. In the new description, the problem can become linearly separable (XOR becomes separable once you have "x₁ AND x₂" as a feature).

### 4.2 Universal approximation: why one hidden layer is "enough"
- **The theorem** (Cybenko 1989; Hornik 1991): one hidden layer with enough units can approximate **any continuous function** on a bounded region as closely as you like.
- **Why: build bumps.**
  - A steep logistic σ(k(x − a)) is a smooth step at x = a.
  - The difference of two steps, **σ(k(x − a)) − σ(k(x − b))**, is ≈ 1 between a and b and ≈ 0 elsewhere: a **bump**.
  - A weighted sum of many narrow bumps approximates any continuous curve, like a histogram approximating a shape.
  - Each bump costs 2 hidden units.
- **The catch:** "enough units" can mean astronomically many. **Deep** networks can represent many functions with far fewer units (Section 11).

### 4.3 Backpropagation (Eqs. 26–31)
Exactly Paper 004's algorithm:
1. **Forward pass**.
2. Output error signal **δ_out = (o − d)·φ′(net_out)**.
3. **Backward:** δ_hidden = φ′(net_h) · W₂ᵀ δ_out (the chain rule).
4. **Update:** Δw = −η · δ · (that weight's input).

**On XOR (4 hidden units), our demo:**
- plain BP needs **4803 epochs**;
- with momentum, **353**;
- with Levenberg–Marquardt, **17**.

Section 8 explains the speed-ups.

### 4.4 Batch vs online
| Mode | Updates after | Modern name | Trade-off |
|---|---|---|---|
| batch | all examples | (full) gradient descent | exact gradient, slow per update |
| online | each example | stochastic gradient descent (SGD) | noisy, but many cheap updates; the noise can shake it out of bad spots |

### 4.5 Momentum (Eq. 32)
```
Δw(t) = −η ∇E(t) + α Δw(t−1)
```
It is a decaying average of past gradients. With a steady gradient the step grows to η/(1 − α) times the gradient (10× for α = 0.9), while zig-zag components cancel.

### 4.6 Variance reduction: SAG, SVRG, SAGA
- SGD's noise never fully disappears, so it can't converge exactly with a fixed step.
- These methods keep a memory of past per-example gradients and correct each new noisy gradient with it.
- The noise shrinks toward zero, so they converge fast at the end.

---

## 5. Generalization: working on new data (pages 11–13)

### 5.1 Overfitting and the bias–variance decomposition
- Suppose the data are y = f(x) + noise (noise variance σ²), and the model trained on a random dataset gives f̂(x).
- Averaged over datasets, the expected squared error at a point splits **exactly** into three parts:
  ```
  E[(y − f̂(x))²] = σ²                       (noise: unavoidable)
                 + (f(x) − E[f̂(x)])²        (bias²: the model is too simple on average)
                 + E[(f̂(x) − E[f̂(x)])²]     (variance: the model jumps around with the data)
  ```
- **Too simple** gives high bias; **too flexible** gives high variance. **Regularization** trades a little bias for a big drop in variance.

### 5.2 Early stopping
Watch the error on held-out data and stop when it starts to rise. Training starts from small weights, so stopping early keeps the weights small, which is a hidden form of weight decay.

### 5.3 Weight decay (Eq. 34)
```
E′ = E + (λ/2) Σ w²
∇E′ = ∇E + λ w
w ← w − η(∇E + λw) = (1 − ηλ) w − η ∇E          "decay": every step shrinks w by a factor (1 − ηλ)
```
- **Small weights make smooth functions.** A logistic unit with small weights stays in its near-linear middle region, so the network can't make sharp wiggles to chase noise.
- **Training with noise** on the inputs has a similar smoothing effect: for small noise it's equivalent to a penalty on the slope of the function.

**Our demo** (fitting a noisy sine wave with a big network):

| | train MSE | test MSE vs the true sine | weight norm |
|---|---|---|---|
| no regularization | 0.0000 | 1.4523 | 63.7 |
| weight decay 1e-3 | 0.0712 | **0.0139** | 4.4 |
| early stopping | 0.0600 | 0.0279 | 14.2 |

The unregularized network fits the noise perfectly and the truth badly.

### 5.4 Choosing λ
Options include cross-validation, information criteria (AIC, BIC and MDL, which penalize the number of parameters), and Bayesian evidence.

---

## 6. Choosing the network size (pages 13–16)

- **Pruning** (start big, remove weights):
  - **Sensitivity:** remove the weights whose removal raises the error least.
  - **Optimal Brain Damage / Surgeon (Eq. 37):**
    - Near a minimum the gradient is ≈ 0, so deleting weight w_i raises E by ≈ ½ H_ii w_i² (OBD, using the diagonal of the curvature matrix H).
    - OBS uses the full H and adjusts the other weights to compensate.
  - **L1 regularization** (λ Σ|w|) pushes useless weights to **exactly** zero. Its "pull" stays constant however small the weight gets, unlike L2's.
- **Growing** (start small): **cascade-correlation** adds hidden units one at a time, each trained to correlate with the remaining error.

---

## 7. Faster first-order training (pages 16–19)

| Problem / trick | The math |
|---|---|
| **flat spots** (7.1) | a saturated logistic has slope o(1 − o) ≈ 0, so learning stalls (Paper 004, section 11.1) |
| **adaptive learning rates** (7.2) | one rate per weight; grow it while successive gradients agree, shrink it when they disagree (delta-bar-delta, SuperSAB, Quickprop) |
| **initialization** (7.3) | small random weights, e.g. uniform in ±3/√(fan-in), so the sums start in the sigmoid's useful middle range; randomness breaks the symmetry between units |
| **RProp** (7.5) | use only the **sign** of each gradient component (below) |

**RProp in detail** (as in our code):
```
for each weight i, keep a step size Δᵢ:
   if gᵢ(t) · gᵢ(t−1) > 0:  Δᵢ ← 1.2 · Δᵢ     (same direction twice: speed up)
   if gᵢ(t) · gᵢ(t−1) < 0:  Δᵢ ← 0.5 · Δᵢ     (sign flipped: we jumped over a minimum, slow down)
   wᵢ ← wᵢ − sign(gᵢ) · Δᵢ
```
**Why it works:** gradient *magnitudes* vary wildly between layers (Paper 004's vanishing gradients), but the *sign* still says which way is downhill. RProp makes every weight move at its own sensible pace.

---

## 8. Second-order methods: using curvature (pages 19–25)

### 8.1 Newton's method: the key idea in one dimension
- **Approximate the error near w by a parabola** (a Taylor expansion):
  ```
  E(w + Δ) ≈ E(w) + g Δ + ½ h Δ²            g = slope E′(w), h = curvature E″(w)
  ```
- **Set its derivative to zero:** g + hΔ = 0, so **Δ = −g/h**.
- **Example:** E(w) = 3(w − 2)², starting at w = 0.
  - g = 6(0 − 2) = −12 and h = 6, so Δ = 2: Newton lands **exactly** at the minimum in one step.
  - Gradient descent with η = 0.1 steps only 1.2 and needs many steps.
- **Newton knows not just which way, but how far.**
- **In many dimensions:** Δ = −H⁻¹ g, where H is the P×P matrix of second derivatives (the **Hessian**).
- **Cost:** storing H takes P² numbers, and inverting it ~P³ operations. That's fine for 50 weights and impossible for 50 million.

### 8.2 Gauss–Newton (8.1.1)
- For a sum of squares E = ½ Σ r_k², where r_k are the residuals (errors) and J is their Jacobian (∂r_k/∂w):
  ```
  ∇E = Jᵀ r
  H  = JᵀJ + Σ r_k ∇²r_k  ≈  JᵀJ            (drop the second term: it's small when the errors are small)
  ```
- So the curvature comes from **first** derivatives only.

### 8.3 Levenberg–Marquardt (8.1.2): Eqs. 47–48
```
(JᵀJ + σ I) Δ = −Jᵀ r
```
- **σ → 0:** Gauss–Newton (fast near a minimum).
- **σ large:** Δ ≈ −(1/σ) Jᵀr, i.e. gradient descent with a small step (safe far from a minimum).
- **Adapting σ:**
  - **after a step that lowers E:** accept it and divide σ by 10, to be bolder;
  - **after a step that raises E:** reject it and multiply σ by 10, to be more careful.
- Our code starts at σ = 0.01.
- **It's the best of both worlds for small networks:** on XOR, 17 epochs vs 4803 for plain BP.

### 8.4 Quasi-Newton: BFGS (8.2.1, Eq. 51)
- **Don't compute H. *Learn* it from how the gradient changes.**
- After a step s = Δw that changes the gradient by y = Δg, a good curvature estimate B must satisfy the **secant condition** B s = y. (For a parabola, the change in slope = curvature × step.)
- BFGS updates B (or its inverse) by the smallest correction that satisfies this.
- **OSS** (one-step secant) is BFGS that forgets its matrix every step, so it needs no P×P storage.

### 8.5 Conjugate gradient (8.3, Eqs. 55–57)
- **The problem with steepest descent:** in a long narrow valley, each new gradient partly **undoes** the previous step, causing zig-zags.
- **The fix:** CG picks directions that are **conjugate**: dᵢᵀ H dⱼ = 0. Moving along a new direction doesn't spoil the progress made along the old ones.
- On a quadratic with P parameters, it reaches the exact minimum in **at most P steps**, using only vectors and no matrix.
- Each new direction is:
  ```
  d_new = −g_new + β d_old
  β (Fletcher–Reeves) = g_newᵀ g_new / gᵀ g
  β (Polak–Ribière)   = g_newᵀ (g_new − g) / gᵀ g
  ```
- **Each step needs a line search,** i.e. finding how far to go along d. **Scaled CG (SCG)** replaces the line search with LM-style damping.

### 8.6 Kalman filter / RLS (8.4)
Treat the weights as a hidden state, and each training example as a noisy measurement of it. Then estimate the weights online, with uncertainty, like tracking a moving object.

**The trade-off:**
- Second-order methods need **10–100× fewer epochs**, but each epoch costs more and memory grows like P².
- The survey also notes they **get stuck in local minima more often** than plain BP. We confirmed this: on XOR from 20 random starts, BP succeeded 19 times and BFGS 12. Their big, confident steps can jump straight into a nearby bad basin.

---

## 9. Other algorithms (pages 25–26)

- EM (expectation–maximization);
- **natural gradient** (steepest descent measured in a geometry where "distance" means how much the network's output distribution changes);
- layer-wise training with linear least squares;
- parameter-wise training;
- fuzzy BP;
- chaotic BP;
- binary MLPs.

---

## 10. Fault tolerance (pages 26–28)

**Myth:** "neural networks are naturally robust to broken parts". **They aren't**, unless trained for it.

### 10.1 Open-node faults
- **The fault:** a hidden unit dies (its output is stuck at 0).
- **The fix, fault injection:** during training, randomly kill nodes. The network learns to spread the work across units, so no single one is critical. This is essentially **dropout** (Paper 009–010).

**Our demo** (one dead hidden node, 5 runs):

| | healthy | one node dead: mean | worst |
|---|---|---|---|
| normal online BP | 100% | 91.9% | 65.3% |
| BP + fault injection | 100% | **99.6%** | **97.3%** |

### 10.2 Multiplicative weight noise
- **The fault:** weights are stored imprecisely, w → w(1 + ε).
- **Why small weights help:** the output error caused by the noise grows with the weights' size, so keeping them small (or penalizing the output's sensitivity, Eq. 63) helps.

---

## 11. The perceptron in the deep-learning era (pages 28–31)

**Every deep network is built from MLP ideas, trained by SGD and backprop.** Three obstacles to *deep* training, and their fixes:

| Obstacle | Fix | Why it works |
|---|---|---|
| overfitting | dropout, data augmentation | noise / more data reduce variance |
| vanishing gradients | **ReLU** = max(0, x) | slope is exactly 1 for x > 0, so gradients pass through unshrunk (unlike σ′ ≤ 0.25) |
| compute | GPUs | massively parallel matrix multiplies |

- **Why huge networks don't overfit** is not fully understood. Candidate explanations: SGD's implicit regularization, sparse ReLU activations, and a bias toward simple functions (see Paper 018).
- **Depth vs width:** some functions need exponentially many units in one hidden layer but only polynomially many with more layers.
- **Second-order methods don't scale** to millions of weights (P² memory), so deep learning uses first-order methods (SGD, Adam: Paper 011).

---

## 12. The experiment and advice (pages 31–34)

### 12.1 Iris classification
- **Data:** 150 flowers, 4 measurements, 3 species; 80% train / 20% test.
- **Network:** 4 → 4 (logistic) → 3 (linear). The targets are +1 for the correct species and −1 for the others.
- **Protocol:** 8 algorithms, 50 runs each, stop at MSE 0.001 or after 1000 epochs.

| Algorithm | Paper: epochs / MSE / accuracy | **Ours: epochs / MSE / accuracy** |
|---|---|---|
| RP (RProp) | 991 / 0.025 / 96.00% | 1000 / 0.040 / 98.40% |
| **LM** | 239 / 0.007 / 100% | **301 / 0.007** / 97.40% |
| **BFGS** | **155** / 0.015 / 93.33% | **194** / 0.020 / 98.93% |
| OSS | 1000 / 0.027 / 96.53% | 999 / 0.027 / 99.87% |
| SCG | 903 / 0.016 / 95.40% | 871 / 0.014 / 99.40% |
| CGB | 439 / 0.028 / 95.27% | 1000 / 0.042 / 100% |
| CGF | 563 / 0.021 / 95.87% | 1000 / 0.036 / 99.73% |
| CGP | 573 / 0.021 / 96.27% | 1000 / 0.040 / 100% |

### 12.2 The survey's advice
| Network size | Use |
|---|---|
| < 1,000 weights | **LM** |
| < 5,000 weights | **BFGS** |
| larger | **conjugate gradient** |
| deep learning (millions) | **first-order SGD** (and Adam-style methods) |

For tabular data (rows of features, not images or sound), the authors argue an MLP is often a better fit than a CNN.

---

## 13. What our reproduction found

See [results.md](results.md) and [figure4_learning_curves.png](figure4_learning_curves.png).

**Matches the paper:**
- **LM and BFGS stop far earlier** than the rest (~190–300 vs ~870–1000 epochs), as in Table 1.
- **RProp and OSS run the full ~1000 epochs.**
- **The learning curves have Figure 4's shape:** LM drops fastest, then BFGS.
- **Training MSEs are in the same range** (0.007–0.04).
- **"Second-order methods get trapped more often":** on XOR, BP 19/20 vs BFGS 12/20.

**Differs, and why:**
- **Our CG variants run all 1000 epochs** (the paper's stop at 440–570). MATLAB's line searches have their own minimum-step stopping rules, which the paper doesn't describe.
- **Test accuracy depends mostly on which 30 flowers are in the test set.** The paper's LM and BFGS accuracies have std 0.000, which means all 50 runs used one fixed split. We also use one fixed split, but ours is a different one, so compare **epochs and MSE**, not exact accuracy.
- **Times:** MATLAB vs Python. Compare them only within a column.

**A small inconsistency in the paper:** Eq. (12) fires at net ≥ 0, Eq. (16) at net > 0. It only matters when net is exactly 0.

---

## 14. Check yourself

1. Do one perceptron update by hand: w = (0, 0, 0), x = (1, 0, 1), d = 1, output = step(0) = 1. Does anything change? (No: the answer is already correct.)
2. Derive the LMS update from E = ½(d − w·x)².
3. Why does the perceptron rule cycle forever on XOR, while LMS settles down?
4. Explain, with two logistic units, how to make a "bump", and why bumps give universal approximation.
5. Write the bias–variance decomposition. Which term does weight decay reduce?
6. Show that weight decay multiplies w by (1 − ηλ) each step.
7. Newton on E = 3(w − 2)² from w = 0: how many steps to the minimum? Gradient descent with η = 0.1?
8. What does σ do in Levenberg–Marquardt, at its small and large extremes?
9. Why does RProp use only the sign of the gradient?
10. How is fault injection related to dropout?
