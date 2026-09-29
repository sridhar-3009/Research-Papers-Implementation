# Du et al. (2022), explained simply

**Paper:** *Perceptron: Learning, Generalization, Model Selection, Fault Tolerance, and Role in the Deep Learning Era*
**Authors:** Ke-Lin Du, Chi-Sing Leung, Wai Ho Mow, M. N. S. Swamy
**Published in:** Mathematics 2022, 10, 4730 ([doi:10.3390/math10244730](https://doi.org/10.3390/math10244730)), open access (CC BY 4.0)
**Page numbers** below are the paper's own ("x of 46").

---

## The big idea in one line

> **One neuron can only draw a straight line. Stack neurons in layers and train them with backpropagation, and they can learn almost anything. This paper reviews 70 years of ideas for doing that training well.**

**This is a review paper (a survey), not a new method.** It summarizes hundreds of other papers. So "implementing it" means building the main ideas it explains, and reproducing its one experiment (Section 12.1).

---

## How it connects to Paper 1

| Paper 1 (1943) | This paper |
|---|---|
| Neuron = on/off threshold unit | Same neuron, now called a **perceptron** (page 2) |
| Weights = whole-number connection counts | Weights = **any real number** |
| **You** set the weights by hand | The network **learns** the weights from examples |
| No learning at all | Learning is the whole topic |

---

## 1. Introduction: the history (pages 1–2)

| Year | What happened |
|---|---|
| 1943 | McCulloch & Pitts: the neuron as a threshold unit (Paper 1) |
| 1958 | **Rosenblatt: the perceptron**, the first neuron that **learns** |
| 1960 | Widrow & Hoff: **Adaline**, trained with **LMS** |
| 1969 | Minsky & Papert: a single perceptron **can't learn XOR**. Interest in neural nets dies for years |
| 1986 | Rumelhart, Hinton & Williams: **backpropagation** trains multi-layer networks. The comeback |
| 1989 | LeCun: the convolutional network (the start of deep learning) |

**Supervised learning:** you show the network an input **and the right answer**. It compares its output with the answer, and uses the **error** to adjust its weights.

---

## 2. Background (pages 2–4)

### The neuron (Eqs. 1–6)
```
net = w1·x1 + w2·x2 + ... + θ          (weighted sum + bias)
o   = φ(net)                           (activation function)
```
The **activation function φ** decides the output:

| Name | Output | Used for |
|---|---|---|
| Hard limiter (step) | 0 or 1 | the original perceptron |
| Logistic (sigmoid) | smooth, 0 to 1 | hidden layers |
| Tanh | smooth, −1 to 1 | hidden layers |
| Linear | just `net` | output layer for regression |

**Why smooth matters:** backpropagation needs to know "if I change this weight a little, how much does the output change?" A step function jumps, so that question has no answer. A smooth function does.

### Linear separability (Eq. 7)
One neuron splits the input space with a **straight line** (a flat plane in more dimensions): `w·x + θ = 0`.
- **AND, OR:** one line can split them → **linearly separable** → one neuron works.
- **XOR:** no single line can → **not separable** → one neuron fails.

### The goal (Eq. 10)
Make the **mean squared error (MSE)** small: the average of (right answer − network output)².

---

## 3. The single-layer perceptron (pages 4–7)

### 3.2 The perceptron learning rule (Eqs. 15–18): the first learning algorithm
Show examples one at a time. For each one:
1. Compute the output (0 or 1).
2. Error = right answer − output (so −1, 0 or +1).
3. **If wrong, nudge the weights:** `w = w + η · x · error`

That's it. If the answer should have been 1, the weights move **toward** that input; if it should have been 0, they move **away**.

**Perceptron convergence theorem:** if the data **can** be split by a line, this rule is **guaranteed** to find a line in a finite number of steps.
**But:** if it can't (XOR), the rule **never stops**. It keeps cycling forever.

**Fixes for inseparable data:**
- **Pocket algorithm:** keep the best weights seen so far "in your pocket."
- **Thermal perceptron:** make the updates smaller over time.

### 3.3 LMS / Adaline (Eqs. 19–21)
Same idea, one difference: the error uses the **raw sum** `net`, not the 0/1 output.
```
error = right answer − net
```
Now the rule is doing **gradient descent on the squared error**. So it settles down even when the classes overlap, at the line with the **smallest squared error**. This is the rule behind modern adaptive signal processing.

---

## 4. Multilayer perceptrons (MLP) and backpropagation (pages 7–11)

### 4.1 Layers (Eqs. 22–25)
Put neurons in layers: **input → hidden → output**. Each layer feeds the next.
A hidden layer can bend the straight line, so **XOR becomes solvable**.

### 4.2 Universal approximation
An MLP with **one hidden layer** and enough hidden neurons can approximate **any** continuous function as closely as you like.

### 4.3 Backpropagation (Eqs. 26–31): the key algorithm
Goal: find how much **each weight** contributes to the error, then nudge it the other way.
1. **Forward:** run the input through the network, get the output.
2. **Error** at the output: `right answer − output`.
3. **Backward:** pass the error back layer by layer with the **chain rule**. Each neuron gets its share of the blame (called **δ**, "delta").
4. **Update:** `weight change = −η × (blame of the neuron) × (input to that weight)`.

**η (the learning rate):** too small → slow, too big → jumps around.

### 4.4 Batch vs online
| Mode | Update after | Modern name |
|---|---|---|
| Batch | seeing **all** examples | gradient descent |
| Online | **each** example | stochastic gradient descent (SGD) |

Online is noisier, but that noise helps it escape bad spots, and it's much faster on big datasets.

### 4.5 Momentum (Eq. 32)
Add a bit of the **previous** step to the current one (usually 0.9 of it). Like a ball rolling downhill: it speeds up on long slopes and doesn't zig-zag as much.

### 4.6 Variance reduction (SAG, SVRG, SAGA)
Tricks that make SGD's noisy steps less noisy, so it converges faster at the end.

---

## 5. Generalization: working on new data (pages 11–13)

**Overfitting:** the network memorizes the training examples, **including their noise**, and does badly on new data. Too many weights or too few examples causes it.

**Bias–variance:**
- **Too simple** a model (high bias) can't fit the pattern.
- **Too flexible** a model (high variance) fits the noise.

**Three fixes:**

| Fix | How | Section |
|---|---|---|
| **Early stopping** | watch the error on held-out data, and stop when it starts going **up** | 5.2 |
| **Weight decay** | add `λ · Σ w²` to the error, which keeps weights small and the function smooth (Eq. 34) | 5.3 |
| **Training with noise** | add small noise to inputs/weights; it acts like weight decay | 5.3 |

**Choosing λ** (how much regularization): cross-validation, information criteria (AIC, BIC, MDL), or Bayesian methods (5.4).

---

## 6. Choosing the network size (pages 13–16)

- **Too small** → underfits. **Too big** → overfits.
- **Pruning** (start big, remove weights):
  - **Sensitivity-based:** remove the weights whose removal changes the error least.
  - **OBD / OBS** ("optimal brain damage / surgeon"): use second derivatives to estimate that change (Eq. 37).
  - **L1 regularization:** pushes useless weights to exactly zero.
- **Growing** (start small, add neurons): e.g. **cascade-correlation** adds hidden neurons one at a time.

---

## 7. Making backpropagation faster (pages 16–19)

| Problem / trick | Idea |
|---|---|
| **Flat-spot problem** (7.1) | when a sigmoid saturates (output near 0 or 1), its slope is ~0, so learning stalls |
| **Adaptive learning rates** (7.2) | change η during training: big at first, smaller later; or one η **per weight** (delta-bar-delta, SuperSAB, Quickprop) |
| **Weight initialization** (7.3) | start with small random weights, e.g. uniform in ±3/√(fan-in); random breaks the symmetry between neurons |
| **Adapt the activation** (7.4) | e.g. start almost linear and slowly make it curved |
| **RProp** (7.5) | ignore the gradient's **size**, use only its **sign**. Each weight has its own step: bigger while the sign stays the same, smaller when it flips. One of the best first-order methods |

---

## 8. Second-order methods: using curvature (pages 19–25)

**Gradient descent** only knows the **slope**. **Second-order** methods also estimate the **curvature** (the **Hessian**, H), so they know how **far** to step, not just which way.

| Method | Idea | Memory |
|---|---|---|
| **Newton** (8.1) | jump straight to the bottom of the local bowl: `step = −H⁻¹ g` | stores H |
| **Gauss–Newton** (8.1.1) | approximate H ≈ JᵀJ using only first derivatives (J = Jacobian) | stores H |
| **Levenberg–Marquardt, LM** (8.1.2) | `(JᵀJ + σI)` makes H always invertible. Big σ acts like gradient descent (safe), small σ like Gauss–Newton (fast). Adjust σ after each step (Eq. 48) | stores H |
| **BFGS** (8.2.1) | build up an estimate of H⁻¹ from how the gradient changes (Eq. 51) | stores H⁻¹ |
| **One-step secant, OSS** (8.2.2) | BFGS that forgets its matrix every step | vectors only |
| **Conjugate gradient, CG** (8.3) | each new direction doesn't undo the previous ones. Needs a line search (Eqs. 55–57) | vectors only |
| **Scaled CG, SCG** (8.3) | CG without a line search, with LM-style damping | vectors only |
| **Kalman filter / RLS** (8.4) | treat the weights as a hidden state and estimate them online | stores P×P |

**Trade-off:** second-order methods converge **10–100× faster** in epochs, but each step costs much more (a P×P matrix for P weights). And they **get stuck in local minima more often** than plain BP.

---

## 9. Other learning algorithms (pages 25–26)

- **EM**
- **Natural gradient:** steepest descent measured in a "curved" parameter space
- **Layer-wise training:** solve one layer at a time with linear equations
- **Parameter-wise training**
- **Fuzzy BP**
- **Chaotic BP**
- **Binary MLPs** (hard-limiter neurons)

---

## 10. Fault tolerance (pages 26–28)

**What if part of the network breaks** (in hardware, say)? People used to assume MLPs were naturally robust. **They're not**, unless you train for it.

- **Open-node fault** (10.1): a hidden neuron dies and its output is stuck at 0.
  - **Fix:** during training, **randomly break nodes** ("fault injection"). The network learns not to depend on any single node. (This is very close to **dropout**.)
  - Other fixes: weight decay (small weights), replicating nodes, or a special training objective.
- **Multiplicative weight noise** (10.2): weights are stored imprecisely. **Fix:** keep weights small, or add a sensitivity penalty (Eq. 63).

---

## 11. The perceptron in the deep learning era (pages 28–31)

- **Every deep network is built from the MLP's ideas.** SGD and backprop are still how everything is trained.
- **Deep training had three problems:** overfitting, the **vanishing gradient**, and heavy compute. They were fixed by:
  - **dropout** (for overfitting)
  - **ReLU** = max(0, x) (for vanishing gradients). It doesn't saturate for positive inputs, so the gradient passes through.
  - **GPUs** (for compute)
  - **data augmentation** (more training examples)
- **Why do huge networks not overfit?** It's not fully known. Explanations include implicit regularization by SGD, ReLU sparsity, and a bias toward simple functions.
- **Deep vs shallow:** one hidden layer **can** approximate anything, but deep networks often need **far fewer** neurons to do it.
- **Second-order methods don't scale** to millions of weights, so deep learning uses first-order SGD.

---

## 12. The experiment and conclusions (pages 31–34)

### 12.1 Iris classification: the experiment this folder reproduces
- **Data:** 150 flowers, 4 measurements each, 3 species. 80% train / 20% test.
- **Network:** 4 inputs → 4 hidden (logistic) → 3 outputs (linear). Target: +1 for the right species, −1 for the others.
- **Compared:** 8 training algorithms, 50 runs each, goal MSE 0.001, at most 1000 epochs.

**Paper's Table 1:**

| Algorithm | Mean epochs | Training MSE | Accuracy |
|---|---|---|---|
| RP (RProp) | 991 | 0.025 | 96.00% |
| **LM** | 239 | **0.007** | **100%** |
| **BFGS** | **155** | 0.015 | 93.33% |
| OSS | 1000 | 0.027 | 96.53% |
| SCG | 903 | 0.016 | 95.40% |
| CGB | 439 | 0.028 | 95.27% |
| CGF | 563 | 0.021 | 95.87% |
| CGP | 573 | 0.021 | 96.27% |

**Their conclusion:** LM and BFGS converge fastest, at the cost of more memory.

### 12.2 Advice: which algorithm to use
| Network size | Use |
|---|---|
| < 1,000 weights | **LM** |
| < 5,000 weights | **BFGS** |
| bigger | **Conjugate gradient** |
| deep learning (millions) | **first-order SGD**: the only practical choice |

### 12.3 Summary
The perceptron is still the core building block. For data that isn't images or signals (tables of features, e.g. network traffic), the authors argue an **MLP** is often a better fit than a CNN.

---

## 13. What our reproduction found

See [results.md](results.md) and [figure4_learning_curves.png](figure4_learning_curves.png).

**Matches the paper:**
- **LM and BFGS stop far earlier** than the others (~190–300 vs ~870–1000 epochs), just like Table 1.
- **RProp and OSS run the full ~1000 epochs**, also like Table 1.
- The **learning curves** have the same shape as Figure 4: LM drops fastest, then BFGS.
- Training MSEs land in the **same range** (0.007–0.04).

**Differs, and why:**
- **Our CG variants run all 1000 epochs**; the paper's stop at 440–570. MATLAB's CG line searches stop on their own minimum-step rules, which the paper doesn't describe.
- **Test accuracy depends mostly on which 30 flowers are in the test set.**
  - The paper's LM (100%) and BFGS (93.33%) have **std 0.000**, which means all 50 runs used **one fixed split**. We do the same by default.
  - With a different split you get different accuracies, so compare **epochs and MSE**, not exact accuracy.
- **Times differ:** the paper used MATLAB, and this is Python.

**A paper claim we confirmed:** "second-order methods become trapped in a local minimum more frequently than BP" (Section 8). On XOR with 20 random starts, plain BP succeeds 19 times, BFGS only 12.

**Small inconsistency in the paper:** Eq. (12) fires the perceptron at net ≥ 0, but Eq. (16) at net > 0. It only matters when net is exactly 0.

---

## 14. Check yourself

1. Why does the perceptron rule never stop on XOR?
2. What's the one difference between the perceptron rule and LMS?
3. Why does backpropagation need a smooth activation function?
4. What does momentum do, in one sentence?
5. Name two ways to prevent overfitting.
6. Why does LM take fewer epochs than RProp, and why isn't it used for deep learning?
7. How is "fault injection" similar to dropout?
