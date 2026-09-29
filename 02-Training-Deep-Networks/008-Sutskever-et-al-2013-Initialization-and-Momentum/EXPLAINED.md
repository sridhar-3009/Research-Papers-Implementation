# Sutskever, Martens, Dahl & Hinton (2013), explained simply

**Paper:** *On the importance of initialization and momentum in deep learning*
**Authors:** Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton
**Published in:** ICML 2013, JMLR W&CP volume 28
**Page numbers** below are the PDF's pages (1–9 plus appendix).

---

## The big idea in one line

> **Deep and recurrent networks were thought to need fancy second-order optimizers. They don't: plain SGD works if you start from a good random initialization and use strong momentum that grows during training, especially Nesterov momentum.**

---

## How it connects to the earlier papers

- **Paper 004** introduced momentum (Eq. 9 there).
- **Paper 006** said momentum helps mostly in batch mode, and was sceptical about it for SGD.
- **Paper 007** fixed initialization for feedforward nets.
- **This paper** shows that **initialization + momentum** together are what make deep nets and RNNs trainable with SGD, matching Hessian-Free optimization (a second-order method).

---

## 1. The claim (pages 1–2)

- Hessian-Free (HF) optimization trained deep autoencoders and RNNs that SGD supposedly couldn't.
- The authors show the gap closes when you use:
  1. a **well-designed random initialization**, and
  2. **momentum** with a **slowly increasing** coefficient µ.
- Why was momentum under-rated? Earlier theory studied the **final** phase of learning, where noise dominates and momentum doesn't help. In deep learning, the **early "transient" phase** (moving far across the error surface) takes most of the time, and there momentum helps a lot.

---

## 2. Two kinds of momentum (pages 2–3)

### Classical momentum (CM), Eqs. (1)–(2)
```
v_{t+1} = µ v_t − ε ∇f(θ_t)
θ_{t+1} = θ_t + v_{t+1}
```
The velocity v builds up in directions where the gradient keeps pointing the same way, which are exactly the **low-curvature** directions that plain gradient descent crawls along.

### Nesterov's accelerated gradient (NAG), Eqs. (3)–(4)
```
v_{t+1} = µ v_t − ε ∇f(θ_t + µ v_t)       ← gradient at the LOOK-AHEAD point
θ_{t+1} = θ_t + v_{t+1}
```
- The only difference: NAG measures the gradient **after** first taking the momentum step.
- If the momentum step overshoots, the look-ahead gradient points back, and NAG **corrects sooner** (Figure 1).

### Theorem 2.1: the precise difference
On a quadratic, along each direction with curvature λ, **NAG = CM with momentum µ(1 − λε)**.
- **Low curvature** (small λ): the same as CM, so full acceleration.
- **High curvature** (large λ): smaller momentum, so fewer oscillations.
- When ε is tiny, the two are the same.

---

## 3. Deep autoencoders (pages 4–5)

**Task:** reconstruct the input through a narrow middle layer (e.g. 784-1000-500-250-30-250-500-1000-784 for MNIST). These are 7–11 layer networks, and a standard benchmark.

**Setup:**
- sigmoid units;
- **sparse initialization** (see Section 4 below);
- 750,000 updates of minibatch 200;
- no regularization;
- the **training** error is reported (this paper is about optimization).

### The momentum schedule, Eq. (5)
```
µ_t = min(1 − 2^(−1 − log₂(⌊t/250⌋ + 1)),  µ_max)
```
In plain words: µ = 1 − 1/(2(k+1)), where k = ⌊t/250⌋. So µ goes 0.5, 0.75, 0.83, 0.875, … toward 1, capped at µ_max ∈ {0, 0.9, 0.99, 0.995, 0.999}.

### Table 1 (squared error, lower is better)

| task | SGD | NAG 0.9 | NAG 0.99 | NAG 0.995 | NAG 0.999 | CM 0.9 | CM 0.99 | CM 0.995 | CM 0.999 | HF |
|---|---|---|---|---|---|---|---|---|---|---|
| Curves | 0.48 | 0.16 | 0.096 | 0.091 | **0.074** | 0.15 | 0.10 | 0.10 | 0.10 | 0.058 |
| MNIST | 2.1 | 1.0 | **0.73** | 0.75 | 0.80 | 1.0 | 0.77 | 0.84 | 0.90 | 0.69 |
| Faces | 36.4 | 14.2 | 8.5 | 7.8 | **7.7** | 15.3 | 8.7 | 8.3 | 9.3 | 7.5 |

- **Bigger µ_max is better.**
- **NAG beats CM**, especially at µ = 0.995 and 0.999.
- **NAG gets close to HF.**

### Table 2: lower µ at the end
Dropping µ to 0.9 for the last 1,000 updates helps. MNIST: 1.20 → 0.73. **High momentum** is for travelling far, **low momentum** for fine settling. But don't lower it too early.

---

## 4. Initialization (page 5)

### Sparse initialization (SI), from Martens (2010)
- Each unit gets exactly **15** non-zero incoming weights, drawn from N(0, 1). All the others are 0.
- The total input doesn't grow with the layer's size, so units don't saturate, and each unit responds to a different small set of inputs.

### Table 3: the scale matters
Multiplying the SI weights by:

| multiplier | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| error (Curves) | 16 | 16 | **0.074** | 0.083 | 0.35 |

- **Too small** is catastrophic: it never learns.
- **2×** is fine.
- **4×** is noticeably worse.

---

## 5. Recurrent networks (pages 5–7)

### Echo-state-style initialization (Section 4.1, Table 4)
- **Hidden-to-hidden:** sparse (15 per unit), rescaled so the **spectral radius** (largest |eigenvalue|) is **1.1**.
  - Much below 1, the network **forgets** its input quickly.
  - Much above 1, it becomes chaotic and gradients **explode**.
  - Just above 1 keeps memory without exploding.
- **Input-to-hidden:** small, N(0,1)·0.001 when there are many irrelevant inputs (otherwise 0.1).
- **Centre** the inputs and outputs.

### Tasks
These are Hochreiter & Schmidhuber's long-range problems (Paper 021, the LSTM). For example, the **addition problem**: sequences of T random numbers, two of them marked, and the target is **their sum**, given at the end. The network must remember across up to 80 steps.

### Table 5 (error rate, lower is better; T = 80 for addition)

| | biases only | µ = 0 | NAG 0.9 | NAG 0.98 | NAG 0.995 | CM 0.9 | CM 0.98 | CM 0.995 |
|---|---|---|---|---|---|---|---|---|
| addition | 0.82 | 0.39 | 0.02 | 0.21 | **0.00025** | 0.43 | 0.62 | 0.036 |

- **Plain SGD RNNs can solve long-range tasks** with this initialization and strong **Nesterov** momentum.
- This was widely believed impossible without LSTMs or second-order methods.

---

## 6. Momentum and Hessian-Free (pages 7–8)
- HF optimizes a quadratic approximation with conjugate gradient (CG), starting each time from the previous solution. That warm start acts like momentum.
- A 1-step-CG version of HF is basically NAG with a curvature-based learning rate.
- So HF **is itself partly a momentum method**.

---

## 7. What our code found

**Scale note:** at your request, the heavy experiments (autoencoder, RNN) were **not run** on this laptop. `experiments.py` runs them on a strong machine. Compare against Tables 1–5 above.

**Checked exactly (tiny computations, done before the pause):**
- **Theorem 2.1 is exact:** NAG on a quadratic equals CM with µ(1 − λε) in each eigendirection, to within **2×10⁻¹⁵**.
- **Our CM and NAG code** gives exactly the iterates of Eqs. (1)–(4).
- **The schedule of Eq. (5):** 0.5, 0.75, 0.833, 0.875, 0.9, …, 0.999.
- **Sparse init** gives exactly 15 weights per unit; the RNN init has spectral radius exactly **1.100**.
- **NAG oscillates much less than CM** (the Figure 2 idea): on a valley with curvatures 1 and 100 (ε = 0.01, µ = 0.9), CM zig-zags **35** times and NAG **once**.

**A finding the paper doesn't mention:**
- When **λε > 1** in some direction, NAG's effective momentum µ(1 − λε) turns **negative**, and NAG can **diverge where CM still converges**.
- Example: ε = 0.015, curvature 100 (λε = 1.5), µ = 0.95. CM converges, while NAG blows up to 10¹².
- So "NAG is more stable" holds only while the learning rate is below 1/λ_max. Pick ε with Paper 006's rule in mind.

**Scaling choice in `experiments.py`:** the paper ran 750,000 updates. A shorter run never lets Eq. (5)'s µ climb to 0.99 or 0.995 (it rises one step every 250 updates). So the code **compresses the schedule's time axis** in proportion to the run length. µ still climbs to about 0.996 by the end.

---

## 8. Check yourself

1. Write CM and NAG. What is the ONE difference?
2. Use Theorem 2.1 to explain why NAG oscillates less in steep directions.
3. Why does the paper say momentum was under-rated before?
4. What does the schedule of Eq. (5) do early vs late in training?
5. Why is a spectral radius of 1.1 a good choice for RNN weights?
6. When can NAG be *worse* than CM?
