# Sutskever, Martens, Dahl & Hinton (2013), explained from scratch

**Paper:** *On the importance of initialization and momentum in deep learning*
**Authors:** Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton
**Published in:** ICML 2013, JMLR W&CP volume 28
**Page numbers** below are the PDF's pages (1–9 plus appendix).

Read Paper 006 (curvature, learning rates) first. This guide analyses momentum exactly on a 1-D parabola. That is enough to see why it gives a huge speed-up, why Nesterov is gentler, and when Nesterov can blow up.

---

## 0. The whole idea in one line

> **Deep and recurrent networks were thought to need fancy second-order optimizers. They don't: plain SGD works if you start from a good random initialization and use strong momentum that grows during training, ideally Nesterov momentum.**

---

## 1. The claim (pages 1–2)

- **The belief at the time:**
  - Hessian-Free (HF) optimization, a second-order method (Paper 023), trained deep autoencoders and RNNs that SGD supposedly couldn't.
  - Momentum was thought to help mostly in batch mode (Paper 006 was sceptical about it for SGD).
- **The authors' claim:** the gap closes with:
  1. a **well-designed random initialization**, and
  2. **momentum** whose coefficient µ **grows slowly** toward 1.
- **Why momentum was under-rated:**
  - Classic theory studied the **final** phase of learning, where gradient noise dominates and momentum can't help.
  - In deep learning, most of the time is spent in the early **transient** phase: travelling far across a badly conditioned error surface. That is where momentum shines.

---

## 2. Two kinds of momentum (pages 2–3)

### 2.1 Classical momentum (CM), Eqs. (1)–(2)
```
v_{t+1} = µ v_t − ε ∇f(θ_t)          velocity: a decaying sum of past gradients
θ_{t+1} = θ_t + v_{t+1}
```

### 2.2 Nesterov's accelerated gradient (NAG), Eqs. (3)–(4)
```
v_{t+1} = µ v_t − ε ∇f(θ_t + µ v_t)  ← the gradient at the LOOK-AHEAD point
θ_{t+1} = θ_t + v_{t+1}
```
- **The only difference:** NAG first imagines taking the momentum step µv_t, then measures the gradient **there**.
- **Why that helps:** if the momentum step is about to overshoot the valley floor, the look-ahead gradient already points back, so NAG brakes **one step earlier** (Figure 1).

---

## 3. The math of momentum on a parabola

Everything below takes one direction with curvature λ: f(θ) = ½ λ θ², so ∇f = λθ, and the minimum is at θ = 0. (Paper 006 showed that any quadratic splits into such independent directions.)

### 3.1 Plain gradient descent (µ = 0)
θ_{t+1} = (1 − ελ) θ_t. The distance shrinks by |1 − ελ| per step.
- Across many directions, the best single ε gives a rate **(κ − 1)/(κ + 1)**, where **κ = λ_max/λ_min** is the condition number.
- **Example, κ = 100:** the rate is 99/101 = 0.980 per step. Reaching 1% of the starting distance takes ln(0.01)/ln(0.980) ≈ **230 steps**.

### 3.2 Classical momentum is a two-step recurrence
- **Substitute** v_{t+1} = θ_{t+1} − θ_t into Eq. (1):
  ```
  θ_{t+1} = θ_t + µ(θ_t − θ_{t−1}) − ελ θ_t = (1 + µ − ελ) θ_t − µ θ_{t−1}
  ```
- **Solve it:** try θ_t = rᵗ. That gives the characteristic equation
  ```
  r² − (1 + µ − ελ) r + µ = 0
  ```
- **The distance shrinks like |r|ᵗ.** When the two roots are complex (an "underdamped" oscillation), both have size |r| = √µ, since the product of the roots is µ. So the speed is **set by µ, not by λ**: every direction with complex roots converges at the same rate √µ.
- **The best tuning** (the "heavy-ball" method, Polyak 1964) chooses ε and µ for the extreme curvatures λ_min and λ_max:
  ```
  µ* = ((√κ − 1)/(√κ + 1))²,      rate = (√κ − 1)/(√κ + 1)
  ```
- **Example, κ = 100:** the rate is 9/11 = 0.818 per step. Reaching 1% takes ln(0.01)/ln(0.818) ≈ **23 steps**, versus 230 for plain GD.
- **Momentum replaces κ by √κ.** That is a **10× speed-up** here, and the gain grows with how badly conditioned the problem is. Deep networks are very badly conditioned.

### 3.3 Theorem 2.1: NAG = CM with smaller momentum in steep directions (exact proof)
- **NAG's velocity update** on f = ½λθ²:
  ```
  v_{t+1} = µ v_t − ελ(θ_t + µ v_t) = µ(1 − ελ) v_t − ελ θ_t
  ```
- That is **CM's update with µ replaced by µ′ = µ(1 − ελ)**. ∎
- **What it means:**
  - **Flat directions** (ελ ≪ 1): µ′ ≈ µ, the full acceleration.
  - **Steep directions:** µ′ is smaller, so there's less overshoot and fewer oscillations.
  - **Tiny ε:** NAG and CM coincide.
- **Our demo** (µ = 0.9, ε = 0.01):

  | curvature λ | NAG's µ′ |
  |---|---|
  | 1 | 0.891 |
  | 10 | 0.810 |
  | 50 | 0.450 |
  | 100 | 0.000 |

### 3.4 Our extra finding: when NAG can blow up
- **When ελ > 1, µ′ = µ(1 − ελ) becomes negative.** Take ε = 0.015, λ = 100 (so ελ = 1.5) and µ = 0.95. Plugging into section 3.2's recurrence gives:
  - **CM:**
    - θ_{t+1} = (1 + 0.95 − 1.5)θ_t − 0.95θ_{t−1} = 0.45θ_t − 0.95θ_{t−1};
    - the roots are complex with |r| = √0.95 ≈ 0.975 < 1, so it **converges**.
  - **NAG** (µ′ = −0.475):
    - θ_{t+1} = (1 − 0.475 − 1.5)θ_t + 0.475θ_{t−1} = −0.975θ_t + 0.475θ_{t−1};
    - the roots of r² + 0.975r − 0.475 = 0 are 0.357 and **−1.332**;
    - |−1.332| > 1, so it **diverges**.
- **Our demo:** CM ends 0.66 from the minimum, and NAG ends at **1.4 × 10¹²**.
- So "NAG is more stable" holds only while ε < 1/λ_max. Choose ε with Paper 006's rule.

### 3.5 Figure 1/2's picture, measured
- **Our test:** a valley with curvatures 1 and 100 (ε = 0.01, µ = 0.9), 100 steps.
- **CM** zig-zags across the valley **35** times and ends 0.026 from the minimum.
- **NAG** zig-zags **once** and ends 0.004 away.

---

## 4. Deep autoencoders (pages 4–5)

**The task:** reconstruct the input through a narrow bottleneck. For MNIST: 784-1000-500-250-**30**-250-500-1000-784. These 7–11-layer nets were a standard "hard optimization" benchmark.

**Setup:**
- sigmoid units and **sparse initialization** (section 5);
- 750,000 updates with minibatch 200, no regularization;
- the **training** error is reported (this is an optimization study).

### 4.1 The momentum schedule, Eq. (5)
```
µ_t = min(1 − 2^(−1 − log₂(⌊t/250⌋ + 1)),  µ_max)
```
- **Simplify:** with k = ⌊t/250⌋, 2^(−1−log₂(k+1)) = 1/(2(k+1)), so **µ = 1 − 1/(2(k+1))**.
- **Values:**
  - k = 0, 1, 2, 3 give µ = 0.5, 0.75, 0.833, 0.875;
  - k = 4 (update 1,000) gives 0.9;
  - at update 50,000, µ = 0.9975;
  - it is capped at µ_max ∈ {0, 0.9, 0.99, 0.995, 0.999}.
- **Why grow µ slowly:**
  - Early on, the gradients change direction quickly (the landscape is unexplored), and high momentum would carry stale directions.
  - Later, the steps follow long consistent valleys, where high momentum pays off. (By section 3.2, the effective "memory" is about 1/(1 − µ) steps: 2 at the start, 1,000 at µ = 0.999.)

### 4.2 Table 1 (squared reconstruction error; lower is better)
| task | SGD | NAG 0.9 | NAG 0.99 | NAG 0.995 | NAG 0.999 | CM 0.9 | CM 0.99 | CM 0.995 | CM 0.999 | HF |
|---|---|---|---|---|---|---|---|---|---|---|
| Curves | 0.48 | 0.16 | 0.096 | 0.091 | **0.074** | 0.15 | 0.10 | 0.10 | 0.10 | 0.058 |
| MNIST | 2.1 | 1.0 | **0.73** | 0.75 | 0.80 | 1.0 | 0.77 | 0.84 | 0.90 | 0.69 |
| Faces | 36.4 | 14.2 | 8.5 | 7.8 | **7.7** | 15.3 | 8.7 | 8.3 | 9.3 | 7.5 |

- **Bigger µ_max is better.** Going from no momentum to µ ≈ 0.99 reduces the error 3–5×.
- **NAG beats CM,** especially at µ = 0.995–0.999, where CM's oscillations in steep directions hurt.
- **NAG nearly matches HF.**

### 4.3 Table 2: lower µ at the end
Dropping µ to 0.9 for the last 1,000 updates helps (MNIST: 1.20 → 0.73). **High momentum is for travelling, low momentum for settling.** Lowered too early, it hurts.

---

## 5. Initialization (page 5)

### 5.1 Sparse initialization (SI), from Martens (2010)
- Each unit gets exactly **15** non-zero incoming weights from N(0, 1). All the rest are 0.
- **Why:**
  - **No saturation:** with dense N(0, 1) weights on 784 inputs, the input sum would have variance ~784 and saturate every sigmoid. With 15, it's ~15 regardless of layer width.
  - **Diversity:** different units look at different small sets of inputs.

### 5.2 Table 3: the scale matters
Multiplying the SI weights by:

| multiplier | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| error (Curves) | 16 | 16 | **0.074** | 0.083 | 0.35 |

- **Too small is catastrophic:** signals vanish through the layers (Paper 007), so the net never learns.
- **Too big hurts** (saturation).

---

## 6. Recurrent networks (pages 5–7)

### 6.1 Why the spectral radius matters
- **A linear RNN:** ignoring inputs, h_t = W h_{t−1}, so h_t = Wᵗ h_0.
- **Eigen-directions:** along the eigenvector with eigenvalue ρ, the component is multiplied by ρ each step.
- **The spectral radius** (the largest |eigenvalue|) decides the fate of a memory over T = 80 steps:
  - **ρ = 0.9:** 0.9⁸⁰ ≈ 0.0002, so the RNN **forgets** within a few dozen steps;
  - **ρ = 1.1:** 1.1⁸⁰ ≈ 2,000, so in a linear net it **explodes**. A tanh RNN's saturation reins this in, leaving rich, long-lasting "echoes".
- **Gradients through time** use the same products (Wᵀ)ᵗ. So the gradients vanish if ρ < 1 and explode if ρ ≫ 1. **Just above 1 is the sweet spot.**

### 6.2 Echo-state-style initialization (Section 4.1, Table 4)
- **Hidden-to-hidden:** sparse (15 per unit), then **rescaled to spectral radius 1.1**.
- **Input-to-hidden:** small, N(0, 1)·0.001 when there are many irrelevant inputs (0.1 otherwise).
- **Centre** the inputs and outputs (Paper 006's mean-subtraction argument).

### 6.3 The tasks
These are Hochreiter & Schmidhuber's long-range problems (Paper 021). For example, in the **addition problem**:
- the input is T random numbers in [0, 1], two of them marked;
- the target is **their sum**, given at the very end;
- the network must remember two values across up to 80 steps.

Our demo shows one example, with target 0.50 + 0.63 = 1.129.

### 6.4 Table 5 (error rate; addition with T = 80)
| | biases only | µ = 0 | NAG 0.9 | NAG 0.98 | NAG 0.995 | CM 0.9 | CM 0.98 | CM 0.995 |
|---|---|---|---|---|---|---|---|---|
| addition | 0.82 | 0.39 | 0.02 | 0.21 | **0.00025** | 0.43 | 0.62 | 0.036 |

**Plain RNNs trained by SGD solve 80-step dependencies** with this init and strong **Nesterov** momentum. This was widely believed to require LSTMs or HF.

---

## 7. Momentum inside Hessian-Free (pages 7–8)

- **How HF works:** it minimizes a local quadratic model with conjugate gradient (CG), **warm-starting** each CG run from the previous solution.
- **Why that is momentum:** the warm start carries over the previous update direction.
- **The paper's argument:** a 1-step-CG version of HF is essentially NAG with a curvature-based learning rate. So HF is itself partly a momentum method, which helps explain why the two perform similarly.

---

## 8. What our code found

**Scale note:** at your request, the heavy experiments (deep autoencoders, RNN tasks) were **not run** on this laptop. `experiments.py` runs them on a strong machine; compare with Tables 1–5.

**Checked exactly:**
- **Theorem 2.1:** NAG on a quadratic equals CM with µ(1 − λε) in each eigen-direction, to **2 × 10⁻¹⁵**.
- **The code's CM and NAG** produce exactly the iterates of Eqs. (1)–(4).
- **Eq. (5)'s schedule:** 0.5, 0.75, 0.833, 0.875, 0.9, …, 0.999.
- **The inits:** sparse init gives exactly 15 weights per unit; the RNN init has spectral radius exactly **1.100**.
- **NAG oscillates much less:** 35 zig-zags for CM vs 1 for NAG on a κ = 100 valley.

**A finding the paper doesn't mention:**
- When ελ > 1, NAG's effective momentum is **negative** and NAG can **diverge where CM converges**: CM ends 0.66 from the minimum, NAG at 10¹².
- The root calculation in section 3.4 shows why.

**Scaling choice in `experiments.py`:**
- The paper ran 750,000 updates.
- A shorter run would never let Eq. (5)'s µ reach 0.99+, because it rises once per 250 updates. So the code **compresses the schedule's time axis** in proportion to the run length. µ still reaches ≈ 0.996 by the end.

---

## 9. Check yourself

1. Write CM and NAG. What is the single difference?
2. Turn CM on f = ½λθ² into the recurrence θ_{t+1} = (1 + µ − ελ)θ_t − µθ_{t−1}.
3. Why is the convergence rate √µ when the roots are complex?
4. For κ = 100, compare the per-step rates of tuned GD and heavy-ball momentum. How many steps does each need for a 100× reduction?
5. Prove Theorem 2.1 in two lines.
6. Show that NAG with ε = 0.015, λ = 100, µ = 0.95 diverges.
7. Simplify Eq. (5) to µ = 1 − 1/(2(k + 1)). Why start low and grow?
8. Why is a spectral radius of 1.1 a good choice for RNN weights? What happens with 0.9 over 80 steps?
