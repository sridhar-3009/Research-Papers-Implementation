# Parallelized Stochastic Gradient Descent, explained simply

**Paper:** Martin A. Zinkevich, Markus Weimer, Alex Smola, Lihong Li (Yahoo! Labs), *Parallelized Stochastic Gradient Descent*, NeurIPS 2010.

**In one sentence:** to train on more data than one machine can read quickly, give each of k machines a random share of the data. Each runs ordinary SGD on its share **with no communication at all**, and at the end you **average** their parameter vectors. The paper proves why this simple one-shot averaging works and shows it speeds up training on a large e-mail dataset.

---

## 1. The problem in 2010
- **Data had outgrown single disks:** reading one 2 TB disk at 100 MB/s takes over 6 hours.
- **SGD is inherently sequential:** each step needs the previous parameters.
- **Earlier ways to parallelise it each had a cost:**

| Approach | Problem |
|---|---|
| **Multicore lock-free SGD** | needs very low latency between processors, so it doesn't work across machines (MapReduce) |
| **Distributed gradients** (compute gradients on each machine, aggregate every step) | many synchronisation rounds and passes over the data |
| **Average exact solutions per machine** (Mann et al.) | one round, but needs an expensive batch solver on each machine; reduces variance but not bias |

---

## 2. The algorithm (Algorithm 3, SimuParallelSGD)
```
T = m / k
randomly split the m examples, T per machine
on every machine i, in parallel:
    shuffle its examples; w_i = 0
    for t = 1..T:   w_i ← w_i − η ∂c_t(w_i)        (fixed learning rate η)
return v = (1/k) Σ_i w_i
```
- **One communication step:** a single average at the end, so **one MapReduce pass**.
- **Each machine only touches its own data,** so the data can stay local.
- **The loss:** regularised risk c_i(w) = (λ/2)‖w‖² + L(x_i, y_i, w·x_i), e.g. Huber, squared or logistic.

---

## 3. Why it works (the analysis)

### (1) A fixed-η SGD step is a contraction (Lemma 3)
- **The condition:** if η ≤ η* = 1/(max‖x‖·c* + λ), one step w ↦ w − η∇c_i(w) brings any two points closer by a factor of at least **(1 − ηλ)**.
- **The consequence:** the *distribution* of w converges **exponentially fast** to a unique **stationary distribution** D*_η (Theorem 8, measured with the Wasserstein / earth-mover's distance).
- **So every machine forgets its starting point,** and all machines end up sampling from the same distribution.

### (2) The stationary distribution is good
- **SGD with a fixed η never stops moving:** it keeps jittering around the optimum.
- **Its mean is nearly optimal (bias):** c(E[w]) − min c ≤ 2ηG² (Theorem 9).
- **Its spread is small (variance):** E‖w − w*‖² ≤ 4ηG²/((2 − ηλ)λ) = O(ηG²/λ) (Theorem 10).

### (3) Averaging k machines cuts the variance by k
- **The k machines are independent draws** from (nearly) D*_η, so their average has about **1/k** of the variance.
- **The bias is not reduced,** but it is already O(η).
- **Theorem 12** combines these: after T = O(log(k)/(ηλ)) steps per machine, the error is about O(ηG²/(√k·λ)) + O(ηG²).
- **Arbitrary precision:** halve η and roughly double T to halve the error.
- **Speed-up:** compared with sequential SGD, the same order of total computation gives a large **wall-clock** speed-up. With k ≈ 1/λ machines, roughly a factor of 1/λ.

### Worked example: averaging
- **Setup:** suppose each machine's final w is the optimum plus independent noise of variance σ² = 43 (our η = 0.5 case).
- **Averaging 10 machines** gives variance σ²/10 ≈ 4.3. We measured 4.19.

---

## 4. Experiments (proprietary Yahoo! e-mail spam data)
- **Data:**
  - 3,189,235 time-stamped e-mails, of which 2,508,220 are for training and the last 681,015 for testing;
  - binary features hashed into **2¹⁸** dimensions, about 313 active per e-mail;
  - unit-normalised.
- **Setup:**
  - Huber and squared loss;
  - **λ = 1e−3** (smooth, "easy") and **λ = 1e−6** (high variance, "hard");
  - **η = 1e−3**;
  - 1, 10 or 100 machines.
- **Measurement:** after each machine has seen n examples, average the models and report the objective and test RMSE, both **normalised so one full sequential pass = 1.0**.
- **Findings:**
  - **more machines give a better model for the same per-machine work** (wall-clock time);
  - parallel training needs slightly more total machine time;
  - **1 → 10 machines helps much more than 10 → 100;**
  - **the high-variance problem (λ = 1e−6) benefits more** from averaging;
  - squared loss, despite unbounded gradients, behaves like Huber.

---

## 5. Why it matters
- **One of the first parallel SGD algorithms with guarantees,** and one designed for **high-latency clusters** (MapReduce) rather than shared-memory multicore machines.
- **"Train independently, then average parameters"** reappears in modern methods:
  - **local SGD / federated averaging** (average every few steps instead of once);
  - **model soups** (averaging fine-tuned models);
  - **SWA / Polyak averaging** (averaging iterates to cut SGD noise).
- **Its analysis of fixed-step SGD as a Markov chain** with a stationary distribution influenced later work on SGD as sampling.

---

## 6. What our code found
- **Data:** synthetic data like the paper's: sparse binary features hashed into 512 dimensions (12 active, unit length), labels from a hidden linear model with noise. Huber loss.
- **Contraction:** with η = 0.5 ≤ η* = 0.999 and λ = 1e−3, two SGD chains fed the same examples from different starts shrink from distance 22.85 to 2.54 in 2,000 steps. That is a per-step factor of **0.99890 ≤ 1 − ηλ = 0.99950**, satisfying Lemma 3 (and faster in practice). The test checks the bound at every step.

**The stationary distribution** (200 chains, λ = 1e−3):

| η | Single chain c − min c | Spread E‖w − mean‖² | Average of 10 chains: spread (c − min c) |
|---|---|---|---|
| 1.0 | 0.124 | 93.1 | 8.97 (0.0136) |
| 0.5 | 0.060 | 43.5 | 4.19 (0.0064) |
| 0.25 | 0.036 | 23.6 | 2.28 (0.0064) |

- **The spread is ∝ η** (Theorem 10), and **averaging 10 chains divides it by ~10.**
- **Honest note:** at η = 0.25 the averaged error didn't fall further, because 3,000 steps weren't enough for the slower chains to forget their start. Smaller η needs a longer T, as the paper says.

**The paper's experiment** (200,000 training examples; values relative to one sequential pass over all data):

| λ | Machines | Objective after 500 / 2,000 per machine | Test RMSE after 500 / 2,000 |
|---|---|---|---|
| 1e−3 | 1 | 1.146 / 0.998 | 1.186 / 1.007 |
| 1e−3 | 10 | 0.991 / **0.850** | 1.110 / 0.938 |
| 1e−3 | 100 | 0.980 / 0.833 | 1.107 / 0.932 |
| 1e−6 | 1 | 1.497 / 1.033 | 1.238 / 1.016 |
| 1e−6 | 10 | 1.323 / **0.840** | 1.138 / 0.904 |
| 1e−6 | 100 | 1.315 / 0.817 | 1.134 / 0.891 |

- **More machines, better model, at equal per-machine work.**
- **1 → 10 machines helps far more than 10 → 100.**
- **The λ = 1e−6 problem gains more.** These are all the paper's findings.
- **With 2,000 examples each, 10 machines beat a single full pass over 200,000,** because averaging removes the noise of the final SGD iterate.
- **Honest difference:** here the parallel run needs *less* total work (20,000 vs 200,000 examples), whereas the paper found it needed slightly more machine time.

**`experiments.py`:**
- **E1:** machines × λ × loss × η;
- **E2:** spread vs η, averaging vs k, and halving η while doubling T;
- **E3:** comparison with averaging exact shard solutions and with synchronous mini-batch SGD (communication rounds);
- **E4:** real RCV1 text data with logistic loss.
- None were run here.

---

## 7. Check yourself

1. How many times do the machines communicate in SimuParallelSGD?
<details><summary>Answer</summary>Once: each machine sends its final parameter vector to be averaged. Everything else is local.</details>

2. What does it mean that an SGD step is a contraction with constant 1 − ηλ?
<details><summary>Answer</summary>For any two parameter vectors, applying the same update (same example) brings them closer: ‖φ(w) − φ(w′)‖ ≤ (1 − ηλ)‖w − w′‖. Repeated steps make the starting point irrelevant exponentially fast.</details>

3. Why doesn't SGD with a fixed learning rate converge to a single point?
<details><summary>Answer</summary>Each step uses one random example, so the update keeps jittering. The iterates converge in distribution to a stationary distribution around the optimum, with spread proportional to η.</details>

4. What does averaging k machines reduce, and what doesn't it reduce?
<details><summary>Answer</summary>It reduces the variance (spread) by about a factor of k, since the machines are independent. It does not reduce the bias, i.e. how far the stationary distribution's mean is from the optimum, which is O(η).</details>

5. If each machine's final w has spread 40 around the stationary mean, what spread does the average of 20 machines have?
<details><summary>Answer</summary>About 40/20 = 2 (variance divides by the number of independent machines).</details>

6. How can you make the error arbitrarily small?
<details><summary>Answer</summary>Decrease η (smaller bias and variance) and increase T accordingly, since contraction is slower with smaller ηλ. Halving η and roughly doubling T halves the error.</details>

7. Why is the method suited to MapReduce while lock-free multicore SGD isn't?
<details><summary>Answer</summary>MapReduce has high latency and limited bandwidth between machines. SimuParallelSGD needs no communication during training (one final reduce), while multicore methods share parameters continuously.</details>

8. How does it differ from averaging exact per-machine solutions (Mann et al.)?
<details><summary>Answer</summary>Each machine runs cheap SGD on a random subset instead of an expensive full-batch solver. The analysis shows both variance and bias are controlled (bias O(η)), whereas the earlier analysis gave variance reduction but no bias reduction.</details>

9. Why did λ = 1e−6 benefit more from more machines in the paper and in our toy?
<details><summary>Answer</summary>Weaker regularisation means a flatter objective and a wider stationary distribution (spread ∝ 1/λ), so there is more variance for averaging to remove.</details>

10. In our toy, 10 machines with 2,000 examples each beat one machine that read all 200,000. How?
<details><summary>Answer</summary>A single fixed-η SGD run ends at a noisy iterate, with spread around the optimum even after seeing all the data. Averaging 10 independent runs cancels much of that noise, giving a better point than one long run.</details>
