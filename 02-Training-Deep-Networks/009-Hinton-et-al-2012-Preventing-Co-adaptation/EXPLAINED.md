# Hinton et al. (2012), explained from scratch

**Paper:** *Improving neural networks by preventing co-adaptation of feature detectors*
**Authors:** Geoffrey Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
**Published:** arXiv:1207.0580, July 2012 (6 pages of main text plus appendices)

This is the **first dropout paper**; Paper 010 is the long journal version. This guide proves the paper's two mathematical claims step by step, and explains the training tricks with numbers.

---

## 0. The whole idea in one line

> **While training, randomly switch off half of the hidden units for every example. No unit can then rely on particular partners, so each learns something useful on its own. At test time, use all units with their outgoing weights halved. That one network equals the average of an astronomically large ensemble.**

---

## 1. The problem: overfitting by co-adaptation (page 1)

- A large network trained on little data has **many** weight settings that fit the training set almost perfectly. Most of them do poorly on new data.
- **The mechanism the authors blame is co-adaptation:**
  - a unit becomes useful only **in combination with** several specific other units;
  - such conspiracies can fit quirks of the training set (even its noise);
  - they don't transfer to new data, because the exact combination of conditions rarely happens again.

---

## 2. Dropout, precisely (pages 1–2)

### 2.1 Training
For each training case and each hidden unit j, flip a biased coin:
```
m_j ~ Bernoulli(p)          m_j = 1 with probability p (keep), 0 with probability 1 − p (drop)
h̃_j = m_j · h_j             the unit's output is replaced by 0 if dropped
```
- **The paper's choices:** p = 0.5 for hidden units, and often p = 0.8 for inputs (drop 20%).
- **Backprop runs through the thinned network.** Dropped units pass no gradient, so their incoming weights get no update from that case.
- **The effect:** a unit can't count on any particular partner, so it must learn a feature that is useful in many contexts.

### 2.2 How many networks is that?
- Each subset of kept units defines a different "thinned" network. With N hidden units there are **2ᴺ** of them.
- **Examples:**
  - the 784-800-800-10 MNIST net has N = 1,600 hidden units, so **2¹⁶⁰⁰ ≈ 10⁴⁸²** thinned networks, far more than atoms in the universe (~10⁸⁰);
  - each training case trains one random member, and **they all share the same weights**.
- So dropout is **training a gigantic ensemble with shared parameters**, almost for free.

### 2.3 At test time: the "mean network"
- **The problem:** we can't run 2¹⁶⁰⁰ networks and average them.
- **The paper's answer:** use **all** units and multiply their outgoing weights by p (halve them).
- **Why scaling by p is natural:** a unit's expected contribution during training was E[m_j h_j w] = p·h_j·w. Halving the weights gives the next layer the **same expected input** it saw during training.

---

## 3. The two mathematical claims, proved

### Claim 1: for one hidden layer + softmax, the mean network is *exactly* the normalized geometric mean of all 2ᴺ thinned networks

**Setup:**
- The hidden activities h (fixed for a given input, since there's nothing below them to drop).
- The output weights w_k for class k.
- A thinned network with mask m predicts:
  ```
  P(k | m) = exp(w_k · (m ⊙ h)) / Z(m)          Z(m) = Σ_l exp(w_l · (m ⊙ h))
  ```

**The geometric mean** over all 2ᴺ masks (each with probability ½ᴺ) is:
```
G(k) = Π_m P(k|m)^(1/2ᴺ) = exp( E_m[ log P(k|m) ] )
     = exp( E_m[ w_k · (m ⊙ h) ] − E_m[ log Z(m) ] )
```

**Two observations:**
1. **The second term doesn't depend on k.** It's the same for every class, so it disappears when we **normalize** (divide by the sum over k).
2. **The first term is linear in m,** so the expectation passes inside: E_m[w_k · (m ⊙ h)] = w_k · (E[m] ⊙ h) = **w_k · (½ h)**.

**So:**
```
normalized G(k) = exp(w_k · ½h) / Σ_l exp(w_l · ½h) = softmax of the network with HALVED outgoing weights  ∎
```

**Our code** checks this by brute force: enumerating all **1,024** sub-networks of a 10-hidden-unit net gives exactly the mean network's output. The demo shows [0.7234, 0.2343, 0.0423] both ways, and the largest difference is **1 × 10⁻¹⁶**.

**With more layers this is no longer exact,** because the hidden units are non-linear functions of dropped inputs. But it is a good approximation (Paper 010 measures it).

### Claim 2: the mean network gives the correct class a higher log-probability than the thinned networks do on average

- Let A(k) = E_m[log P(k|m)]. Then the unnormalized geometric mean is e^{A(k)} = G(k).
- **The key step:** for each m, the geometric mean of numbers is at most their arithmetic mean (AM–GM). So:
  ```
  Σ_k G(k) ≤ Σ_k E_m[P(k|m)] = E_m[Σ_k P(k|m)] = 1
  ```
- **Therefore**, writing the mean network's prediction as G(k)/Σ_l G(l):
  ```
  log [ G(k) / Σ_l G(l) ] = A(k) − log Σ_l G(l) ≥ A(k)          (since log of a number ≤ 1 is ≤ 0)
  ```
  The mean network's log-probability ≥ the average log-probability of the thinned networks. ∎
- **Our tests:** −1.80 vs −1.99, and −1.88 vs −2.17.
- **In words:** averaging an ensemble never hurts the log-likelihood, and usually helps.

---

## 4. Max-norm instead of weight decay (page 2)

- **The rule:** after each update, if a hidden unit's incoming weight vector w has squared length ‖w‖² > l (the paper uses l = 15), **project it back**:
  ```
  w ← w · √l / ‖w‖           (same direction, length exactly √l)
  ```
- **Why it pairs well with dropout:**
  - However big the learning rate, the weights can't explode, because every step ends inside a ball.
  - So training can start with a **huge learning rate** (10!) and explore much more of weight space, then decay it.
  - Dropout's noise keeps the search from settling in a sharp, overfit spot.
- **Compared with L2 weight decay:** L2 pulls *every* weight toward 0 all the time. Max-norm does nothing until the cap is hit.

---

## 5. The MNIST training recipe (Appendix A.1), decoded

| Setting | Value | Why |
|---|---|---|
| architecture | 784-800-800-10 (and larger) | big enough to overfit without dropout |
| dropout | 50% hidden, 20% input | inputs carry more unique information, so drop fewer |
| batch, loss | 100, cross-entropy | |
| learning rate | starts at **10**, × 0.998 per epoch | after 1,000 epochs: 10 · 0.998¹⁰⁰⁰ ≈ 1.35 |
| momentum | 0.5 → 0.99 over 500 epochs | Paper 008's "grow µ" idea |
| gradient scaling | the gradient is multiplied by (1 − µ) | keeps the effective step ε/(1 − µ)·(1 − µ) = ε as µ grows (see Paper 008, section 3) |
| max-norm | ‖w‖² ≤ 15 | makes lr 10 safe |
| init | N(0, 0.01²) | small, random |
| epochs | **3,000** | dropout learns slowly: each weight sees only part of the network |

---

## 6. Results (pages 2–5)

| Task | Without dropout | With dropout |
|---|---|---|
| **MNIST** (plain nets, no tricks) | ~160 errors (best published) | ~130 (50% hidden), **~110** (+20% input) |
| MNIST, pretrained deep belief net | 118 errors | 92 |
| MNIST, pretrained deep Boltzmann machine | ~94 | **79** (record) |
| TIMIT speech (frame recognition) | 22.7% | 19.7% |
| CIFAR-10 (conv net) | 16.6% | 15.6% |
| ImageNet 2010 | 48.6% | **42.4%** (record; this is AlexNet, Paper 014) |
| Reuters text | 31.05% | 29.62% |

The gains appear in **every** domain: vision, speech and text. That is the paper's strongest evidence.

---

## 7. Why it works: several views (pages 5–6)

- **Ensemble view:** 2ᴺ models with shared weights, averaged at test time (sections 2.2 and 3).
- **vs Bayesian averaging:** Bayes weights each model by its posterior probability, which is very expensive to compute. Dropout gives every model equal weight, which is cheap.
- **vs bagging:** bagging trains separate models on resampled data. Dropout trains each "model" on about one example, and weight sharing regularizes them all heavily.
- **Naive Bayes** is extreme dropout: every feature is trained alone, which famously works well with little data.
- **Sex in evolution:** mixing genes breaks up co-adapted gene complexes, so a function gets done robustly by several genes. Similarly, dropout breaks up co-adapted units.
- **A regularization view (later work):** for linear models, dropout acts like an **adaptive L2 penalty** that penalizes weights on features that vary a lot. (Wager et al. 2013; see Paper 010.)

---

## 8. What the features look like (Appendix A.3, Figure 5)

First-layer features of a 784-500-500 net:
- **Standard backprop:** noisy and hard to interpret;
- **dropout:** clean, **stroke-like** detectors.

Each unit must be useful on its own, so it learns a meaningful local feature.

---

## 9. What our code found

**Scale note:** at your request, the full MNIST experiments were **not fully run** on this laptop. `experiments.py` runs them (≈ 25 minutes on a GPU; the paper used 3,000 epochs, the script 100 with a time-compressed schedule).

**Checked exactly:**
- **Claim 1:** enumerating all **1,024** sub-networks of a 10-hidden-unit net reproduces the mean network to **1 × 10⁻¹⁶**.
- **Claim 2:** the mean network's correct-class log-probability is higher than the thinned networks' average: **−1.80 vs −1.99** and **−1.88 vs −2.17**.

**Partial MNIST results** (784-800-800-10, 100 epochs):

| method | test errors (of 10,000) |
|---|---|
| standard backprop, best learning rate (3.0) | **171** (paper: ~160 best published) |
| max-norm only, no dropout | 225 |
| dropout runs | not finished |

**The demo** (20,000 training images, 15 epochs):

| | training errors | test errors | test − train error rate |
|---|---|---|---|
| standard backprop | 858 / 20000 | **620** / 10000 | 1.91 points |
| dropout + max-norm | 773 / 20000 | **500** / 10000 | 1.13 points |
| average of 20 sampled dropout nets (instead of the mean net) | | 515 | |

- **Dropout narrows the generalization gap.**
- **The single mean network beats averaging 20 random samples,** consistent with Claim 2.

**A detail the paper leaves implicit:** without max-norm, its learning rate of 10 makes a plain network blow up to chance (~9,000 errors). Max-norm is what makes those huge rates safe, so the no-dropout baseline needs its own, smaller rate.

---

## 10. Check yourself

1. What is co-adaptation, and why does it hurt on new data?
2. Write the dropout mask equation. What happens to a dropped unit's gradient?
3. How many thinned networks does a net with 1,600 hidden units contain?
4. Prove that halving the outgoing weights gives the normalized geometric mean (one hidden layer, softmax). Which term vanishes on normalization, and why?
5. Use AM–GM to show the mean network's log-probability is ≥ the average thinned network's.
6. Write the max-norm projection. Why does it make lr = 10 safe?
7. Why multiply the gradient by (1 − µ) when momentum grows?
8. How is dropout like bagging, and how is it different?
