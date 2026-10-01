# Jozefowicz, Zaremba & Sutskever (2015), explained from scratch

**Paper:** *An Empirical Exploration of Recurrent Network Architectures*
**Authors:** Rafal Jozefowicz, Wojciech Zaremba, Ilya Sutskever (Google)
**Published at:** ICML 2015

Read Paper 021 (LSTM) and Paper 024 first. The GRU is Paper 026; its equations are repeated here. This guide:
- derives why an *additive* state keeps gradients alive;
- derives what the forget gate does to them;
- counts each cell's parameters;
- works the search's fitness function with numbers.

---

## 0. The whole idea in one line

> **Is the LSTM's hand-designed cell optimal? An evolutionary search over 10,000 architectures found GRU-like cells that beat the LSTM on some tasks. But the biggest practical lesson is tiny: initialize the LSTM's forget-gate bias to 1. That alone closes most of the gap. Of the LSTM's gates, the forget gate matters most and the output gate least.**

---

## 1. Why LSTMs work (Section 2)

### 1.1 Exploding vs vanishing
- **The cause:** an RNN's gradient over T steps is roughly a **product of T Jacobians** (Paper 021, section 2), so it **explodes** or **vanishes** exponentially.
- **Exploding is easy:** clip the gradient norm (Paper 024, section 4.2).
- **Vanishing is the hard one:** the long-term components of the gradient are tiny next to the short-term ones, so RNNs learn short-term structure only.

### 1.2 The additive trick
- **The idea:** compute a **change** ΔS_t and **add** it:
  ```
  S_t = S_{t−1} + ΔS_t      ⇒      S_1000 = S_0 + Σ_{t=1}^{1000} ΔS_t
  ```
- **Every past change enters the final state with coefficient 1:**
  ```
  ∂S_1000/∂ΔS_t = I        for every t, even t = 1
  ```
  The gradient reaching any ΔS_t is the full gradient at S_1000. It can't shrink by passing through the sum.
- **The paper's words:** "It may become smeared, but it will never be negligibly small."
- **The catch:** ΔS_t itself depends on S_{t−1}, so other paths do multiply Jacobians. But the direct additive path always exists.

### 1.3 The LSTM used (no peepholes)
```
i = tanh(W_xi x + W_hi h + b_i)       candidate input
j = σ(W_xj x + W_hj h + b_j)          input gate
f = σ(W_xf x + W_hf h + b_f)          forget gate
o = tanh(W_xo x + W_ho h + b_o)       output gate (tanh in the paper's equations)
c_t = c_{t−1} ⊙ f + i ⊙ j,      h_t = tanh(c_t) ⊙ o
```
- **Two states:** a **slow** state c (an additive memory) and a **fast** state h (short-term decisions).
- **Against attractors:**
  - Bengio et al. (1994) proved that storing a bit in a stable attractor implies vanishing gradients.
  - LSTMs don't have vanishing gradients, so they **can't** be storing information in attractors. They store it in the near-linear c instead.

### 1.4 The forget gate, and its bias (Section 2.2)
- **With a forget gate, the memory path is multiplied, not just added:**
  ```
  ∂c_T/∂c_t = Π_{k=t+1}^{T} f_k         (elementwise)
  ```
- **The problem with small random weights:** every f ≈ σ(b_f).
  - **b_f = 0** gives f = 0.5. That's "a vanishing gradient with a factor of 0.5 per timestep".
  - **The fix,** b_f = 1 or 2, gives f = 0.73 or 0.88.

| forget bias | f at init | kept after 20 steps (f²⁰) | after 100 steps |
|---|---|---|---|
| 0 | 0.500 | 9.5·10⁻⁷ | 8·10⁻³¹ |
| 1 | 0.731 | 1.9·10⁻³ | 2.5·10⁻¹⁴ |
| 2 | 0.881 | 7.9·10⁻² | 3·10⁻⁶ |

- **The fix isn't new:** Gers et al. (2000) said it, but "many practitioners" didn't know.
- **Why it matters:** without it, "we may erroneously conclude that the LSTM is incapable of learning to solve problems with long-range dependencies".
- **Learning can still close the gate later.** The bias only sets the starting point.

### 1.5 The GRU (Section 2.3; Paper 026)
```
r = σ(W_xr x + W_hr h + b_r),     z = σ(W_xz x + W_hz h + b_z)
h̃ = tanh(W_xh x + W_hh (r ⊙ h) + b_h),     h_t = z ⊙ h_{t−1} + (1 − z) ⊙ h̃
```
- **An interpolation:** the new state mixes the old state and a candidate. With z = 1 it **copies** the old state (factor 1, like the CEC); with z = 0 it **replaces** it.
- **The reset gate r:** it decides how much of the past the candidate may see.
- **Compared with the LSTM,** the GRU merges c and h and has 3 gate blocks instead of 4.

### 1.6 Parameter counts (64 units, a 40-symbol vocabulary)
- **One gate block** reading [x; h] (64 + 64) into 64 units has 64·128 + 64 = **8,256** parameters.
- **Embedding + output:** 40·64 + (64·40 + 40) = **5,160** parameters.

| Cell | Gate blocks | Total (our code agrees) |
|---|---|---|
| tanh RNN | 1 block | 8,256 + 5,160 = **13,416** |
| LSTM (and -f/-i/-o/-b: gates removed but weights kept) | 4 blocks | 33,024 + 5,160 = **38,184** |
| GRU | 3 blocks | 24,768 + 5,160 = **29,928** |

---

## 2. The architecture search (Section 3)

### 2.1 Procedure (3.1)
- **The list:** keep the **100 best** architectures, starting with only the LSTM and the GRU (both fully tuned).
- **Fitness (Eq. 1):**
  ```
  fitness(A) = min over tasks T of  [ best accuracy of A on T / best accuracy of the GRU on T ]
  ```
  - **Worked example:** an architecture at 0.92 / 0.47 / 0.090 vs the GRU's 0.896 / 0.460 / 0.0907 gives ratios 1.027, 1.022, 0.992.
  - **Its fitness is 0.992**, set by its **worst** task.
  - **The min forces "good everywhere":** a cell can't win by excelling on one task and failing another.
- **Each step does one of:**
  - **Re-evaluate** a random top-100 architecture on 20 new random hyperparameter settings per task, so noisy lucky scores get re-tested.
  - **Mutate** a random top-100 architecture into a candidate, then filter it in stages:
    1. **The memorization filter:** read 5 symbols (26 possibilities) and reproduce them, reaching ≥ 95%. The LSTM does this in < 4 minutes with 25K parameters.
    2. **Task 1** with 20 random settings must reach ≥ 90% of the GRU, else stop. Then task 2, then task 3.
    3. **Enter the top-100** if good enough.
- **Totals:** 10,000 architectures evaluated, 1,000 past the first filter, ~220 settings each, **230,000 settings overall**.

### 2.2 Mutations (3.2)
- **Representation:** an architecture is a **computation graph** with inputs h, c, … (the previous state) and x, outputs (the new state), and all nodes the same size.
- **Mutation:** pick p ~ U[0, 1] (so the size of the change varies), then apply 1–3 random transformations:
  1. **swap an activation** for another, from {tanh, σ, ReLU, Linear(0, ·), Linear(1, ·), Linear(0.9, ·), Linear(1.1, ·)}. Linear(a, x) = a fresh W·x + b with a added to W's diagonal; Linear(1, ·) starts near the identity;
  2. **swap an element-wise op** (+, ×, −);
  3. **insert** a random activation between a node and a parent;
  4. **remove** a node with one input and one output;
  5. **replace** a node with one of its ancestors;
  6. **replace** a node with the sum, product or difference of an ancestor of it and an ancestor of another node.
- **Every mutant keeps the state interface,** so it is still a runnable cell (our demo runs 4 random mutants).

### 2.3 Tasks (3.5)
- **Arithmetic:**
  - add or subtract two numbers of up to 8 digits, with **random distractor letters** in between;
  - e.g. `3e36d9-h1h39f94eeh43keg3c=-13991064` means 3369 − 13994433;
  - after "=", the answer is predicted digit by digit.
- **XML:** synthetic nested tags (2–10 letters). With probability ½ (or after 50 steps), close the latest tag (or stop if none is open); otherwise open a new one.
- **Penn Treebank:** word-level language modelling (10k vocabulary), without dropout during the search.
- **Music** (Nottingham, Piano-midi): binary note vectors. **Not used in the search**; it checks generalization.

**Training:**
- batch 20, unroll 35, with the state carried over (Paper 024);
- during the search, halve the learning rate per epoch for 4 epochs after 3 epochs without improvement.

**Hyperparameter ranges:**
- init scale {0.3 … 2.8}/√units;
- learning rate {0.1 … 5};
- clip {1 … 20};
- 1–4 layers;
- parameter budget 250K or 1M (5M for PTB).

---

## 3. Results (Section 4)

### 3.1 The three best architectures
```
MUT1: z = σ(W_xz x + b_z)                       ← z doesn't look at h at all
      r = σ(W_xr x + W_hr h + b_r)
      h_{t+1} = tanh(W_hh (r ⊙ h) + tanh(x) + b_h) ⊙ z + h ⊙ (1 − z)
MUT2: z = σ(W_xz x + W_hz h + b_z);   r = σ(x + W_hr h + b_r)
      h_{t+1} = tanh(W_hh (r ⊙ h) + W_xh x + b_h) ⊙ z + h ⊙ (1 − z)
MUT3: z = σ(W_xz x + W_hz tanh(h) + b_z);   r = σ(W_xr x + W_hr h + b_r)
      h_{t+1} = tanh(W_hh (r ⊙ h) + W_xh x + b_h) ⊙ z + h ⊙ (1 − z)
```
All three are close relatives of the GRU: the same interpolation h_{t+1} = (new)·z + h·(1 − z), with small rewiring.

### 3.2 Table 1: best next-step accuracy (higher is better)
| Arch | Arith | XML | PTB |
|---|---|---|---|
| tanh RNN | 0.295 | 0.321 | 0.0878 |
| LSTM | 0.892 | 0.425 | 0.0891 |
| LSTM-f (no forget gate) | **0.293** | **0.234** | 0.0881 |
| LSTM-i (no input gate) | 0.751 | 0.414 | 0.0866 |
| LSTM-o (no output gate) | 0.867 | 0.421 | 0.0893 |
| **LSTM-b (forget bias 1)** | 0.902 | 0.444 | 0.0895 |
| GRU | 0.896 | 0.460 | 0.0907 |
| **MUT1** | **0.921** | **0.475** | 0.0897 |
| MUT2 | 0.897 | 0.473 | 0.0904 |
| MUT3 | 0.907 | 0.465 | **0.0916** |

### 3.3 Table 3: PTB with dropout (20M parameters), test perplexity
| Arch | Test perplexity |
|---|---|
| **LSTM-b** | **79.8** |
| LSTM | 81.4 |
| LSTM-o | 82.3 |
| MUT3 | 89.5 |
| GRU | 91.7 |
| tanh | 97.7 |
| LSTM-f | 100.8 |

With dropout (Paper 024), the LSTM, especially LSTM-b, wins language modelling clearly.

### 3.4 Findings
- **The GRU beat the LSTM on everything but language modelling.** MUT1 matched the GRU on language modelling and beat it everywhere else.
- **LSTM-b beat both the LSTM and the GRU on almost all tasks.**
- **Gate importance:**
  - **forget gate:** essential on Arithmetic and XML. Removing it is about as bad as a plain tanh RNN, which is consistent with section 1.4's products. It barely matters for PTB (matching Mikolov et al. 2014: a plain RNN with a fixed integrator matches the LSTM on language modelling);
  - **input gate:** second most important;
  - **output gate:** least important (h = tanh(c) is mostly enough).
- **Recommendation:** **"adding a bias of 1 to the forget gate of every LSTM in every application"**. PyTorch and TensorFlow users still do this by hand today.

---

## 4. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` reproduces:
  - E1: Table 1 for all 10 cells on Arithmetic, XML and PTB, with Section 3.7's random hyperparameters;
  - E2: LSTM vs LSTM-b over several seeds;
  - E3: a small run of the actual search (mutations + Eq. 1 + a top-100 list).

**Checked (tests and demo, about a second):**
- **All 10 cells implement the paper's equations.** Explicitly checked: the GRU update, MUT1's full update (its z ignores h), and LSTM-f keeping c′ = c + i·j.
- **The forget bias:** the table in section 1.4. Our `SequenceModel` re-applies the +1 after its uniform init, so LSTM-b really starts with an open forget gate.
- **Parameter counts:** 13,416 / 38,184 / 29,928 for tanh / LSTM / GRU (section 1.6).
- **The search machinery:**
  - the **LSTM and GRU written as graphs** compute exactly what the hand-written cells do;
  - **200 random mutations** (all six operators) always yield valid, runnable cells with the same state interface;
  - Eq. 1's fitness is the worst task relative to the GRU;
  - the staged filter rejects candidates below 90% of the GRU at the first task.
- **Task generators:**
  - arithmetic answers are correct after removing the distractors;
  - generated XML is always well-formed;
  - memorization repeats its 5 symbols.

**A quirk of the paper's XML rule:** taken literally, it produces an **empty** string 50% of the time (the first coin flip can say "close" when nothing is open). Our generator follows the rule; empty samples simply add nothing to the training stream.

---

## 5. Check yourself

1. Show that ∂S_T/∂ΔS_t = I for an additive state. Why does that keep the gradient from vanishing?
2. With a forget gate, what is ∂c_T/∂c_t? Compute it for f = 0.5 and f = 0.88 over 20 steps.
3. Why is σ(0) = 0.5 a bad starting forget gate? What bias gives f ≈ 0.73?
4. Derive the LSTM's 38,184 and the GRU's 29,928 parameters (64 units, 40 symbols).
5. Compute Eq. 1's fitness for ratios 1.10, 0.95, 1.02. Why use the min?
6. Name three of the six mutation operators.
7. Which LSTM gate matters most and which least (Table 1)? Why does the forget gate barely matter for PTB?
8. How does MUT1 differ from the GRU?
