# Hochreiter & Schmidhuber (1997), explained from scratch

**Paper:** *Long Short-Term Memory*
**Authors:** Sepp Hochreiter, Jürgen Schmidhuber
**Published in:** Neural Computation 9(8), 1735–1780, 1997
**What to read:** Sections 1–4 (the problem and the architecture) and Section 5.4 (the adding problem). Appendix A.1 has the exact algorithm.

Read Paper 004 (backprop, and backprop-through-time in its section 10) first. This guide:
- derives *why* plain recurrent nets forget;
- derives the exact condition for a unit that never forgets;
- runs one LSTM memory cell forward by hand;
- shows the paper's truncated learning rule as an equation.

---

## 0. The whole idea in one line

> **Ordinary recurrent nets can't learn to connect events more than ~10 steps apart, because the error signal is multiplied by a number < 1 (or > 1) at every step back in time. LSTM adds a memory cell whose state is carried forward by a linear self-connection of weight exactly 1.0, the "constant error carrousel". Errors flow back through it unchanged. Multiplicative gates decide when to write into it and when to read from it.**

---

## 1. Recurrent nets in one paragraph

- **A recurrent net** has units that feed back into themselves. At each time step t:
  ```
  net_j(t) = Σ_u w_ju · y_u(t − 1)   (+ the current external inputs)
  y_j(t)   = f_j(net_j(t))
  ```
- **Its state y(t) is a memory** of everything seen so far.
- **Training uses backprop through time (BPTT):** unroll the net into one layer per time step (all layers sharing the same weights), then backpropagate through the unrolled net (Paper 004, section 10).

---

## 2. The problem: vanishing (or exploding) error (Section 3.1)

**The error recursion.** In BPTT, a unit's error at time t comes from the errors one step later:
```
δ_j(t) = f′_j(net_j(t)) · Σ_i w_ij δ_i(t + 1)
```

**Follow one path back q steps,** from unit u at time t to unit v at time t − q (Eq. 2). The scaling is a **product of q factors**:
```
∂δ_v(t−q) / ∂δ_u(t) = Σ over paths  Π_{m=1..q}  f′(net_{l_m}(t−m)) · w_{l_m l_{m−1}}
```
- **If each |f′·w| > 1,** the product **explodes** exponentially, and the weights oscillate wildly.
- **If each |f′·w| < 1,** it **vanishes** exponentially, so long lags can't be learned "in acceptable time".

**For the logistic sigmoid, f′ ≤ 0.25:**
- With |w| < 4 the factor is < 1 at **every** step.
- **Our numbers** (best case 0.25|w| per step, 100 steps):

  | \|w\| | per step | after 100 steps |
  |---|---|---|
  | 1 | 0.25 | **6·10⁻⁶¹** |
  | 3 | 0.75 | 3·10⁻¹³ |
  | 3.99 | 0.998 | 0.78 |
  | 6 | 1.5 | **4·10¹⁷** (explodes) |

**No easy escape:**
- **Big weights don't help:** they push units into saturation, where f′ → 0 even faster.
- **A bigger learning rate doesn't help:** it scales short-range and long-range errors alike, so their **ratio** is unchanged.

**The upshot:** the net only learns from recent inputs. (Bengio et al. 1994 published "a very similar analysis".)

---

## 3. The fix: a unit whose error never changes (Section 3.2)

**What we want:** a unit j connected only to itself (weight w_jj) to pass its error back **unchanged**:
```
δ_j(t) = f′_j(net_j(t)) · w_jj · δ_j(t+1)     and we want    f′_j(net_j(t)) · w_jj = 1.0
```
**Solving the condition:**
- f′ must equal the constant 1/w_jj for **every** input. Integrating, **f_j(x) = x/w_jj**: f must be **linear**.
- The simplest choice is **f(x) = x with w_jj = 1.0**.
- Then y_j(t) = y_j(t − 1): **the unit keeps its value forever.**

This is the **Constant Error Carrousel (CEC)**, LSTM's heart.

**But a bare CEC has two conflicts:**
1. **Input weight conflict:** the same incoming weight must sometimes **store** an input and sometimes **ignore** inputs (to protect what's stored).
2. **Output weight conflict:** the same outgoing weight must sometimes **read** the memory and sometimes **hide** it (so it doesn't disturb other units).

One fixed weight can't do both jobs.

---

## 4. The LSTM memory cell (Section 4, Figure 1)

**Put two multiplicative gates around the CEC:**
```
y_in(t)  = f(net_in(t))                         input gate  ∈ [0, 1]   "write now?"
y_out(t) = f(net_out(t))                        output gate ∈ [0, 1]   "read now?"
s_c(t)   = s_c(t−1) + y_in(t) · g(net_c(t))     internal state: the CEC (self-weight exactly 1.0)
y_c(t)   = y_out(t) · h(s_c(t))                 the cell's output
```
- **Inputs:** every net input is Σ_u w·y_u(t − 1) over the previous step's inputs, gates, cells, ….
- **Squashing functions (Appendix A.1):**
  - f = logistic, range [0, 1];
  - **g(x) = 4/(1 + e⁻ˣ) − 2**, range [−2, 2];
  - **h(x) = 2/(1 + e⁻ˣ) − 1**, range [−1, 1].
- **Multiplying by a gate value between 0 and 1 is a soft switch.** Gate ≈ 0 blocks the signal, gate ≈ 1 passes it. Each gate **learns** when to open, from context, which solves both conflicts.
- **Memory blocks:** S cells sharing one input gate and one output gate.
- **No forget gate in 1997:** the state can only be reset between sequences. The forget gate, s(t) = **f_t**·s(t − 1) + …, came in **Gers, Schmidhuber & Cummins (2000)**, and that is the "LSTM" everyone uses today.

### 4.1 One cell, two steps, by hand
**Step 1, a write:** with s(0) = 0, net_in = 2, net_c = 1, net_out = 0:
```
y_in = σ(2) = 0.881                 g(1) = 4σ(1) − 2 = 4(0.731) − 2 = 0.924
s(1) = 0 + 0.881 · 0.924 = 0.814
y_out = σ(0) = 0.5                  h(0.814) = 2σ(0.814) − 1 = 0.386
y_c(1) = 0.5 · 0.386 = 0.193
```
**Step 2, the gate closes:** net_in = −6, so y_in = σ(−6) = 0.0025:
```
s(2) = 0.814 + 0.0025 · 0.924 = 0.816         (almost nothing added; nothing lost)
```
- **The stored value survives.**
- **Our demo** writes once with an open gate, s = g(1) = 0.924234, and reads **exactly 0.924234 after 999 more steps** with the gate closed.

### 4.2 Why the error flows: ∂s(t)/∂s(t−1) = 1
- **Along the state path,** s(t) = s(t − 1) + (something that does not involve s(t − 1) directly). So the derivative is exactly **1**.
- **An error that reaches s:**
  1. it is scaled once on the way in, by y_out·h′(s), at the cell output;
  2. it travels back through **any** number of steps unchanged;
  3. it is scaled once more on the way out, by y_in·g′, to update the incoming weights.
- **Autograd confirms it:** ∂s(500)/∂s(0) = **1.000**.

### 4.3 The paper's learning rule: truncated, forward-mode gradients (Appendix A.1)
- **Forward-mode, not backward:** instead of BPTT, the paper carries derivatives **forward in time** (RTRL style), local in space and time. For a weight w into the cell input (from unit u):
  ```
  ∂s_c(t)/∂w = ∂s_c(t−1)/∂w + y_in(t) · g′(net_c(t)) · y_u(t−1)
  ```
  (and similarly for the input-gate weights, with f′ and g).
- **The truncation:** the recurrent inputs y_u(t − 1) are treated as constants, i.e. errors are **truncated** whenever they would leave a memory cell.
- **So errors travel back in time only inside the CEC,** where nothing scales them.
- **Cost:** O(W) per time step, with no storage of the whole history (unlike BPTT).
- **Our check:** this gradient equals PyTorch autograd's to **10⁻¹⁰** when the recurrent inputs are detached. That's exactly what "truncated" means; it differs from the full BPTT gradient.

### 4.4 Two practical problems (Section 4)
- **The abuse problem:**
  - Early in training, a cell may be used as a constant "bias" unit (its output never changes).
  - **Remedies:** add cells one at a time, or start the **output gates with negative biases** (closed).
- **Internal state drift:**
  - **The cause:** with no forget gate, s keeps accumulating input, |s| grows, h saturates, and h′(s) → 0 at the output.
  - **The effect:** the error can't even *enter* the CEC.
  - **Remedy:** **negative input-gate biases**, so the gates start nearly closed and s stays small.

---

## 5. Experiments (Section 5)

- **Training:** all on-line (one sequence at a time), logistic gates, initial weights in [−0.2, 0.2] or [−0.1, 0.1].
- **Baselines:** RTRL, BPTT, Recurrent Cascade-Correlation, Elman nets and the Neural Sequence Chunker.
- **Every task except the first has only long lags:** no short-lag examples to bootstrap from.

| Exp. | Task | Result |
|---|---|---|
| **1** | **Embedded Reber grammar:** predicting the second-to-last symbol requires remembering the second one across the whole inner string | LSTM solves it in almost all trials (2 failures in 150), and much faster; the others rarely solve it |
| **2a** | (y, a₁…a_{p−1}, y) vs (x, a₁…a_{p−1}, x): remember the first symbol for p steps | RTRL 79% at p = 4, **0% at p = 10**; BPTT fails; **LSTM 100% even at p = 100** |
| **2b/2c** | random distractors; **2c: lags up to 1000** | LSTM solves both; the others fail |
| **3** | Bengio et al.'s "2-sequence problem" | solved, including a harder real-valued version |
| **4** | **The adding problem:** sum two marked numbers seen ≥ T/2 steps before the end; target 0.5 + (X₁ + X₂)/4, correct if the error < 0.04 | T = 100: 1/2560 wrong (74K training sequences); T = 1000: 1/2560 (853K) |
| **5** | **multiplication** instead of addition | solved, but needs longer training for precision |
| **6** | **Temporal order:** classify by the order (XX/XY/YX/YY) of two symbols ~40 steps apart among distractors | solved; never solved by earlier RNNs |

**Why the adding target is 0.5 + (X₁ + X₂)/4:** with X_i ∈ [−1, 1], the sum lies in [−2, 2]. Dividing by 4 and adding 0.5 maps it into [0, 1], the logistic output's range.

---

## 6. Limitations (Section 6)

- **Strongly delayed XOR:**
  - Truncated LSTM can't solve it: storing just *one* of the two inputs doesn't reduce the error at all, so there's no incremental path for gradient descent to follow.
  - Compare Paper 003: XOR has no "partial credit".
- **Precise counting:** telling 99 steps from 100 is hard (3 vs 11 is fine).
- **Size:** each block adds 2 gate units, up to ~9× more weights than a plain RNN of the same size.

---

## 7. Why it matters

- **LSTM (with the 2000 forget gate) was the workhorse of sequence learning until Transformers (2017):** speech recognition, machine translation (Papers 027–028), handwriting and text generation.
- **"Carry state along an additive, gated path so gradients don't vanish"** reappears in:
  - GRUs (Paper 026);
  - ResNets' identity shortcuts (Paper 016; compare ∂y/∂x = I + ∂F/∂x with ∂s/∂s = 1 here);
  - the residual stream of Transformers.

---

## 8. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` reproduces Experiments 1 (Reber), 2a (p = 4, 10, 100), 4 (adding), 5 (multiplication) and 6a (temporal order), each against a plain RNN with BPTT. It also runs a modern LSTM with a forget gate on the adding problem.

**Checked (tests and demo, about a second):**
- **The paper's truncated rule = autograd with detached recurrent inputs,** to 10⁻¹⁰.
- **The CEC really has factor 1.0:** ∂s(500)/∂s(0) = 1.000, and a stored 0.924234 is unchanged after 999 steps.
- **Section 3.1's numbers:** the table in section 2.
- **The error from the end reaching the FIRST input** (untrained nets):

  | Lag | Plain RNN | LSTM, default init (\|s\| at end) | LSTM, input-gate biases −3, −6 (\|s\| at end) |
  |---|---|---|---|
  | 10 | 1.7·10⁻¹⁰ | 2.6·10⁻³ (1.7) | 2.1·10⁻⁴ (0.2) |
  | 50 | **0** (underflow) | 2.1·10⁻³ (10.1) | 1.9·10⁻⁴ (0.5) |
  | 100 | 0 | 1.4·10⁻⁶ (20.0) | 1.2·10⁻⁴ (1.3) |
  | 200 | 0 | **2.2·10⁻⁹** (44.0) | **8.3·10⁻⁵** (2.9) |

  - **The plain RNN's signal vanishes completely.**
  - **The default-init LSTM also fades, but because of state drift** (|s| → 44, so h′(s) → 0), not because of the CEC.
  - **With the paper's remedy** the signal barely fades over 200 steps.
- **The tasks are generated exactly as described:**
  - Reber strings are valid, and each step's targets are its legal next symbols;
  - adding: one marker in the first 10 steps, one before T/2, and the target 0.5 + (X₁ + X₂)/4;
  - temporal order: t₁ ∈ [10, 20], t₂ ∈ [50, 60].

---

## 9. Check yourself

1. Write the BPTT error recursion. Why does following it back q steps give a *product* of q factors?
2. Why must that product vanish for logistic units with |w| < 4? Compute 0.25¹⁰.
3. Derive the CEC condition f′(net)·w = 1. What does it force f to be?
4. What are the input-weight and output-weight conflicts? How do the gates solve them?
5. Run the memory cell by hand: s(0) = 0, net_in = 0, net_c = 0, net_out = 2. What is y_c(1)? (s = 0.5·0 = 0, so y_c = 0.)
6. Why is ∂s(t)/∂s(t − 1) = 1 exactly? Where are the only places an error gets scaled?
7. What does "truncated" mean in Appendix A.1? Write the forward-mode derivative recursion.
8. What is internal state drift? Why does a negative input-gate bias help?
9. Why can't truncated LSTM solve a strongly delayed XOR?
