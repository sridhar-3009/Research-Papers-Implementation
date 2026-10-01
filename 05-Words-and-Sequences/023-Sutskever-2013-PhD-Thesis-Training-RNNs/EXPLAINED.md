# Sutskever (2013), explained from scratch

**Thesis:** *Training Recurrent Neural Networks* (PhD thesis, University of Toronto, 2013)
**Author:** Ilya Sutskever (advisor: Geoffrey Hinton)
**Length:** 101 pages. **Optional** in the reading order.

**Coverage:**
- **Already in this repo:** Chapter 5 (the multiplicative RNN) is **Paper 022**, and Chapter 7 (momentum and initialization) is **Paper 008**.
- **Implemented in this folder:** the two new chapters, **Chapter 3 (the RTRBM)** and **Chapter 4 (Hessian-free training with structural damping)**.

**What this guide does:**
- explains Restricted Boltzmann Machines from zero, since Chapter 3 assumes them;
- shows why heavy damping cripples Hessian-free;
- derives the structural-damping fix.

---

## 0. The whole idea in one line

> **RNNs are not hopeless: with the right model (Ch. 3), the right optimizer (Ch. 4: Hessian-free with structural damping) and the right initialization and momentum (Ch. 7), plain RNNs learn long-range structure that was believed to need LSTMs.**

---

## 1. Background you need: Restricted Boltzmann Machines

### 1.1 The model
- **The structure:** an RBM has binary **visible** units v (the data, e.g. pixels) and binary **hidden** units h (features), with weights W between the layers and **no connections within a layer**.
- **It defines an energy:**
  ```
  E(v, h) = −b_vᵀv − b_hᵀh − vᵀWh
  P(v, h) = e^{−E(v,h)} / Z,         Z = Σ_{all v,h} e^{−E(v,h)}     (the "partition function")
  ```
- **Low energy = high probability.** The weights make good (v, h) combinations low-energy.

### 1.2 Given one layer, the other is easy (the "restricted" part)
- **No hidden–hidden connections** means the hidden units are **independent given v**:
  ```
  P(h_j = 1 | v) = σ( b_h,j + Σ_i W_ij v_i )          and symmetrically   P(v_i = 1 | h) = σ( b_v,i + Σ_j W_ij h_j )
  ```
- **Tiny example:** a hidden unit with bias −1 and weights (2, 2) to two visible units, both on: P(h = 1) = σ(−1 + 4) = σ(3) = 0.95.
- **Gibbs sampling** alternates: sample h given v, then v given h, and so on.

### 1.3 Learning: "data minus model"
- **The gradient** of the log-likelihood of a data vector has a famous form:
  ```
  ∂ log P(v) / ∂W_ij = ⟨v_i h_j⟩_data − ⟨v_i h_j⟩_model
  ```
  - **The first term:** h comes from P(h|v) on real data, which is easy.
  - **The second term:** an average over the model's own samples, which needs Z. That's intractable for big RBMs.
- **Contrastive divergence (CD-k)** approximates the model term:
  - start a Gibbs chain **at the data**;
  - run only k steps;
  - use the result as a stand-in for a true model sample.

  It's cheap and works well in practice. Our test: the CD-20 gradient correlates > 0.9 with the exact one.

---

## 2. Chapter 3: the Recurrent Temporal RBM (Sutskever, Hinton & Taylor 2009)

### 2.1 The problem with the TRBM
- **What a Temporal RBM is:** a sequence model (for videos, motion capture) with **one RBM per frame**, whose hidden biases depend on the **previous hidden states**.
- **The trouble:** the hidden states are random. The posterior over past hiddens depends on *future* frames too, so **inference is intractable**, and learning needs crude approximations.

### 2.2 The RTRBM fix: make the carried state deterministic
Replace the sampled hidden state by its **mean-field value** (its expected value, a number in [0, 1]):
```
r_t = σ(W v_t + b_h + W′ r_{t−1})        (3.20)    the "recurrent" state
RBM at time t: hidden bias b_t = b_h + W′ r_{t−1}   (visible bias b_v, weights W)
```
- **Sampling (Algorithm 3):**
  1. Sample v_t from the RBM with bias b_t (Gibbs).
  2. **Set** r_t ← σ(…). Deterministic, not sampled.
- **Inference (Algorithm 4):** given the frames, r₁ … r_T follow from **one forward pass**. The posterior is a point mass.

### 2.3 Learning = an RNN with an RBM loss
```
log P(v₁…v_T) = Σ_t log P_RBM(v_t | r_{t−1})          (3.22)
```
- **The structure:** an **RNN** (the r's) whose per-step loss is an **RBM's log-probability**.
- **The gradient:**
  - each step's RBM term gives a gradient for W, b_v and its bias b_t, by CD;
  - the gradient on b_t flows into W′, b_h and back through r by **BPTT** (Eqs. 3.23–3.25).
- **With exact RBM gradients instead of CD, this is the exact gradient** (Section 3.10). Our test confirms it to 10⁻⁸.

### 2.4 Results
- **Bouncing balls** (3 balls, 30×30 pixels, 400 hidden units):
  - **The samples:** RTRBM samples move persistently instead of in a random walk.
  - **Next-frame squared error:** **0.007** (RTRBM) vs **0.04** (TRBM).
  - **The hidden units:** some have no visible connections and strong recurrent weights. They act as a dedicated "memory" layer.
- **Motion capture:** 0.42 vs 0.47.
- **Training:**
  - 100,000 updates, one sequence each, momentum 0.9;
  - CD-10, then CD-25;
  - W pre-trained with CD-5.

---

## 3. Chapter 4: Hessian-free training of RNNs (Martens & Sutskever 2011)

### 3.1 HF recap (see Paper 022, section 4)
Each step minimizes a quadratic model with:
- the **Gauss–Newton** matrix G;
- **damping** λI;
- **conjugate gradient** with Martens' early stopping;
- **Levenberg–Marquardt** adaptation of λ.

### 3.2 Why a large λ cripples HF
- **The HF step** solves (G + λI)d = −g.
- **When λ is huge** compared with G's eigenvalues:
  ```
  d = −(G + λI)⁻¹ g ≈ −g/λ          ← plain gradient descent with learning rate 1/λ
  ```
- **What's lost:**
  - HF's power is scaling up the **low-curvature** directions: an eigenvalue γ gets step factor 1/(γ + λ) instead of a uniform one;
  - big λ erases that, so the low-curvature directions are "masked out";
  - **on long-range tasks, those are exactly the directions** that carry information across many time steps.

### 3.3 The cause: hidden states are hypersensitive to W_hh
- **What Martens & Sutskever saw:** on long-range tasks, the quadratic model became **inaccurate** as CG grew the step, λ rose, and the net got stuck where "little-to-no long-term information was being propagated".
- **The cause:** a small change in W_hh is applied at **every** time step, so over hundreds of steps it can change the hidden-state sequence h **a lot, and very nonlinearly**. A quadratic model in θ can't track that.

### 3.4 The fix: structural damping (Section 4.2.3)
**Penalize changes to the hidden states themselves:**
```
R(θ) = ½ λ‖θ − θ_n‖² + λμ · S(θ),       S(θ) = D( h(θ), h(θ_n) )           (4.7)
```
- **What S is:** how far the new hidden-state sequence is from the current one, measured by a distance D that matches the units.

**Three properties make this work:**
1. **Use S's Gauss–Newton matrix:** G_S = J_xᵀ(D∘e)″J_x, where x are the hidden pre-activations. For tanh units with the matching D, (D∘e)″ = diag(1 − h²). The curvature becomes:
   ```
   G_f + λI + λμ G_S
   ```
2. **It costs almost nothing:**
   - one forward-mode pass gives both J_o v (outputs) and J_x v (hidden states);
   - one reverse pass applies both transposes.
3. **It changes the step, not the destination:**
   - **at θ = θ_n:** S = 0, and **∇S = 0** (S is a distance minimized at θ_n);
   - **away from it:** it adds curvature, which shortens the steps in sensitive directions, but doesn't move the optimum.

**Our measurement** (curvature along unit-length weight changes):

| Change to … | Curvature without | Curvature with |
|---|---|---|
| W_hh (recurrent) | 0.003 | **5.79** |
| W_oh (output) | 0.16 | 0.16 (unchanged) |

**Only the directions that scramble the dynamics become "expensive".** So λ needn't rise, and the other directions keep their full Newton-like steps.

---

## 4. Chapter 4's experiments

### 4.1 The pathological problems (Hochreiter & Schmidhuber 1997, plus memorization)
Each has a target at the **end** that depends on inputs far in the past (length T′ ~ i[T, 11T/10]):

| Problem | What |
|---|---|
| addition / multiplication | two marked numbers among noise; target (u_I + u_J)/2 or u_I·u_J |
| XOR | marked **bits**; target u_I xor u_J (one input alone gives no partial credit) |
| temporal order | 2 special symbols among 6; output their order (4 classes) |
| 3-bit temporal order | 3 special symbols; 8 classes |
| random permutation | 100 symbols, first = last ∈ {1, 2}; predict the last |
| 5-bit / 20-bit memorization | read 5 bits (or 10 symbols of 5 values), wait T steps, reproduce them after a trigger |

- **Marker positions:** I ~ i[1, T′/10] and J ~ i[T′/10, T′/2], so at least T′/2 steps separate the last marker from the end.
- **Memorization is hardest:** the net must keep information alive and then *replay* it in order, so the hidden state must evolve "on its own" after the trigger.

### 4.2 Setup and results
- **Setup:**
  - an RNN with 100 tanh units (~10,000 parameters), sparse init (15 per unit);
  - gradient on 10,000 sequences, curvature on 1,000, ≤ 300 CG steps;
  - λ₀ = 0.1 and μ = 1/30 with structural damping (λ₀ = 0.3 without);
  - **success:** < 1% of test sequences wrong (|error| > 0.04 counts as wrong for continuous outputs).
- **Figure 4.1:**
  - **HF solves the problems for T up to 200**, with plain RNNs.
  - **Structural damping** helps slightly on problems 1–6, and is **essential** for memorization with lags > 50.
  - HF needs far more computation than LSTM on these tasks: LSTM's gates make memory easy, while a plain RNN must build it from general units.

### 4.3 Natural problems (Table 4.1)
An HF-trained RNN (300 units) vs an LSTM trained with gradient descent (30 blocks × 10 cells, similar size):

| Dataset | RNN + HF | LSTM + GD |
|---|---|---|
| bouncing balls (error) | **22** | 35 |
| MIDI music (log-likelihood) | **−569** | −683 |
| speech (error) | **22** | 41 |

---

## 5. Chapters 5–8 (brief)

- **Ch. 5:** the multiplicative RNN for text (Paper 022).
- **Ch. 6:** "augmented" HF to train RNN **controllers** for a simulated plant (model predictive control).
- **Ch. 7:** **momentum + good initialization is enough.** SGD with Nesterov momentum and a schedule nearly matches HF (Paper 008).
- **Ch. 8:** conclusions. RNNs, and deep nets in general, are trainable with the right tricks. That line of work led to Seq2Seq (Paper 027).

---

## 6. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` reproduces:
  - E1: Figure 4.1, HF with and without structural damping on all 8 problems for T = 30, 50, 100;
  - E2: SGD + momentum on addition;
  - E3: the RTRBM vs a no-recurrence baseline on bouncing balls.

**Checked (tests and demo, a couple of seconds):**
- **The RTRBM learning rule is exact with exact RBM gradients:** on a 4-visible, 3-hidden RTRBM, our BPTT rule with enumerated RBM gradients equals **autograd of the exact log-likelihood to 10⁻⁸**.
  - CD-20 correlates > 0.9 with exact.
  - Exact RBM probabilities sum to 1, and their gradients match finite differences.
- **Structural damping as derived:**
  - our one-jvp/one-vjp product equals (G_f + λμG_S)·v built from full Jacobians, and G_f·v with μ off;
  - the W_hh vs W_oh table in section 3.4.
- **HF learns short memorization:** 1.57 → 0.21 in 6 steps with structural damping, 1.57 → 0.24 without. Both work on a tiny problem, as the thesis found.
- **All 8 problems are generated exactly per Section 4.4:**
  - lengths;
  - marker windows;
  - special-symbol windows;
  - the memorization trigger at step T + 5.

**Simplifications:**
- **The TRBM baseline isn't implemented** (it needs approximate inference). E3 compares with the RTRBM at W′ = 0, i.e. independent RBMs per frame.
- **Our CG stopping follows Martens (2010).** We don't backtrack over CG iterates.

---

## 7. Check yourself

1. Write an RBM's energy. Why are the hidden units independent given v? Compute P(h = 1) for bias 0, weights (1, −1), v = (1, 1). (σ(0) = 0.5.)
2. Write the RBM gradient "data minus model". Which term is hard, and how does CD approximate it?
3. What makes TRBM inference hard and RTRBM inference trivial?
4. Why is an RTRBM's log P an RNN loss? Where do CD and BPTT each come in?
5. Show that a large λ turns the HF step into gradient descent. Why is that bad for long-range tasks?
6. What does structural damping's S measure? Why is its gradient 0 at θ_n?
7. Why does changing W_hh move the hidden states far more than changing W_oh?
8. Describe 5-bit memorization. Why is it the hardest problem?
