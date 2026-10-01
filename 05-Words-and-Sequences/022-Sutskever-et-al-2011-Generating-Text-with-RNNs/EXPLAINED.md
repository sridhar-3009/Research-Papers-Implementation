# Sutskever, Martens & Hinton (2011), explained from scratch

**Paper:** *Generating Text with Recurrent Neural Networks*
**Authors:** Ilya Sutskever, James Martens, Geoffrey Hinton (University of Toronto)
**Published at:** ICML 2011

Read Paper 021 (LSTM) first. This paper takes the other route to making RNNs trainable: keep a plain-style RNN, and train it with a **second-order optimizer**. This guide:
- explains how language models are scored (bits per character);
- derives every parameter count;
- shows with a two-number example why products of weights confuse gradient descent;
- walks through one Hessian-free step.

---

## 0. The whole idea in one line

> **Character-level language modelling: predict the next character. Let the current character choose the RNN's hidden-to-hidden matrix (a factored tensor, the "multiplicative RNN"), and train it with Hessian-free optimization. It beat the best non-parametric character model and generated surprisingly fluent text: plausible new names, and parentheses balanced over 30 characters.**

---

## 1. Language modelling and how to score it

- **A character-level language model** gives, at each position, a probability distribution over the next character: P(x_{t+1} | x₁ … x_t).
- **By the chain rule,** the probability of a whole text is the product of these (Paper 031, Eq. 1).
- **Bits per character (bpc)** = the average number of bits an ideal compressor using the model would need per character:
  ```
  bpc = −(1/N) Σ_t log₂ P(x_{t+1} | x₁…x_t)
  ```
- **Worked numbers:**
  - giving the correct next character probability ½ costs 1 bit, ¼ costs 2 bits, and 0.9 costs 0.15 bits;
  - a uniform model over M characters costs **log₂ M** bits every time: 86 characters gives 6.43 bpc, and our demo's 4-symbol toy gives 2 bpc.
  - Good English models reach ~1.3–1.6 bpc.
- **Modelling = compression:** a better model means smaller files. That's why the paper compares against the compressor PAQ.

---

## 2. Background: RNNs are hard to train (Sections 1–2)

**The standard RNN:**
```
h_t = tanh(W_hx x_t + W_hh h_{t−1} + b_h)          (1)
o_t = W_oh h_t + b_o                               (2)       P(x_{t+1} | x_≤t) = softmax(o_t)
```
- **The difficulty:** the gradient through time **vanishes or explodes** (Paper 021, section 2), so plain gradient descent struggles.
- **Earlier fixes:**
  - change the architecture (**LSTM**, Paper 021);
  - don't train the recurrent weights at all (**Echo State Networks**);
  - or keep the RNN and use **curvature** (**Hessian-free**; Martens 2010, Martens & Sutskever 2011, Paper 023).

---

## 3. The multiplicative RNN (Section 3)

### 3.1 Why multiplicative? A conjunction
- **An RNN as a tree:** each character labels an edge that turns the previous hidden state into the next.
- **Example:** after a verb stem ("fix", "break"), "i" should lead to a state predicting "n" (→ "ing"); elsewhere it shouldn't. What matters is **context AND character**, together.
- **Addition can't express that:** an additive RNN computes tanh(A h + B x), where the input can only **shift** the state. **Multiplication** lets the input change *how the past state is used*.

### 3.2 The tensor RNN (Eqs. 3, 5)
- **Give each character its own recurrent matrix:**
  ```
  h_t = tanh(W_hx x_t + W_hh^(x_t) h_{t−1} + b_h),        W_hh^(x_t) = Σ_m x_t^(m) W_hh^(m)
  ```
- **The cost:** with one-hot characters that's **M separate H×H matrices**. For H = 1500 and M = 86: 86·1500² = **193.5M** parameters. Far too many.

### 3.3 Factor the tensor (Eq. 6)
```
W_hh^(x_t) = W_hf · diag(W_fx x_t) · W_fh
```
- **F "factors",** each a **rank-one** matrix: (a column of W_hf) × (a row of W_fh).
- **The character sets each factor's gain** (W_fx x_t), so every character's matrix is a blend of the same F rank-one pieces.
- **Two consequences:**
  - characters **share structure**;
  - every W^(c) has rank ≤ F (our demo: rank 4 for F = 4).

### 3.4 The MRNN equations (Eqs. 7–9)
```
f_t = diag(W_fx x_t) · W_fh h_{t−1}       factor units: (character gain) × (projection of the past)
h_t = tanh(W_hf f_t + W_hx x_t)
o_t = W_oh h_t + b_o
```
That's **two non-linear-ish steps per character**, so unrolled over 250 characters it acts like a ~500-layer network.

### 3.5 Parameter counts, derived (M = 86, H = F = 1500)
| Model | Pieces | Total |
|---|---|---|
| standard RNN | W_hx 1500·86 + W_hh 1500² + W_oh 86·1500 + biases ≈ 129K + 2.25M + 129K | **≈ 2.5M** |
| tensor RNN | 86·1500² + the rest | **≈ 194M** |
| **MRNN** | W_fx 1500·86 + W_fh 1500² + W_hf 1500² + W_hx + W_oh ≈ 129K + 2.25M + 2.25M + 258K | **≈ 4.9M** (the paper's number) |

Factoring buys per-character dynamics for ~2× an RNN's size instead of ~80×.

**A same-size comparison (ML dataset):** an RNN with 500 units gets **1.65 bpc** vs **MRNN 350 × 350: 1.56**, and the RNN is the (slightly) bigger model.

### 3.6 Why products of weights need a second-order method (Section 3.3)
- **The effective weight** W_ij^(c) = Σ_f W_if·W_fc·W_fj is a **product** of parameters.
- **A two-number example:** take a product p = a·b with a = 0.01 and b = 100. The gradients are ∂p/∂a = b = **100** and ∂p/∂b = a = **0.01**.
  - **The small weight gets the huge gradient,** so one step changes it by 100η, a giant *relative* change.
  - **The big weight gets the tiny gradient,** so it barely moves.
- **The scales are exactly backwards.** Gradient descent zig-zags or crawls.
- **Curvature-aware methods fix this:** they divide by the curvature in each direction (Paper 006's Newton step), which rescales the steps sensibly.

---

## 4. Hessian-free optimization (Section 5.2; Martens 2010)

### 4.1 One HF step
1. **Gradient** g on a big batch (millions of characters).
2. **A quadratic model** of the loss near the current weights θ:
   ```
   L(θ + d) ≈ L(θ) + gᵀd + ½ dᵀ(G + λI)d
   ```
   - **G is the Gauss–Newton matrix,** a positive semi-definite stand-in for the Hessian.
   - **λI is damping:** don't trust the model far away.
3. **Minimize the model:** set its gradient to zero, (G + λI)d = −g, and solve with **conjugate gradient (CG)**.
   - **CG never needs G itself,** only products G·v (Paper 005, section 8.5).
4. **Adjust λ (Levenberg–Marquardt):**
   - compare the actual and predicted changes: ρ = (actual change)/(predicted change);
   - **ρ > 3/4:** the model was good, so trust it more: λ ← (2/3)λ;
   - **ρ < 1/4:** the model was poor, so be careful: λ ← (3/2)λ.

### 4.2 The Gauss–Newton product, without the matrix
- **The decomposition:** loss = L(z(θ)), where z = the logits. Then G = Jᵀ H_L J, with:
  - **J = ∂z/∂θ**;
  - **H_L = ∂²L/∂z²**, which for softmax + cross-entropy is **diag(p) − ppᵀ**.
- **H_L is positive semi-definite.** It is the covariance matrix of a one-hot sample drawn from p, so vᵀH_Lv = Var(v·onehot) ≥ 0. That makes G positive semi-definite too, and CG works.
- **Computing G·v takes two cheap passes:**
  1. **a forward-mode pass** (jvp) gives Jv;
  2. **multiply by H_L;**
  3. **a backward pass** (vjp) gives Jᵀ(H_L J v).
- **Cost:** never forming G, which would have (4.9M)² entries.

### 4.3 The paper's setup
- **The schedule:** 160 HF steps of ≤ 150 CG iterations, with structural damping μ = 0.1 (Paper 023) and initial λ = 10.
- **The data per step:** the gradient on 48,000 sequences of 250 characters; the curvature on 2,400.
- **The hardware:** 5 days on 8 GPUs.
- **The model:** **1500 units, 1500 factors (4.9M parameters)**, with sparse init (15 nonzero incoming weights per unit, Paper 008).
- **Context:** only the **last 200** characters of each 250 are predicted, so every prediction has at least 50 characters of context and no prediction is judged without history.

---

## 5. Results (Section 5, Table 1)

- **The data:** ~100 MB per dataset with an 86-character alphabet; the last 10M characters are the test set.
- **Test bpc** (lower is better):

| Dataset | Sequence memoizer | PAQ (dictionary off) | **MRNN** | MRNN (1 GB of training text) |
|---|---|---|---|---|
| Wikipedia | 1.66 | 1.51 | **1.60** | 1.55 |
| NYT | 1.49 | 1.38 | **1.48** | 1.47 |
| ML papers | 1.33 | 1.22 | **1.31** | — |

- **It beats the sequence memoizer** (the best single character model at the time) everywhere.
- **It's behind PAQ,** a huge hand-tuned mixture of context models with word-level knowledge.
- **10× more data helps only a little:** at 100 MB the model is already "fairly well-trained".

**Debagging (Section 5.4):**
- **The task:** given a bag of 7 words (plus 2 context words on each side), find the original order among 7! = 5040 orders.
- **The result:** MRNN **34%** vs memoizer 27%. Word order needs long character contexts.

---

## 6. What the MRNN learned (Section 6)

- **Samples** ("The meaning of life is …") show a large vocabulary, mostly real words, plausible invented names and grammatical fragments, but no coherent meaning.
- **Balanced parentheses and quotes over ~30 characters.** An n-gram model would need 31-grams; the memoizer and PAQ can't do it.
- **Completions:**
  - "England, Spain, France, Germany," continues as a **list of places**;
  - "(ABC et al" continues "(ABC et al., 2003)", even though "ABC" never appeared. It **generalizes** to an unseen name.

---

## 7. Why it matters

- **It started "generate text with an RNN":** Graves (2013), Karpathy's char-RNN (2015), and eventually LLMs.
- **Multiplicative interactions** (one input gating another's weights) reappear in LSTM/GRU gates and attention.
- **In practice,** LSTMs with SGD/Adam plus **gradient clipping** (Pascanu et al. 2013) replaced HF, which was too expensive per step.

---

## 8. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` covers:
  - E1: RNN-500 vs MRNN-350×350 at equal size;
  - E2: Hessian-free vs Adam on a small MRNN;
  - E3: samples and list completion;
  - E4: debagging 500 bags from Anna Karenina, all 5040 orders in one batched pass.
- The paper's datasets aren't available, so it uses **text8** (100 MB of Wikipedia, lowercase letters). E1 uses Adam + gradient clipping; E2 compares HF.

**Checked (tests and demo, about a second):**
- **Sizes:**
  - the 1500×1500 MRNN for 86 characters has **4.9M** parameters;
  - RNN-500 has slightly more (< 1%) than MRNN-350×350;
  - the same-size tensor RNN has **194M** (~40× more).
- **Eq. 6:**
  - each MRNN step equals tanh(W^(c)h + W_hx x) with W^(c) = W_hf·diag(W_fx e_c)·W_fh;
  - an MRNN *is* a tensor RNN with those matrices;
  - each W^(c) has rank ≤ F.
- **Hessian-free:**
  - our G·v (jvp → softmax Hessian → vjp) equals an explicitly built JᵀHJ;
  - G is positive semi-definite, and CG solves SPD systems.
  - **The demo** goes 1.49 → 0.55 nats in 6 steps on a repeating pattern. ρ stays > 3/4, so λ shrinks by 2/3 each step (0.667, 0.444, 0.296, …).
- **Evaluation:** an all-zero (uniform) model scores exactly log₂ M bpc.
- **Debagging:** a pure bigram model can't order "sat the cat" vs "the cat sat", because every letter pair occurs in both. That's why long contexts are essential.

---

## 9. Check yourself

1. A model gives the right characters probabilities 0.5, 0.25 and 0.9. What is its bpc on these 3? ((1 + 2 + 0.152)/3 ≈ 1.05.)
2. What does "the input chooses the recurrent matrix" mean? Why is it a conjunction?
3. Derive the 4.9M and 194M parameter counts.
4. For p = a·b with a = 0.01 and b = 100, compute ∂p/∂a and ∂p/∂b. Why is that bad for gradient descent?
5. Write the quadratic model of an HF step. What equation does CG solve?
6. Why is diag(p) − ppᵀ positive semi-definite? Why does that matter for CG?
7. How is λ adjusted, and what is ρ?
8. What does balancing parentheses over 30 characters show that an n-gram model can't do?
