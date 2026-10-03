# Scaling Laws for Neural Language Models, explained simply

**Paper:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu & Dario Amodei, *Scaling Laws for Neural Language Models*, OpenAI, 2020.

**In one sentence:** a language model's test loss falls as a smooth **power law** in model size, dataset size and compute, over 6–8 orders of magnitude. The exact architecture barely matters. So you can **predict** how good a model will be before training it, and choose the best size for your budget.

---

## 1. The big idea

- Before this paper, choosing model sizes was guesswork.
- The authors trained hundreds of Transformer language models, from 768 to 1.5 billion non-embedding parameters, on WebText2 (23B tokens, GPT-2's BPE).
- They measured how the test loss L (cross-entropy in nats per token) depends on:

| symbol | what it is |
|---|---|
| **N** | model size: **non-embedding** parameters (embeddings excluded, which gives much cleaner laws) |
| **D** | dataset size in tokens |
| **C** | training compute, C ≈ 6NBS (B = batch size in tokens, S = steps), measured in PF-days (1 PF-day = 8.64·10¹⁹ FLOPs) |

**The surprise:** the loss depends **strongly on scale** (N, D, C) and **very weakly on shape** (depth, width, number of heads, feed-forward ratio). A 6-layer and a 48-layer model with the same N have nearly the same loss.

---

## 2. Background: what is a power law?

**y = (x_c / x)^α.** On log–log axes this is a **straight line** with slope −α:

```
log y = α log x_c − α log x
```

**What α means:** multiplying x by k multiplies y by k^(−α).
- With α = 0.076, doubling N multiplies the loss by 2^(−0.076) = **0.949** (5% better).
- 10× N multiplies it by **0.839**.

**Why it's useful:** fit the line on small, cheap models, then **extrapolate** to huge ones.

---

## 3. The laws

### 3.1 The three single-variable laws (Eqs. 1.1–1.3)
Each applies when that resource is the **only bottleneck**:

```
L(N)     = (N_c / N)^α_N,          α_N ≈ 0.076,      N_c ≈ 8.8·10¹³ params      (big data, train to convergence)
L(D)     = (D_c / D)^α_D,          α_D ≈ 0.095,      D_c ≈ 5.4·10¹³ tokens      (big model, early stopping)
L(C_min) = (C_c / C_min)^α_C,      α_C ≈ 0.050,      C_c ≈ 3.1·10⁸ PF-days      (best model size for each budget)
```

**Examples:**

| model size N | L(N) |
|---|---|
| 10⁶ | 4.02 nats |
| 10⁸ | 2.83 nats |
| 10¹⁰ | 1.99 nats |

- **Mind the constants:** N_c, D_c and C_c depend on the tokenizer and have no deep meaning. The **exponents** are what matter.
- **A warning:** these laws must eventually stop, since the loss can't go below the entropy of natural language. The paper saw no sign of a floor yet.

### 3.2 Model size and data together (Eq. 1.5)

```
L(N, D) = [ (N_c / N)^(α_N/α_D) + D_c / D ]^α_D
```

Table 2 fit: α_N = 0.076, α_D = 0.103, N_c = 6.4·10¹³, D_c = 1.8·10¹³.

**Check the limits:**
- **D → ∞:** L → (N_c/N)^α_N, the model-size law.
- **N → ∞:** L → (D_c/D)^α_D, the data law.

**Overfitting** (Eq. 4.3): the relative excess loss from finite data is

```
δL = L(N, D)/L(N, ∞) − 1 ≈ [1 + (N/N_c)^(α_N/α_D) · D_c/D]^α_D − 1
```

- It depends on N and D only through **N^(α_N/α_D)/D = N^0.74/D**.
- To keep δL below the ~2% run-to-run noise, you need

  ```
  D ≳ 5·10³ · N^0.74      (Eq. 4.4)
  ```

- **Sub-linear:** an 8× bigger model needs only 8^0.74 ≈ 4.7× more data.

**Our demo's overfitting table:**

| N | D = 10⁹ tokens | D = 10¹¹ tokens |
|---|---|---|
| 10⁶ | 0.32% | 0.00% |
| 10⁹ | **20.5%** | 0.52% |

### 3.3 Learning curves (Eq. 1.6)

```
L(N, S) = (N_c / N)^α_N + (S_c / S_min)^α_S,          α_S ≈ 0.76, S_c ≈ 2.1·10³
```

- **The first term** is the loss you'd reach with infinite training.
- **The second term** shrinks as you train longer.
- S_min is the number of steps you would need at a very large batch size (Section 3.4).

### 3.4 The critical batch size (Eqs. 1.4, 5.1–5.5)
For a target loss, there is a trade-off between **steps** S and **examples processed** E = B·S:

```
(S / S_min − 1)(E / E_min − 1) = 1          (Eq. 5.1)
```

| batch size | effect |
|---|---|
| tiny | E ≈ E_min (no wasted data), but many steps |
| huge | S ≈ S_min (fewest steps), but wasted compute |
| **B_crit = E_min / S_min** | **twice the minimum steps AND twice the minimum compute**: the balanced choice |

**B_crit depends only on the loss, not on the model:**

```
B_crit(L) = B* / L^(1/α_B),       B* ≈ 2·10⁸ tokens, α_B ≈ 0.21
```

| loss | B_crit |
|---|---|
| 4 nats | 2.7·10⁵ tokens |
| 3 nats | 1.1·10⁶ tokens |
| 2 nats | 7.4·10⁶ tokens |

B_crit doubles for every ~13% drop in loss: (0.87)^(−1/0.21) ≈ 1.94.

### 3.5 Shape barely matters (Section 3.1, Figure 5)
At fixed N, changing the depth/width aspect ratio by 40× changes the loss by only a few percent. **Size is what counts.**

---

## 4. Spending a compute budget (Section 6)

**The question:** with C FLOPs, should you train a big model briefly or a small one for a long time?

**The answer:** combine L(N, S) with C ≈ 6·N·B_crit·S_min and minimise over N. This gives (Eqs. 1.7–1.8):

```
N_opt ∝ C^(α_C/α_N),   B ∝ C^(α_C/α_B),   S ∝ C^(α_C/α_S),     with α_C = 1/(1/α_S + 1/α_B + 1/α_N)
```

**Plugging in:**
- α_C = 1/(1/0.76 + 1/0.21 + 1/0.077) = 1/(1.32 + 4.76 + 12.99) = **0.052** (the directly fitted value is 0.050);
- N_opt ∝ C^**0.68**, B ∝ C^**0.25**, S ∝ C^**0.07**.

**The paper's empirical recipe (Table 5):**

```
N_opt = 1.3·10⁹ · C^0.73 params,  B = 2.0·10⁶ · C^0.24 tokens,  S = 5.4·10³ · C^0.03 steps,  D = 2·10¹⁰ · C^0.27 tokens
```

with C in PF-days.

**The striking conclusions:**
1. **Almost all extra compute should go into a bigger model** (exponent 0.73). The number of steps barely grows (0.03).
2. **Compute-efficient training stops long before convergence.** At the optimum, the loss is about **α_N/α_S ≈ 10% above** where the same model would end up if trained forever.
3. **Bigger models are more sample-efficient:** they reach a given loss with fewer tokens.
4. Data needs grow slowly: D ∝ C^0.27.

**Our numerical check** solves the frontier from the paper's own L(N, S) and B_crit fits:
- N_opt ∝ C^**0.66** (theory 0.68, empirical 0.73), within ~2× of Table 5's sizes;
- the loss at the optimum is exactly **1.103×** the converged loss;
- the frontier loss matches L(C_min) within a few percent.

**GPT-3 (paper 050)** followed this philosophy: a 175B model trained on "only" 300B tokens. **Chinchilla (paper 052)** later re-measured the trade-off and concluded **models and data should grow equally** (both ∝ C^0.5). This paper had under-weighted data, partly because of how the learning-rate schedule interacted with training length.

---

## 5. Other findings
- **Transformers vs LSTMs:** Transformers win, especially on later tokens in the context; LSTMs plateau after about 100 tokens.
- **Generalisation:** the loss on other text distributions tracks the training-distribution loss with a roughly constant offset, so the laws transfer.
- **Universality:** the smooth trends held across eight orders of magnitude of compute.

---

## 6. Why power laws? (an intuition, not from this paper)

**A simple mechanism:**
- Suppose the data is made of "patterns" with power-law importance: a few very common (frequent words, grammar), and a long tail of rarer ones (rare words, specific facts).
- If the importance of the k-th pattern falls as k^(−α), a model that can represent N patterns leaves the tail error

  ```
  Σ_{k>N} k^(−α) ≈ N^(1−α)/(α − 1)
  ```

  which is a power law in N.
- The same reasoning applies to data: each extra example lets you pin down a few more rare patterns.

**Our toy:** 4,000 patterns with importance k^(−1.5).
- **Limited capacity:** loss ∝ N^(−0.60) (theory 0.5);
- **limited data:** loss ∝ D^(−0.56).

Natural language's heavy-tailed (Zipfian) statistics fit this picture.

---

## 7. What our code found

All numbers come from `demo.py` (instant) and `test_scaling.py` (9 tests, about 1.5 s). They use the paper's fitted constants, not new training.

1. **Exact formulas:**
   - 2^(−0.076) = 0.949 per doubling of N;
   - L(N, D) reduces to L(N) and L(D) in the limits;
   - training at B = B_crit costs exactly 2·S_min steps and 2·C_min compute;
   - B_crit grows by 0.87^(−1/0.21) per 13% loss drop.
2. **The overfitting rule** D = 5·10³·N^0.74 keeps the predicted penalty between 0.5% and 5% for N from 10⁶ to 10⁹, around the paper's ~2% noise level.
3. **Eq. 1.8:** α_C = 0.052 from the other exponents (fitted: 0.050); N ∝ C^0.68.
4. **The compute frontier, solved numerically:**
   - **a bug fixed during development:** a naive fixed-point iteration oscillated and always picked the biggest model, so we replaced it with bisection (the defining equation is monotonic);
   - N_opt ∝ C^0.66, within ~2× of Table 5;
   - the loss at the optimum is 1.103× the converged loss, matching 1 + α_N/α_S;
   - the frontier loss agrees with L(C_min).
5. **GPT-3's budget** (~3,646 PF-days): Table 5 suggests about 5·10¹¹ parameters and 1.8·10¹¹ tokens. GPT-3 chose 1.75·10¹¹ parameters and 3·10¹¹ tokens.
6. **The "why power laws" toy:** exponents 0.60 (capacity) and 0.56 (data).

**Not run (too heavy):** on OpenWebText or Gutenberg with paper 049's GPT-2 code,
- E1: L(N) for 8 sizes, with and without embedding parameters;
- E2: depth/width/heads at fixed N;
- E3: L(N, D) grid and the Eq. 1.5 fit;
- E4: compute frontier from learning curves;
- E5: critical batch size from steps-vs-examples curves.

---

## 8. Check yourself

1. With α_N = 0.076, by what factor does the loss change if N grows 100×?
   <details><summary>Answer</summary>100^(−0.076) = 10^(−0.152) ≈ 0.705: about 30% lower.</details>
2. Why exclude embedding parameters from N?
   <details><summary>Answer</summary>Embeddings scale with vocabulary size, not modelling power. Counting only non-embedding parameters makes models of different depths and vocabularies fall on one clean power law.</details>
3. A 10⁹-parameter model. Roughly how many tokens to avoid overfitting, by Eq. 4.4?
   <details><summary>Answer</summary>5·10³ × (10⁹)^0.74 = 5·10³ × 10^6.66 ≈ 2.3·10¹⁰ tokens.</details>
4. What does training at B = B_crit cost compared with the minimum steps and minimum compute?
   <details><summary>Answer</summary>From (S/S_min − 1)(E/E_min − 1) = 1 with equal factors: S = 2·S_min and E = 2·E_min.</details>
5. Under this paper's recipe, if compute grows 10×, by how much should the model grow, and the data?
   <details><summary>Answer</summary>N by 10^0.73 ≈ 5.4×; data by 10^0.27 ≈ 1.9×.</details>
6. Why does compute-efficient training stop "early"?
   <details><summary>Answer</summary>Near convergence each extra step buys little (the S-term flattens). The same compute buys more loss reduction by making the model bigger and training it less.</details>
