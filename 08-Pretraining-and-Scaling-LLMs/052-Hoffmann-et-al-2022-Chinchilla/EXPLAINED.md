# Chinchilla: Training Compute-Optimal Large Language Models, explained simply

**Paper:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals & Laurent Sifre (DeepMind), *Training Compute-Optimal Large Language Models*, NeurIPS 2022.

**In one sentence:** for a fixed compute budget, you should grow the **model size and the training data equally**, about **20 tokens per parameter**. By that rule, GPT-3 and Gopher were far too big and trained on far too little data. A 4× smaller model, Chinchilla (70B), trained on 4× more data (1.4T tokens), beats Gopher (280B) at the same cost.

---

## 1. The big idea

### 1.1 The question
- A training run costs C ≈ **6·N·D** FLOPs (N = parameters, D = training tokens; paper 051).
- **With C fixed:** a bigger model means fewer tokens, and more tokens means a smaller model.
- **Which split gives the lowest loss?**

### 1.2 Two answers
| | N_opt ∝ C^a | D_opt ∝ C^b | 10× compute means |
|---|---|---|---|
| Kaplan et al. 2020 (paper 051) | a = **0.73** | b = **0.27** | model ×5.4, data ×1.9 |
| **Chinchilla** | a ≈ **0.50** | b ≈ **0.50** | model ×3.2, data ×3.2 |

Following Kaplan, the large models of 2020–21 were all trained on about 300B tokens (Table 1):

| model | parameters | training tokens |
|---|---|---|
| LaMDA | 137B | 168B |
| GPT-3 | 175B | 300B |
| Jurassic | 178B | 300B |
| Gopher | 280B | 300B |
| MT-NLG | 530B | 270B |
| **Chinchilla** | **70B** | **1.4T** |

### 1.3 Why Kaplan got a different answer
The main methodological difference was the **learning-rate schedule**:
- Kaplan used a fixed schedule length for all runs, so shorter runs ended with a learning rate that hadn't decayed. That made **training longer look less useful** than it is.
- Chinchilla always sets the **cosine schedule length to match the run**.
- **Appendix B:** overshooting the cycle length by more than ~25% clearly hurts.

Chinchilla also includes larger models and counts parameters including embeddings, but the schedule is the key point.

---

## 2. Three ways to find the optimum (Section 3)
They trained **over 400 models** from 70M to 16B parameters, on 5B to 500B tokens.

### 2.1 Approach 1: the minimum over training curves
1. Train each model size with several schedule lengths, and record loss vs FLOPs throughout.
2. At every FLOP value, find which model's curve is lowest. That is the **lower envelope**.
3. Fit power laws to the winning (N, D) pairs.

**Result: a = 0.50, b = 0.50.**

### 2.2 Approach 2: IsoFLOP profiles
1. Fix 9 budgets (6·10¹⁸ to 3·10²¹ FLOPs). For each, train models of many sizes, with D = C/(6N).
2. The loss vs log N is a **valley**: too small a model underfits, too big a model sees too little data.
3. Fit a **parabola** in log N; its minimum is N_opt(C).

**Result: a = 0.49, b = 0.51.**

**The parabola minimum:** for L = c₂x² + c₁x + c₀ with x = log N, the minimum is at x* = −c₁/(2c₂).

### 2.3 Approach 3: fit a formula for the loss

```
L(N, D) = E + A / N^α + B / D^β                     (Eq. 2)
```

| term | meaning |
|---|---|
| **E** | the irreducible loss of a perfect model: the "entropy of natural text" |
| **A/N^α** | the extra loss from a model too small to represent the ideal predictor |
| **B/D^β** | the extra loss from finite training (finite data and optimisation steps) |

**Fitted values (Eq. 10):** E = **1.69**, A = **406.4**, B = **410.7**, α = **0.34**, β = **0.28**.

**How it's fitted** (Eqs. 3, 11):
- minimise Σ Huber_δ(log L̂ − log L) with δ = 10⁻³, using **L-BFGS**, from a grid of starting points;
- the robust Huber loss downweights outliers;
- the model is written in log-sum-exp form, log L̂ = LSE(a − α log N, b − β log D, e), so A, B, E = exp(a), exp(b), exp(e) stay positive.

**The closed-form frontier (Eq. 4).** Minimise L subject to 6ND = C.
1. Substitute D = C/(6N). Setting dL/dN = 0 gives α·A/N^α = β·B/D^β: **the two gaps are balanced** (weighted by their exponents).
2. Solving:

   ```
   N_opt = G (C/6)^a,   D_opt = G⁻¹ (C/6)^b,   G = (αA / βB)^(1/(α+β)),   a = β/(α+β),   b = α/(α+β)
   ```

3. With α = 0.34 and β = 0.28: **a = 0.28/0.62 = 0.45**, b = 0.55.

**Result: a = 0.46, b = 0.54.**

**All three approaches agree:** parameters and data should scale roughly equally.

---

## 3. What this predicts (Table 3)

| parameters | FLOPs | tokens | tokens per parameter |
|---|---|---|---|
| 400M | 1.92·10¹⁹ | 8.0B | 20 |
| 1B | 1.21·10²⁰ | 20.2B | 20 |
| 10B | 1.23·10²² | 205.1B | 21 |
| 67B | 5.76·10²³ | 1.5T | 22 |
| 175B | 3.85·10²⁴ | 3.7T | 21 |
| 280B | 9.90·10²⁴ | 5.9T | 21 |
| 1T | 1.27·10²⁶ | 21.2T | 21 |

- **The rule of thumb: about 20 training tokens per parameter.**
- A 175B model would have needed about 3.7T tokens, not 300B.
- Getting enough high-quality data becomes the bottleneck.

### 3.1 Worked example
Gopher's budget is C = 5.76·10²³ FLOPs.
- **Approach 1's answer:** 67B parameters on 1.5T tokens. Check: 6 × 67·10⁹ × 1.5·10¹² = 6.0·10²³ ✓.
- **Chinchilla** used 70B and 1.4T: 6 × 70·10⁹ × 1.4·10¹² = 5.9·10²³, the same budget as Gopher's 280B × 300B.

---

## 4. Chinchilla vs Gopher (Section 4)
Same compute, same data source (MassiveText) and same architecture family. The differences:
- **AdamW** instead of Adam;
- a slightly modified tokenizer (no NFKC normalisation);
- **bfloat16 weights with an fp32 master copy**.

**Under the fitted law** (Eq. 10):

| model | entropy E | model gap | data gap | total |
|---|---|---|---|---|
| Gopher (280B, 300B tokens) | 1.69 | 0.052 | 0.251 | **1.993** |
| Chinchilla (70B, 1.4T tokens) | 1.69 | 0.083 | 0.163 | **1.937** |

Gopher wasted compute shrinking an already-small model gap while leaving a large data gap.

**Benchmarks:**

| benchmark | Chinchilla | Gopher | other |
|---|---|---|---|
| **MMLU** (5-shot, 57 subjects) | **67.6%** | 60.0% | GPT-3 43.9%; expert forecast for June 2023: 63.4% |
| BIG-bench (average) | **65.1%** | 54.4% | |
| reading comprehension, closed-book QA, common sense | better on most | | |

Chinchilla is also better than GPT-3, Jurassic-1 and MT-NLG.

**A bonus:** a 4× smaller model is **4× cheaper to run and fine-tune**, which matters because inference costs often dominate.

---

## 5. Caveats (including ones found later)
1. **The paper's own limits:**
   - only two large-scale points (Chinchilla, Gopher);
   - most data seen less than once (behaviour with many epochs is unknown);
   - slight curvature in the frontier at the largest scales hints the optimal model may be even smaller.
2. **Exponent sensitivity:** the closed-form frontier depends strongly on α and β.
   - With the rounded values (0.34, 0.28) we get a 32B optimum for Gopher's budget.
   - With more precise values quoted in later work (0.3392, 0.2849) we get **40.3B**, which matches the paper's quoted Approach 3 value of ~40B.
3. **Approach 3 is inconsistent with Approaches 1–2:**
   - its frontier implies **59–93 tokens per parameter**, against ~20 for Approaches 1–2;
   - a 2024 replication (Besiroglu et al., "Chinchilla Scaling: A replication attempt") refitted the data and found estimates consistent with ~20 tokens per parameter, attributing the gap to the original Approach 3 fit;
   - the headline "a ≈ b ≈ 0.5, about 20 tokens per parameter" rests mainly on Approaches 1–2.
4. **"Compute-optimal" ≠ "best to deploy."**
   - If a model will serve billions of requests, a *smaller* model trained on *more* tokens than "optimal" can be cheaper overall.
   - Later models like LLaMA (paper 056) deliberately train far past 20 tokens per parameter.

---

## 6. Why it works (the intuition)

1. **Two error sources, both shrinking as power laws.**
   - A bigger model reduces approximation error.
   - More data and more steps reduce estimation and optimisation error.
   - With a fixed budget, you should keep cutting whichever error is currently cheaper to cut. At the optimum, the marginal returns are equal.
2. **The exponents are similar (0.34 vs 0.28).** Neither resource dominates, so both should grow at similar rates. If α were much larger than β, model size would deserve most of the budget, and vice versa.
3. **A matched cosine schedule tells the truth.** Each run anneals fully, so its loss reflects what that token count can really achieve.

---

## 7. What our code found

All numbers come from `demo.py` (about 0.5 s) and `test_chinchilla.py` (7 tests, about 1 s).

1. **The closed form is right:**
   - Eq. 4 matches brute-force minimisation of L(N, C/6N) within 1%;
   - at the optimum, αA/N^α = βB/D^β exactly;
   - a = 0.452, b = 0.548.
2. **Table 3 is self-consistent:** 6ND matches the listed FLOPs (within 5%) at 20–22 tokens per parameter.
3. **Chinchilla vs Gopher under the law:** 1.937 vs 1.993 nats at the same budget. Kaplan's recipe would build ~800B parameters for that budget.
4. **The three approaches on synthetic runs** from the fitted law (true a = 0.452):

   | approach | recovered |
   |---|---|
   | 1 (40 sizes) | a = **0.457** |
   | 2 (with 0.2% noise) | a = **0.451** |
   | 3 (Huber + L-BFGS) | E = 1.690, α = 0.339, β = 0.279 |

   **Honest note:** with only 5 model sizes, Approach 1's envelope gave a ≈ 0.93 in our tests; the envelope needs many sizes.
5. **FLOPs:** our Appendix F count is 1.68× 6ND for a 70M model at context 2048 (attention over the context and the embeddings dominate at that width), but only 1.05× at 65B.
6. **Exponent sensitivity:** with rounded vs precise exponents, Gopher's optimum is 32B vs 40B (93 vs 59 tokens per parameter).

**Not run (too heavy):** small-scale real-model versions of
- E1: IsoFLOP profiles;
- E2: the training-curve envelope;
- E3: the parametric fit;
- E4: cosine cycle length;
- E5: Chinchilla-style vs Kaplan-style allocation at equal FLOPs.

---

## 8. Check yourself

1. A budget of 1.2·10²¹ FLOPs. Using 20 tokens per parameter and C = 6ND, what N and D?
   <details><summary>Answer</summary>6·N·20N = 120N² = 1.2·10²¹, so N = √(10¹⁹) ≈ 3.2·10⁹ parameters and D ≈ 63B tokens.</details>
2. Why do the exponents α and β decide how to split the budget?
   <details><summary>Answer</summary>The optimum balances αA/N^α against βB/D^β, so the share going to N is a = β/(α+β). If the data term shrinks more slowly (β < α), data deserves a bigger share of the scaling.</details>
3. With α = 0.34 and β = 0.28, by what factor should N grow if compute grows 100×?
   <details><summary>Answer</summary>a = 0.28/0.62 = 0.452, so 100^0.452 ≈ 8×; D grows 100^0.548 ≈ 12.5×.</details>
4. What is E, and why can't any amount of scaling remove it?
   <details><summary>Answer</summary>E is the loss of the ideal predictor: the inherent unpredictability (entropy) of text. Even a perfect model with infinite data pays it.</details>
5. Why did Kaplan et al. underestimate the value of data?
   <details><summary>Answer</summary>Mainly because they used one fixed learning-rate schedule length, so shorter runs weren't fully annealed. Their measured loss at a given token count was worse than achievable, making extra tokens look less valuable.</details>
6. Why might you train a model with *more* than 20 tokens per parameter on purpose?
   <details><summary>Answer</summary>Inference cost scales with model size. If the model will be used heavily, a smaller model trained longer, though not compute-optimal for training, can be cheaper over its whole lifetime (the LLaMA philosophy).</details>
