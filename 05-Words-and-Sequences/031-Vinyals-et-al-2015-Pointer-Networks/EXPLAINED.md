# Pointer Networks (Vinyals, Fortunato & Jaitly, 2015), explained from scratch

**Paper:** *Pointer Networks*
**Authors:** Oriol Vinyals (Google Brain), Meire Fortunato (UC Berkeley), Navdeep Jaitly (Google Brain)
**Published at:** NIPS 2015 (arXiv:1506.03134)

This guide assumes only high-school math. Every symbol is defined before it's used, and every formula is worked through with small numbers. If you have read Papers 027 (Seq2Seq) and 028 (attention), parts 2–3 will be a refresher.

---

## 0. The whole idea in one picture

Imagine handing someone 6 dots on a sheet of paper and asking: *"Which dots form the outer fence around all the others?"* They don't invent new dots. They **point** at some of the dots you gave them, in order: "this one, then this one, then this one…".

A Pointer Network does the same. Its answer is a list of **positions in its own input**: "dot 2, then dot 4, then dot 3…". That sounds simple, but before this paper, sequence models couldn't do it properly. This guide explains why, and how one small change to attention fixes it.

---

## 1. The three problems in the paper (the science of the tasks)

All three take **n points in a square** as input. Each point P_j = (x_j, y_j) has coordinates between 0 and 1. All three answers are lists of point numbers.

### 1.1 Convex hull

**What it is:** the convex hull is the smallest convex polygon that contains all the points. Picture stretching a rubber band around nails hammered at the points and letting it snap tight. The nails it touches are the hull.

**How a computer finds it:** the key tool is the **cross product**, which tells whether three points make a left turn or a right turn. For points o, a, b:
```
cross(o, a, b) = (a.x − o.x)·(b.y − o.y) − (a.y − o.y)·(b.x − o.x)
```
- **> 0:** going o → a → b turns **left** (counter-clockwise).
- **< 0:** it turns **right**.
- **= 0:** the three points are on one line.

*Example:* o = (0,0), a = (1,0), b = (1,1). Then cross = (1−0)(1−0) − (0−0)(1−0) = 1 > 0. Walk east, then turn north: a left turn. ✔

**Andrew's monotone chain** (the algorithm in `tasks.py`) uses this:
1. Sort the points by x.
2. Walk left to right. Each time the last three points make a right turn, the middle one can't be on the hull, so throw it away.

That builds the bottom half. Do the same right to left for the top half. Sorting costs O(n log n), and every point is added and removed at most once, so the total is **O(n log n)**.

**The answer format (Section 3.1):**
- start from the hull point with the **lowest index**;
- go **counter-clockwise**;
- repeat the starting point at the end to "close" the polygon.

The paper's Figure 2a gives the example ⇒ 2, 4, 3, 5, 6, 7, 2 ⇐ (⇒ and ⇐ mark the start and end of the output).

**How it's scored (Table 1):**
- **Accuracy:** is the predicted polygon exactly the true hull? Starting at a different vertex of the same cycle still counts as correct.
- **Area coverage:** the predicted polygon's area divided by the true hull's area. Any simple (non-self-crossing) polygon on these points lies inside the hull, so this is at most 100%.

  Area is computed with the **shoelace formula**:
  ```
  Area = ½ · | Σ_k (x_k · y_{k+1} − y_k · x_{k+1}) |      (with the last vertex wrapping around to the first)
  ```
  *Example:* the unit square (0,0), (1,0), (1,1), (0,1):
  - the four terms are:
    - (x₁y₂ − y₁x₂) = 0·0 − 0·1 = 0;
    - (x₂y₃ − y₂x₃) = 1·1 − 0·1 = 1;
    - (x₃y₄ − y₃x₄) = 1·1 − 1·0 = 1;
    - (x₄y₁ − y₄x₁) = 0·0 − 1·0 = 0;
  - the sum is 2, so the area is ½·2 = **1** ✔.
- **"FAIL":** if more than 1% of the outputs are not simple polygons, the paper reports FAIL instead of an area.

### 1.2 Delaunay triangulation

**What it is:** connect the points into triangles so that **no point lies inside the circle through any triangle's three corners** (its "circumcircle").

**Why people care:**
- Among all ways to triangulate the points, it **maximizes the smallest angle**, so it avoids thin sliver triangles. That makes it the standard mesh for physics simulations.
- Its outer edges are exactly the convex hull, which is why it's the "next level up" from task 1.

**The answer format:** a list of triangles. Each triangle is a triple of point numbers, written in increasing order: (1, 2, 4), not (2, 4, 1). The triangles are sorted by their **incenter** (the centre of the circle that fits inside the triangle).

**Why sorting matters:** a triangulation is a **set**, so any order is equally correct. But a sequence model must output *some* order, and "many correct answers" makes the training signal noisy. The authors fixed one canonical order, and found that without it "the models learned were not as good". The companion paper, 032 *Order Matters*, is entirely about this.

**How it's scored:**
- **Accuracy:** the exact set of triangles is right.
- **Triangle coverage:** the percentage of true triangles that were predicted.

### 1.3 The Travelling Salesman Problem (TSP)

**What it is:** visit every city exactly once and return home, using the **shortest** total route.

**Why it's hard:**
- With n cities there are (n−1)!/2 different round trips:
  - n = 10: 181,440 trips;
  - n = 20: about 6 × 10¹⁶ trips.
- TSP is **NP-hard**: no known algorithm solves every instance quickly as n grows.

**Exact method: Held–Karp dynamic programming.**
- Let C(S, j) be the length of the shortest path that starts at city 1, visits exactly the cities in set S, and ends at city j. Then:
  ```
  C(S, j) = min over k in S∖{j} of [ C(S∖{j}, k) + dist(k, j) ]
  ```
  In words: the best way to end at j is to arrive from the best previous city k.
- There are 2ⁿ sets S, n endpoints j and n choices of k, so the cost is **O(2ⁿ · n²)**:
  - fine for n = 10 (about 10⁵ steps);
  - painful for n = 20 (about 4 × 10⁸ steps).

  The paper used it up to n = 20 to create perfect training answers.

**Approximate methods (A1, A2, A3 in Table 2)** are used to label bigger problems:
- **Greedy / nearest neighbour:** always go to the closest unvisited city. Fast, but often about 25% too long.
- **2-opt:** if two edges of the tour cross, reversing the segment between them uncrosses them and makes the tour shorter. Repeat until nothing improves.
- **Christofides:** guaranteed to be **at most 1.5 × the optimum**. Sketch of why:
  1. Build a minimum spanning tree (MST). Deleting one edge of the optimal tour gives a spanning tree, so MST ≤ OPT.
  2. Pair up the odd-degree vertices with a minimum matching. That costs ≤ OPT/2.
  3. Walk the combined graph (an Euler tour), skipping repeated cities. Thanks to the triangle inequality, skipping never makes it longer.

  In total: ≤ OPT + OPT/2 = 1.5·OPT.

**The answer format:** a permutation of the cities, always starting at city 1. The score is the average tour length (lower is better).

---

## 2. Background: how a sequence-to-sequence model makes an output

### 2.1 The chain rule of probability (Eq. 1)

We want the probability of a whole output sequence C = (C₁, C₂, …, C_m) given the input points P. Any joint probability can be split into one-step-at-a-time pieces:
```
p(C | P) = p(C₁ | P) · p(C₂ | C₁, P) · p(C₃ | C₁, C₂, P) · …  =  Π_i p(C_i | C₁ … C_{i−1}, P)       (1)
```
This is not an approximation. It's an exact identity, and it means a model only ever has to answer one question: **"given everything so far, what comes next?"**

### 2.2 Training by maximum likelihood (Eq. 2)

Choose the parameters θ (all the weights) to make the correct answers as likely as possible:
```
θ* = argmax_θ  Σ over training pairs  log p(C | P; θ)                                                     (2)
```
- **We take logs** because the log of a product is a sum of logs. That's numerically safer and easier to differentiate.
- **Maximizing log p is the same as minimizing −log p**, which is called the **cross-entropy loss**.

*Tiny example:* the model gives the correct next point probability 0.5. The loss for that step is −ln 0.5 = 0.69. If it gives 0.9, the loss is 0.105. If it gives 0.01, the loss is 4.6. **Confident and wrong is punished hard.**

### 2.3 The encoder and the decoder (Section 2.1)

**The encoder** is an LSTM (Paper 021) that reads the points one at a time:
```
e_j = LSTM_enc(P_j, e_{j−1})          j = 1 … n
```
After step j, the vector e_j is a summary of the points seen so far, with special focus on P_j itself.

**The decoder** is another LSTM that writes the answer. Its state d_i at output step i summarizes the input and the outputs produced so far.

**The classic seq2seq output layer:**
```
p(C_i | …) = softmax(W · d_i)          W has a FIXED number of rows = the size of the output vocabulary
```
- **The softmax** turns any list of numbers into probabilities: softmax(u)_k = e^{u_k} / Σ_l e^{u_l}. Every result is positive, and they sum to 1.
- **The problem the paper solves:** the vocabulary here is "point 1, point 2, …, point n". With 10 points you need 10 classes, and with 50 you need 50. A softmax layer has a fixed number of rows, so **one model can only handle one n**. The paper says: "we need to train a separate model for each n". And a model built for n = 10 cannot even *say* "point 15".

---

## 3. Attention, the ingredient (Section 2.2)

Bahdanau-style attention (Paper 028) lets the decoder look back at **all** encoder states instead of one summary. At output step i, it gives every input j a score:
```
u_ij = vᵀ · tanh(W₁ e_j + W₂ d_i)              j = 1 … n                                                 (3)
a_ij = softmax_j(u_ij)                          the attention weights: positive, sum to 1
d'_i = Σ_j a_ij · e_j                           a weighted average of the encoder states
```

**Reading Eq. 3 slowly:**
- **W₁ e_j:** "what input j offers". It's computed once per input and reused for every output step.
- **W₂ d_i:** "what the decoder is looking for right now".
- **tanh(…):** adds them and squashes each entry into (−1, 1). This "additive" form can express "j matches what I'm looking for" even when the two vectors are very different.
- **vᵀ · (…):** collapses that vector into a single number, the score u_ij.

**Worked example** (hidden size 2):
1. Suppose W₁e₁ = [0.5, −1.0] and W₂d_i = [0.3, 0.8].
2. Their sum is [0.8, −0.2].
3. tanh gives [0.664, −0.197].
4. With v = [1.0, 2.0], the score is u_i1 = 1.0·0.664 + 2.0·(−0.197) = **0.270**.
5. Do this for every j, then apply the softmax.

**In the classic attention model,** d'_i is glued to d_i, and the pair [d_i ; d'_i] goes into the usual fixed-size softmax. It is also fed into the next decoder step.

**The cost:** each output step scores n inputs, and there are about n output steps, so the cost is **O(n²)**.

---

## 4. The pointer: attention *is* the output (Section 2.3)

The whole contribution fits on one line. **Stop using the attention weights to blend the encoder states. Use them directly as the answer:**
```
u_ij = vᵀ · tanh(W₁ e_j + W₂ d_i)        j = 1 … n
p(C_i = j | C₁ … C_{i−1}, P) = softmax(u_i)_j
```

**Why this solves the problem:**
- The softmax runs over **u_i1 … u_in**, one score per input. So the number of choices is automatically **the input length n**, whatever n is.
- The same weights (v, W₁, W₂ and the LSTMs) work for n = 5, 50 or 500. A model trained on 5–50 points can be *run* on 500 (Table 1's bottom rows).

**Worked example:**
- Say there are 3 input points plus an "end" option (index 0), with scores u = [0.2, 1.5, −0.3, 0.9] for (end, P₁, P₂, P₃).
- e^u = [1.221, 4.482, 0.741, 2.460], which sums to 8.904.
- The probabilities are **[0.137, 0.503, 0.083, 0.276]**.

So the network "points" at P₁ with probability ½. If P₁ is the correct next answer, the loss is −ln 0.503 = 0.687.

**How learning adjusts the scores:**
- The gradient of the cross-entropy with respect to the scores is wonderfully simple: **∂loss/∂u = p − onehot(correct)** = [0.137, −0.497, 0.083, 0.276].
- Gradient descent subtracts this, so the correct score u₁ goes **up** and all the wrong ones go **down**, each in proportion to how much probability it wrongly took.
- Backpropagation then passes this signal into v, W₁, W₂ and both LSTMs.

**Feeding back what was chosen:**
- To condition on C_{i−1} (as Eq. 1 requires), the decoder's next input is **a copy of the chosen point's coordinates**, P_{C_{i−1}}.
- So the decoder literally "holds the point it just picked" while choosing the next one.

**Start and stop:**
- A special start symbol ⇒ is the first decoder input.
- The model must be able to *say* "stop" (⇐). The paper doesn't spell out how. **In our code we add one extra learned vector e₀ to the encoder states**, so "pointing at position 0" means stop.

---

## 5. Inference: beam search and constraints

**The search problem:** we want the most likely whole output, argmax_C p(C | P). Trying every sequence is impossible, so the paper uses **beam search**:
- keep the B most probable partial answers;
- extend each one by every possible next pointer;
- keep the best B again;
- repeat.

B = 1 is "greedy" decoding.

**For TSP,** the decoder is **constrained** to valid tours:
- cities already visited are masked out (their probability is set to 0);
- "stop" is only allowed after every city has been visited.

The paper notes that without this, for n > 20, "at least 10% of instances would not produce any valid tour". For convex hull and Delaunay the decoder is unconstrained.

---

## 6. Training details (Section 4.1)

- **The same setup for every task** (no tuning per problem, to keep the message clean):
  - a 1-layer LSTM with 256 or 512 units;
  - SGD with learning rate 1.0, batch 128;
  - weights initialized uniformly in [−0.08, 0.08];
  - gradient norm clipped at 2.0 (if the gradient vector is longer than 2, scale it down to length 2; this stops rare huge steps).
- **Data:** 1M training pairs per model, with points sampled uniformly. Some overfitting was seen for small n.
- **Curricula:** training on lengths 5–50 sampled uniformly worked; "other forms of curriculum learning [were] not effective".

---

## 7. Results

### Table 1: convex hull
| Method | Trained n | Tested n | Accuracy | Area |
|---|---|---|---|---|
| LSTM (seq2seq) | 50 | 50 | 1.9% | FAIL |
| + attention | 50 | 50 | 38.9% | 99.7% |
| **Ptr-Net** | 50 | 50 | **72.6%** | 99.9% |
| LSTM | 5 | 5 | 87.7% | 99.6% |
| Ptr-Net | 5–50 | 5 | 92.0% | 99.6% |
| LSTM | 10 | 10 | 29.9% | FAIL |
| Ptr-Net | 5–50 | 10 | 87.0% | 99.8% |
| Ptr-Net | 5–50 | 50 | 69.6% | 99.9% |
| Ptr-Net | 5–50 | 100 | 50.3% | 99.9% |
| Ptr-Net | 5–50 | 200 | 22.1% | 99.9% |
| Ptr-Net | 5–50 | 500 | 1.3% | 99.2% |

**How to read it:**
- **At n = 50, pointing beats blending:** 72.6% vs 38.9%. The plain LSTM fails completely.
- **At n = 5 the models are close** (92.0 vs 87.7). The advantage grows with n.
- **Extrapolation:** trained on at most 50 points, it still gets **99.2% of the area at n = 500**, ten times longer than anything it saw. Exact accuracy falls, because one wrong point out of hundreds makes the whole answer "wrong". But the polygons are almost right, which "indirectly indicate[s] that the model has learned more than a simple lookup".
- **The inspiration:** while building the attention baseline, the authors noticed that its attention was already "pointing" at the correct answer. That observation led to Ptr-Net.
- **Input order matters:** when hull points appear late in the input, accuracy drops. The encoder has too few steps left to "update" its idea of the hull. Paper 032 studies this.
- **Typical mistakes:** nearly collinear points, which are hard for exact algorithms too.

### Delaunay (Section 4.3)
| n | Accuracy (exact) | Triangle coverage |
|---|---|---|
| 5 | 80.7% | 93.0% |
| 10 | 22.6% | 81.3% |
| 50 | 0% | 52.8% |

Harder than the hull: the output is about 2n triangles (6n pointers), and they all must be right.

### Table 2: TSP average tour length (lower is better)
| n | Optimal | A1 | A2 | A3 | **Ptr-Net** |
|---|---|---|---|---|---|
| 5 | 2.12 | 2.18 | 2.12 | 2.12 | **2.12** |
| 10 | 2.87 | 3.07 | 2.87 | 2.87 | **2.88** |
| 50 (trained on A1) | – | 6.46 | 5.84 | 5.79 | **6.42** |
| 50 (trained on A3) | – | 6.46 | 5.84 | 5.79 | **6.09** |
| 20 (trained on 5–20) | 3.83 | 4.24 | 3.86 | 3.85 | 3.88 |
| 30 (trained on 5–20) | – | 5.11 | 4.63 | 4.60 | 4.72 |
| 50 (trained on 5–20) | – | 6.46 | 5.84 | 5.79 | 7.66 |

**How to read it:**
- **Small n: essentially optimal.**
- **Imitating a weak teacher:** trained on A1's tours, the student (6.42) is *better* than A1 (6.46).
- **Generalizing to longer tours** works up to about 30 cities, then breaks down at 40–50. The hull model extrapolated 10×, but TSP's underlying algorithm is far more complex than the hull's O(n log n), which the authors suggest is the reason.

---

## 8. Why this paper matters

- **"Point at the input" became a standard building block:**
  - **copy mechanisms** in summarization (pointer-generator networks copy rare names from the source);
  - extractive question answering (point at the answer span's start and end words);
  - neural combinatorial optimization (Bello et al. 2016 trained Ptr-Nets on TSP with reinforcement learning, which removed the need for an exact solver to make labels).
- **Variable-size outputs:** it showed that attention is not just a helper for translation. It's a general way to let a network's output size **follow its input**, which matters whenever the answer is "some of the things you were given".

---

## 9. What our code found (honest notes)

**Scale note:**
- At your request, no heavy training was run on this laptop.
- `experiments.py` reproduces Table 1, the Delaunay numbers and Table 2 with the paper's setup (1M examples per model, SGD lr 1.0, batch 128, clip 2). It is **not run**.

**Checked in tests and the demo (about 20 seconds):**
- **The solvers are correct:**
  - `convex_hull` matches scipy's hull, starts at the lowest index and runs counter-clockwise;
  - every Delaunay triangle passes the empty-circumcircle test;
  - Held–Karp equals brute force for n ≤ 7;
  - Christofides + 2-opt is within its 1.5× guarantee;
  - 2-opt never makes a tour longer.
- **The pointer's dictionary really grows with the input.** One model outputs n + 1 probabilities for n = 3, 7 or 20. The seq2seq baseline always outputs n_max + 1.
- **Sorting** (a pointer task that trains in seconds), trained on 5–10 numbers:
  - **75%** exactly sorted at n = 5 and 20% at n = 10;
  - at lengths **never seen**, it still puts 58.5% of positions right at n = 15 and 31% at n = 20.

  The single model *runs* at any length, which a seq2seq model can't. But in a few seconds of training it learned a rough rule, not the exact algorithm.
- **Convex hull at n = 5,** about 77,000 examples (the paper used 1M): Ptr-Net **46.5%** vs seq2seq **46.0%**, both FAIL on area. That matches the paper's small gap at n = 5 (92.0 vs 87.7). The big gap only appears at n = 50, which is too big to train here.
- **Training is fragile at this scale:**
  - the loss sits on a long plateau, then suddenly drops;
  - with lr 3e-3 instead of 1e-2, the Ptr-Net was still stuck (0% accuracy, the same output for every input) after the same budget.
- **TSP heuristics at n = 8** (mean of 20): optimal 2.623, greedy-edge 2.776, NN + 2-opt 2.624, Christofides + 2-opt 2.637. The same pattern as Table 2's n = 10 row.

**Where we differ from the paper, and why:**
- **Input embedding:** the paper feeds raw (x, y) into the LSTM. With raw inputs and the small uniform init, our short runs hardly learned (sorting loss 8.2 vs **1.2** with a learned linear embedding, same 300 steps). So we embed by default. `embed=False` gives the literal version.
- **The stop token:** pointing at a learned extra vector e₀ is our choice; the paper doesn't describe it.
- **A1/A2/A3** are three GitHub solvers in the paper. We use greedy-edge, nearest-neighbour + 2-opt, and Christofides + 2-opt as stand-ins.
- **Exact TSP labels:** the experiments use exact labels up to n = 12 by default (the paper went to 20), to keep data generation feasible.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| encoder / decoder | the LSTM that reads the input / the LSTM that writes the output |
| e_j, d_i | encoder state after input j; decoder state at output step i |
| softmax | turns scores into probabilities: e^{u_k} / Σ e^{u_l} |
| attention | a softmax over input positions, used to weight them |
| pointer | using that softmax *as the answer* |
| cross-entropy | −log(probability given to the correct answer) |
| beam search | keep the B best partial answers at each step |
| NP-hard | no known fast exact algorithm for all instances |
| teacher forcing | during training, feed the *correct* previous answer, not the model's guess |

---

## 11. Check yourself

1. Why can't a normal seq2seq model, trained on 10 points, answer for 15 points?
2. Write the pointer equation. Which part makes the number of choices equal to n?
3. Scores are [0, ln 3] for two options. What are the probabilities? (Answer: ¼ and ¾.)
4. If the correct option got probability ¾, what is the loss, and what is ∂loss/∂u? (Answer: −ln ¾ ≈ 0.288, and [¼, −¼].)
5. Why must Delaunay triangles be put in a fixed order before training?
6. What does constraining the TSP beam search prevent?
7. Why can a student trained on A1's tours beat A1?
8. Explain the cost O(n²) of attention in one sentence.
