# Vinyals, Bengio & Kudlur (2016), explained from scratch

**Paper:** *Order Matters: Sequence to Sequence for Sets*
**Authors:** Oriol Vinyals, Samy Bengio, Manjunath Kudlur (Google Brain)
**Published at:** ICLR 2016 (arXiv:1511.06391)

Read Papers 027 (Seq2Seq), 028 (attention) and 031 (Pointer Networks) first. This paper is **optional** in the reading order. This guide:
- explains why the order of data matters to models that "in theory" don't care;
- shows how to build an encoder that provably ignores order;
- shows how a model can *choose* its own output order.

---

## 0. The whole idea in one line

> **Seq2seq models read and write sequences, but many inputs and outputs are really sets (numbers to sort, objects in an image, a triangulation). The order you pick still changes how well the model learns. For input sets, use an attention-based encoder whose summary cannot depend on order (Read–Process–Write). For output sets, let the model search over orders during training (Eq. 9).**

---

## 1. Why order should not matter, and why it does

### 1.1 The chain rule works in any order
- **Any joint probability can be factored in any order:**
  ```
  p(y₁, y₂, y₃) = p(y₁) p(y₂|y₁) p(y₃|y₁,y₂) = p(y₃) p(y₁|y₃) p(y₂|y₁,y₃) = …        (Eq. 8, and Bayes' rule)
  ```
- **So with a powerful enough model,** every order represents the same distribution exactly.
- **An RNN encoder** is a universal approximator: in principle it can compute any feature of its input, whatever order it reads it in.

### 1.2 But learning is not "in principle"
Training is **non-convex optimization with finite data**. Some orders give the model **easy intermediate steps**:
- **Seq2Seq** (Paper 027): reversing the source added 5 BLEU (short time lags at the start).
- **Parsing** (Paper 030): reversing the sentence added 0.5 F1.
- **Convex hull** (Paper 031): feeding the points sorted by angle turns an O(n log n) problem into an O(n) one, and accuracy rose by up to 10%.

**The paper's thesis:** *"often for optimization purposes, the order in which input data is shown to the model has an impact on the learning performance."*

### 1.3 Two kinds of order
- **Input-independent:** e.g. always reverse the words.
- **Input-dependent:** e.g. sort the points by angle. Each example gets its own order.

---

## 2. Input sets: Read–Process–Write (Section 4)

### 2.1 What "set encoder" must mean
- **The requirement:** swapping any two elements x_i, x_j must leave the encoding **unchanged** (permutation invariance).
- **The simplest invariant encoder** is a sum: Σ_i embed(x_i), as in "bag of words".
- **The paper's objection:** a fixed-size sum must hold a set of any size, but "the amount of memory required to encode a length T set should increase as a function of T".

### 2.2 The fix: keep a memory, read it with attention (Figure 1)
**Read:** embed each element with the **same** small network: m_i = MLP(x_i). That gives one memory per element, so the memory **grows with the set**.

**Process:** an LSTM **with no inputs** runs T steps. At each step it queries the memory with content-based attention:
```
q_t   = LSTM(q*_{t−1})                         (3)    the query evolves
e_i,t = f(m_i, q_t)            (we use m_i·q_t) (4)    score each memory
a_i,t = softmax_i(e_i,t)                        (5)
r_t   = Σ_i a_i,t m_i                           (6)    read: a weighted average
q*_t  = [q_t ; r_t]                             (7)    query + what was read
```
**Why q*_T ignores order:**
- **The memories enter only through (4)–(6):** a score for each memory, then a softmax and a weighted **sum**.
- **Permuting the memories** permutes the scores and the weights together, and leaves the sum unchanged.
- **The LSTM never sees the elements one by one,** only r_t.
- **So by induction over t,** every q*_t, including the final summary, is identical for every ordering.
- **Our demo:** shuffling 6 inputs changes the Read–Process encoding by 3·10⁻⁸ (round-off), but changes an LSTM encoder's by 10⁻³.

**Why "process" steps help:**
- **Each step is another round of reasoning:** the LSTM can attend to one element, update its state, then attend somewhere else.
- **More steps give a richer summary of the set,** like reading it several times. With **P = 0** steps, the writer starts with **no summary at all** and must "blindly" point.

### 2.3 Write: a pointer network with glimpses
- **The output:** a pointer network (Paper 031) started from q*_T produces indices into the set, e.g. the sorted order.
- **A "glimpse"** is one extra attention read over the memories **before** each pointer step, so the decoder can look at the set again while deciding where to point.
- **Our implementation detail:** we **add** the glimpse readout to the decoder state, rather than replacing the state with it. A literal replacement stalled our set model: at the start, attention is nearly uniform, so the readout is the *average* memory at every step, and the pointer can't tell the steps apart.

### 2.4 Table 1: sorting N numbers (out-of-sample accuracy: the whole list sorted exactly)
| Length N | Ptr-Net (no glimpse / glimpse) | P = 0 | P = 1 | P = 5 | P = 10 |
|---|---|---|---|---|---|
| 5 | 81% / 90% | 65% / 84% | 84% / 92% | 88% / 94% | 90% / 94% |
| 10 | 8% / 28% | 7% / 30% | 14% / 44% | 17% / 57% | 19% / 50% |
| 15 | 0% / 4% | 1% / 2% | 0% / 5% | 2% / 4% | 0% / 10% |

- **With P = 0, the order-blind model is worse than the Ptr-Net.**
- **With ≥ 1 process step, it is better,** and more steps help.
- **Glimpses** roughly **double** the accuracy in the hard cases.
- **Accuracy collapses with N:** exact-match accuracy demands all N outputs right. With per-step accuracy p, the chance of a perfect list is about pᴺ (0.95¹⁵ = 0.46, 0.9¹⁵ = 0.21).

---

## 3. Output sets: order matters there too (Section 5)

### 3.1 Language modelling (5.1.1)
- **The experiment:** PTB with the regularized "medium" LSTM (Paper 024), trained on:
  - natural order ("This is a sentence .");
  - reversed (". sentence a is This");
  - **3-word reversal** ("a is This <pad> . sentence": every block of 3 words flipped).
- **Results:** natural and reversed both reach **86** perplexity. 3-word reversal reaches **96**, and its *training* perplexity is also 10 points higher, so it's an **optimization** difficulty, not overfitting.

### 3.2 Parsing (5.1.2, Figure 2)
- **The same tree linearized two ways:**
  ```
  depth first:   S NP DT !DT !NP VP VBZ !VBZ NP DT !DT NN !NN !NP !VP . !. !S
  breadth first: S LEV NP VP . LEV DT PAR VBZ NP LEV PAR PAR DT NN DONE
  ```
- **How breadth-first is written:** level by level (LEV starts a level). Within a level, it lists the children of each node of the previous level, with groups separated by PAR. Trailing empty groups are dropped.
- **Result:** depth-first **89.5 F1**, breadth-first **81.5**.
- **Why:** depth-first keeps each word's subtree together and close to the word itself, while breadth-first scatters it.

### 3.3 Combinatorial problems: shrink the equivalence class (5.1.3)
- **The problem:** sorting's answer is a deterministic function, but if you write the output as a *set* of (index, rank) pairs, any of the **n! orders** of those pairs is correct.
- **Training on random orders** forces the model to spread probability over n! equally valid outputs:
  - for n = 5 that's 120, so each gets at most 1/120;
  - the target is then **much noisier** to learn.
- **Paper 031 already fixed one canonical order** (TSP from city 1, counter-clockwise; Delaunay triangles sorted) and gained ≥ 5%.
- **Here:** for sorting with random output orders, "convergence for n as small as 5 never reached the same performance".

### 3.4 Graphical models (5.1.4)
- **The setup:** a **star**: one "head" variable y₀ ~ Cat(π), and each other y_j depends **only** on the head, y_j ~ Cat(P_j[y₀]).
- **The true joint:**
  ```
  p(y₀, …, y_n) = p(y₀) · Π_j p(y_j | y₀)
  ```
- **Head first,** the LSTM's chain rule matches this structure: after reading y₀, each p(y_j | everything so far) is just a table lookup.
- **Head last,** the LSTM must model p(y₁), then p(y₂ | y₁), …, mixtures over the unseen head. That's correct in principle, but far harder to learn from limited data.
- **Paper:**
  - with enough data (20,000) or nearly deterministic tables, either order works;
  - **otherwise head-first is always easier**.
- **Our demo** (10 variables, 500 samples, 3 random models): **head first is 0.26 nats better on average,** and better in every model.

---

## 4. Letting the model choose its output order (Section 5.2, Eq. 9)

```
θ* = argmax_θ Σ_i max_π log p(Y_π(X_i) | X_i ; θ)                     (9)
```
For each training example, train on **the order the model currently finds most likely**.

**Two problems, and the paper's fixes:**
1. **n! is huge.** Instead of the max, **sample** an order π with probability ∝ p(Y_π | X), using **ancestral sampling**:
   - pick the first element in proportion to the model's p(element);
   - then the second given the first;
   - and so on.

   That's one left-to-right pass instead of n! evaluations.
2. **Lock-in:** a pure max picks whatever order the random initialization slightly prefers, then reinforces it forever. So **pre-train with a uniform prior** over orders (a fresh random order every time) for 1000 steps, and only then switch to sampling.

**A subtlety (our analysis):**
- **The sampled probability:** ancestral sampling picks π with probability Π_t p(y_{π_t} | prefix)/Z_t, where Z_t sums the model's probability over the elements **not yet emitted**.
- **That equals p(Y_π)/Σ_π′ p(Y_π′) only when every Z_t = 1,** i.e. the model never wastes probability on elements outside the set. A trained model gets close; early on it's an approximation.
- **Our test** checks the sampling frequencies against this formula.

### 4.1 Table 2: 5-grams as sets of (word, position)
- **The setup:** "This is a five gram" becomes {(This,1), (is,2), (a,3), (five,4), (gram,5)}. Any order of the pairs still determines the sentence.

| Task | Orders considered | Perplexity |
|---|---|---|
| (1,2,3,4,5) | 1 | 225 |
| (5,1,3,4,2) | 1 | 280 |
| easy (data in either of the two orders) | 2 | 225 |
| hard (data in any order) | 5! = 120 | 225 |

- **In the easy setting,** Eq. 9 quickly settles on the natural order.
- **In the hard setting,** it settles on (1,2,3,4,5), (5,4,3,2,1) and small variations, all reaching the natural-order perplexity **without being told any order**.

**Our demo** (sets of 4 tokens from a Markov chain, 600 updates):

| Training | nats/token in the model's best order |
|---|---|
| natural order | 0.59 |
| uniform random orders | 1.22 |
| **Eq. 9 (uniform pretraining, then sampling)** | **0.77** |

Sampling concentrates the model on a few orders. It hasn't settled on one in 600 updates; the paper trained far longer.

---

## 5. Why it matters

- **Set encoders:** "embed each element, combine with an order-free operation" became **Deep Sets** (Zaheer et al. 2017). Attention-based set encoders led to the **Set Transformer** (2019).
- **The Transformer:** its self-attention is permutation-*equivariant*, and it has to *add* position information precisely because attention ignores order.
- **Object detection:** DETR (2020) predicts a **set** of objects and solves the output-order problem with a **matching** loss (Hungarian matching), the same "pick the best assignment during training" idea as Eq. 9.
- **The broader lesson:** what a model *can* represent and what it can *learn* are different questions. Data ordering, curriculum and decomposition order all shape the second.

---

## 6. What our code found

**Scale note:**
- At your request, nothing heavy was run on this laptop.
- `experiments.py` reproduces:
  - E1: Table 1;
  - E2: the PTB orderings;
  - E3: depth- vs breadth-first parsing, reusing Paper 030's parser;
  - E4: sorting as an output set;
  - E5: the graphical-model grid;
  - E6: Table 2 on PTB 5-grams.

**Checked (tests and demo, ~13 seconds):**
- **Permutation invariance:** the Read–Process encoding changes by 3·10⁻⁸ under shuffling, and the LSTM encoder's by 10⁻³. The process attention is a proper distribution, and r_t is exactly the weighted average (Eq. 6). With P = 0 the writer's start state doesn't depend on the input at all.
- **Table 1 in miniature** (N = 5, 600 updates each):

  | model | sorted exactly |
  |---|---|
  | Ptr-Net, 1 glimpse | 74.6% |
  | **P = 0**, 1 glimpse | **65.2%** |
  | P = 1, no glimpse | 76.4% |
  | **P = 1, 1 glimpse** | **84.4%** |

  The same pattern as the paper: P = 0 is worst, processing beats the LSTM encoder, and glimpses help.
- **Figure 2 linearizations, character for character,** and breadth-first is invertible (50 random trees round-trip).
- **3-word reversal** gives exactly "a is This <pad> . sentence".
- **The star model:** an exact, normalized distribution; head-first beats head-last by 0.26 nats on average.
- **Eq. 9:**
  - the exact max equals brute force;
  - ancestral sampling's frequencies match its stated probability;
  - all four training modes give valid permutations and reduce the loss;
  - sampling beats uniform orders (0.77 vs 1.22 nats/token).

**Honest notes:**
- **Our glimpse adds its readout to the decoder state,** because the literal "replace" version stalled (section 2.3).
- **On our toy Markov chain, a fixed scrambled order learns almost as well as the natural one.** A simple chain is easy in many orders. The paper's real 5-grams show the bigger gap (225 vs 280), which E6 measures on PTB.

---

## 7. Check yourself

1. Why does the chain rule say any output order is fine? Why does order still matter in practice?
2. Prove that the Read–Process summary q*_T is unchanged when the inputs are permuted.
3. Why is a plain sum Σ embed(x_i) a weak set encoder, according to the paper?
4. What does a process step add? What happens with P = 0?
5. If each pointer step is right with probability 0.95, what is the chance of sorting 15 numbers perfectly? (0.95¹⁵ ≈ 0.46.)
6. Linearize the Figure 2 tree breadth-first, and explain each LEV and PAR.
7. In the star model, why is head-first easier for an LSTM?
8. Write Eq. 9. Why sample orders instead of taking the max, and why pre-train with uniform orders?
9. When does ancestral sampling give orders exactly in proportion to p(Y_π)?
