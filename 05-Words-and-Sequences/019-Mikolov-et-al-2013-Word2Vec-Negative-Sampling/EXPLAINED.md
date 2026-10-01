# Mikolov et al. (2013), explained from scratch

**Paper:** *Distributed Representations of Words and Phrases and their Compositionality*
**Authors:** Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean (Google)
**Published at:** NIPS 2013 (arXiv:1310.4546)

This is the paper behind **word2vec**, improving the Skip-gram model from Mikolov et al.'s earlier 2013 paper. This guide:
- does one training step of negative sampling by hand;
- proves the hierarchical softmax is a real probability distribution;
- explains the statistics behind subsampling, the 3/4 power and phrase detection;
- explains *why* vector arithmetic on words works at all.

---

## 0. The whole idea in one line

> **Give every word a vector, and train the vectors so that a word's vector predicts the words around it. Words used in similar contexts end up with similar vectors, and relationships become directions: vec("Madrid") − vec("Spain") + vec("France") ≈ vec("Paris"). Two tricks, negative sampling and subsampling of frequent words, make this fast enough for billions of words.**

---

## 1. Background: why vectors, and the "distributional hypothesis"

- **One-hot codes are useless for similarity.** A one-hot code (Paper 004's family-tree inputs) makes every pair of words equally different: "cat" is as far from "dog" as from "carburettor".
- **The distributional hypothesis** (Harris, Firth: "you shall know a word by the company it keeps"): words that appear in **similar contexts** have **similar meanings**. "Paris" and "Madrid" both appear near "capital", "city", "visit".
- **So:** learn a dense vector per word whose job is to predict context. Words with similar contexts are pushed toward similar vectors.
- **Similarity is measured by cosine:** cos(a, b) = a·b/(‖a‖‖b‖), between −1 and 1. It is 1 when the two vectors point the same way, whatever their lengths.

---

## 2. The Skip-gram model (Section 2)

**The objective.** Given a text w₁ … w_T, maximize
```
(1/T) Σ_t  Σ_{−c ≤ j ≤ c, j ≠ 0}  log p(w_{t+j} | w_t)                         (Eq. 1)
```
**From each word, predict the words within c positions of it.** For example, with c = 2, from "Paris" in "…the capital Paris is beautiful…" predict "the", "capital", "is", "beautiful".

**Two vectors per word:**
- v_w, the **input** vector, used when w is the centre word. **This is the word vector you keep.**
- v′_w, the **output** vector, used when w is a context word.

**The plain probability model** is a softmax over the whole vocabulary (W words):
```
p(w_O | w_I) = exp(v′_{w_O} · v_{w_I}) / Σ_{w=1..W} exp(v′_w · v_{w_I})          (Eq. 2)
```
- **The problem:** the denominator has **W = 10⁵–10⁷ terms**, and so does its gradient. That's for **every** (centre, context) pair, and there are billions of pairs. Far too slow.

---

## 3. Hierarchical softmax (Section 2.1, Eq. 3)

- **The tree:** put the W words at the **leaves of a binary tree**, and give each inner node n its own vector v′_n.
- **A word's probability is a walk from the root:** at each node, go "left" with probability σ(v′_n · v_{w_I}) and "right" with σ(−v′_n · v_{w_I}). (σ(x) = 1/(1 + e^{−x}) is the logistic function.)
  ```
  p(w | w_I) = Π over the nodes on w's path of σ( ±v′_n · v_{w_I} )      (+ or − depending on the turn taken)
  ```

**Why the leaves sum to 1 (the paper says "it can be verified"):**
- **The key identity:** σ(x) + σ(−x) = 1, because σ(−x) = e^{−x}/(1+e^{−x}) = 1 − σ(x).
- **At the root:** the probability mass 1 splits into σ(·) + σ(−·) = 1 between the two subtrees.
- **At every node below:** each subtree's mass is split the same way, without loss.
- **So the leaves' probabilities always add up to exactly 1.** Our code checks this numerically.

**The cost:**
- A word's cost is its **path length**: ~log₂ W ≈ 20 sigmoid evaluations instead of W = 10⁶ exponentials.
- **A Huffman tree** gives frequent words short paths:
  - **How it's built:** start with every word as a node weighted by its frequency, then repeatedly merge the **two least frequent** nodes into a parent. Rare words end up deep and frequent words near the root.
  - **Average path length:** Huffman coding gives the shortest possible average path for these frequencies; it is at most (entropy of the word distribution) + 1 ≤ log₂ W + 1.
  - **Our demo (29 words):** "in" needs **3** decisions, "the" 4, "paris" 6, and "yen" **8**. Most training words are frequent, so most updates are cheap.

---

## 4. Negative sampling, NEG (Section 2.2, Eq. 4): the main new method

Replace log p(w_O | w_I) by:
```
log σ(v′_{w_O} · v_{w_I})  +  Σ_{i=1..k} E_{w_i ~ P_n(w)} [ log σ(−v′_{w_i} · v_{w_I}) ]
```
**Read it as logistic regression,** "is this (word, context) pair real or fake?":
- push the **true** context word's score up, so σ → 1;
- push **k random "noise" words'** scores down, so σ(−·) → 1.

The cost is **k + 1 dot products**: k = 5–20 for small data, 2–5 for big data.

### 4.1 The gradient, derived
- **The calculus fact:** d/dx log σ(x) = 1 − σ(x), and d/dx log σ(−x) = −σ(x).
- **Write** u_O = v′_{w_O}, u_i = v′_{w_i}, v = v_{w_I}. The loss to **minimize** is ℓ = −[log σ(u_O·v) + Σ_i log σ(−u_i·v)]. Its gradients are:
  ```
  ∂ℓ/∂v   = −(1 − σ(u_O·v)) · u_O  +  Σ_i σ(u_i·v) · u_i
  ∂ℓ/∂u_O = −(1 − σ(u_O·v)) · v
  ∂ℓ/∂u_i =  σ(u_i·v) · v
  ```
- **In words:** move v toward the true context's vector and away from each noise vector, each in proportion to how wrong the current score is.

### 4.2 One training step by hand
**Setup:** v = (0.5, 0.2), true context u_O = (0.4, 0.6), one noise word u₁ = (−0.3, 0.5), learning rate 0.5.
```
u_O·v = 0.32   → σ = 0.579        u₁·v = −0.05  → σ(−u₁·v) = σ(0.05) = 0.512
loss ℓ = −log 0.579 − log 0.512 = 0.546 + 0.668 = 1.214
∂ℓ/∂v = −(1 − 0.579)(0.4, 0.6) + σ(−0.05)(−0.3, 0.5) = (−0.168, −0.253) + 0.488·(−0.3, 0.5) = (−0.315, −0.009)
v ← v − 0.5·∂ℓ/∂v = (0.657, 0.204)          new loss = 1.166  (lower ✔)
```
Our tests check that the code's update equals −lr × the numerical gradient.

### 4.3 NEG vs NCE
- **NCE** (noise-contrastive estimation) is the principled version. Its classifier's logit is v′·v **− log(k·P_n(w))**.
  - With that shift, the optimal score equals the **true** log-probability, so NCE approximately maximizes the softmax log-likelihood.
  - **Where the shift comes from:** a real pair occurs with probability p(w | w_I), and a noise pair with k·P_n(w). The optimal "real vs noise" classifier is σ(log p − log(k P_n)).
- **NEG drops the shift.** It no longer approximates the softmax, but "the Skip-gram model is only concerned with learning high-quality vector representations", so that's fine.
- **What NEG learns instead** (Levy & Goldberg 2014): at its optimum,
  ```
  v′_c · v_w = PMI(w, c) − log k,         PMI(w, c) = log [ P(w, c) / (P(w) P(c)) ]
  ```
  **Skip-gram with NEG implicitly factorizes a shifted pointwise-mutual-information matrix.** PMI is positive when two words co-occur **more than chance**: "Paris"–"capital" is high, "Paris"–"the" is ≈ 0.

### 4.4 The noise distribution: the 3/4 power
- **The choice:** P_n(w) = U(w)^{3/4}/Z (unigram frequency to the 3/4). It worked much better than the plain unigram or uniform distributions.
- **The effect is flattening:**
  - a word 100× more frequent is sampled only 100^{3/4} = **31.6×** more often;
  - rare words get more chances to serve as negatives, and the ubiquitous ones fewer.
- **Our demo:** "the" goes from 8.0% (unigram) to 6.8%; "yen" goes from 0.55% to 0.90%.

---

## 5. Subsampling frequent words (Section 2.3, Eq. 5)

- **The problem:** "the", "in" and "a" occur hundreds of millions of times and carry little information. Seeing "France" next to "the" teaches nothing; seeing "France" next to "Paris" does.
- **The rule:** discard each occurrence of word w with probability
  ```
  P(discard w) = 1 − √( t / f(w) )             t ≈ 10⁻⁵,  f(w) = relative frequency
  ```
- **What it does to counts:**
  - a word with count n_w = f(w)·N keeps on average n_w·√(t/f(w)) = **√(t·N·n_w)** occurrences;
  - so the kept counts grow only like **√(count)**: frequent words are tamed, and words rarer than t are untouched;
  - "the" at f = 5% is kept √(10⁻⁵/0.05) = **1.4%** of the time.
- **Two benefits:**
  - **2–10× faster** training;
  - **better vectors for rare words**, because with frequent words removed, the context window effectively spans more meaningful words.

---

## 6. Results on the analogy test (Section 3, Table 1)

**The test:** "Germany : Berlin :: France : ?". The answer is the word x whose vector is closest (cosine) to
```
vec(Berlin) − vec(Germany) + vec(France)            (the 3 question words are excluded)
```
- **Geometrically,** it completes a **parallelogram**: if "country → capital" is (roughly) the same vector offset for every country, then Berlin − Germany + France lands near Paris.
- **Question types:** **syntactic** (quick : quickly) and **semantic** (country : capital).
- **Setup:** 1B words of Google News, a 692K vocabulary (min count 5), 300-d vectors.

| Method | Time (min) | Syntactic | Semantic | Total |
|---|---|---|---|---|
| NEG-5 | 38 | 63% | 54% | 59% |
| NEG-15 | 97 | 63% | 58% | 61% |
| HS-Huffman | 41 | 53% | 40% | 47% |
| NCE-5 | 38 | 60% | 45% | 53% |
| **with 10⁻⁵ subsampling:** | | | | |
| NEG-5 | **14** | 61% | 58% | 60% |
| NEG-15 | 36 | 61% | 61% | **61%** |
| HS-Huffman | 21 | 52% | 59% | 55% |

- **NEG beats HS** and slightly beats NCE.
- **Subsampling is 2–3× faster** and often more accurate.

---

## 7. Phrases (Section 4, Eq. 6)

- **The problem:** some phrases mean more than their parts: "Boston Globe" (a newspaper), "Air Canada".
- **The fix:** merge frequent, "sticky" bigrams into single tokens:
  ```
  score(w_i, w_j) = (count(w_i w_j) − δ) / (count(w_i) × count(w_j))
  ```
- **Why this formula:** without δ, the score is ∝ P(w_i w_j)/(P(w_i)P(w_j)). That's how much more often the pair occurs than if the words were independent: the exponentiated PMI again.
  - "this is" is common, but both words are very common, so the ratio is modest.
  - "new york" is far more common than "new" × "york" would predict.
- **δ** stops rare words from forming phrases from a single chance co-occurrence.
- **Passes:** run 2–4 passes with a decreasing threshold to build longer phrases ("new_york_times").
- **Our demo:** in random text with "new york" mixed in, "new york" scores **0.0078**, 3.6× the best random pair (0.0022). It is the only pair merged.

**Phrase analogies** (3,218 questions like Montreal : Montreal Canadiens :: Toronto : Toronto Maple Leafs; Table 3):

| Method | No subsampling | 10⁻⁵ subsampling |
|---|---|---|
| NEG-5 | 24% | 27% |
| NEG-15 | 27% | 42% |
| HS-Huffman | 19% | **47%** |

- **HS becomes the best** once subsampling is on.
- **Data size matters a lot:** with **33B words**, 1000-d vectors and the whole sentence as context, accuracy reaches **72%** (66% with 6B words).

---

## 8. Additive compositionality (Section 5, Table 5)

**Simple addition works:**

| sum | closest to |
|---|---|
| vec("Czech") + vec("currency") | "koruna" |
| vec("Vietnam") + vec("capital") | "Hanoi" |
| vec("Russian") + vec("river") | "Volga River" |
| vec("French") + vec("actress") | "Juliette Binoche" |

**Why addition means "AND":**
- **The training objective makes v_w·v′_c behave like a log-probability** of context c given w (up to constants; section 4.3).
- **Adding two word vectors adds their log-probabilities,** which means **multiplying** their context distributions:
  ```
  (v_a + v_b) · v′_c  =  v_a·v′_c + v_b·v′_c  ∝  log p(c|a) + log p(c|b) = log [ p(c|a) · p(c|b) ]
  ```
- **A product is large only where both factors are large,** so the sum picks out contexts likely near **both** words. "Volga River" is likely near both "Russian" and "river".

---

## 9. Why it matters

- **Word embeddings** became the standard input to NLP models until contextual embeddings (ELMo, BERT, GPT) took over.
- **"Contrast real pairs with random pairs"** (negative sampling) is now everywhere: contrastive learning, CLIP, recommendation systems.
- **The PMI connection** (Levy & Goldberg) tied neural embeddings back to classic count-based statistics.

---

## 10. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` trains on **text8** (17M words of Wikipedia; the paper used 1B words of news) and tests on the paper's own `questions-words.txt`. It covers:
  - Table 1: NEG-5, NEG-15, HS, NCE-5, each with and without subsampling;
  - phrases;
  - vector addition;
  - the Figure 2 PCA.

**Checked (tests and demo, about a second):**
- **Eq. 2 and Eq. 3 are both true distributions:** the hierarchical softmax over our Huffman tree sums to exactly 1.
- **Huffman:** frequent words get shorter codes; the codes are prefix-free; the average length is ≤ log₂ W + 1 ("in" 3 decisions, "yen" 8).
- **The hand-written gradients are right:** one NEG step and one HS step each equal −lr × the numerical gradient. The batched update and the NCE variant equal the single-pair versions.
- **Subsampling:** 5% frequency → kept 1.4%; rarer than t → always kept.
- **The 3/4 power:** a 100× more frequent word is sampled 31.6× more often.
- **The dynamic window** (as in the C code) uses distance-1 pairs most and distance-5 least.
- **Phrases:** "new york" 0.0078 vs the best random pair 0.0022 (3.6×); it is the only bigram merged.
- **Learning on a tiny made-up corpus:**
  - the nearest neighbours come out right (paris → madrid, rome; euro → yen, yuan);
  - but **0 of 3 analogies** are right. Analogies need the *same* offset for every country → capital pair, which only emerges from huge, varied text (the paper used ~1B words for 61%).

**Notes:** the released C code uses a slightly different subsampling formula from Eq. 5. We implement Eq. 5 as written, plus the C code's dynamic window.

---

## 11. Check yourself

1. Write the Skip-gram objective. Which of the two vectors per word do you keep?
2. Why is the full softmax too slow? Prove that hierarchical-softmax leaves sum to 1.
3. Derive ∂ℓ/∂v for the NEG loss. Redo section 4.2's step with learning rate 1.
4. How is NCE's logit different from NEG's? What does NEG converge to instead (PMI)?
5. Why raise the unigram to the 3/4? How much more often is a 1000× more frequent word sampled? (≈ 178×.)
6. Show that subsampling keeps ~√(count) occurrences. What happens to "the"?
7. Why does Eq. 6 merge "new york" but not "this is"?
8. Why does vec(a) + vec(b) act like an AND of the two words' contexts?
