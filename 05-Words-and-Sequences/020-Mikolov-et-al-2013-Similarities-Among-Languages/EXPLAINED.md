# Mikolov, Le & Sutskever (2013), explained from scratch

**Paper:** *Exploiting Similarities among Languages for Machine Translation*
**Authors:** Tomas Mikolov, Quoc V. Le, Ilya Sutskever (Google)
**Published:** arXiv:1309.4168, 2013

Read Paper 019 (word2vec) first. This short paper is **optional** in the reading order. This guide:
- derives the translation matrix as a least-squares problem, with a 2-D example you can do in your head;
- counts how many dictionary pairs are needed;
- explains the baselines (edit distance, co-occurrence) and the confidence trick.

---

## 0. The whole idea in one line

> **Word vectors trained separately on English and on Spanish text have the same shape: "one, two, three" or "cat, dog, horse" are arranged the same way in both spaces. So a single linear map W, learned from a small dictionary, turns English vectors into Spanish ones, and translates words that were never in the dictionary.**

---

## 1. The observation (Section 3, Figure 1)

- **The experiment:**
  - train word vectors (CBOW or Skip-gram) on English text, and **separately** on Spanish text;
  - project one … five / uno … cinco and dog, cat, horse, cow, pig / perro, gato, caballo, vaca, cerdo to 2-D with PCA.
- **They look the same, up to a rotation and scaling.**
- **Why:**
  - all languages describe the same world;
  - "cat" and "dog" occur in similar contexts in every language;
  - vectors are built from contexts (Paper 019), so the **relationships** between concepts carry over, even though the coordinates themselves are arbitrary.
- **Comparing vectors directly is meaningless:** our demo's cosine between an "English" word and its own "Spanish" translation is +0.03. The two spaces have unrelated axes.

---

## 2. The method (Section 4)

**Step 1:** train monolingual vectors for each language on lots of text. That's cheap: no translations are needed.

**Step 2:** from a small dictionary of n pairs {x_i (source vector), z_i (target vector)}, learn W:
```
min_W  Σ_{i=1..n}  ‖W x_i − z_i‖²                (Eq. 3)
```
W is d₂ × d₁, so the two languages may use **different vector sizes**. The paper solves it with SGD.

**Step 3: translate any word.**
- compute z = W x;
- output the target word whose vector is **closest to z by cosine similarity**;
- the top 5 make a candidate list.

### 2.1 The exact solution (least squares)
- **Stack the pairs as columns:** X (d₁ × n) and Z (d₂ × n). The loss is ‖WX − Z‖².
- **The gradient** is 2(WX − Z)Xᵀ. Setting it to 0:
  ```
  W X Xᵀ = Z Xᵀ      ⇒      W = Z Xᵀ (X Xᵀ)⁻¹          ("normal equations"; needs n ≥ d₁ pairs)
  ```
- **SGD reaches this same W:** each step moves W by −η·2(W x_i − z_i)x_iᵀ. Our test finds a max difference of 0.03 from the exact formula.

### 2.2 A 2-D example you can do by hand
**Setup:** suppose "Spanish" is "English" rotated by 90°. Two dictionary pairs:
```
x₁ = (1, 0) → z₁ = (0, 1)          x₂ = (0, 1) → z₂ = (−1, 0)
```
**Solve:**
- X = I, so W = Z = [[0, −1], [1, 0]], a 90° rotation.
- **A word never in the dictionary,** x = (1, 1), maps to Wx = (−1, 1). That's exactly where its translation sits if the spaces really are rotated copies.
- **The point:** this is how the method translates unseen words.

### 2.3 How many dictionary pairs are needed?
- **Unknowns:** W has d₂·d₁ numbers.
- **Equations:** each pair gives d₂ (one per output coordinate).
- **So you need at least n ≥ d₁ pairs** (enough for X Xᵀ to be invertible). In practice you need many more, because real languages aren't exactly linear images of each other.
- **Our synthetic test** (d₁ = 50, d₂ = 30, a random linear map plus noise; P@1 on 1,000 unseen words):

  | pairs | P@1 |
  |---|---|
  | 20 | 17.8% |
  | 40 | 76.0% |
  | 60 | 97.5% |
  | 100+ | 100% |

- **The paper used 5,000 pairs.**

### 2.4 The model: CBOW (Section 2)
- **CBOW** predicts the middle word from the **sum** of its context vectors: the opposite direction to Skip-gram.
- **Trade-off:** it's faster on big data; Skip-gram is better for rare words.
- **The gradient:** the context is a sum, so every context word receives the **same** gradient. Our test checks this against numerical gradients.

---

## 3. Baselines (Section 5.2)

### 3.1 Edit distance (spelling)
- **Levenshtein distance** = the fewest insertions, deletions and substitutions that turn one word into another, computed by dynamic programming:
  ```
  D[i][j] = min( D[i−1][j] + 1,                 delete
                 D[i][j−1] + 1,                 insert
                 D[i−1][j−1] + [a_i ≠ b_j] )    substitute (free if equal)
  ```
- **Similarity** = 1 − distance/(longer length). Our demo:

  | pair | similarity |
  |---|---|
  | information ~ informacion | 1 change in 11 → 0.91 |
  | university ~ universidad | 0.73 |
  | house ~ casa | 0.20 |
  | dog ~ perro | **0.00** |

- **The catch:** it only helps where spellings are related (English–Spanish share Latin roots), and almost not at all for English–Czech.

### 3.2 Word co-occurrence
- **For each word,** count how often each **dictionary** word appears near it, then map those counts through the dictionary into the other language.
- **Normalize:** adjust for corpus size, take the log, L2-normalize.
- **Translate:** pick the most similar count vector. This is a count-based cousin of the embedding method.

---

## 4. Experiments (Sections 5–6)

**Data (WMT11):** English 575M tokens (127K vocabulary), Spanish 84M (107K), Czech 155M (505K).
- **Cleaning:** capitalized words (named entities) are removed, numbers become one token, and short phrases are built with Paper 019's method.
- **Dictionary:** the 5K most frequent source words, translated by Google Translate, for training; the **next 1K** for testing.
- **Scoring:** only an **exact** match counts, so a correct synonym is scored wrong (P@1 is underestimated).

### Table 2 (precision@1 / precision@5, %)
| | Edit Distance | Word Co-occurrence | **Translation Matrix** | **ED + TM** |
|---|---|---|---|---|
| En → Sp | 13 / 24 | 19 / 30 | **33 / 51** | **43 / 60** |
| Sp → En | 18 / 27 | 20 / 30 | 35 / 52 | 44 / 62 |
| En → Cz | 5 / 9 | 9 / 17 | 27 / 47 | 29 / 50 |
| Cz → En | 7 / 11 | 11 / 20 | 23 / 42 | 25 / 45 |

- **The translation matrix works from meaning,** so it beats both baselines in all four directions.
- **Edit distance is complementary:** it adds 10 points for English–Spanish, and ~2 for English–Czech.
- **Vector sizes:** the **source** vectors should be **2–4× bigger** than the target vectors (the best En → Sp used 800-d English and 200-d Spanish). A richer source space gives W more information to map from.

### More data, rarer words (Figures 3–4)
- **More monolingual text keeps helping,** up to billions of words (Google News).
- **Rare words need far more text:** words ranked 15K–19K reach ~**60% P@5** with big data, but only 25% with WMT11.

### Confidence (Section 6.1, Table 3)
- **The score:** confidence = max_i cos(W x, z_i). If W x lands far from every real word, the translation is probably wrong.

| Threshold | Coverage | P@1 | P@5 |
|---|---|---|---|
| 0.0 | 92.5% | 53% | 75% |
| 0.5 | 78.4% | 59% | 82% |
| 0.6 | 54.0% | 71% | 90% |
| 0.7 | 17.0% | 78% | 91% |

- **The trade:** coverage for precision. About half the words can be translated at ~90% P@5.
- **Another use:** the same score can flag **wrong entries** in existing dictionaries.

---

## 5. Why it matters (and what came next)

- **The first strong demonstration** that separately trained embedding spaces are nearly **isomorphic**.
- **Xing et al. (2015):** constrain W to be **orthogonal** (a pure rotation, no stretching). The best rotation has a closed form (**orthogonal Procrustes**): take the SVD ZXᵀ = UΣVᵀ, then **W = UVᵀ**. This preserves distances and improved the results.
- **Conneau et al. (2018, MUSE):** learn the map with **no dictionary**, using adversarial training plus Procrustes refinement.
- **This led to:** multilingual embeddings and unsupervised machine translation.

---

## 6. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` reproduces Table 2 (E1), the vector-size observation (E2), Table 3 (E3) and Figure 1 (E4).
- The paper's WMT11 + Google Translate setup isn't freely available, so it uses the **English and Spanish sides of Europarl** with **Facebook's MUSE en-es dictionaries** (5K train / 1.5K test).

**Checked (tests and demo, about a second):**
- **SGD on Eq. 3 reaches the exact least-squares W** (max difference 0.03).
- **Synthetic languages** (50-d "English" → 30-d "Spanish" via a random linear map plus noise): W from 500 pairs translates **1,000 unseen words at 100% P@1 and P@5**. Pairs vs P@1: the table in section 2.3.
- **Confidence filtering** (a noisier language): the same trade-off as Table 3.

  | Threshold | Coverage | P@1 |
  |---|---|---|
  | none | 100% | 41.6% |
  | 0.6 | 81% | 48.0% |
  | 0.7 | 22% | 84.2% |
  | 0.8 | 1.6% | 100% |

- **Edit distance:** the values in section 3.1.
- **CBOW's hand-written gradient** matches the numerical gradient, and every context word gets the same gradient.

**Note:** the paper doesn't say how it handles log(0) in the co-occurrence baseline; we use log(1 + count).

---

## 7. Check yourself

1. Why would vectors trained separately on English and Spanish have the same "shape"? Why is their direct cosine meaningless?
2. Derive W = Z Xᵀ(X Xᵀ)⁻¹ by setting the gradient of ‖WX − Z‖² to zero.
3. In the 2-D example, translate x = (2, −1). (Wx = (1, 2).)
4. With d₁ = 300 and d₂ = 200, what is the minimum number of dictionary pairs? Why use far more?
5. Compute the edit distance between "cat" and "gato". (2: substitute c → g, then insert o.)
6. What is the confidence score, and what does raising the threshold trade?
7. What is the orthogonal Procrustes solution, and why might a rotation be a better model than a general linear map?
