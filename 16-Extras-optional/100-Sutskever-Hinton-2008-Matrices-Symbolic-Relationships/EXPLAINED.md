# Using matrices to model symbolic relationships, explained simply

**Paper:** Ilya Sutskever and Geoffrey Hinton (University of Toronto), *Using matrices to model symbolic relationships*, NIPS 2008. ([PDF](https://papers.nips.cc/paper/2008/file/cfa0860e83a4c3a763a7e62d825349f7-Paper.pdf))

**In one sentence:** represent **every** symbol, both objects (like the number 3 or "Colin") and relations (like "+3" or "has_father"), as a small **matrix**, and model "A stands in relation R to B" as **R × A ≈ B**. Because relations are matrices too, they can be arguments of other relations, so the system can learn what "+3" means purely from facts like "(3, +3) ∈ plus", without ever seeing an example of +3 itself.

---

## 1. The idea in plain words

- **The trick behind logarithms:** they turn multiplication into addition. Mapping things into another space where operations become simple is a powerful idea.
- **Earlier, Linear Relational Embedding (LRE)** learned:
  - a **vector** for each object;
  - a **matrix** for each relation;
  - so that R · A lands near B whenever (A, B) ∈ R.
  - For arithmetic modulo 10, LRE puts the numbers on a circle and makes "+k" a **rotation**.
- **The limitation:** in LRE, relations (matrices) and objects (vectors) are different kinds of thing. The system can't see that the relation "+3" has anything to do with the number 3.
- **Matrix Relational Embedding (MRE)** makes **objects matrices too**, say 4 × 4. Then:
  - relations can be applied to relations: (+3, +9) ∈ inverse;
  - relations can link objects and relations: (3, +3) ∈ plus;
  - **higher-order** relations can link relations: (has_father, has_mother) ∈ higher_oppsex.

---

## 2. The cost function (Eq. 1)

For every training fact (A, B) ∈ R, compute M = R · A and score every object C by how close it is:
```
p(C | A, R) = exp(−‖M − C‖²) / Σ_C′ exp(−‖M − C′‖²)          ‖·‖ = sum of squared matrix entries
cost       = − Σ log p(B | A, R)  +  0.01 · Σ (all matrix entries)²
```

- **This is a softmax over "how close is RA to each object".** It is *discriminative*: B must be closer than all the alternatives. That also stops the trivial solution where every matrix is zero.
- **Training:** conjugate-gradient descent on all facts, starting from random Gaussian matrices, exactly as in the paper.

**Worked example (1 × 1 "matrices", i.e. numbers).**
- **Setup:** objects 0, 1, 2 are the numbers 0.0, 1.0 and 2.0; relation R = 2.0; fact (A = 1, B = 2).
- **The product:** M = 2.0 · 1.0 = 2.0.
- **Squared distances** to the objects: 4, 1, 0, so the scores are e⁻⁴, e⁻¹, e⁰ = 0.018, 0.368, 1.
- **Probability of the right answer:** p(B = 2) = 1 / 1.386 = 0.72, and the cost for this fact is −ln 0.72 = **0.33**.

### The gradient (what our code computes)
- With M = RA and probabilities p_c:
  ```
  ∂cost/∂M = 2 ( Σ_c p_c C − B )
  ∂cost/∂C = 2 ( p_c − [c = B] ) ( M − C )
  ∂cost/∂R = (∂cost/∂M) Aᵀ        ∂cost/∂A = Rᵀ (∂cost/∂M)
  ```
- **The first line** says: move RA toward the right answer B, and away from the probability-weighted average answer.

### Answering a query
- To answer (A, ?) ∈ R, rank the objects by their distance to RA.
- **An answer counts as correct** if the true B is among the k closest, where k is the number of correct answers. (Some people have two aunts.)

---

## 3. The tasks

- **Modular arithmetic (base 12):**
  - objects 0–11; relations +0…+11 and ×0…×11;
  - 12 · 24 = **288** facts, e.g. (11, 2) ∈ +3 because 11 + 3 = 14 ≡ 2.
- **Family trees:**
  - two isomorphic families (English and Italian), 12 people each;
  - 12 relations: husband, wife, son, daughter, father, mother, brother, sister, nephew, niece, uncle, aunt;
  - **112** cases when a query with two correct answers counts twice.

---

## 4. Higher-order relations, and why Eq. 1 isn't enough for them

- **The higher-order arithmetic task:**
  - remove **every** example of one basic relation, e.g. +4;
  - add **36** higher-order facts: (n, +n) ∈ plus, (n, +(−n)) ∈ minus, (+a, +(−a)) ∈ inverse.
  - The system must answer (x, ?) ∈ +4 knowing about +4 only from (4, +4) ∈ plus, (8, +4) ∈ minus and (+8, +4) ∈ inverse.
- **First attempt: training the higher-order facts with the same discriminative cost.**
  - Generalisation was only slightly better than chance.
  - **Why:** Eq. 1 only requires plus · 4 to be **closer** to +4 than to the other relations, not **equal** to it. A matrix that is merely "nearest to +4" doesn't work properly when you multiply a number by it.
- **The fix (Eq. 2):** train higher-order facts with plain squared error,
  ```
  cost_higher = Σ ‖ R̃ · A − B ‖²
  ```
  This forces plus · 4 ≈ +4 exactly. The discriminative cost on basic facts still prevents collapse to zero.
- **Incremental learning:** first learn everything else and **freeze** it. Then learn only the new relation's matrix from its higher-order facts. The paper shows this works about as well as learning everything together: new knowledge from a definition, without relearning.

---

## 5. The paper's results (4 × 4 matrices, mean errors over 5 runs)

| Task | Held out | Mean test errors |
|---|---|---|
| Arithmetic (Table 1) | 30 / 60 / 90 of 288 | **0.0** / 6.8 / 24.0 |
| Family trees (Table 2) | 10 / 20 / 30 of 112 | **0.4** / 1.2 / 2.0 |
| Higher-order arithmetic (Table 3), all of +1 / +4 / +6 / +10 removed | 12 queries each | 1.0 / 2.6 / 2.8 / 3.6 |
| Higher-order family (Table 4), has_father / has_aunt / has_sister / has_nephew | 12 / 8 / 6 / 8 | 2.4 / 4.0 / 0.4 / 1.6 |
| Incremental (Table 5), arithmetic +1 / +4 / +6 / +10 | 12 each | 1.2 / 5.8 / 2.6 / 4.4 |
| Incremental, family | as above | 2.0 / 1.6 / 0.0 / 0.0 |

- **The individual runs are often all-or-nothing:** e.g. has_father errors in five runs: 0, 12, 0, 0, 0.
- **The training error was always zero.**
- **For comparison,** the original feed-forward network for family trees usually got 1–2 test cases wrong with only 4 held out.

---

## 6. What our code found (3 runs each)

| Task | Our test errors per run | Paper's mean |
|---|---|---|
| Arithmetic, 30 held out | 0, 0, 1 | 0.0 |
| Arithmetic, 60 held out | 5, 7, 11 | 6.8 |
| Arithmetic, 90 held out | 21, 16, 19 | 24.0 |
| Family, 10 / 20 / 30 held out | 0,0,0 / 1,1,0 / 4,1,3 | 0.4 / 1.2 / 2.0 |
| Higher-order +4 (Eq. 2) | 12, 1, 0 | 2.6 |
| Higher-order +10 (Eq. 2) | 0, 9, 0 | 3.6 |
| … with the discriminative higher-order cost | +4: 3, 8, 8; +10: 12, 10, 6 | "slightly better than chance" |
| Incremental +4 / +10 | 7, 0, 0 / 0, 0, 0 | 5.8 / 4.4 |
| Higher-order has_father (joint / incremental) | 0, 0, 12 / 0, 0, 0 | 2.4 / 2.0 |
| Higher-order has_sister | 0, 0, 0 / 0, 0, 0 | 0.4 / 0.0 |

- **The analytic gradient matches finite differences to 3·10⁻⁸.** Our family-tree definitions (including aunts and uncles by marriage) give exactly the paper's **112** cases.
- **The numbers match the paper's closely:** errors grow with the number held out; there are no training errors; and runs are often all-or-nothing (a run either fully discovers +4 or fails completely, as in the paper's "0, 12, 0, 0, 0").
- **The discriminative higher-order cost is worse on average** (7.8 vs 3.7 mean errors out of 12 across +4 and +10). That supports the paper's fix, though ours is better than the paper's "slightly better than chance" (chance ≈ 11 of 12 wrong).
- **Incremental learning works:** the frozen-everything-else runs were often *better* than joint training here.
- **Note:** our Italian names follow the paper's Figure 1, but the exact placement of each name in the tree is our own reading. Only the tree structure matters for the experiment.

**`experiments.py`:**
- **E1:** Table 1 with 5 runs and matrix sizes 2–6;
- **E2–E5:** Tables 2–5 with 5 runs.
- Only E2 and E4 smoke runs were done here (family 10 held out: 0, 0 errors; has_sister: 0, 0).

---

## 7. Why it matters

- **MRE is an early example of learning distributed representations** in which relations and objects live in the same space and compose by multiplication.
- **The same idea appears in many later models:**
  - knowledge-graph embeddings (RESCAL, which uses a bilinear form with relation matrices; TransE; ComplEx);
  - "rotations as relations" (RotatE, which echoes the circle-of-numbers solution);
  - and, more broadly, transformers composing learned linear maps.
- **The higher-order trick,** understanding a relation from an analogy without direct examples, foreshadows few-shot "learning from definitions" in modern models.

---

## 8. Check yourself

1. How does MRE represent "Colin's father is James"?
<details><summary>Answer</summary>The matrix product has_father · Colin should be closer to the James matrix than to any other person's matrix.</details>

2. What is the main limitation of LRE that MRE removes?
<details><summary>Answer</summary>In LRE objects are vectors and relations are matrices, so relations can't be arguments of relations. In MRE everything is a matrix, so (3, +3) ∈ plus and (has_father, has_mother) ∈ higher_oppsex can be represented.</details>

3. Compute the Eq. 1 cost for 1 × 1 matrices: objects 0 and 1 are the numbers 0 and 1, R = 0.5, and the fact is (A = 1, B = 0).
<details><summary>Answer</summary>M = 0.5 · 1 = 0.5. Distances to the objects: 0.25 and 0.25, so p(B = 0) = 0.5 and the cost is −ln 0.5 ≈ 0.69. The model can't tell the answers apart.</details>

4. Why did the discriminative cost fail for higher-order facts?
<details><summary>Answer</summary>It only requires plus · 4 to be closer to +4 than to the other relations. The product need not equal +4, so when it is used as a relation on a number it gives wrong results.</details>

5. Why doesn't Eq. 2 (squared error) collapse all matrices to zero?
<details><summary>Answer</summary>The discriminative Eq. 1 is still used for the basic facts, and it requires the objects to stay distinguishable. A collapse would make all answers equally likely, which costs a lot.</details>

6. How is a query with two correct answers scored?
<details><summary>Answer</summary>As correct if the designated answer is among the 2 closest objects to RA; each of the two answers counts as a separate case.</details>

7. What does incremental learning freeze, and what does it learn?
<details><summary>Answer</summary>It freezes all previously learned object and relation matrices, and learns only the new relation's matrix from the higher-order facts that mention it.</details>

8. Why might LRE solve mod-10 addition with 2 × 2 rotation matrices?
<details><summary>Answer</summary>Numbers placed evenly on a circle can be shifted by k steps by a rotation through 360°·k/10. Rotations compose like addition, so "+k" becomes a rotation matrix.</details>

9. Why were some runs all-or-nothing (0 or 12 errors)?
<details><summary>Answer</summary>The cost is non-convex. Either the run discovers a representation in which plus · 4 really acts like +4, and all queries are right, or it lands in a poor local minimum where the inferred +4 matrix doesn't work, and all are wrong.</details>

10. What is the "equivalent neural network" view of MRE?
<details><summary>Answer</summary>One-hot inputs select the R and A matrices (the first-layer weights); "sigma-pi" units multiply and sum activities to form RA; a softmax output layer with weights 2B and biases −‖B‖² gives probabilities proportional to exp(−‖RA − B‖²).</details>
