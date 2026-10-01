# Minsky & Papert (1969), explained from scratch

**Book:** *Perceptrons: An Introduction to Computational Geometry* (Expanded Edition)
**Authors:** Marvin Minsky and Seymour Papert (MIT)
**This PDF:** only the book's **Introduction (Chapter 0)**, pages [1]–[20] in the book's numbering.
**Page numbers** below are the book's, printed in brackets on each page.

This guide assumes school algebra. Every argument is worked through with small pictures or numbers.

---

## 0. The whole idea in one line

> **A perceptron decides by adding up many small, local pieces of evidence with weights and comparing the total to a threshold. Some properties of a picture (like "is it convex?") can be decided that way. Others (like "is it all one piece?") provably cannot, whatever weights you choose or learn.**

---

## 1. Where this fits

| Paper | Says |
|---|---|
| 001 McCulloch & Pitts (1943) | neurons are logic gates; nets of them can compute anything |
| 002 Rosenblatt (1958) | a randomly wired perceptron can **learn** to recognize patterns |
| **003 Minsky & Papert (1969)** | **what can a perceptron compute at all?** Some simple things, never |
| 004 Rumelhart, Hinton & Williams (1986) | backpropagation trains hidden layers, which escapes these limits |

This book is blamed for the first "AI winter". As the Du et al. survey (Paper 005) puts it: *"Interest in neural networks diminished in the 1970s when Minsky and Papert proved that the simple perceptron model was unable to perform complex logic function and could not solve linearly inseparable problems."* Section 9 below explains what the book **actually** proved.

---

## 2. Background math: what a threshold unit can and can't separate

The book assumes this. It's the key to everything else.

### 2.1 A weighted vote is a straight line
A threshold unit with inputs x₁, x₂ answers YES when:
```
w₁x₁ + w₂x₂ > θ
```
- The boundary w₁x₁ + w₂x₂ = θ is a **straight line** in the (x₁, x₂) plane. YES points are on one side and NO points on the other.
- With n inputs, the boundary is a flat "hyperplane".
- So **a single threshold unit can only separate classes that a straight cut can separate**. They are called **linearly separable**.

**Example: AND** (YES only at (1,1)). Take w₁ = w₂ = 1 and θ = 1.5:

| x₁ | x₂ | sum | > 1.5? |
|---|---|---|---|
| 0 | 0 | 0 | no ✔ |
| 0 | 1 | 1 | no ✔ |
| 1 | 0 | 1 | no ✔ |
| 1 | 1 | 2 | **yes** ✔ |

### 2.2 XOR is impossible for one unit: a 4-line proof
XOR says YES at (0,1) and (1,0), and NO at (0,0) and (1,1). Suppose some w₁, w₂, θ worked:
```
(0,0) NO :  0        ≤ θ        →  θ ≥ 0
(0,1) YES:  w₂       > θ
(1,0) YES:  w₁       > θ
(1,1) NO :  w₁ + w₂  ≤ θ
```
- Add the two YES lines: w₁ + w₂ > 2θ.
- The θ ≥ 0 line gives 2θ ≥ θ, so w₁ + w₂ > θ.
- That contradicts the last line. **No weights exist.**

Geometrically: XOR's YES points sit on opposite corners of the square, and no straight line can cut them off from the other two corners.

### 2.3 The way out: better features
- **The fix:** add a feature φ₃ = x₁·x₂ ("are both on?"). Then:
  ```
  XOR = x₁ + x₂ − 2·x₁x₂ > 0.5
  ```
  This is linear **in the features** (x₁, x₂, x₁x₂). Check it:

  | x₁ | x₂ | sum |
  |---|---|---|
  | 0 | 0 | 0 |
  | 0 | 1 | 1 |
  | 1 | 0 | 1 |
  | 1 | 1 | 1 + 1 − 2 = 0 |

- **The book's whole question becomes:** *how complicated must the features be?* A feature that looks at **k** input points has **order k**. x₁x₂ looks at 2 points, so XOR needs order 2.

---

## 3. Chapter 0 section by section

### 0.0 Who it's for (page [1])
- People working on pattern recognition and learning machines.
- Mathematicians.
- Above all, people who want a **theory of computation**, including how brains compute.

**The real goal:** understand **parallel** computation. The perceptron is the simplest parallel machine.

### 0.1 "We know shamefully little about our computers" (pages [1]–[3])
- Basic questions are unanswered. How many steps does solving n linear equations *really* need? How much do parallel machines help?
- Words like "parallel vs serial" and "local vs global" are used loosely, and the folklore about them is often **drastically wrong**.

### 0.2 Strategy (pages [3]–[4])
Study **particular, well-chosen cases** thoroughly: the simplest truly parallel machines, with **no loops and no feedback** (unlike Paper 001's circles).

### 0.3 Cybernetics and romanticism (page [4])
- They name the machines "perceptrons" in honour of Rosenblatt. A perceptron **"adds up evidence obtained from many small experiments"**.
- **Sharp criticism:** most perceptron writing is *"without scientific value"*; *"now the time has come for maturity"*.

### 0.4 Parallel computation (page [5], Figure 0.1)
```
            ┌─ φ₁(X) ─┐
   X ──────►├─ φ₂(X) ─┼──► Ω(φ₁, φ₂, …, φₙ) ──► ψ(X)
            └─ φₙ(X) ─┘
   Stage I: many simple tests,         Stage II: combine them
   each computed independently         into one answer
```
- **Without limits this is meaningless:** let φ₁ be the whole answer and let Ω copy it.
- So you must **restrict** what the φ's may look at and how Ω combines them.

### 0.5 Pictures and predicates (pages [5]–[7])
- **A figure X** is the set of black points on the retina.
- **A predicate** ψ(X) is a yes/no question about it:

| Predicate | Asks |
|---|---|
| ψ_CONVEX | is X convex (no dents)? |
| ψ_CONNECTED | is X all one piece? |
| φ_p | is the single point p black? (order 1) |
| φ_A | are **all** points of a set A black? (a "mask", order = the size of A) |

---

## 4. Convexity is local; connectedness is not (Section 0.6)

### 4.1 Convexity: check triples of points
> **X is not convex ⇔ there are points p, q, r, with q on the segment from p to r, where p and r are black but q is white.**

So build one small test for every such triple (p, q, r):
```
φ_pqr(X) = 1  if  p black, r black, q white       (a "violation" detector, order 3)
```
X is convex ⇔ **no** test fires. As a perceptron:
```
ψ_CONVEX(X) = 1  ⇔  Σ (−1)·φ_pqr(X)  >  −1          (weights −1, threshold −1: even one violation fails)
```

**Definition:** ψ is **conjunctively local of order k** if it is "all of some tests that each see ≤ k points say OK". ψ_CONVEX is conjunctively local of order **3**.

**Tiny example:** on the row ■□■ (black, white, black), the triple (left, middle, right) fires a violation, so the row is not convex. On ■■■ no triple fires, so it is convex.

### 4.2 Theorem 0.6.1: connectedness is not conjunctively local of any order
**Proof by the "ladder" picture:**
1. **Y₀** is two separate horizontal bars, so it is **not** connected. Some test must say "no" to it, and that test looks at only **k** points.
2. Build the picture with **more than k** places where a vertical rung could join the bars. Some rung position contains **none** of the test's k points.
3. Add **only that rung**. This is Y₂, and it **is** connected.
4. The test sees exactly the same k points as before, so it still says "no". It wrongly rejects a connected figure. **Contradiction.** ∎

**Why this matters:** to a person, convexity and connectedness feel equally simple. To a parallel machine of local tests they are worlds apart. **Connectedness is a global property**: it can depend on a single pixel arbitrarily far from any given test.

---

## 5. Other meanings of "local" (Section 0.7)

- **Limited order:** each φ looks at ≤ k points, which may be far apart.
- **Limited diameter:** each φ looks only at points within a small distance of each other, like a small window. This is how real retinas (and CNN filters) work.

From here on, Stage II is a **weighted vote**, not a unanimous one.

---

## 6. Perceptrons defined precisely (Section 0.8)

> **ψ is linear with respect to a family Φ** if there are weights α_φ and a threshold θ such that
> ψ(X) = 1 ⇔ Σ_{φ∈Φ} α_φ · φ(X) > θ.
> **A perceptron** computes the predicates that are linear in some fixed Φ.

### 6.1 Worked example: the seesaw (Figure 0.3)
- There are 7 positions i = 1 … 7 on a plank, with the pivot at position 4. φ_i(X) = 1 if a pebble sits at position i.
- Physics: the plank tips right if the **torque** (lever arm × weight) is positive:
  ```
  Tips right ⇔ Σ_i (i − 4) · φ_i(X) > 0
  ```
  Each φ_i sees one point, so this is an **order-1** perceptron.
- *Example:* pebbles at positions 1 and 6 give torque (1 − 4) + (6 − 4) = −3 + 2 = −1 < 0, so it tips **left**.
- **Our code checks all 2⁷ = 128 pebble arrangements:** the perceptron agrees with physics on every one.

### 6.2 Families of perceptrons
| Family | Restriction on the φ's |
|---|---|
| diameter-limited | each φ sees a small window |
| order-restricted | each φ sees ≤ k points |
| Gamba | each φ is itself an order-1 perceptron (really a 2-layer network) |
| random | the φ's are random Boolean functions (Rosenblatt's machines, Paper 002) |
| bounded | infinitely many φ's but finitely many weight values |

---

## 7. Theorem 0.8: no diameter-limited perceptron computes CONNECTED

### 7.1 The four figures
Each figure is long, much longer than any window, and its two **ends** can each be closed in one of two ways (type 0 or type 1):

| Figure | Left end | Right end | Connected? |
|---|---|---|---|
| X₀₀ | 0 | 0 | **no** (two separate loops) |
| X₀₁ | 0 | 1 | **yes** |
| X₁₀ | 1 | 0 | **yes** |
| X₁₁ | 1 | 1 | **no** |

### 7.2 The algebra
- No window can see both ends. So each feature φ is one of three kinds:
  - **L:** it sees the left end (it only cares whether the left end is type 0 or 1);
  - **R:** it sees the right end;
  - **N:** it sees neither (its value is the same on all four figures).
- **For every single feature** this gives the identity:
  ```
  φ(X₁₁) − φ(X₁₀) − φ(X₀₁) + φ(X₀₀) = 0
  ```
  **Check each kind:**
  - an **L** feature gives a − a − b + b = 0, where a is its value on left-type-1 figures and b on left-type-0 figures;
  - an **R** feature is the same with the ends swapped;
  - an **N** feature gives c − c − c + c = 0.
- The total S(X) = Σ α_φ φ(X) is a weighted sum of features, so it obeys the same identity:
  ```
  S(X₁₁) = S(X₁₀) + S(X₀₁) − S(X₀₀)
  ```
- **For a correct perceptron:**
  - S(X₁₀) > θ and S(X₀₁) > θ (connected);
  - S(X₀₀) ≤ θ (not connected).

  Then:
  ```
  S(X₁₁) = S(X₁₀) + S(X₀₁) − S(X₀₀) > θ + θ − θ = θ
  ```
  So the perceptron calls X₁₁ connected, but it isn't. **No weights can work.** ∎

**Notice:**
- **No learning and no probability** appear anywhere. It's a statement about the *machine*, not the training algorithm.
- It is **XOR in disguise**: "connected" = "the two ends are of different types" = left XOR right.

### 7.3 What our code shows
- The identity holds for windows of width 5, 15 and 30, i.e. for **every** feature they can define.
- Rosenblatt's learning rule, given all of those features, is **still wrong on all 4 figures after 2000 passes**.
- With width-40 windows (as wide as the whole 36-column figure), the identity fails and the rule learns all four in **4 passes**.

**Biology note:** if the brain's first-stage detectors are diameter-limited, then seeing connectedness needs **more than one stage of summation**. The authors remark that only "the most advanced animals" do it well.

---

## 8. The order of parity (from Chapter 3, beyond this PDF)

- **Parity** ψ_PARITY(X) = 1 iff an **odd** number of points are black. XOR is parity on 2 points.
- **The book's theorem:** parity on n points has order **n**. Some feature must look at **every** point.
- **Intuition:**
  - **Any** feature that misses one point p can't tell X from X with p flipped.
  - Flipping p **always** changes the answer.
  - The book's "group-invariance" argument turns this into a proof: averaging the weights over all re-orderings of the points shows a lower-order perceptron would have to be a function of the *number* of black points only, of degree < n. No such polynomial can flip sign n times.
- **Our code** finds the smallest order by exact linear programming: parity has order 2, 3, 4 for n = 2, 3, 4, while AND always has order 1.

---

## 9. Why perceptrons are seductive, and what's wrong (Section 0.9)

**The seductive vision:**
- build the machine once with random features;
- "programming" is just setting the weights;
- an error signal sets them automatically ("learning");
- a **convergence theorem** guarantees success.

**The hidden costs:**
1. **Some simple predicates aren't in the repertoire at all** (connectedness, parity with low-order features). No learning rule can find weights that don't exist.
2. **The weights can be astronomically large.** For some predicates, the ratio of the largest to smallest weight grows so fast that storing the weights takes more memory than listing every possible picture.
3. **"It will eventually converge" is empty.** Trying every weight setting also "eventually" converges. The real question is **how fast**, and sometimes the answer is slower than exponential.
4. **Parallel isn't free.** **All** the features must be computed, even irrelevant ones. A clever *serial* program that decides what to look at next (like tracing an outline to check connectedness) can do far less work.

**Their position on learning:** they are not against learning machines. But *"significant learning at a significant rate presupposes some significant prior structure."* Learning works when the features fit the task.

### 9.1 Common misunderstanding
- People say "Minsky & Papert proved neural networks can't do XOR". They didn't.
- They proved limits for **single-stage** perceptrons whose features are **restricted** (in order or diameter).
- A network with a **learned hidden layer** (Paper 004) builds its own features, so it solves XOR and connectedness-like problems.
- Their deeper point about **structure** has aged well. CNNs (local windows + many layers) and Transformers succeed precisely because of well-chosen built-in structure.

---

## 10. What our reproduction found

See `python3 demo.py` and the tests in `test_minsky.py`.
- **Seesaw (Figure 0.3):** the order-1 perceptron matches physics on **all 128** arrangements.
- **Convexity:** a perceptron with **372** order-3 violation detectors agrees with a direct convexity check on 300 random 6×6 figures.
- **Theorem 0.6.1, checked constructively:** for random tests looking at 1, 2, 5, 10 or 20 points (100 tests each), the ladder recipe **always** finds a connected figure the test can't tell from Y₀.
- **Theorem 0.8:** the identity φ(X₁₁) − φ(X₁₀) − φ(X₀₁) + φ(X₀₀) = 0 holds for every window position. The perceptron rule stays wrong on all 4 figures after 2000 passes, until the windows are as wide as the figure; then it learns in 4 passes.
- **Order of parity:** n for n = 2, 3, 4 (by linear programming), vs 1 for AND.

**About the scan:** the OCR reads the title as "Perceptions" and the author as "Papen". It is *Perceptrons* by Minsky and **Papert**.

---

## 11. Check yourself

1. Prove that no single threshold unit computes XOR (4 inequalities).
2. Add one feature that makes XOR linear. What is its order?
3. Why is convexity "order 3"? What does each violation detector check?
4. Explain the ladder argument for connectedness in three sentences.
5. Show that φ(X₁₁) − φ(X₁₀) − φ(X₀₁) + φ(X₀₀) = 0 for a feature that only sees the left end.
6. From that identity, derive S(X₁₁) > θ.
7. Why isn't "it will eventually converge" a useful guarantee?
8. Did the book prove that multi-layer networks can't learn XOR? What did it prove?
