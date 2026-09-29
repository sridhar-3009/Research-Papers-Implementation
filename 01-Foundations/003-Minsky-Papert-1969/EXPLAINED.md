# Minsky & Papert (1969), explained simply

**Book:** *Perceptrons: An Introduction to Computational Geometry* (Expanded Edition)
**Authors:** Marvin Minsky and Seymour Papert (MIT)
**This PDF:** only the book's **Introduction (Chapter 0)**, pages [1]–[20] in the book's numbering.
**Page numbers** below are the book's, printed in brackets on each page.

---

## The big idea in one line

> **A perceptron makes a decision by adding up many small, local pieces of evidence. Some patterns (like "is this shape convex?") can be decided that way. Others (like "is this shape all one piece?") provably cannot, however you set the weights.**

---

## How it connects to the earlier papers

| Paper | Says |
|---|---|
| 1. McCulloch & Pitts (1943) | Neurons are logic gates, and nets of them can compute anything |
| 2. Rosenblatt (1958) | A randomly wired perceptron can **learn** to recognize patterns |
| **3. Minsky & Papert (1969)** | **But what can a perceptron compute at all?** Some simple things, never, whatever it learns |
| 4. Rumelhart, Hinton & Williams (1986) | Backpropagation trains hidden layers, which gets around these limits |
| 5. Du et al. (2022) | A survey of everything since: MLPs, training algorithms, deep learning |

This book is famous for the claim that led to the first "AI winter". As Paper 5 (the Du et al. survey) puts it (page 1): *"Interest in neural networks diminished in the 1970s when Minsky and Papert [5] proved that the simple perceptron model [2] was unable to perform complex logic function and could not solve linearly inseparable problems."*

---

## 0.0 Who it's for (page [1])

Three kinds of readers:
1. People working on **pattern recognition** and **learning machines**.
2. People who like **abstract mathematics** (geometry, topology, algebra).
3. The one they care about most: people interested in a **general theory of computation**, including how the brain computes.

**The real goal:** understand **parallel computation**. The perceptron is the simplest parallel machine, so it's the place to start.

---

## 0.1 "We know shamefully little about our computers" (pages [1]–[3])

- Even simple questions have no answers. How many steps does solving n linear equations **really** need? How much faster are **parallel** machines than **serial** ones?
- Words like "parallel vs serial", "local vs global" and "digital vs analog" are used **loosely**, "as if" they were precise. They aren't, and the folklore about them is often **drastically wrong**.

---

## 0.2 Strategy (pages [3]–[4])

- Don't build a grand theory first. Study **particular, well-chosen cases** very thoroughly.
- They pick the **simplest machines that are truly parallel**: **no loops, no feedback** (unlike Paper 1's circles), yet still able to do non-trivial things.
- The book reads like "a mathematical novel where characters appear, reappear, and develop."

---

## 0.3 Cybernetics and romanticism (page [4])

- They name the machines **"perceptrons"** "in recognition of the pioneer work of Frank Rosenblatt".
- A perceptron **"adds up evidence obtained from many small experiments"**.
- **Sharp criticism:** most perceptron writing is *"without scientific value"*. The early excitement was fine, but *"now the time has come for maturity"*.

---

## 0.4 Parallel computation (page [5])

**Figure 0.1**, the general two-stage scheme:
```
            ┌─ φ1(X) ─┐
   X ──────►├─ φ2(X) ─┼──► Ω(φ1, φ2, …, φn) ──► ψ(X)
            └─ φn(X) ─┘
   Stage I: compute many           Stage II: combine
   simple things independently     them into one answer
```
- **Without restrictions this is meaningless:** just let φ1 = ψ and have Ω pass it along.
- So you have to **limit** what the φ's can be and what Ω can do.

---

## 0.5 Geometric patterns = predicates (pages [5]–[7])

- A **figure X** = the set of black points on the retina (the plane).
- A **predicate** ψ(X) = a yes/no question about the figure:

| Predicate | Asks |
|---|---|
| ψ_CIRCLE | Is X a circle? |
| ψ_CONVEX | Is X **convex** (no dents or holes)? |
| ψ_CONNECTED | Is X **all one piece**? |
| φ_p | Is point p black? (the simplest) |
| φ_A | Are **all** points of set A black? ("A ⊆ X", a "mask") |

---

## 0.6 CONVEX vs CONNECTED: the key contrast (pages [7]–[8])

### Convexity is local
> **X is not convex ⇔ there are three points p, q, r, with q on the line from p to r, where p and r are black but q is white.**

So you can test convexity by checking **triplets of points**, each **independently**. X is convex if **every** triplet test passes.

**Definition: conjunctively local of order k.** ψ is computed by predicates that each look at **at most k points**, and ψ = 1 only if **all** of them say yes.
→ **ψ_CONVEX is conjunctively local of order 3.**

### Connectedness is not (Theorem 0.6.1)
> **ψ_CONNECTED is not conjunctively local of any order.**

**The proof, simply:**
1. Take **Y0**: two separate horizontal bars. Not connected, so at least one local test must say "no" to it.
2. That test looks at only **k points**. Make a ladder with **more than k** possible middle "rungs". At least one rung contains **none** of the test's points.
3. Add **just that rung** → **Y2**, which is connected.
4. But the test sees **exactly** what it saw on Y0, so it still says "no".
5. So it wrongly rejects a connected figure. Contradiction.

**Why it matters:** convexity and connectedness **look** equally simple to a person, but they're completely different for a parallel machine.

---

## 0.7 Other ideas of "local" (pages [9]–[10])

Two ways to limit "local":
- **Limited order:** each φ looks at no more than **k points** (anywhere).
- **Limited diameter:** each φ looks only at points **within a small distance** of each other.

The two stages, stated generally:
- **Stage I:** compute many **easy** features.
- **Stage II:** combine them with an **easy** rule.

"Unanimous vote" (all must say yes) is too narrow. From here on, Stage II is a **weighted vote**.

---

## 0.8 Perceptrons, defined precisely (pages [10]–[14])

> **ψ is linear with respect to Φ if there are numbers α_φ (weights) and θ (threshold) such that ψ(X) = 1 ⇔ Σ α_φ · φ(X) > θ.**
>
> **A perceptron** computes every predicate that is linear in some given set Φ of partial predicates.

- **Intuition:** each φ gives some **evidence**. The weight α says how much, and whether it's for (positive) or against (negative).
- **Example 1:** any conjunctively local predicate is a perceptron: give every "failure test" weight −1 and set θ = −1.
- **Example 2, the seesaw (Figure 0.3):** 7 pebble positions, with the pivot at position 4.
  *"Tips right"* ⇔ **Σ (i − 4) · φ_i(X) > 0**. Each φ_i looks at **one** point, so this has **order 1**.

**Families of perceptrons:**

| Family | Restriction |
|---|---|
| **Diameter-limited** | each φ looks at points within a fixed small distance |
| **Order-restricted** | each φ looks at ≤ n points |
| **Gamba** | each φ may see everything, but must itself be an order-1 perceptron (a 2-layer network!) |
| **Random** | the φ's are random Boolean functions: **Rosenblatt's perceptrons (Paper 2)** |
| **Bounded** | infinitely many φ's, but with weights from a finite set |

### Theorem 0.8: no diameter-limited perceptron computes CONNECTED (pages [12]–[14])

Take **four figures**, each long enough that no φ can see **both ends**:

| Figure | Left end | Right end | Connected? |
|---|---|---|---|
| X00 | type 0 | type 0 | **no** |
| X01 | type 0 | type 1 | **yes** |
| X10 | type 1 | type 0 | **yes** |
| X11 | type 1 | type 1 | **no** |

Split the φ's into 3 groups: those that see the **left** end, those that see the **right** end, and those that see **neither**.
- X00 → X10 changes only the left end: only group 1's sum changes, and it must go **up** (to say "connected").
- X00 → X01 changes only the right end: only group 2's sum goes **up**.
- X00 → X11 changes **both** ends, and **locally** each end looks exactly as it did in X10 or X01. So **both** sums go up by the **same** amounts.
- So X11's total is **even bigger** than X01's or X10's, and the perceptron says X11 is connected. **Wrong!**

**Notice:** the proof uses **no learning and no probability**. It's pure geometry plus algebra. Whatever weights you learn, they can't work.

**Biology note:** if the brain's receptor cells are diameter-limited, then recognizing connectedness needs **more than simple summation**, and indeed "only the most advanced animals" can do it.

---

## 0.9 Why perceptrons are seductive, and what's wrong (pages [14]–[20])

**The seductive vision:** build it once with random φ's; "programming" is just setting the weights; attach an error signal and it programs itself ("learning"); the **convergence theorem** guarantees it will find the weights.

### 0.9.1 The hidden costs of "homogeneous programming"
1. **Some simple predicates aren't in the repertoire at all**, like CONNECTED. Thinking only about weight vectors hides this.
2. **Weights can be huge.** Sometimes the ratio of the largest to smallest weight is meaninglessly big, needing more storage than just listing all the figures.
3. **Convergence time.** "It will eventually converge" is empty: trying **every** setting would also converge eventually. The real question is **how fast**, and sometimes it's **faster than exponentially slow**.

**They are not against learning machines.** But *"significant learning at a significant rate presupposes some significant prior structure."* Learning works when the φ's are **well matched** to the task (Samuel's checkers program).

### 0.9.2 Parallel computation has a cost
**All** φ's must be computed, even the irrelevant ones. A smart **serial** program that decides what to check next can do far less work. For CONNECTED on a 100×100 retina, each φ would need to look at **hundreds** of points, so "local" becomes meaningless.

### 0.9.3 Simple analog devices
Storing the weights in simple analog hardware fails when the weights must span huge ranges. For retinas of more than ~20 points, no simple device could store them.

### 0.9.4 Brains, Gestalt and "distributed" memory
- The appeal of perceptrons comes from an image of the brain as **randomly connected simple parts** with **distributed** memory (like Rosenblatt's conclusion 9).
- The **hundreds** of perceptron projects were *"generally disappointing"*: they worked on easy problems and got **much worse** as tasks got harder, and making them bigger didn't help.
- They suspected that even the successes came from **a small part of the network**, not "global, distributed activity".
- Their motive: to dispel a **"holistic" or "Gestalt" misconception** before it haunted AI the way it had haunted psychology.

---

## 0.10 Plan of the book (page [20])

- **Part I:** algebra: general properties of linear predicates.
- **Part II:** geometry: what perceptrons can recognize in pictures.
- **Part III:** perceptrons as practical pattern recognizers and learners.
- **Final chapter:** history and future directions.

*(This PDF stops here; the later chapters aren't included.)*

---

## Common misunderstanding

People often say Minsky & Papert proved that **neural networks can't do XOR**. What they actually studied:
- **Single-stage** perceptrons with **restricted** φ's (limited order or diameter).
- The limits are real for **those** machines.
- A network with a **learned hidden layer** (Papers 4–5) is not restricted this way. It learns its own φ's, and it solves XOR and much more.
- Their point about **structure** still stands. Today's successful networks (CNNs, transformers) have lots of built-in structure, which is exactly "significant prior structure".

---

## What our reproduction found

See `python3 demo.py`, and the tests in `test_minsky.py`.

- **Seesaw (Figure 0.3):** the order-1 perceptron matches the physics on **all 128** pebble arrangements.
- **Convexity:** a perceptron with **372** partial predicates, each looking at **3 points**, agrees with the direct convexity check on 300 random 6×6 figures.
- **Theorem 0.6.1** is checked constructively: for tests that look at 1, 2, 5, 10 or 20 points (100 random tests each), the proof's recipe **always** finds a connected figure the test can't tell apart from the disconnected Y0.
- **Theorem 0.8** is checked for **every window position**, and so for **every** predicate of limited diameter, whatever its rule: the identity φ(X11) − φ(X10) − φ(X01) + φ(X00) = 0 holds.
  - As a result, Rosenblatt's learning rule, given **all** the local features, is **still wrong on all 4 figures after 2000 passes**.
  - Once a window is wide enough to see both ends, it learns them in 4 passes.
- **Extension (from Chapter 3, not in this PDF):** by exact linear programming, **parity on n points has order n** (for n = 2, 3, 4), while AND has order 1. XOR is parity on 2 points.

**Note about the scan:** the OCR reads the title as "Perceptions" and the author as "Papen". The book is *Perceptrons* by Minsky and **Papert**. The last page contains a handwritten note that the scan can't read.

---

## Check yourself

1. What are Stage I and Stage II of a perceptron?
2. Why is convexity "order 3"? What does each partial predicate check?
3. Explain in your own words why no fixed-size test can check connectedness (the ladder argument).
4. In the four-figure proof, why must X11's total be bigger than X01's?
5. Why do the authors say "it will eventually converge" isn't a useful guarantee?
6. Did Minsky & Papert prove that multi-layer networks can't learn XOR?
