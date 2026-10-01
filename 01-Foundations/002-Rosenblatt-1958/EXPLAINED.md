# Rosenblatt (1958), explained from scratch

**Paper:** *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*
**Author:** Frank Rosenblatt (Cornell Aeronautical Laboratory)
**Published in:** Psychological Review, Vol. 65, No. 6, 1958, pp. 386–408
**Page numbers** below are the journal's (386–408).

This guide assumes only school math: fractions, powers and a little probability. Every formula is worked through with real numbers, and those numbers match what our code prints.

---

## 0. The whole idea in one line

> **Wire a network at random. Then let experience change how much each unit's "vote" counts. That alone lets the machine learn to recognize patterns, including examples it has never seen.**

This is the first neural network that **learns from examples**. McCulloch & Pitts (Paper 001) had to *design* every connection; Rosenblatt's machine *learns* its own connection strengths.

---

## 1. Where it comes from

| McCulloch & Pitts (1943) | Rosenblatt (1958) |
|---|---|
| every connection designed by hand | connections are **random** |
| logic: true/false formulas | **probability**: "how likely is a unit to fire?" |
| nothing is learned | **learning** from examples |
| the exact wiring matters | only the **statistics** of the wiring matter |

- Rosenblatt calls designed networks "logical contrivances". They show how a brain *could* compute, but they need perfect wiring, and real brains are messy and partly random.
- **His question:** can a *randomly* wired network still learn?

### 1.1 Three questions about the brain (pages 386–388)
1. How does the brain **sense** the world? (Physiology had mostly answered this.)
2. **In what form is information stored?**
3. **How does stored information affect recognition and behaviour?**

### 1.2 Two answers to question 2
| "Coded memory" | Connectionism (Rosenblatt) |
|---|---|
| store a **picture** of each thing seen | store **no picture at all** |
| recognizing = comparing the input with stored pictures | memory = **changed connections**. The input simply flows down paths that experience strengthened |
| a memory could be read back out | it can't be read back; it is a **tendency to respond** |

**His five assumptions (from Hebb and others):**
1. Networks start wired **largely at random**.
2. Connections can **change** with activity ("plasticity").
3. **Similar** stimuli come to activate the **same** cells.
4. **Reward and punishment** can help or block this.
5. "Similar" is defined by the **network itself**, not by geometry.

---

## 2. The machine (pages 389–392)

```
  RETINA (S-units)         ASSOCIATION (A-units)              RESPONSES (R-units)
  points: lit 1 / dark 0 ──random──▶  fire if e − i ≥ θ  ──values V──▶  strongest wins
```

### 2.1 S-units: the retina
A grid of points. A stimulus is a 0/1 pattern s = (s₁, …, s_N). In the demo the grid is 20 × 20 = 400 points.

### 2.2 A-units: random feature detectors
Each A-unit j picks, **at random and once and for all**, x **excitatory** points and y **inhibitory** points of the retina. For a stimulus s:
```
e_j = number of its excitatory points that are lit          (0 … x)
i_j = number of its inhibitory points that are lit          (0 … y)
a_j = 1  if  e_j − i_j ≥ θ,  else 0                          θ = threshold
```
This is a McCulloch–Pitts neuron with weights +1 and −1. **Its wiring never changes.** In modern language, the A-layer is a layer of **random, fixed features**.

### 2.3 R-units: the responses
- Each A-unit belongs to the **source-set** of one response.
- Each A-unit also carries a **value** V_j, which starts at 0.
- **The value is the only thing that learning changes.**

Response r's strength is computed in one of two ways:
```
Σ-system:  strength_r = Σ_{j in source-set r, a_j = 1}  V_j               (sum of active values)
μ-system:  strength_r = (that sum) / (number of active units in source-set r)   (their mean)
```
**The strongest response wins.** Feedback then inhibits the other responses' A-units, so responses are mutually exclusive.

### 2.4 The two phases (Figure 3)
1. **Predominant phase:** the stimulus activates some A-units; the responses are still undecided.
2. **Postdominant phase:** one response takes over and suppresses the rest.

**The modern view:** the response "computes a weighted sum of features and takes the largest". That is a **linear classifier** on top of random features. The same idea came back decades later as *random features* and *extreme learning machines*, and it still works well.

---

## 3. The probability theory: how often do A-units fire? (pages 392–394)

The wiring is random, so we can't say exactly which units fire. But we **can** compute **probabilities**. This is the mathematical heart of the paper.

### 3.1 A 30-second probability refresher: the binomial
- Flip a coin that lands heads with probability p, n times.
- The number of heads k follows the **binomial distribution**:
  ```
  P(k heads) = C(n, k) · pᵏ · (1 − p)ⁿ⁻ᵏ          C(n, k) = n! / (k!(n−k)!) = "n choose k"
  ```
- *Example:* n = 3, p = ½. Then P(2 heads) = 3 · ¼ · ½ = 3/8 (the outcomes HHT, HTH, THH).

### 3.2 Pa: the chance that an A-unit fires (Eq. 1)
- A stimulus lights a fraction R of the retina.
- On a large retina, each of the unit's connection points is lit independently with probability R. So:
  - e ~ Binomial(x, R);
  - i ~ Binomial(y, R).
- The unit fires when e − i ≥ θ:
```
Pa = Σ over all (e, i) with e − i ≥ θ   of   P(e) · P(i)                         (1)
```

**Worked example** (the demo's setting): x = y = 5, θ = 3, R = ½.
- **A neat trick:** count the inhibitory points that are *dark* instead of lit. That number, 5 − i, is also Binomial(5, ½).
- So e + (5 − i) is the number of heads in 10 fair flips: Binomial(10, ½).
- e − i ≥ 3 is the same as e + (5 − i) ≥ 8. Therefore:
  ```
  Pa = P(at least 8 heads in 10 flips) = [C(10,8) + C(10,9) + C(10,10)] / 2¹⁰ = (45 + 10 + 1)/1024 = 56/1024 = 0.0547
  ```
- **Our code:** the formula gives 0.0547, and the simulated machine measures 0.0554. ✔

So about 5% of the A-units respond to a typical stimulus: a **sparse** code.

**What Figure 4 shows (reproduced):**
- Pa falls as θ rises (pickier units).
- Pa falls as inhibition y grows.
- With x ≈ y, Pa hardly depends on the stimulus size R. That's useful, because the machine then behaves similarly for big and small stimuli.

### 3.3 Pc: do the same A-units fire for two stimuli? (Eq. 2)
- Stimulus S₁ changes into S₂:
  - a fraction **L** of S₁'s lit points goes dark ("lost");
  - a fraction **G** of S₁'s dark points lights up ("gained").
- Pc is a **conditional probability**:
  ```
  Pc = P(unit fires for S₂ | it fired for S₁) = P(fires for both) / P(fires for S₁) = P(both) / Pa
  ```
- **How the code computes it:** for a unit with (e, i) lit for S₁:
  - l_e ~ Bin(e, L) of its lit excitatory points go dark;
  - g_e ~ Bin(x − e, G) of its dark excitatory points light up;
  - the same for the inhibitory points (l_i, g_i);
  - for S₂ the unit sees e − l_e + g_e excitatory and i − l_i + g_i inhibitory points;
  - sum the probabilities of all cases where it still reaches θ.

**Worked example:** two stimuli, each lighting half the retina, that share 70% of their lit points.
- L = 1 − 0.7 = **0.3**.
- G = R(1 − C)/(1 − R) = 0.5 · 0.3/0.5 = **0.3**.
- The formula gives **Pc = 0.205**: only 1 in 5 of the units that fired for S₁ also fire for S₂.
- So 30% different pixels → 80% different active A-units. **High-threshold units exaggerate differences.** That helps tell similar things apart, but it hurts generalization when the differences are just noise.

### 3.4 Pc_min (Eq. 3): the floor
- **At a very high threshold,** a unit keeps firing only if **none** of its connections change state. That means:
  - each of its x excitatory points avoids being "lost", with probability (1 − L) each;
  - each of its y inhibitory points avoids being "gained", with probability (1 − G) each.
- So:
  ```
  Pc_min = (1 − L)ˣ · (1 − G)ʸ = 0.7⁵ · 0.7⁵ = 0.7¹⁰ = 0.028          (same example)
  ```

**What Figures 5–6 show (reproduced):**
- Pc = 1 for identical stimuli.
- Pc is still **> 0** for completely different stimuli, because random units sometimes fire for both.
- Pc drops sharply as θ grows.

### 3.5 Why Pa and Pc decide everything
- **When learning will work:** a response learns a class by raising the values of the units that fire for that class's members. That works if:
  - **members of the same class share many active units:** high Pc within a class, written **Pc₁₁**;
  - **members of different classes share few:** low **Pc₁₂**.
- **Rosenblatt's condition** for better-than-chance learning is **Pc₁₂ < Pa < Pc₁₁**.

---

## 4. Learning: how the values change (page 391, Table 1)

When a stimulus is shown and response r is reinforced, only **active** units (a_j = 1) in r's source-set are rewarded. The three "value systems" differ in **who pays**:

| System | Rule | Consequence |
|---|---|---|
| **α** | every active unit gains +1 (permanently) | responses trained more often grow bigger values and **win too often** |
| **β** | each source-set gains a fixed total, shared among its active units | totals still grow |
| **γ** | active units gain +1; **the inactive units of the same source-set lose the same total** | each source-set's total value **never changes**, so there's no bias toward frequently trained responses |

**The γ rule as an equation:**
- Let the source-set have n_all units, of which n_on are active. Then:
  ```
  active units:      V_j ← V_j + 1
  inactive units:    V_j ← V_j − n_on / (n_all − n_on)
  ```
- **The total change is zero:** n_on · 1 − (n_all − n_on) · n_on/(n_all − n_on) = n_on − n_on = 0. ✔
- **Example:** 10 units, 2 active. The 2 active units gain +1 each (+2 in total), and the 8 inactive units lose 2/8 = 0.25 each (−2 in total).

### 4.1 Σ vs μ
- **Σ** (the sum) favours responses whose source-sets happen to have **more** active units.
- **μ** (the mean) removes that accident, so it is usually better.
- In the γ system the two behave the same in our tests (page 398 says so too).

### 4.2 Bivalent systems: reward *and* punishment (pages 402–403)
- In a bivalent system, values can **decrease** too:
  - the machine answers;
  - the active units of the correct response gain;
  - the active units of the wrong responses lose.
- **This is trial-and-error learning**, and it is the direct ancestor of the modern **perceptron rule**:
  ```
  w ← w + η (y − ŷ) · a          (change the weights only when the guess ŷ is wrong, in the direction of the answer)
  ```
- **A later result** (Novikoff, 1962), not in this paper: if the classes **can** be separated by a linear rule with margin γ, and the feature vectors have length at most R, this rule makes **at most (R/γ)² mistakes** and then stops making them. That is the famous **perceptron convergence theorem**.

---

## 5. Learning curves: the statistics of memory (pages 394–402)

### 5.1 Two tests
- **P_r** (recall): train on some stimuli, then test on **the same** ones. *Did it memorize?*
- **P_g** (generalization): test on **new** stimuli from the same classes. *Did it learn the concept?*

### 5.2 Why every curve has the same shape (Eq. 4)
```
P = P(N_ar > 0) · Φ(Z),        Z = (c₁n + c₂) / √(c₃n + c₄n²)
```
**Where this comes from (the science):**
1. **The winner is decided by the difference** "strength of the correct response − strength of the other". This difference is a sum of **many small random contributions**, one from each A-unit that every past training stimulus touched.
2. **The Central Limit Theorem** says a sum of many independent random bits is approximately **normal** (bell-shaped), whatever the bits look like individually.
3. So the probability that the difference is positive (a correct answer) is **Φ(mean/standard deviation)**. Φ is the normal "area to the left" function: Φ(0) = 0.5, Φ(1) ≈ 0.84, Φ(2) ≈ 0.98.
4. **With n stimuli learned per response:**
   - the "signal" (the mean) grows linearly in n (c₁n + c₂);
   - the "noise" variance grows like c₃n + c₄n², because the learned stimuli interfere with one another.
5. **P(N_ar > 0)** is the chance that at least one relevant unit fires at all.

The constants c₁…c₄ come from Pa, Pc and the network sizes (Eqs. 5–11). In our scan those equations are too damaged to rebuild, so we **measured** the curves by simulation.

### 5.3 The "ideal environment": random patterns
The stimuli are random dot patterns with no classes. The paper predicts, and our code confirms:
- **it can memorize**, but recall **falls toward 50%** as memory fills (interference grows like n², while the signal grows like n);
- **generalization stays at 50%:** there is nothing to generalize.

| patterns learned | recall P_r | new patterns P_g |
|---|---|---|
| 40 | 1.000 | 0.507 |
| 400 | 0.927 | 0.482 |
| 4000 | 0.623 | 0.510 |

(These are our demo's numbers.)

### 5.4 The "differentiated environment": real classes
- The stimuli are noisy versions of class prototypes. Now the units that fire for one class overlap a lot (high Pc₁₁).
- **P_r and P_g approach the same limit.** After enough experience, it no longer matters whether a particular example was seen before.
- **More A-units push that limit toward 100%**, because averaging over more random features reduces the noise.
- **Demo:** trained on 300 noisy examples (30% of pixels flipped), tested on 400 **new** ones: P_r = P_g = **1.000**.

### 5.5 Binary response coding (page 402)
Instead of 100 separate responses for 100 classes, use **7 yes/no responses**, since 2⁷ = 128 ≥ 100. This is an early idea of distributed output codes.

---

## 6. Extensions and limits (pages 403–405)

**What else it can do:**
- **Sequences in time:** possible if activity leaves a fading trace.
- **Contours:** A-units wired to *nearby* points (a "projection area") instead of random ones. This is an early form of **local receptive fields**, the core idea of CNNs (Paper 013).
- **Spontaneous concept formation:** with slowly decaying values, it can split two classes **without labels**: unsupervised learning.

**Where it stops:**
- **Relations** like "the object left of the square" or "the pattern that came before the circle". *"Some system, more advanced in principle than the perceptron, seems to be required."* Minsky & Papert (Paper 003) turned this into theorems.

---

## 7. Conclusions (pages 405–407)

1. A randomly wired network can learn to link responses to stimuli.
2. With random stimuli, recall falls toward chance as more is learned, and there is nothing to generalize.
3. With real classes, performance approaches a better-than-chance limit that more A-units push toward 100%. **P_g approaches the same limit as P_r.**
4. Trial-and-error learning works (bivalent systems).
5. **Memory is distributed:** removing A-units causes a small *general* loss, not the loss of specific memories.
6. **The theory is scientific:** everything is predicted from six measurable numbers (x, y, θ, ω, N_A, N_R), so it can be proven wrong. Compare the joke: "with four parameters I can fit an elephant".

---

## 8. What our reproduction found

See [results.md](results.md) and [figures/](figures/).

**Matches the paper:**
- **Eqs. (1)–(3) are exact.** Pa and Pc from the formulas agree with direct simulation of 20,000 random A-units to 2–3 decimals (e.g. Pa 0.0547 in theory vs 0.0554 measured).
- **Figures 4–6:** the trends reproduce (Pa falls with θ and with inhibition; Pc = 1 for identical stimuli and > 0 for disjoint ones).
- **Ideal environment:** recall falls toward 50%, and P_g ≈ 0.5.
- **μ beats Σ in the α system; they tie in γ.** **γ beats α** when responses get unequal training (Figure 10).
- **Differentiated environment:** P_r and P_g meet at the same limit, and more A-units raise it (Figure 11).
- **Distributed memory:** removing 25 / 50 / 75% of the A-units leaves P_g at 1.000 / 0.990 / 0.958. The damage is spread thin.
- **Trial and error:** mistakes fall from 162 to about 50 per pass.

**Found along the way:**
- **Moving shapes fail.** Squares vs circles: perfect when they stay put, chance when they shift by 1–2 pixels. Random wiring gives **no translation invariance**, exactly what Minsky & Papert attacked and what CNNs later fixed.
- **Pc₁₂ < Pa < Pc₁₁ is sufficient, not necessary.** For two unrelated classes, Pc₁₂ ≈ Pa, so the condition barely fails, yet learning is perfect. What really predicts success is the **gap** between Pc₁₁ and Pc₁₂.

**Couldn't reproduce exactly:**
- The c₁…c₄ formulas (Eqs. 5–11) are unreadable in the scan, so we measured the learning curves instead.
- Some of the paper's parameters (retina size, stimulus shapes) aren't given.

---

## 9. Check yourself

1. In your own words: what is random in a perceptron, and what is learned?
2. Compute Pa for x = y = 5, θ = 5, R = ½. (Hint: you need 10 heads out of 10, so 1/1024.)
3. What is Pc? Why is it a *conditional* probability?
4. Show that the γ rule keeps a source-set's total value constant.
5. Why does recall of random patterns fall as more are learned? Use the words "signal" and "interference".
6. Why is P_g stuck at 0.5 for random patterns?
7. Why does a randomly wired perceptron fail when a shape moves by two pixels?
8. Write the modern perceptron update rule. When does it change the weights?
