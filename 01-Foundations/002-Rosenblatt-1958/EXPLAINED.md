# Rosenblatt (1958), explained simply

**Paper:** *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*
**Author:** Frank Rosenblatt (Cornell Aeronautical Laboratory)
**Published in:** Psychological Review, Vol. 65, No. 6, 1958, pp. 386–408
**Page numbers** below are the journal's (386–408), printed on each page.

---

## The big idea in one line

> **Wire up a network at random, then let experience change how strongly each unit counts. That's enough for the network to learn to recognize patterns, and even to recognize examples it has never seen.**

---

## How it connects to Paper 1

| McCulloch & Pitts (1943) | Rosenblatt (1958) |
|---|---|
| Every connection is **designed** by hand | Connections are **random** |
| Logic: true/false formulas | **Probability**: "how likely is a unit to fire?" |
| Nothing is learned | The network **learns** from examples |
| Exact wiring matters | Only **statistics** of the wiring matter |

Rosenblatt cites Paper 1 (page 387). He says models like it are **"logical contrivances"**: they show how a brain **could** compute something, but they need perfect wiring, which real brains don't have.

---

## 1. The question (pages 386–388)

He asks three questions:
1. How does the brain **sense** the world? (That's physiology, largely answered.)
2. **In what form is information stored?**
3. **How does stored information affect recognition and behavior?**

**Two possible answers to question 2:**

| "Coded memory" | "Connectionist" (Rosenblatt's choice) |
|---|---|
| The brain stores a **picture** (like a photo negative) | The brain stores **no picture at all** |
| Recognizing = **comparing** the new input with stored pictures | Memory = **changed connections**. A new input simply flows down the paths that experience made stronger |
| You could read a memory back out | You can't read it back; it's a **preference for a response** |

**Five assumptions** (from Hebb and others, page 388):
1. At birth, the important networks are wired **largely at random**.
2. Connections can **change** with activity (plasticity).
3. **Similar** stimuli come to activate the **same** responding cells.
4. **Reward and punishment** can help or block this.
5. What counts as "similar" depends on the **network itself**, not on geometry.

---

## 2. How a perceptron is built (pages 389–392)

```
  RETINA              ASSOCIATION AREA              RESPONSES
  S-units   ──────▶   A-units   ──────────────▶     R-units
 (points)   random     fire if                      compete:
  on/off    wiring     excitation − inhibition ≥ θ  the strongest wins
```

| Part | What it is |
|---|---|
| **S-units (sensory points)** | The **retina**. Each point is lit (1) or dark (0) |
| **A-units (association cells)** | Each is connected to **random** retina points ("origin points"): some **excitatory** (x of them), some **inhibitory** (y of them). It fires if (lit excitatory) − (lit inhibitory) **≥ θ** |
| **R-units (responses)** | Each response listens to its own group of A-units, called its **source-set** |

**Feedback between responses** (page 390): when one response wins, it **inhibits** the A-units of the other responses. So only one response can be active: they're **mutually exclusive**.

### What learning changes: the "value" V (page 391)
- The wiring **never** changes.
- What changes is each A-unit's **value V**: how strong its signal is. Active cells gain value.
- The response whose source-set sends the **strongest total** (or strongest average) wins.

### Three ways values can change (Table 1, page 391)

| System | Rule | Problem / advantage |
|---|---|---|
| **α (alpha)** | Every active unit gains +1 and keeps it forever | Simplest. But responses trained more often grow bigger and win too much |
| **β (beta)** | Each source-set gains a fixed amount, shared among its active units | Total value keeps growing |
| **γ (gamma)** | Active units gain, **inactive units of the same source-set pay for it** | A source-set's total value **never changes**, so there's no bias. **Best** |

### Two phases of a response (Figure 3, page 392)
1. **Predominant phase:** some A-units fire. The responses are still quiet.
2. **Postdominant phase:** one response takes over and suppresses the others.

**Learning** = when the same stimulus comes back, the response that won (and was reinforced) is more likely to win again.

---

## 3. The probability theory: how often do A-units fire? (pages 392–394)

This is the mathematical heart of the paper. It uses **probability** instead of logic, because the wiring is random.

### Pa: how many A-units fire (Eq. 1)
> **Pa** = the chance that a random A-unit fires for a stimulus that lights a proportion **R** of the retina.

An A-unit's excitatory points each land on a lit spot with chance R, and the same goes for its inhibitory points. Pa = the chance that (lit excitatory) − (lit inhibitory) ≥ θ.

**What Figure 4 shows:**
- **Higher θ** → fewer A-units fire.
- **More inhibitory connections** → fewer fire.
- With **about equal** excitation and inhibition, Pa hardly changes with stimulus size. That's useful, because the system needs Pa near a good value for every stimulus.

### Pc: do the same A-units fire for two stimuli? (Eq. 2)
> **Pc** = the chance that an A-unit that fired for stimulus S1 **also** fires for stimulus S2.

It depends on how much S1 and S2 overlap:
- **L** = the part of S1's lit points that go dark in S2 ("lost")
- **G** = the part of S1's dark points that light up in S2 ("gained")

**What Figures 5–6 show:**
- **Identical stimuli** → Pc = 1.
- **Completely separate stimuli** → Pc is still **above 0**.
- **Higher θ** → Pc drops sharply. A-units become very **picky**, which helps tell similar things apart.

### Pc_min (Eq. 3)
The lowest Pc can go is **(1 − L)^x · (1 − G)^y**: the chance that **nothing** about the unit's connections changes.

**Why Pa and Pc matter:** together they predict the **learning curves**. If similar stimuli share A-units (high Pc) and different ones don't (low Pc), the network can tell them apart.

---

## 4. The learning theory (pages 394–402)

### Two ways to test (page 395)
- **P_r** (recall): train on some stimuli, then test on **the same** stimuli. *Did it memorize?*
- **P_g** (generalization): test on **new** stimuli from the same classes. *Did it learn the concept?*

### Two ways to pick the winning response (page 394)
- **Σ-system (sum):** the response with the largest **total** value of active inputs wins.
- **μ-system (mean):** the response with the largest **average** wins. Usually better, because it's less affected by how many units happen to fire.

### The learning-curve law (Eq. 4)
Every learning curve has the same form:
```
P = P(N_ar > 0) · Φ(Z),     Z = (c1·n + c2) / √(c3·n + c4·n²)
```
- n = number of stimuli learned per response
- Φ = the normal (bell-curve) probability
- c1…c4 depend on the network and the environment

### "Ideal environment": random stimuli (pages 396–399)
The stimuli are **random dot patterns**, with no real classes.
- The perceptron **can** memorize them.
- But the more it memorizes, the **worse** recall gets (Figures 7–8). Memory fills up.
- P_g stays at **0.5** (chance). There's nothing to generalize, because random patterns share no structure.
- **μ beats Σ**, and **γ beats α and β** (Figure 10).

### "Differentiated environment": real classes (pages 400–402)
The stimuli come in **classes of similar things** (squares vs circles, letters).
- The perceptron now **generalizes**: P_g rises above chance.
- **Key result:** P_r and P_g approach **the same limit**. After enough experience, it doesn't matter whether it has seen a particular example before.
- **More A-units** push that limit toward **100%**.
- **The condition** for better-than-chance performance: **Pc12 < Pa < Pc11**. Members of the same class must share more active A-units (Pc11) than members of different classes do (Pc12).

### Binary coding of responses (page 402)
Instead of 100 separate responses for 100 classes, use **7 yes/no features** (2⁷ = 128). This is an early version of binary output coding.

---

## 5. Bivalent systems: reward and punishment (pages 402–403)

- In the systems so far, active units only **gain** value.
- In a **bivalent** system, value can go **up or down**:
  - **Reward** (positive reinforcement) → the active units of the "on" response gain.
  - **Punishment** (negative reinforcement) → they lose.
- This allows **trial-and-error learning**: the machine answers, then gets told right or wrong. This is the direct ancestor of the **perceptron learning rule** taught today.
- He simulated this on an **IBM 704** computer, and it matched the theory.

---

## 6. Improved perceptrons and limits (pages 403–405)

**What else it can do:**
- **Time:** it can learn sequences, if activity leaves a short-lasting trace.
- **Contour detection:** with A-units that look at nearby points instead of random ones (the "projection area" of Figure 1). This is an early form of **local receptive fields**, as in CNNs.
- **Spontaneous concept formation:** if values slowly decay, it can separate two classes **without being told which is which**. That's unsupervised learning.
- **Selective attention and recall**, e.g. "name the object on the left".

**Where it stops:**
- It can't handle **relations**: "the object left of the square" or "the pattern before the circle".
- He compares this to brain-damaged patients who handle concrete things but not abstract relations.
- *"Some system, more advanced in principle than the perceptron, seems to be required."*

---

## 7. Conclusions (pages 405–407)

1. A **randomly wired** network can learn to link specific responses to specific stimuli.
2. With random stimuli, recall **falls toward chance** as more is learned.
3. With random stimuli, there's **nothing to generalize**.
4. With real classes, performance approaches a **better-than-chance limit**, which can be pushed toward 100% with more A-units.
5. With real classes, **P_g approaches the same limit as P_r**.
6. **Contour-sensitive** A-units and **binary response coding** improve performance.
7. **Trial-and-error** learning works in bivalent systems.
8. **Time sequences** can be learned too.
9. **Memory is distributed**: removing some A-units causes a small **general** loss, not the loss of particular memories.
10. **Relations in space and time** seem to be the perceptron's limit.

**Why he thinks it's a better theory than older psychology** (pages 406–407): everything is predicted from **six physical numbers**:
- **x, y:** connections per A-unit (excitatory, inhibitory)
- **θ:** the A-unit threshold
- **ω:** the share of responses each A-unit connects to
- **N_A, N_R:** the numbers of A-units and responses

These can be **measured** independently, so the theory can be **proven wrong**. That makes it scientific, unlike curve fitting ("given seven parameters, I could fit an elephant").

---

## 8. What our reproduction found

See [results.md](results.md) and [figures/](figures/).

**Matches the paper:**
- **Eqs. (1)–(3) are exact.** Pa and Pc from the formulas agree with a direct simulation of 20,000 random A-units to 2–3 decimal places.
- **Figures 4–6** reproduce the paper's trends: Pa falls with θ and with inhibition, and Pc → 1 for identical stimuli but stays > 0 for separate ones.
- **Ideal environment:** recall falls toward 50% as more is learned, and P_g ≈ 0.5 (conclusions 1–3).
- **μ beats Σ in the α system; in the γ system they're the same.** Both are stated on pages 394–398.
- **γ beats α** when responses get different amounts of training (Figure 10).
- **Differentiated environment:** P_r and P_g meet at the same limit, and more A-units raise it (conclusions 4–5, Figure 11).
- **Distributed memory:** removing 20% of A-units costs about 3 points of accuracy, spread over all classes (conclusion 9).
- **Trial and error:** mistakes fall from 162 to about 50 per pass (conclusion 7).

**Found along the way:**
- **Moving shapes fail.** Squares vs circles work perfectly when they stay in place, but drop to chance when they shift by 1–2 pixels. Random wiring gives **no position invariance**. That's exactly the kind of weakness Minsky & Papert attacked in 1969 (Paper 3).
- **The condition Pc12 < Pa < Pc11 is sufficient, not necessary.** For two unrelated classes, Pc12 comes out almost **equal** to Pa, so the condition barely fails, yet learning is perfect. What predicts success is the **gap** between Pc11 and Pc12.

**Couldn't reproduce exactly:**
- The formulas for c1…c4 (Eqs. 5–11) are too damaged in the scan to rebuild, so we **measured** the learning curves by simulation instead of computing them from the formulas.
- The paper's numbers came from its own parameter choices, several of which the text doesn't give (e.g. retina size, stimulus shapes).

---

## 9. Check yourself

1. What's the difference between "coded memory" and the connectionist view?
2. What do the A-units do, and what exactly changes when the perceptron learns?
3. Why is the γ system better than the α system?
4. What do P_r and P_g measure? Why is P_g stuck at 0.5 for random stimuli?
5. What does "memory is distributed" mean, and how would you test it?
6. What kind of problem did Rosenblatt himself say the perceptron couldn't solve?
