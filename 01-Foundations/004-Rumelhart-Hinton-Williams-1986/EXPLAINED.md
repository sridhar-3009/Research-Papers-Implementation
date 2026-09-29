# Rumelhart, Hinton & Williams (1986), explained simply

**Paper:** *Learning Representations by Back-Propagating Errors*
**Authors:** David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
**Published in:** Nature, Vol. 323, 9 October 1986, pp. 533–536
**Page numbers** below are the journal's (533–536).

---

## The big idea in one line

> **Give a network hidden units, measure how wrong its output is, and send that error backwards through the network to work out how much each weight is to blame. Adjust every weight a little in the direction that reduces the error. The hidden units then invent their own useful features.**

That's **backpropagation**. Almost every neural network trained since, including every LLM, is trained with it.

---

## How it connects to the earlier papers

| Paper | Problem it left |
|---|---|
| 001 McCulloch & Pitts | Networks can compute anything, but **you** must design the weights |
| 002 Rosenblatt | Networks can **learn**, but only the last layer. The A-units are random and fixed |
| 003 Minsky & Papert | Without learnable features, some things (symmetry, parity, connectedness) are impossible |
| **004 This paper** | **Learn the hidden features too, with backpropagation** |

The paper says it directly (page 533): perceptrons have "feature analysers" between input and output, *"but these are not true hidden units because their input connections are fixed by hand … they do not learn representations."*

---

## 1. The problem: hidden units (page 533)

- **Easy case:** inputs connected straight to outputs. Learning rules for this already existed (the perceptron rule, LMS).
- **Hard case:** **hidden units**, whose correct values aren't given by the task. Nobody tells you what a hidden unit *should* do.
- So the learning procedure must itself **decide what the hidden units should represent**.
- **The claim:** one simple, general procedure is powerful enough to build useful internal representations.

---

## 2. The network (page 533)

- **Layers:** input units at the bottom, any number of hidden layers, output units at the top.
- **Connections go upward only.** They may **skip layers**, but never go sideways or down.
- To run it, set the inputs, then compute each layer in turn, **bottom to top**.

### Eq. (1): total input to unit j
```
x_j = Σ_i  y_i · w_ji
```
This is the weighted sum of the outputs y_i of the units below.

**Bias:** an extra input that is always 1, with its own weight. It's the same as a threshold with the opposite sign, and it's learned like any other weight.

### Eq. (2): output of unit j (the logistic function)
```
y_j = 1 / (1 + e^(−x_j))
```
- It's a **smooth** version of the on/off threshold (compare Paper 001's step function): large negative x → 0, x = 0 → 0.5, large positive x → 1.
- **Why smooth?** You need to know *how much* the output changes when a weight changes a little. A step function jumps, so there's no useful answer. The paper: any function with a **bounded derivative** will do.

---

## 3. The error (page 534)

### Eq. (3)
```
E = ½ Σ_c Σ_j (y_j,c − d_j,c)²
```
- c = each training case, j = each output unit, y = actual output, d = desired output.
- **Goal:** find weights that make E small. That's **gradient descent**: find the slope ∂E/∂w for every weight and step downhill.

---

## 4. The backward pass: the heart of the paper (pages 534–535)

The **forward pass** computes every unit's output. The **backward pass** computes every weight's share of the blame. It goes **from the top down**.

### Eq. (4): how the error depends on an output unit
```
∂E/∂y_j = y_j − d_j
```
Too high → positive; too low → negative.

### Eq. (5): from the output to the total input
```
∂E/∂x_j = ∂E/∂y_j · y_j (1 − y_j)
```
**y(1 − y)** is the slope of the logistic (Eq. 2). It's largest (0.25) at y = 0.5 and nearly 0 when y is close to 0 or 1.
**Remember this:** a unit stuck near 0 or 1 barely learns. We hit exactly this problem below.

### Eq. (6): the gradient for a weight
```
∂E/∂w_ji = ∂E/∂x_j · y_i
```
A weight's blame = the blame of the unit it feeds × how active its input was. If the input unit was off (y_i = 0), the weight couldn't have contributed, so it gets no blame.

### Eq. (7): passing the error down a layer
```
∂E/∂y_i = Σ_j  ∂E/∂x_j · w_ji
```
**This is the key step.** A hidden unit's blame is the sum of the blames of all the units it feeds, weighted by the connections. Now you know ∂E/∂y for the layer below, so you can **repeat Eqs. (5)–(7)** all the way down to the inputs.

**Why it's called back-propagation:** the error signal **propagates backwards**, from outputs to inputs, through the same connections the activity went forward through.

---

## 5. Changing the weights (page 535)

### Eq. (8): simple gradient descent
```
Δw = −ε · ∂E/∂w
```
ε (epsilon) is the **learning rate**.

**Two ways to apply it:**
- change the weights **after every case**, or
- **add up** ∂E/∂w over **all** cases first, then change once per **sweep** (epoch). The paper uses this one.

### Eq. (9): with momentum
```
Δw(t) = −ε · ∂E/∂w(t) + α · Δw(t−1)
```
- Part of the **previous** step is added on (α between 0 and 1, typically 0.9), like a ball rolling downhill.
- It speeds things up on long gentle slopes and damps zig-zagging, **without** second derivatives, so it stays simple and local.

**Breaking symmetry:** start from **small random** weights. If all weights started equal, every hidden unit would compute the same thing and get the same update forever.

**Credit:** variants were found independently by **David Parker** and **Yann LeCun** (page 535).

---

## 6. Experiment 1: detecting mirror symmetry (Figure 1, page 534)

**Task:** 6 binary inputs. Is the pattern the same read backwards? (e.g. `101101`: yes; `100000`: no.)

**Why a hidden layer is needed:** one input bit *alone* says nothing about symmetry, so simply adding up evidence from single inputs can't work (Minsky & Papert, ref. 2).

**Setup:** 6 inputs → **2 hidden units** → 1 output. All 64 patterns per sweep, ε = 0.1, α = 0.9, initial weights uniform in [−0.3, 0.3]. It took **1,425 sweeps**.

**The solution it found:**
- For each hidden unit, weights at **mirror positions are equal in size and opposite in sign**. A symmetric input cancels out, so the hidden unit gets 0.
- Both hidden units have **negative biases**, so with 0 input they're **off**. The output has a **positive bias**, so it's **on** → "symmetric".
- The weights on each side are in ratio **1 : 2 : 4** (like binary place values), so every pattern on one side gives a **unique** sum. Only its exact mirror image can cancel it.
- The two hidden units are **sign-flipped copies**. For any non-symmetric input, one of them turns on and switches the output off.

**Nobody designed this. Backprop discovered it.**

---

## 7. Experiment 2: family trees (Figures 2–4, pages 534–535)

**Data (Figure 2):** two family trees with **the same shape**, one English and one Italian (12 people each). Facts are triples:
```
(person 1) (relationship) (person 2)      e.g.  Colin  has-aunt  Jennifer
```
The **12 relationships**: father, mother, husband, wife, son, daughter, uncle, aunt, brother, sister, nephew, niece. That gives **104** facts in total.

**Network (Figure 3), five layers:**
```
24 person units ──► 6 units ─┐
                             ├──► 12 central ──► 6 ──► 24 output person units
12 relation units ─► 6 units ┘
```
Input: one person unit and one relation unit on. Output: switch on **every** correct person 2 (Colin has two aunts, so two outputs are on).

**Training:**
- 1,500 sweeps: ε = 0.005 and α = 0.5 for the first 20, then ε = 0.01 and α = 0.9.
- **Weight decay** of 0.2% after each change.
- **"Close enough" rule:** no error if an "on" unit is above 0.8, or an "off" unit is below 0.2.
- **100** of the 104 facts used for training; **4 held back** for testing.

**Results:**
- It **generalized correctly to all 4 unseen facts**.
- **Figure 4:** the 6 units that encode people learned meaningful **features that were never in the input**:
  - unit 1: **English vs Italian**
  - unit 2: **which generation**
  - unit 6: **which branch of the family**
- Because most units ignore nationality, an English person's code is **very similar** to their Italian twin's. The network **shares structure** between the two trees, and that's why it generalizes.

**Why this matters:** the inputs are one unit per person, which contains **no** information about generation or branch. The network **invented** those concepts because they help predict relationships. That's what "learning representations" means in the title.

---

## 8. Recurrent networks (Figure 5, page 535)

- A **recurrent** net feeds its units back into themselves over time steps.
- Running it for 3 steps = a **3-layer layered net** in which every layer has **the same weights**.
- So you can train it with backprop, with two changes:
  1. **Store** every unit's output at every time step (the backward pass needs them).
  2. The copies of a weight must stay equal, so **combine** their gradients and change them together.
- This is **backpropagation through time (BPTT)**, the method behind RNNs and LSTMs (Stage 05).

---

## 9. Limits (pages 535–536)

- **Local minima:** gradient descent can get stuck in a poor spot. In practice they say it **rarely** happens, mostly in nets with **just enough** connections. *"Adding a few more connections creates extra dimensions in weight-space and these dimensions provide paths around the barriers."*
- **Not how the brain learns:** *"The learning procedure, in its current form, is not a plausible model of learning in brains."* But it shows interesting representations **can** be learned by gradient descent, so it's worth looking for brain-like ways to do it.

---

## 10. What our reproduction found

See [results.md](results.md) and [figures/](figures/). Every number below comes from running the code.

### ✅ Reproduced
- **Backprop is exactly right:** Eqs. (1)–(7) match finite differences on a network with skip connections and two input groups, and also for the recurrent net (Figure 5).
- **Symmetry (Figure 1):** solved in **18/20** random starts with the paper's settings, median **2,208 sweeps** (the paper's run: 1,425).
- **The learned solution matches Figure 1 in every run that solves it:**
  - mirror weights equal and opposite (error < 1%);
  - the two hidden units are exact sign-flipped copies (correlation −1.000);
  - negative hidden biases and a positive output bias;
  - magnitudes follow powers of two: one run learned **1 : 2.01 : 3.90**, the paper's 1 : 2 : 4.
- **No hidden layer → symmetry is impossible:** the best is 57/64, about the same as always saying "no" (56/64).
- **Family trees:** the network learns **98.7%** of the training facts, and **generalizes above chance**.
- **Feature discovery:** code units track **nationality** (correlation 0.83), **generation** (0.73) and **branch** (0.61), learned from one-unit-per-person inputs.
- **Recurrent net:** trained through time to remember the first of 3 bits, it got all 8 sequences right.

### ⚠️ Different from the paper, and why
1. **The paper's exact family-tree recipe doesn't learn at all.**
   - With weights starting in ±0.3 (the paper gives this range only for Figure 1), ε = 0.01 and 0.2% decay, **0% of the training facts** are learned.
   - **Why:** the signal passes through **4 layers** of logistic units. Each multiplies the backward error by y(1 − y) ≤ 0.25 (Eq. 5), so small initial weights make the error vanish before it reaches the bottom layers. Also, 0.2% decay per sweep outweighs such small gradients.
   - **Fix:** start weights in ±1.0, use ε = 0.05, and a 10× smaller decay (0.02%). Then it learns.
   - This is the **vanishing gradient** problem. It's exactly what Paper **007 (Glorot & Bengio, 2010)** explains and fixes with better initialization.
2. **Generalization is weaker than the paper's single run:**
   - over 20 runs, **36 of 80** held-out facts (45%) are answered exactly right;
   - the paper reports 4/4 for its run, and our demo run (seed 0) also gets **4/4**;
   - English–Italian twin codes are only **somewhat** closer than random pairs (ratio 0.84, where 1.0 = no closer); the paper describes them as "very similar".
3. **Local minima vs step size.** The runs that failed on symmetry weren't in local minima of the kind the paper means:
   - They all ended at **E = 4.0**, the "always say not symmetric" plateau. 56 of 64 patterns aren't symmetric, so the output **saturates** near 0 and Eq. (5)'s y(1 − y) ≈ 0 kills the gradient.
   - With the paper's ε = 0.1, **bigger** nets hit this more often (8 hidden units: 4/20 solved).
   - With **ε = 0.03, every net size solves it 20/20**. The problem is the step size, not the number of connections.

---

## 11. Check yourself

1. Why were Rosenblatt's A-units not "true hidden units"?
2. Write Eqs. (4)–(7) from memory. Which one sends the error **down** a layer?
3. Why must the activation function be smooth?
4. What does y(1 − y) do when a unit is saturated near 0 or 1? Which two of our findings does that explain?
5. How does the symmetry network use weights in ratio 1 : 2 : 4?
6. What features did the family-tree network discover, and why do they help it generalize?
7. How can a recurrent network be trained with backprop?
