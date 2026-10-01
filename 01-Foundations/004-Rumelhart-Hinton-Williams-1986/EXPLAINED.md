# Rumelhart, Hinton & Williams (1986), explained from scratch

**Paper:** *Learning Representations by Back-Propagating Errors*
**Authors:** David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
**Published in:** Nature, Vol. 323, 9 October 1986, pp. 533–536
**Page numbers** below are the journal's (533–536).

This guide assumes school algebra plus the idea of a slope. Every step of the calculus is shown, and a full forward + backward + update is done by hand with real numbers.

---

## 0. The whole idea in one line

> **Give a network hidden units. Measure how wrong its output is. Send that error backwards through the network to work out how much each weight is to blame, and nudge every weight a little in the direction that reduces the error. The hidden units then invent their own useful features.**

That's **backpropagation**. Almost every neural network trained since, including every LLM, is trained this way.

---

## 1. The problem it solves

| Paper | Problem left open |
|---|---|
| 001 McCulloch & Pitts | nets compute anything, but **you** must design the weights |
| 002 Rosenblatt | nets **learn**, but only the last layer; the A-units are random and fixed |
| 003 Minsky & Papert | without learnable features, symmetry, parity and connectedness are impossible |
| **004 this paper** | **learn the hidden features too** |

The paper, page 533: perceptrons have "feature analysers", *"but these are not true hidden units because their input connections are fixed by hand … they do not learn representations."*

**Why hidden units are hard:**
- For an **output** unit, the task says what it should be (the target), so its error is obvious.
- For a **hidden** unit, nobody says what it *should* do. How do you know whether to make it more or less active?
- Backprop's answer: **ask how the final error would change if this unit's activity changed.** That number is a derivative, and the chain rule computes it.

---

## 2. The math toolkit (a gentle refresher)

### 2.1 The derivative = the slope = "how much does the output change per unit change in the input?"
- If f(w) = w², then f′(w) = 2w. At w = 3, nudging w up by 0.01 raises f by about 2·3·0.01 = 0.06.
- **Notation:** ∂E/∂w ("partial derivative") means the same thing when E depends on many variables: the slope in the w direction, with all the others held fixed.

### 2.2 Gradient descent: why "−slope" goes downhill
- **The idea:** for a small step Δw, the error changes by about:
  ```
  E(w + Δw) ≈ E(w) + (∂E/∂w) · Δw
  ```
- Choose **Δw = −ε · ∂E/∂w** (ε > 0 small). Then:
  ```
  E(w + Δw) ≈ E(w) − ε · (∂E/∂w)²   ≤  E(w)
  ```
  A square is never negative, so **the error goes down** whenever the slope isn't zero.
- **This holds for every weight at once:** the change is −ε Σ (∂E/∂w)². That's Eq. (8).

### 2.3 The chain rule: slopes multiply along a chain
If E depends on y, y depends on x, and x depends on w, then:
```
∂E/∂w = ∂E/∂y · ∂y/∂x · ∂x/∂w
```
*Example:* if doubling x triples y, and tripling y adds 5 to E, the effects multiply through the chain. **Backprop is the chain rule, organized so that no product is computed twice.**

---

## 3. The network (page 533)

- Layers: inputs at the bottom, any number of hidden layers, outputs at the top.
- **Connections go upward only.** They may skip layers, but never go sideways or down.
- Compute the layers **bottom to top**: the **forward pass**.

### Eq. (1): total input to unit j
```
x_j = Σ_i  y_i · w_ji          (plus a bias: a weight from an input that is always 1)
```

### Eq. (2): output of unit j, the logistic ("sigmoid") function
```
y_j = σ(x_j) = 1 / (1 + e^(−x_j))
```
- σ(−∞) = 0, σ(0) = 0.5, σ(+∞) = 1. It's a **smooth** version of Paper 001's on/off threshold.
- **Why smooth?** The step function's slope is 0 everywhere except at the jump, where it's infinite. Gradient descent needs useful slopes. The paper: any function with a **bounded derivative** will do.

### The logistic's slope (needed for Eq. 5): derivation
```
σ(x) = (1 + e^(−x))^(−1)
σ′(x) = −(1 + e^(−x))^(−2) · (−e^(−x)) = e^(−x) / (1 + e^(−x))²
      = [1/(1 + e^(−x))] · [e^(−x)/(1 + e^(−x))] = σ(x) · (1 − σ(x))
```
So **dy/dx = y(1 − y)**: the slope comes for free from the output.
- At y = 0.5 it's **0.25**, its maximum.
- At y = 0.99 it's 0.0099: almost flat. **A saturated unit barely learns.** Remember this; it explains several findings below.

---

## 4. The error (Eq. 3)
```
E = ½ Σ_c Σ_j (y_j,c − d_j,c)²         c = training case, j = output unit, d = desired output
```
- This is the "sum of squared errors".
- The ½ is there so the 2 from differentiating the square cancels.

---

## 5. The backward pass (pages 534–535): derived step by step

**Goal:** ∂E/∂w for every weight. Work **top down**. For one case:

### Eq. (4): the error's slope at an output unit
```
E = ½(y_j − d_j)²   ⇒   ∂E/∂y_j = y_j − d_j
```
It is positive if the output is too high, negative if too low.

### Eq. (5): through the logistic, to the unit's total input
```
∂E/∂x_j = ∂E/∂y_j · dy_j/dx_j = ∂E/∂y_j · y_j(1 − y_j)          (chain rule + section 3's slope)
```
Call this δ_j, the unit's **error signal**.

### Eq. (6): to a weight feeding the unit
x_j = Σ_i y_i w_ji, so ∂x_j/∂w_ji = y_i. Therefore:
```
∂E/∂w_ji = δ_j · y_i
```
**A weight's blame = the error signal of the unit it feeds × how active its input was.** If y_i = 0, the weight had no influence, so it gets no blame.

### Eq. (7): down to the unit below (the key step)
Unit i feeds **every** j above it, and changing y_i changes all of their x_j's (by w_ji each). The effects add up:
```
∂E/∂y_i = Σ_j  ∂E/∂x_j · ∂x_j/∂y_i = Σ_j δ_j · w_ji
```
Now unit i's ∂E/∂y_i is known, so **apply Eq. (5) to it and repeat** all the way down. The error **propagates backwards** through the same connections the activity went forward through.

### Why this is cheap
- The backward pass costs about **the same as one forward pass**: each weight is used once going up and once going down.
- **The naive alternative:** nudge each weight separately and re-run the network to see how E changes (finite differences). That needs **one forward pass per weight**: millions of passes for millions of weights. **Backprop gets all gradients for the price of about two passes.** That is why big networks are trainable at all.

---

## 6. A complete example by hand

**The network:**
- input u = 1 → hidden unit h (weight w₁ = 0.5, no bias);
- h → output o (weight w₂ = −0.4, bias b = 0.3);
- target d = 1, learning rate ε = 0.5.

**Forward:**
```
x_h = 0.5 · 1 = 0.5               y_h = σ(0.5)     = 0.6225
x_o = 0.3 + (−0.4)(0.6225) = 0.051  y_o = σ(0.051) = 0.5128
E = ½ (0.5128 − 1)² = 0.1187
```

**Backward:**
```
Eq. 4:  ∂E/∂y_o = 0.5128 − 1                     = −0.4872
Eq. 5:  δ_o = −0.4872 · 0.5128 · 0.4872          = −0.1217
Eq. 6:  ∂E/∂w₂ = δ_o · y_h = −0.1217 · 0.6225    = −0.0758
        ∂E/∂b  = δ_o · 1                         = −0.1217
Eq. 7:  ∂E/∂y_h = δ_o · w₂ = −0.1217 · (−0.4)    = +0.0487
Eq. 5:  δ_h = 0.0487 · 0.6225 · 0.3775           = +0.0114
Eq. 6:  ∂E/∂w₁ = δ_h · u = 0.0114 · 1            = +0.0114
```

**Update (Eq. 8, Δw = −ε ∂E/∂w):**
```
w₂: −0.4 + 0.5·0.0758 = −0.3621      b: 0.3 + 0.5·0.1217 = 0.3609      w₁: 0.5 − 0.5·0.0114 = 0.4943
```

**Forward again:** y_h = 0.6211, y_o = 0.5339, **E = 0.1086** (down from 0.1187 ✔).

**Read the signs:**
- The output was too low, so b went up.
- w₂ was negative and was dragging o down, so it moved toward zero.
- **w₁ went *down*:** h was hurting (through the negative w₂), so the network made h a bit *less* active.

**Nobody told the hidden unit what to do. The chain rule did.** (Our `test_backprop.py` checks gradients like these against finite differences on bigger networks.)

---

## 7. Changing the weights (page 535)

### Eq. (8): gradient descent
```
Δw = −ε · ∂E/∂w
```
The paper **adds up** ∂E/∂w over all cases and updates once per **sweep** (epoch), i.e. "batch" gradient descent.

### Eq. (9): momentum
```
Δw(t) = −ε · ∂E/∂w(t) + α · Δw(t−1)          (0 ≤ α < 1, typically 0.9)
```
**Unrolling it shows what it does:**
```
Δw(t) = −ε [ g(t) + α g(t−1) + α² g(t−2) + … ]
```
- It is a **running, decaying average of past gradients**.
- On a long gentle slope the gradients agree, so the steps add up. With a steady gradient g, the step grows to −ε g/(1 − α), which is **10× bigger** for α = 0.9.
- In a zig-zagging valley the alternating gradients cancel.
- It's a ball rolling downhill with friction, using only first derivatives. (Paper 008 revisits momentum in depth.)

### Breaking symmetry
- Start from **small random** weights.
- If all the weights started equal, every hidden unit would compute the same thing, receive the same δ, and stay identical forever.

**Credit:** variants were found independently by **David Parker** and **Yann LeCun** (and earlier by Werbos, 1974).

---

## 8. Experiment 1: mirror symmetry (Figure 1, page 534)

**The task:** 6 bits; answer "yes" if the pattern reads the same backwards (101101 yes, 100000 no).

**Why a hidden layer is needed:**
- **Any single input bit is useless on its own.** Flip one bit of a symmetric pattern and you get an asymmetric one, and vice versa.
- So no weighted sum of single bits works (Paper 003's argument).

**Setup:** 6 → **2 hidden** → 1. All 64 patterns per sweep, ε = 0.1, α = 0.9, initial weights in [−0.3, 0.3]. The paper's run took **1,425 sweeps**.

**The solution it found (and the math of why it works):**
- **Mirror weights are equal and opposite.** Hidden unit 1's weights are (−a, −b, −c, +c, +b, +a).
  - For a symmetric input, bit k equals bit 7−k, so each pair contributes −a·s + a·s = 0, and the total input is **exactly 0**.
  - The **negative bias** then keeps the hidden unit **off**.
- **The weights on one side are in ratio 1 : 2 : 4.**
  - For an asymmetric input, the input is a·(s₆ − s₁) + 2a·(s₅ − s₂) + 4a·(s₄ − s₃). Each difference is −1, 0 or +1, and like binary digits, **no combination of ±1, ±2, ±4 except all-zero sums to 0**.
  - So any asymmetry gives a nonzero input.
- **The second hidden unit is the sign-flipped copy.** Whatever sign the asymmetry has, one of the two units gets a large **positive** input, turns on, and drives the output (through a big negative weight) to "no".
- **The output's positive bias** says "yes" when both hidden units are off.

**Nobody designed this. Gradient descent found a binary-number trick.**

---

## 9. Experiment 2: family trees (Figures 2–4, pages 534–535)

**The data:**
- Two family trees with **identical shape**, English and Italian, 12 people each.
- Facts are triples (person₁, relation, person₂), e.g. *Colin has-aunt Jennifer*.
- 12 relations (father, mother, husband, wife, son, daughter, uncle, aunt, brother, sister, nephew, niece) give **104** facts.

**The network (Figure 3):**
```
24 person units ──► 6 units ─┐
                             ├──► 12 central ──► 6 ──► 24 output person units
12 relation units ─► 6 units ┘
```
- **Input:** a **one-hot** code, meaning exactly one person unit and one relation unit are on.
- **Output:** every correct person₂ (Colin has two aunts, so two outputs are on).

**Why the 6-unit bottleneck matters:**
- **One-hot inputs carry no similarity:** every two people are equally different.
- Squeezing 24 people into 6 numbers forces the network to place people with similar "roles" near each other. That is a **learned embedding**, the ancestor of word vectors (Paper 019).

**Training:**
- 1,500 sweeps: ε = 0.005, α = 0.5 for the first 20, then ε = 0.01, α = 0.9.
- **Weight decay 0.2%** per update: every weight shrinks a little toward zero, so only useful weights survive.
- "Close enough": an "on" unit above 0.8, or an "off" unit below 0.2, counts as correct.
- 100 facts for training, **4 held out**.

**Results:**
- **All 4 unseen facts were answered correctly.**
- **Figure 4:** the 6 person-code units learned features that **were never in the input**:
  - unit 1 = English vs Italian;
  - unit 2 = generation;
  - unit 6 = branch of the family.
- Most units ignore nationality, so an English person's code resembles their Italian "twin". The network **shares structure** between the two trees, and that's why it generalizes.

---

## 10. Recurrent networks (Figure 5, page 535)

- A **recurrent** net feeds its units back to themselves over time.
- Unrolled for T steps, it's a T-layer feed-forward net in which **every layer has the same weights**.
- Train it with backprop, with two changes:
  1. **Store** every unit's output at every time step; the backward pass needs them.
  2. Each copy of a weight gets its own gradient. **Add them up** and change the shared weight once. (Chain rule: a weight used in T places affects E through all T paths.)
- This is **backpropagation through time (BPTT)**, the engine of RNNs and LSTMs (Stage 05).

---

## 11. Limits (pages 535–536)

- **Local minima:** gradient descent can stop in a poor valley. The authors say it rarely happens, mainly in nets with *just enough* connections. *"Adding a few more connections creates extra dimensions in weight-space and these dimensions provide paths around the barriers."*
- **Not a brain model:** *"not a plausible model of learning in brains"*. But it shows that rich representations **can** be learned by gradient descent.

### 11.1 The vanishing-gradient arithmetic (the reason behind our biggest finding)
- **Every logistic layer multiplies the backward signal by y(1 − y) ≤ 0.25** (Eq. 5), times the weights (Eq. 7).
- With small weights (≈ 0.3), each connection passes back at most 0.25 × 0.3 ≈ 0.075 of the error. Summing over a handful of units in the layer above doesn't make up for that, so the signal **shrinks layer after layer**.
- From the logistic slopes alone, the factor is at most 0.25⁴ ≈ 0.004 after 4 layers and 0.25¹⁰ ≈ 10⁻⁶ after 10.
- **The bottom layers hardly receive any error, so they barely learn.** Papers 006–008 and 012 are largely about fixing this.

---

## 12. What our reproduction found

See [results.md](results.md) and [figures/](figures/). Every number below comes from running the code.

### ✅ Reproduced
- **The backprop equations are exactly right:** Eqs. (1)–(7) match finite differences on a net with skip connections and two input groups, and also on the recurrent net.
- **Symmetry (Figure 1):**
  - solved in **18/20** random starts with the paper's settings, median **2,208 sweeps** (the paper's single run: 1,425; our demo run: 1,121);
  - **the learned solution matches Figure 1 in every successful run:** mirror weights equal and opposite (error < 1%), the two hidden units exact sign-flipped copies (correlation −1.000), negative hidden biases with a positive output bias, and magnitudes like **1 : 2.01 : 3.90** (the paper's 1 : 2 : 4).
- **No hidden layer, no symmetry:** the best is 57/64, about as good as always answering "no" (56/64).
- **Family trees:**
  - **98.7%** of training facts learned;
  - generalization above chance (the demo run gets 4/4 held-out facts);
  - code units track **nationality** (correlation 0.83), **generation** (0.73) and **branch** (0.61).
- **Recurrent net:** trained through time to remember the first of 3 bits, it gets all 8 sequences right.

### ⚠️ Different from the paper, and why
1. **The paper's exact family-tree recipe learns nothing:**
   - with ±0.3 initial weights, ε = 0.01 and 0.2% decay, **0%** of the training facts are learned;
   - this is the vanishing gradient of section 11.1: 4 logistic layers with small weights, plus decay that outweighs the tiny gradients;
   - **fix:** initial weights ±1.0, ε = 0.05 and 10× smaller decay;
   - Paper 007 (Glorot & Bengio) explains this and fixes it with better initialization.
2. **Generalization is weaker on average than the paper's single run:** over 20 runs, 36 of 80 held-out facts (45%) are exactly right. English–Italian twin codes are only somewhat closer than random pairs (ratio 0.84, where 1.0 = no closer).
3. **"Local minima" were really saturation:**
   - The failed symmetry runs all ended at **E = 4.0**, the "always say not symmetric" plateau. 56 of 64 patterns are not symmetric, so the output saturates near 0, and y(1 − y) ≈ 0 kills the gradient.
   - With ε = 0.1, *bigger* nets hit this more often (8 hidden units: 4/20 solved).
   - With **ε = 0.03, every net size solves it 20/20**. The culprit is the step size, not too few connections.

---

## 13. Check yourself

1. Derive dy/dx = y(1 − y) for the logistic.
2. Show why Δw = −ε ∂E/∂w lowers E for small ε.
3. Write Eqs. (4)–(7). Which one sends the error down a layer, and why is it a **sum**?
4. Redo section 6's example with target d = 0. Which way does w₁ move now?
5. Why is backprop's cost about two forward passes, while finite differences cost one pass per weight?
6. Explain how weights in ratio 1 : 2 : 4 let the symmetry net detect *any* asymmetry.
7. What does momentum do to a steady gradient g, with α = 0.9?
8. Why do deep logistic nets with small weights barely learn in their bottom layers?
