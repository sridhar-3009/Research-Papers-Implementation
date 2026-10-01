# McCulloch & Pitts (1943), explained from scratch

**Paper:** *A Logical Calculus of the Ideas Immanent in Nervous Activity*
**Authors:** Warren McCulloch (a neurophysiologist) and Walter Pitts (a self-taught logician, aged 20 at the time)
**Published in:** Bulletin of Mathematical Biophysics, Vol. 5, 1943, pp. 115–133 ([doi:10.1007/BF02478259](https://doi.org/10.1007/BF02478259))
**Page numbers** below follow the 21-page 2008 seminartext reprint, not the journal's page numbers.

This guide assumes no background. Every symbol is defined before it is used, and every rule is tried on small examples.

---

## 0. The whole idea in one line

> **A neuron is either ON or OFF. A statement is either TRUE or FALSE. So a network of neurons is a machine that does logic, and with memory added, a machine that can compute anything a computer can.**

This is the first mathematical model of a neural network. Every network in this repository, up to today's Transformers, descends from the little "threshold unit" defined here.

---

## 1. The biology they started from (pages 1–4)

### 1.1 What a real neuron does
A neuron is a cell with three parts:
- **Dendrites** (input branches) receive signals from other neurons at contact points called **synapses**.
- **The cell body** adds the incoming effects up.
- **The axon** (the output cable) sends a pulse, the **spike** or **action potential**, to other neurons.

### 1.2 Four facts known in 1943
1. **All-or-none.**
   - A neuron either fires a full spike or nothing.
   - The spike's size does not depend on how strong the input was. (Stronger inputs make spikes *more frequent*, not bigger.)
   - So at any instant, the output is like a bit: 0 or 1.
2. **A threshold.**
   - One incoming spike is usually not enough to fire the neuron.
   - Its effects must add up past a fixed level, the **threshold**, within a short time window. This adding-up is called **summation**.
3. **A delay.**
   - It takes roughly a millisecond for a signal to cross a synapse.
   - So the output at one moment depends on inputs slightly *before* it.
4. **Inhibition.**
   - Some synapses make firing *harder* instead of easier.

### 1.3 The leap: neurons ↔ logic
McCulloch and Pitts made a bold translation:

| Neuron | Logic |
|---|---|
| fires (1) | the proposition is TRUE |
| silent (0) | the proposition is FALSE |
| connections | logical connectives (AND, OR, NOT…) |

**Two complications, and how they handled them:**
- **Real neurons change:** they tire, become sensitized, and learn.
- **Their claim:** these effects can always be *imitated* by a fixed network with extra neurons and loops, so they add no new logical power (Theorems IV–VII, section 8).
- **They say clearly that this is a mathematical idealization.** The claim is that real nets *behave* like these, not that they *are* these.

---

## 2. The model neuron, as math (page 4)

### 2.1 The five assumptions
| # | Assumption | In plain words |
|---|---|---|
| 1 | All-or-none | Each neuron's output is 0 or 1 |
| 2 | Fixed threshold | A neuron fires iff at least θ excitatory synapses were active in the previous tick |
| 3 | The only delay is synaptic | Time moves in discrete **ticks** t = 0, 1, 2, … and each synapse costs exactly one tick |
| 4 | Absolute inhibition | If **any** inhibitory input is active, the neuron cannot fire (a veto) |
| 5 | Fixed structure | Connections never change |

### 2.2 The neuron equation
Let:
- x_i(t) ∈ {0, 1} be input neuron i at tick t;
- w_i be **how many** excitatory synapses input i makes onto our neuron (a whole number, 0, 1, 2, …);
- θ be the threshold (a positive whole number);
- I be the set of inhibitory inputs.

Then:
```
          ⎧ 1   if  Σ_i w_i · x_i(t−1) ≥ θ   and   x_k(t−1) = 0 for every inhibitory k in I
y(t)  =   ⎨
          ⎩ 0   otherwise
```

**Read it slowly:**
- **Σ_i w_i·x_i(t−1)** counts how many excitatory "votes" arrived one tick ago. An input that is ON and makes 2 synapses contributes 2 votes.
- **≥ θ** means "enough votes".
- **The inhibitory condition** is a hard veto: one inhibitory input beats any number of votes.

**This is the ancestor of every artificial neuron.**
- Modern neurons replace whole-number counts with real-valued **weights** (which may be negative), and the hard step with a smooth function.
- They also replace absolute inhibition with negative weights.
- The *shape* has stayed the same: **weighted sum → threshold**.

### 2.3 Worked example
Take θ = 2, input A making 1 synapse, input B making 1 synapse, and an inhibitory input C.

| A(t−1) | B(t−1) | C(t−1) | votes | y(t) |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 | 0 (1 < 2) |
| 1 | 1 | 0 | 2 | **1** |
| 1 | 1 | 1 | 2 | 0 (vetoed) |

---

## 3. The notation (page 5)

The paper uses the logic notation of Russell & Whitehead's *Principia Mathematica* (Carnap's version):

| Paper writes | Means |
|---|---|
| N_i(t) | "neuron i fires at tick t" (a proposition: true or false) |
| `.` | AND |
| `∨` | OR |
| `~` | NOT |
| S | the "shift" operator: S N_i(t) means N_i(t−1), one tick earlier |
| ≡ | "if and only if", "exactly when" |

**Example:** N₃(t) ≡ N₁(t−1) . N₂(t−1). This reads: "neuron 3 fires exactly when neurons 1 and 2 both fired one tick ago." That is an AND gate with a one-tick delay.

**About the dots:** the paper also uses dots as brackets (more dots means a bigger bracket). Read a lone `.` between propositions as AND.

---

## 4. Vocabulary (pages 5–7)

| Term | Meaning |
|---|---|
| **peripheral afferents** | input neurons: nothing in the net feeds them (like sensors in the skin) |
| **solution of a net** | a formula for every neuron in terms of the inputs only |
| **realizable** | "some net computes this formula" |
| **in the extended sense** | "…if we allow the answer to arrive a few ticks late". The paper always allows this |
| **circle** | a loop: a path of connections that comes back to where it started |
| **order of a net** | the fewest neurons whose removal breaks every loop. Order 0 = loop-free |
| **TPE** (temporal propositional expression) | a formula built from input propositions with S (delay), ∨, `.`, and "A . ~B". **Plain NOT is not allowed** |

---

## 5. The four building blocks (Figure 1a–d, page 18)

Each block is **one neuron with θ = 2** and costs **one tick**. Only the number of synapses changes.

| Figure | Function | Wiring | Why it works |
|---|---|---|---|
| **1a** | delay: N₂(t) ≡ N₁(t−1) | A makes 2 synapses | A alone gives 2 votes ≥ 2 |
| **1b** | OR: N₃(t) ≡ N₁(t−1) ∨ N₂(t−1) | A makes 2, B makes 2 | either one alone gives 2 votes |
| **1c** | AND: N₃(t) ≡ N₁(t−1) . N₂(t−1) | A makes 1, B makes 1 | only both together give 2 votes |
| **1d** | A AND NOT B: N₃(t) ≡ N₁(t−1) . ~N₂(t−1) | A makes 2, B **inhibits** | A gives 2 votes unless B vetoes |

**Truth tables** (inputs at t−1, output at t):

| A | B | OR (1b) | AND (1c) | A·~B (1d) |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 | 0 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 | 0 |

**The key insight:** OR and AND are **the same neuron**, differing only in connection strengths. **Changing weights changes the computed function.** Seventy years of machine learning is about finding good weights automatically.

---

## 6. The three main theorems for loop-free nets (pages 7–9)

### Theorem I: every loop-free net computes a TPE
**Idea of the proof** (a substitution argument):
1. Write each neuron's rule as a formula of the neurons feeding it, one tick earlier.
2. Replace those neurons by *their* formulas, and keep going.
3. With no loops, every path back from a neuron reaches the inputs after finitely many steps, so the substitution stops.
4. The result is a formula built from inputs, delays, AND, OR and AND-NOT. That is a TPE.

### Theorem II: every TPE is computed by some loop-free net
**Idea of the proof** (induction on the formula's structure):
- A single input is just that input neuron.
- If nets for formulas F and G exist, then F ∨ G, F . G and F . ~G are obtained by feeding both nets into one more neuron of type 1b, 1c or 1d.
- **Timing:** if F's answer arrives at tick 3 but G's at tick 5, insert two delay neurons (1a) after F so that both arrive together. That's why "in the extended sense" (late answers are allowed) matters.

**Theorems I + II together:** loop-free nets and TPEs are the same thing in two languages.

### Theorem III: exactly which formulas are realizable
**Statement:** a formula is realizable iff it is **FALSE when every input is OFF**. The paper calls these formulas "not identically true when all inputs are silent".

**Why (the science):** the threshold θ ≥ 1, so a neuron with zero input votes stays silent. By induction from the inputs upward, **if every input is 0, every neuron is 0**. So no net can output TRUE on all-silent inputs.

| Formula | value when all inputs are 0 | realizable? |
|---|---|---|
| A ∧ B | 0 | ✅ |
| A ∨ B | 0 | ✅ |
| A ∧ ¬B | 0 | ✅ |
| A ⊕ B (XOR) | 0 | ✅ (with 2 layers) |
| ¬A | **1** | ❌ |
| A → B (= ¬A ∨ B) | **1** | ❌ |

**Counting how much is lost:**
- With n inputs there are 2ⁿ input patterns.
- A truth table assigns 0 or 1 to each pattern, so there are 2^(2ⁿ) Boolean functions.
- Realizable ones must output 0 on the all-zero pattern, leaving the other 2ⁿ − 1 patterns free: **2^(2ⁿ − 1)** functions. That is **exactly half**.

  *Example:* for n = 2 there are 16 functions, and 8 are realizable.
- The fix is cheap in practice: add one "always ON" input (a constant 1). Then ¬A = (always ON) . ~A becomes realizable. Modern networks do the same with a **bias** term.

### Worked example: XOR needs two layers
XOR(A, B) = (A . ~B) ∨ (B . ~A).
- **Layer 1:** neuron u = A . ~B (block 1d) and neuron w = B . ~A (block 1d).
- **Layer 2:** the output = u ∨ w (block 1b).

The answer arrives after **2 ticks**.

**Why one neuron can't do it:** one threshold neuron splits the input patterns with a single straight line (more on this in Papers 002–003). XOR's TRUE points (0,1) and (1,0) sit on opposite corners of the square, so no single line separates them from (0,0) and (1,1). The absolute veto doesn't help either: AND-NOT can only *remove* cases, and XOR needs two separate "pockets" of TRUE.

---

## 7. The famous example: the heat/cold illusion (pages 9–10)

### 7.1 The real phenomenon
- If a **cold** object touches your skin for a moment and is removed, you feel **heat**.
- If the cold stays, you feel **cold**.

The first instant is identical in both cases, so the brain must **wait and see** what happens next before deciding what you feel. This shows that **what we perceive depends on timing and wiring, not just on the stimulus**.

### 7.2 The formulas
**Neurons:**
- N₁ = heat receptor;
- N₂ = cold receptor;
- N₃ = "feel heat";
- N₄ = "feel cold".

They assume cold must persist for 2 ticks to be felt:
```
N₃(t) ≡ N₁(t−1) ∨ [ N₂(t−3) . ~N₂(t−2) ]        feel heat: heat now, OR cold that came and went
N₄(t) ≡ N₂(t−2) . N₂(t−1)                        feel cold: cold two ticks in a row
```

### 7.3 Built from the four blocks (Figure 1e)
1. a = delay(N₂): "cold one tick ago".
2. N₄ = a AND N₂: "cold, and cold before it".
3. b = a AND NOT N₂: "it was cold, now it isn't".
4. N₃ = N₁ OR b.

### 7.4 Hand simulation (verified by our code)
**Brief cold** (cold only at tick 0):

| tick | cold N₂ | a | b | feel cold N₄ | feel heat N₃ |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 1 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | **1 ← heat!** |

**Long cold** (cold at every tick):

| tick | cold | a | b | feel cold | feel heat |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 0 | 0 | 0 |
| 2 | 1 | 1 | 0 | **1 ← cold** | 0 |
| 3+ | 1 | 1 | 0 | 1 | 0 |

**A detail the paper doesn't mention:** cold lasting **exactly 2 ticks** gives "cold" at tick 2 *and then* "heat" at tick 4. Its formula for N₃ predicts this too.

---

## 8. "More realistic neurons add nothing" (pages 10–12)

Each realistic feature can be imitated by a fixed net with some extra neurons. The answer may arrive later, but the *logic* is the same.

| Theorem | Realistic feature | Imitated by | Figure |
|---|---|---|---|
| **IV** | relative inhibition (inhibition raises the threshold instead of vetoing) | an equivalent net with absolute vetoes | 1f |
| **V** | extinction / fatigue (a neuron is less excitable after firing) | loops that feed back and block it | 1g |
| **VI** | temporal summation (votes from different ticks add up) | delay chains that make them arrive together | 1h |
| **VII** | **learning** (a synapse that becomes effective after pairing) | **a loop that remembers** the pairing | 1i |

### Theorem VII is historically remarkable
**Their rule:** if input A is active *while* the neuron fires because of B, A's synapse becomes permanent. Afterwards, A alone is enough.

- This is **"cells that fire together wire together"**, stated **six years before Hebb (1949)**.
- It is also Pavlovian conditioning: bell + food → later the bell alone works.

**Their trick:** don't change the wiring. Add a **latch** neuron L that turns ON the first time A and B coincide and then keeps itself ON through a self-loop. From then on, L's "help" lets A alone reach the threshold.

**Our code (Figure 1i):** input 1 alone does nothing at tick 0. At tick 3 it is paired with 2, the latch turns ON at tick 4, and from then on input 1 alone makes neuron 3 fire (tick 7). ✔

---

## 9. Nets with loops: memory and computation (pages 12–17)

**This section is very hard to read; the authors call it "very sketchy".** Here is the core idea in modern words.

### 9.1 Loops = memory = states
- A neuron that excites itself keeps firing forever once started. That is **one bit of memory**.
- With p neurons inside loops, the net has at most **2ᵖ** possible memory states.
- At each tick, the next state is a fixed function of (current state, current inputs).

That is exactly a **finite-state machine**, a term that didn't exist yet. In 1956 Kleene turned this section into the theory of **finite automata and regular expressions**, which is now basic computer science.

### 9.2 Three tiny loops (Theorem X)
| Loop | Rule | Answers the question |
|---|---|---|
| **Latch (memory)** | M(t) = P(t−1) ∨ M(t−1) | "Did P **ever** happen?" Once ON, it stays ON |
| **Always-so-far** | A(t) = P(t−1) . A(t−1) | "Has P been true **every** tick?" Once OFF, it stays OFF |
| **Clock** | a ring of n neurons, each copying the previous | a pulse every n ticks |

**Our code:**
- the latch turns ON at tick 3 after a single pulse at tick 2, and never turns off;
- a ring of 3 fires c₀ at ticks 0, 3, 6, 9. ✔

### 9.3 What loops can and cannot do
- **Theorem VIII:** a net with loops can still be described, but the formulas must quantify over the past ("at *some* earlier time…").
- **No net can see the future:** outputs depend only on past inputs.
- **Theorem IX:** an exact characterization of what loop nets can detect. Correct, but unusable in practice.

### 9.4 The Turing connection (page 17)
- A net **plus an unlimited tape** (external memory it can read and write) computes **exactly what a Turing machine computes**, i.e. anything computable.
- Without the tape, memory is limited to 2ᵖ states, so it is strictly weaker. (A finite machine can't, for example, check that arbitrarily long bracket strings are balanced.)
- **The consequence:** "brain-like" networks and digital computers have the same computational power. **John von Neumann** used McCulloch–Pitts neurons to describe the logic of the first stored-program computer (EDVAC, 1945).
- *A small slip in the paper:* it says "primitive recursive". Turing-equivalence corresponds to *general* recursive functions, a larger class.

---

## 10. The philosophy (pages 17–21)

1. **The future is predictable, the past is not recoverable.** An OR neuron that fired doesn't tell you *which* input caused it. Information is lost going forward.
2. **Losing detail is how general ideas form.** A neuron for "red" must ignore *which* red thing you saw. In machine-learning terms: **abstraction and generalization**.
3. **Perception depends on wiring:** change the net (disease, drugs) and you change experience (hallucinations, tinnitus).
4. **The atom of mind is one neuron firing,** so mental events obey the two-valued logic of propositions.
5. **Purpose from feedback:** a system that acts to reduce the difference between what it senses and a goal *looks* purposeful (thermostats, hunger, attention). In ML, that is **minimizing a loss**.
6. **Many nets, same behaviour:** you can't infer someone's exact wiring from their behaviour.

---

## 11. What was missing (and came later)

| Missing in 1943 | Added by |
|---|---|
| Real-valued weights, negative weights | Rosenblatt's perceptron (1958, Paper 002) |
| **Learning weights from examples** | perceptron learning rule (002); backpropagation (1986, Paper 004) |
| Smooth outputs (for gradients) | sigmoid units (004), later ReLU |
| Bias (a constant input) | the perceptron's threshold-as-weight trick |

---

## 12. What our code found

The code in this folder simulates the model **exactly as defined**: 0/1 outputs, whole-number synapse counts, an absolute veto, and one tick per synapse.
- **The four blocks (Figure 1a–d)** reproduce their truth tables, each after one tick.
- **The heat/cold net (Figure 1e)** gives heat at tick 3 for brief cold and cold from tick 2 for long cold, exactly the tables above.
- **The learning loop (Figure 1i)** makes input 1 effective on its own only after one pairing with input 2.
- **The memory latch and the 3-neuron clock** behave as in Theorem X.
- **A TPE compiler (Theorem II as code):** give it a formula such as XOR = (p·~q) ∨ (q·~p) and it builds the net, inserting delay neurons to align timing. The XOR net is correct on all 4 inputs and needs **2 layers**.
- **Theorem III** is built into the compiler's language: it has no plain NOT, only AND-NOT. A formula like ¬A can't even be written, and anything that isn't a TPE raises `TypeError: not a TPE`.

---

## 13. Check yourself

1. Write the neuron equation. What do w_i, θ and the veto mean?
2. Why can't any net compute ¬A? Give the one-line proof.
3. How many of the 16 two-input Boolean functions are realizable? Why exactly half?
4. OR and AND are the same neuron. What differs?
5. Build XOR from the four blocks. How many ticks does it take?
6. In the heat/cold net, what is the delay neuron a for?
7. How does a loop give a net memory? With p loop neurons, how many states at most?
8. What extra ingredient makes these nets as powerful as a Turing machine?
