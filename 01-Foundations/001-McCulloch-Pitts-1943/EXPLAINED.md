# McCulloch & Pitts (1943), explained simply

**Paper:** *A Logical Calculus of the Ideas Immanent in Nervous Activity*
**Published in:** Bulletin of Mathematical Biophysics, Vol. 5, 1943, pp. 115–133 ([doi:10.1007/BF02478259](https://doi.org/10.1007/BF02478259))
**Page numbers** below follow the 21-page 2008 seminartext reprint, not the journal's page numbers.

---

## The big idea in one line

> **A neuron is either ON or OFF. So is a true/false statement. So a brain is a machine that does logic.**

That's the whole paper. Everything else is proving it carefully.

---

## 1. Why they wrote this (pages 1–4)

In 1943, scientists knew a few facts about neurons:

- A neuron either **fires fully or not at all**. There is no half-firing. This is called **"all-or-none."**
- A neuron needs **several inputs at the same time** to fire. One input alone is not enough.
- There is a **small delay** between a neuron getting input and firing.
- Some connections **stop** a neuron from firing. This is **inhibition**.

McCulloch and Pitts noticed that on/off works like **true/false**. So:

- "Neuron fires" = **"this statement is true"**
- "Neuron is quiet" = **"this statement is false"**

And if neurons are true/false statements, then connections between neurons work like **AND, OR, NOT**.

**Two problems they had to deal with:**
1. **Neurons change with use.** They get tired or more sensitive for a while.
2. **Brains learn.** Connections change permanently.

**Their answer:** these changes don't matter for the logic. You can always replace a changing network with a **fixed** network that behaves the same way (proved later, pages 10–12).

**Important:** they say this is a **math trick, not biology**. They don't claim the brain really works this way inside. Only that it **behaves** the same.

---

## 2. The neuron model: 5 simple rules (page 4)

This is the famous **McCulloch–Pitts neuron**. It's the grandfather of every neural network.

| # | Rule | What it means |
|---|---|---|
| 1 | All-or-none | Output is only **0 or 1** |
| 2 | Fixed threshold | Fires only if **enough inputs** are ON at the same time |
| 3 | Only delay is at the connection | Time moves in **ticks**. Output at tick `t` depends on inputs at tick `t−1` |
| 4 | Inhibition is absolute | If **any** blocking input is ON, the neuron **cannot** fire. It's a veto |
| 5 | Nothing changes | Connections never change. **No learning** |

**Example.** A neuron needs 2 input signals to fire:
- Input A sends 1 signal, input B sends 1 signal → both ON = 2 → **fires**
- Only A ON = 1 → **doesn't fire**
- Both ON, but a blocking input is also ON → **doesn't fire** (veto)

**Note:** one input neuron can connect **more than once** to the same neuron. Two connections = counts as 2. That's how an input gets more "weight." (There are no decimal weights here, only whole counts.)

---

## 3. The notation (page 5): just 4 things to know

The paper's symbols look scary but are simple:

| Paper writes | Means |
|---|---|
| `N₁(t)` | Neuron 1 fires at time t |
| `.` | AND |
| `v` | OR |
| `~` | NOT |
| `S` | "one tick earlier". So `S N₁` = neuron 1 one tick ago |
| `≡` | "exactly when" |

**Example:** `N₃(t) ≡ N₁(t−1) . N₂(t−1)`
→ "Neuron 3 fires **exactly when** neuron 1 **and** neuron 2 fired one tick ago."

**Tip:** the paper uses **dots as brackets**. More dots = bigger bracket. Don't worry about it, just read `.` as AND.

---

## 4. Key words (pages 5–7)

- **Input neurons** (paper calls them "peripheral afferents"): neurons that nothing else feeds into. Like sensors in your skin.
- **Solving a net:** writing what each neuron does using **only the inputs**.
- **Realizable:** "we can build a network that does this."
- **Extended sense:** "it's OK if the answer comes a few ticks late." The paper always allows this.
- **Circle:** a loop in the network (A → B → A).
- **Order:** how many neurons you must remove to break all loops. Order 0 = no loops.
- **TPE:** a formula built only from **delay, AND, OR, and "A AND NOT B"**. Notice: **no plain NOT**. (Why? See Theorem III.)

---

## 5. The four building blocks (Figure 1a–d, page 18)

Every loop-free network is built from just these four. Each one takes **one tick**.

| Figure | Does | How it's wired (fires when ≥ 2 signals arrive) |
|---|---|---|
| **1a** | **Delay** (copy input one tick later) | A connects twice to the output |
| **1b** | **OR** | A connects twice, B connects twice. Either one is enough |
| **1c** | **AND** | A connects once, B connects once. Needs both |
| **1d** | **A AND NOT B** | A connects twice. B **blocks** |

**Notice:** OR and AND use the **same neuron**. The only difference is **how many connections** each input makes. That's exactly what "weights" do in modern neural networks.

---

## 6. The three main theorems (pages 7–9)

### Theorem I: every loop-free network does logic
Any network without loops computes some logic formula of its inputs.
**Why:** write each neuron's rule, then keep replacing neurons with their own rules until only inputs remain. Without loops, this always ends.

### Theorem II: any such logic can be built as a network
Give me a formula made of delay/AND/OR/AND-NOT, and I can build a network for it.
**How:** plug the four building blocks together, just like the formula is built. If two signals arrive at different times, add delay neurons so they line up.

**So Theorem I + II:** loop-free networks and these logic formulas are **the same thing**.

### Theorem III: which logic can NOT be built
A formula can be built **only if it's FALSE when all inputs are OFF**.

**Why?** A neuron needs input to fire. **Nothing in → nothing out.** So a network can never say "true" when everything is silent.

| Formula | All inputs OFF gives | Can build? |
|---|---|---|
| A AND B | false | ✅ |
| A OR B | false | ✅ |
| A AND NOT B | false | ✅ |
| A XOR B | false | ✅ (needs 2 layers) |
| NOT A | **true** | ❌ |
| A → B | **true** | ❌ |

That's why there's **no plain NOT gate**. Only "A AND NOT B."

---

## 7. The example: hot/cold illusion (pages 9–10)

**This is the most important example in the paper. It's the one to reproduce in code.**

**The real effect:**
- Touch something **cold briefly** and remove it → you feel **heat**. (Strange!)
- Hold something **cold longer** → you feel only **cold**.

The first moment is the same in both cases. So the brain must **wait and see** if the cold continues before deciding what you feel.

**Neurons:**
- 1 = heat sensor
- 2 = cold sensor
- 3 = "I feel heat"
- 4 = "I feel cold"

**Rules** (they assume cold must last 2 ticks to be felt):
- **Feel heat** = heat sensor fired 1 tick ago, **OR** cold fired 3 ticks ago **and then stopped** at 2 ticks ago.
- **Feel cold** = cold fired **2 ticks in a row**.

**The network (Figure 1e), built from the 4 blocks:**
1. **a** = delay of cold → "cold 1 tick ago"
2. **4 (feel cold)** = a AND cold → "cold two ticks in a row"
3. **b** = a AND NOT cold → "cold was on, then turned off"
4. **3 (feel heat)** = heat OR b

**Try it by hand:**

**Brief cold** (cold only at tick 0):

| tick | cold | a | b | feel cold | feel heat |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 1 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | **1** ← feels heat! |

**Long cold** (cold at every tick):

| tick | cold | a | b | feel cold | feel heat |
|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 0 | 0 | 0 |
| 2 | 1 | 1 | 0 | **1** ← feels cold | 0 |
| 3+ | 1 | 1 | 0 | 1 | 0 |

**Bonus finding** (not in the paper): if cold lasts **exactly 2 ticks**, you feel cold at tick 2 and then **heat at tick 4**. Good thing to mention when you write it up.

**Their point:** what you feel depends on **how your neurons are wired**, not just on the real world.

---

## 8. "Realistic neurons don't add power" (pages 10–12)

Real neurons are messier than the 5 rules. The authors show each messy feature can be copied by a simple network. (The answer may just come a few ticks later.)

| Theorem | Messy real feature | Replaced by | Figure |
|---|---|---|---|
| **IV** | Blocking only makes firing *harder* (not a full veto) | Full veto neurons | 1f |
| **V** | Neuron gets **tired** after firing | Loops that feed back and block it | 1g |
| **VI** | Signals at **different times** add up | Delay paths so they arrive **at the same time** | 1h |
| **VII** | **Learning** (a new connection forms) | A **loop** that remembers | 1i |

**Theorem VII is special.** Their learning rule is:
> If input A is active **while** the neuron fires, the A connection becomes permanent.

That's **"neurons that fire together wire together"**: Hebb's famous rule, but **6 years before Hebb** (1949). It's also like Pavlov's dog: bell + food together → later the bell alone works.

In Figure 1i, a loop neuron switches ON the first time A and B fire together, then **stays ON forever**. That's how a fixed network "remembers" it learned something.

---

## 9. Networks with loops (pages 12–17)

**Skim this part.** It's very hard to read, and the authors admit it's "very sketchy" (page 16).

### The main idea (simple version)
- A loop can keep a signal going around **forever**.
- So a loop = **memory**. It remembers **that** something happened, but not **when**.
- With `p` loop neurons, the network has at most `2^p` possible states.
- The network moves from state to state based on inputs.

→ That's a **finite-state machine**. (The paper doesn't use that name. It didn't exist yet. In 1956, Kleene cleaned up this section and it became **finite automata and regular expressions**.)

### What they prove
- **Theorem VIII:** you can describe any network with loops. Its current state = "some history of states and inputs led here."
- **No network can see the future.** Output only depends on past inputs (page 14).
- **Theorem IX:** an exact description of what loop networks can detect. Correct but **impossible to use in practice** (too many cases to check).
- **Theorem X: the useful one.** Three tiny loops give you three new powers:

| Loop | Rule | What it does |
|---|---|---|
| **Memory / latch** | ON if (input now) OR (I was ON before) | Once ON, stays ON forever. "Did X **ever** happen?" |
| **Always-so-far** | ON if (input now) AND (I was ON before) | Turns OFF forever the first time input fails. "Has X **always** been true?" |
| **Clock** | A ring of n neurons passing a signal around | Fires every n ticks |

### The Turing connection (page 17): very important historically
- Network + **unlimited external memory** (a tape) = **exactly as powerful as a Turing machine** (any computer).
- Network **without** a tape = less powerful (limited memory).
- **So:** anything a brain can compute, a computer can compute too, and the other way around.

This inspired **von Neumann**, who used these neurons to describe the design of early computers.

*(Small mistake in the paper: it says "primitive recursiveness." The correct term is "general recursiveness." Primitive recursive is a smaller class.)*

---

## 10. What it all means (pages 17–21)

This part is philosophy, no math. The main points:

1. **You can predict the future, but not recover the past.** An OR neuron firing doesn't tell you *which* input caused it. Information is lost.
2. **Losing detail is useful.** Forgetting details is what lets you form general ideas. A "red" neuron must ignore *which* red thing it saw. (This is **generalization** in ML.)
3. **Perception depends on wiring.** Change the network (illness, drugs) and you get hallucinations, ringing ears, and so on.
4. **The smallest unit of mind is one neuron firing.** So mental life follows true/false logic.
5. **Purpose comes from feedback.** A system that tries to **reduce the difference** between what it senses and a goal *looks* purposeful. Examples: a thermostat, hunger, attention. (In ML: reducing a **loss**.)
6. **Many networks can produce the same behavior.** So you can't tell someone's exact brain wiring just by watching what they do.

---

## 11. What's missing (and came later)

| Missing in 1943 | Added by |
|---|---|
| Weights that **learn from data** | Perceptron (Rosenblatt, 1958) |
| Training algorithm | Backpropagation (1986) |
| Decimal weights, negative weights | Perceptron |
| Smooth outputs (not just 0/1) | Sigmoid, ReLU (later) |

---

## 12. Check yourself

1. Why can't a network build plain NOT A?
2. What's the only difference between the OR neuron and the AND neuron?
3. In the heat/cold network, why is the delay neuron **a** needed?
4. How does a loop give a network memory?
5. Can one neuron do XOR? Can a network? Why?

---

## 13. What to build in code

1. A neuron + network simulator: 0/1 outputs, ticks, whole-number connections, veto blocking
2. The 4 building blocks (1a–d), each tested with a truth table
3. The heat/cold network (1e): test brief cold, long cold, and 2-tick cold
4. The learning loop (1i) and the memory latch
5. XOR as a small network
