# The code, explained simply

This explains how the code in this folder implements the McCulloch & Pitts (1943) paper.
Read [EXPLAINED.md](EXPLAINED.md) first for the paper itself.

---

## 1. The files at a glance

| File | What it is | Paper part |
|---|---|---|
| `mp_neuron.py` | The **neuron** and the **network simulator** | The 5 rules (page 4) |
| `circuits.py` | Ready-made **networks from Figure 1** + memory loops | Figure 1, Theorem X |
| `tpe.py` | A **compiler**: logic formula → network | Theorem II |
| `test_mp.py` | **21 tests**, each checks one claim of the paper | Everything |
| `demo.py` | Prints networks running **tick by tick** | Heat/cold, loops, XOR |

**How the files depend on each other:**
```
mp_neuron.py   ← the core, depends on nothing
   ↑      ↑
circuits.py  tpe.py
   ↑      ↑
 demo.py, test_mp.py
```

**Run it** (from the `01-McCulloch-Pitts-1943` folder):
```
python3 demo.py           # see the networks run
python3 -m pytest -q      # run all tests
```

---

## 2. `mp_neuron.py`: the heart of the code

### 2.1 The `Neuron`

A neuron stores just **three things**:

```python
@dataclass
class Neuron:
    threshold: int = 2                 # how many active connections it needs to fire
    excite: dict[str, int] = ...       # who excites it, and how many connections each
    inhibit: set[str] = ...            # who can block (veto) it
```

**Example:** `Neuron(threshold=2, excite={"A": 1, "B": 1}, inhibit={"C"})`
- A connects **once**, B connects **once**
- C can **block** it
- So it fires when **A and B** are on, and **C is off** → that's "A AND B AND NOT C"

**Why `excite` is a dictionary of counts:** in the paper, one neuron can connect to another **more than once**. Two connections count as 2. That count is the only "weight" in 1943: always a whole number, never negative.

### 2.2 How a neuron decides to fire

```python
def fires(self, prev):
    if any(prev[src] for src in self.inhibit):   # Step 1: any blocker on? → 0
        return 0
    total = sum(n for src, n in self.excite.items() if prev[src])   # Step 2: count
    return int(total >= self.threshold)          # Step 3: enough? → 1, else 0
```

`prev` = the state of **every** neuron **one tick ago** (e.g. `{"A": 1, "B": 0, "C": 0}`).

| Step | Code | Paper rule |
|---|---|---|
| 1 | Check blockers first | Rule 4: blocking is a total veto |
| 2 | Add up connections from neurons that fired | Rule 2: threshold |
| 3 | Return only 0 or 1 | Rule 1: all-or-none |
| (all) | Look only at `prev` (last tick) | Rule 3: one tick delay |

**Worked example:** `excite={"A": 1, "B": 1}`, `inhibit={"C"}`, threshold 2

| prev | Blocked? | Total | Fires? |
|---|---|---|---|
| A=1, B=1, C=0 | no | 1+1 = 2 | **1** |
| A=1, B=0, C=0 | no | 1 | 0 |
| A=1, B=1, C=1 | **yes** | (not counted) | 0 |

### 2.3 The `Network`

A network holds two things:
- `inputs`: a list of input neuron names. **The outside world sets these.** (The paper calls them "peripheral afferents.")
- `neurons`: a dictionary `{name: Neuron}` for all the other neurons.

You build a network like this:
```python
net = Network(["1", "2"])                       # two inputs
net.add("3", excite={"1": 1, "2": 1})           # neuron 3 = 1 AND 2
```

`add()` refuses **duplicate names** (they would overwrite each other). `_check()` catches **typos** (connecting from a neuron that doesn't exist) before running.

### 2.4 How the simulation runs: `run()`

```python
h = net.run({"1": [1, 0, 1], "2": [1, 1, 0]}, steps=4)
```

**Inputs:** for each input neuron, a list of 0/1 values, one per tick. Past the end of the list = 0.
**Output:** the **history**: `{neuron name: [value at tick 0, tick 1, tick 2, ...]}`.

**What happens inside:**
```
tick 0:  inputs take their first value; every other neuron starts at 0
         (or whatever `initial` says)

each next tick:
   1. inputs  → read next value from the lists
   2. others  → each neuron calls fires(OLD state)
   3. OLD state = NEW state
   4. save everything to history
```

**The most important detail:** all neurons look at the **old** state and update **at the same time**. The code builds a brand-new dictionary `new` instead of changing `state` as it goes.

Why? If neuron 3 updated first and neuron 4 then read neuron 3's **new** value, a signal would travel through two neurons in one tick. That breaks rule 3 (one tick per connection).

**What is `initial` for?** Networks with loops sometimes need to **start** with a neuron already on. Example: the clock needs one spike to start it going round. `run(..., initial={"c0": 1})` does that.

### 2.5 `show()`: printing

Turns a history into a table. `#` = fires, `.` = silent:
```
  |  0  1  2  3
---------------
2 |  #  .  .  .
3 |  .  .  .  #
```

---

## 3. `circuits.py`: the paper's networks

Every function builds one `Network` and returns it. All neurons use threshold 2, so:

| Wiring | Meaning |
|---|---|
| `excite={"x": 2}` | x **alone** can fire it |
| `excite={"x": 1}` | x needs **one partner** |
| `inhibit={"x"}` | x **blocks** it |

### 3.1 The four building blocks (Figure 1a–d)

| Function | Wiring | Does |
|---|---|---|
| `delay()` | `"2": excite {"1": 2}` | copies 1, one tick later |
| `or_gate()` | `"3": excite {"1": 2, "2": 2}` | 1 OR 2 |
| `and_gate()` | `"3": excite {"1": 1, "2": 1}` | 1 AND 2 |
| `and_not_gate()` | `"3": excite {"1": 2}, inhibit {"2"}` | 1 AND NOT 2 |

**Look at OR vs AND:** same neuron, same threshold. The only change is **2 connections vs 1**. That's the whole idea of weights.

### 3.2 `heat_cold()`: Figure 1e

```python
net = Network(["1", "2"])                        # 1 = heat sensor, 2 = cold sensor
net.add("a", excite={"2": 2})                    # a = cold, one tick ago
net.add("4", excite={"a": 1, "2": 1})            # 4 = cold twice in a row → FEEL COLD
net.add("b", excite={"a": 2}, inhibit={"2"})     # b = cold was on, then stopped
net.add("3", excite={"1": 2, "b": 2})            # 3 = heat OR brief cold → FEEL HEAT
```

**Trace it: brief cold** (cold only at tick 0). Each tick reads the row above it:

| tick | 2 (cold) | a | b | 4 (cold) | 3 (heat) | Why |
|---|---|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 | 0 | cold arrives |
| 1 | 0 | 1 | 0 | 0 | 0 | a copies cold from tick 0 |
| 2 | 0 | 0 | 1 | 0 | 0 | a was on, cold was off → b fires |
| 3 | 0 | 0 | 0 | 0 | **1** | b was on → **feel heat** |

Neuron 4 never fires: it needs `a` and cold on **together**, and they never are.

### 3.3 `two_in_a_row()`: Figure 1h

```python
net.add("d", excite={"1": 2})             # d = input, one tick ago
net.add("2", excite={"1": 1, "d": 1})     # needs "input now" AND "input a tick ago"
```
Two spikes at **different times** are turned into two signals arriving **at the same time**.

### 3.4 `learned_association()`: Figure 1i (learning)

```python
net.add("L", excite={"1": 1, "2": 1, "L": 2})    # memory
net.add("3", excite={"2": 2, "1": 1, "L": 1})    # output
```

- **L** turns on when 1 and 2 fire **together**. Then L feeds **itself** 2 connections, so it **stays on forever**. This is the loop.
- **3** fires from 2 alone. It fires from 1 **only if L is on**.
- So: **before** 1 and 2 fire together, 1 does nothing. **After**, 1 alone works. The network "learned."

### 3.5 Memory loops (Theorem X)

| Function | Key line | Behavior |
|---|---|---|
| `latch()` | `M: excite {"P": 2, "M": 2}` | P turns M on, M keeps itself on **forever** |
| `latch(with_reset=True)` | adds `inhibit {"R"}` | R switches M **off** (set/reset memory) |
| `always_so_far()` | `A: excite {"P": 1, "A": 1}` | needs P **and** itself. Once P fails, off forever. Start with `initial={"A": 1}` |
| `clock(n)` | each `c{i}` copies `c{i-1}`, last feeds first | one spike runs round the ring. Start with `initial={"c0": 1}` |

`(i - 1) % n` is what closes the ring: for `i = 0`, it gives `n - 1`, so the first neuron listens to the last one.

---

## 4. `tpe.py`: the formula-to-network compiler

This is **Theorem II** as code: *any formula made of delay, AND, OR, AND-NOT can be built as a network.*

### 4.1 Writing a formula

Five small classes, one per rule of the paper's grammar:

| Class | Means |
|---|---|
| `Var("p")` | input p |
| `Delay(x)` | x, one tick earlier (the paper's `S`) |
| `Or(a, b)` | a OR b |
| `And(a, b)` | a AND b |
| `AndNot(a, b)` | a AND NOT b |

A formula is a **tree** of these:
```python
xor = Or(AndNot(Var("p"), Var("q")), AndNot(Var("q"), Var("p")))
#     (p AND NOT q)  OR  (q AND NOT p)
```

There's **no plain `Not`** class on purpose. The paper proves a network can't build it (Theorem III).

### 4.2 `evaluate()`: the answer key

Computes a formula **directly**, without any network:
- `Var` → read the input at tick t
- `Delay` → evaluate the inside at **t − 1**
- `Or` / `And` / `AndNot` → combine with `|`, `&`, `1 - x`

The tests use it as the **correct answer** to check the networks against.

### 4.3 `compile_tpe()`: building the network

Returns three things:
```python
net, out, lag = compile_tpe(formula)
```
- `net`: the network
- `out`: the name of the neuron that gives the answer
- `lag`: **how many ticks late** the answer comes

**Rule:** `net output at tick t  ==  formula at tick (t − lag)`.

The paper allows this lateness ("realizable in the extended sense"). Every neuron takes one tick, so a deeper formula answers later.

**How `build()` works: one case per piece, called recursively:**

| Piece | What it builds | Lag |
|---|---|---|
| `Var(p)` | nothing: the input neuron already exists | 0 |
| `Delay(x)` | one delay neuron after x | **same** as x (see below) |
| `Or/And/AndNot(a, b)` | build a, build b, **line them up**, add one gate | max + 1 |

**Why `Delay` doesn't add lag:** every neuron naturally takes one tick. For `Delay`, that tick **is** the delay we asked for, so the answer isn't late. For a gate, the tick is **extra**, so the answer is one tick later.

**Why "line them up"?** A gate must see both signals **at the same tick**. If one side is faster, `pad()` adds delay neurons to it until both sides match.

**Example 1: XOR.** The compiler builds:
```
n0 = p AND NOT q      (excite p×2, inhibit q)
n1 = q AND NOT p      (excite q×2, inhibit p)
n2 = n0 OR n1         (excite n0×2, n1×2)      ← output, lag = 2
```
Two layers, which is why XOR needs more than one neuron.

**Example 2: padding.** `And(Var("p"), Or(Var("q"), Var("r")))`
- `p` is ready at lag 0, `q OR r` is ready at lag 1 → **p is too early**
- `pad()` adds one delay neuron after p
- Then the AND gate → final lag 2
```
n0 = q OR r           (lag 1)
n1 = delay of p       (padding, now lag 1)
n2 = n1 AND n0        (lag 2)  ← output
```

**Small detail:** in the `And` case, connections are **added up** (`ends[nb] = ends.get(nb, 0) + 1`). For `And(p, p)`, p must get **2** connections. A plain dictionary would merge them into 1 and the gate would never fire. A test guards this.

---

## 5. `test_mp.py`: what each test proves

| Test | Proves |
|---|---|
| `test_delay`, `test_gates` | Figure 1a–d give the right truth tables |
| `test_inhibition_is_a_veto...` | 100 connections still lose to 1 blocker (rule 4) |
| `test_brief_cold_feels_hot` | Brief cold → heat at tick 3 |
| `test_long_cold_feels_only_cold` | Long cold → only cold |
| `test_real_heat_is_felt_after_one_tick` | Real heat → heat at tick 1 |
| `test_two_tick_cold_gives_cold_then_heat` | 2-tick cold → cold, then heat (not in the paper) |
| `test_heat_cold_matches_the_papers_formulas...` | Network = paper's formulas for **all 4,096** input patterns |
| `test_two_in_a_row` | Figure 1h |
| `test_learned_association` | Figure 1i: learning works through a loop |
| `test_latch...`, `test_always_so_far...`, `test_clock...` | Theorem X loops |
| `test_compiled_net_equals_formula` | Theorem II on **300 random formulas** |
| `test_compiler_handles_repeated_input` | The `And(p, p)` bug stays fixed |
| `test_silence_in_means_silence_out` | Theorem III: no input → no output |
| `test_xor_needs_two_layers` | XOR works with 2 layers |
| `test_no_single_neuron_computes_xor` | Tries **every** small neuron: none does XOR |

**Two testing tricks used:**
- **Brute force:** try every possible input (`itertools.product`). If all pass, the claim is proven for that size.
- **Random with a fixed seed:** `random.Random(1943)` gives the same "random" formulas every run, so a failure can always be repeated.

---

## 6. Paper → code map

| In the paper | In the code |
|---|---|
| Neuron fires or not | `Neuron.fires()` returns 0 or 1 |
| Threshold | `Neuron.threshold` |
| Number of connections | values in `Neuron.excite` |
| Absolute inhibition | `Neuron.inhibit` + the check that runs first |
| Synaptic delay | `run()` updates from the **old** state |
| Peripheral afferents | `Network.inputs` |
| `N_i(t)` | `history[name][t]` |
| `S` (one tick earlier) | `Delay` / a neuron with `excite={x: 2}` |
| Realizable in extended sense | `lag` returned by `compile_tpe` |
| Figure 1a–e, h, i | functions in `circuits.py` |
| Theorem II proof | `compile_tpe()` |
| Theorem III | `test_silence_in_means_silence_out` |
| Theorem X loops | `latch`, `always_so_far`, `clock` |

---

## 7. Try it yourself

Small exercises to check you understand the code:

1. **Build a 3-input AND** with one neuron. (Hint: 1 connection each, threshold 3.)
2. **Build "at least 2 of 3"** (majority) with one neuron.
3. **Change the brief cold** in `demo.py` to 3 ticks. Predict the table first, then run it.
4. **Compile** `And(Var("p"), Delay(Delay(Var("p"))))`. What does it detect? Check the `lag`.
5. **Add Figure 1f** (relative inhibition): add a `relative` option to `Neuron` where each blocker **raises the threshold by 1** instead of vetoing. Then write a test.
