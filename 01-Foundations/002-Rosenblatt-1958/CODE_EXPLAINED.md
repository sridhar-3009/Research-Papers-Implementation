# The code, explained simply

How the code in this folder implements Rosenblatt (1958).
Read [EXPLAINED.md](EXPLAINED.md) first for the paper itself.

---

## 1. The files at a glance

| File | What it is | Paper part |
|---|---|---|
| `theory.py` | The **probability formulas**: Pa, Pc, Pc_min, the learning-curve law | Eqs. (1)–(4), pages 392–396 |
| `perceptron.py` | The **machine**: retina → random A-units → competing responses, with α / γ / bivalent learning | pages 389–392, 402–403 |
| `experiments.py` | Reproduces **Figures 4, 5, 6, 7, 10, 11** and conclusions 1–10 | pages 392–405 |
| `demo.py` | A short guided tour | everything |
| `test_rosenblatt.py` | **13 tests**, each checks an equation or conclusion | everything |
| `results.md`, `figures/` | Output of `experiments.py` | |

**How they depend on each other:**
```
theory.py        perceptron.py
      ↑               ↑
        experiments.py
              ↑
    demo.py, test_rosenblatt.py
```

**Run it** (from `01-Foundations/002-Rosenblatt-1958`):
```
python3 demo.py              # the tour (~3 s)
python3 experiments.py       # all figures + results.md (~15 s)
python3 -m pytest -q         # 13 tests (~1 s)
```
Needs `numpy`; the figures need `matplotlib`.

**Suggested reading order:**
1. `perceptron.py`
2. `theory.py` (`Pa` first)
3. `experiments.py`

---

## 2. `perceptron.py`: the machine

### 2.1 Building it
```python
p = Photoperceptron(n_points=400, n_assoc=2000, n_resp=2, x=5, y=5, theta=3)
```
| Stored | Meaning |
|---|---|
| `exc` (2000 × 5) | For each A-unit, the 5 **random** retina points it's excited by |
| `inh` (2000 × 5) | …and the 5 it's inhibited by |
| `source` (2000) | Which response each A-unit feeds (its **source-set**) |
| `V` (2000) | Each A-unit's **value**. The **only** thing learning changes. Starts at 0 for everyone |
| `alive` (2000) | Set to False to "remove" A-units (the distributed-memory test) |

**Compare with Paper 5's MLP:** there, learning changes the **weights on the wires**. Here the wiring is **random and frozen**; each A-unit has **one value** that's shared across all its outputs.

### 2.2 `activate`: which A-units fire (the predominant phase)
```python
net = S[:, self.exc].sum(-1) - S[:, self.inh].sum(-1)   # lit excitatory − lit inhibitory
return (net >= self.theta) & self.alive
```
`S[:, self.exc]` looks up, for every stimulus, whether each A-unit's excitatory points are lit. That's the whole rule from page 389, done for all stimuli and all units at once.

### 2.3 `strengths`: which response wins (the postdominant phase)
For each response r: take the A-units that are **active** and **in r's source-set**, then
- `mode="sigma"`: **add up** their values;
- `mode="mu"`: take their **average**.

The biggest wins. `respond` adds a tiny random number so that ties are broken fairly.

### 2.4 `reinforce`: how learning changes V
```python
active = a & src                               # active units in the reinforced source-set
alpha:  V[active] += 1                         # everyone active gains, forever
gamma:  V[active] += 1
        V[src & ~a] -= n_on / (n_all - n_on)   # the inactive ones pay, so the total stays fixed
```
**Why γ's subtraction matters:** in α, a response trained 300 times piles up far more value than one trained 100 times, and starts winning everything. In γ, every source-set keeps the **same total**, so no response can take over. A test checks that this total stays at exactly 0.

### 2.5 `train_forced` and `train_bivalent`
- **Forced** (page 395): the experimenter **makes** the correct response happen, and it's reinforced. Used for most experiments.
- **Bivalent** (page 402): the machine **answers first**. Then the correct response's active units gain, and the other responses' active units **lose**. That's reward plus punishment, and it's trial-and-error learning.

### 2.6 `p_correct`: P_r and P_g
Checks whether the correct response's strength **beats every other** response, and counts a tie as half.
- Call it with the **training** stimuli → it's **P_r**.
- Call it with **new** ones → it's **P_g**.

---

## 3. `theory.py`: the probability formulas

### 3.1 `Pa`: Eq. (1)
```python
sum(binom(x, e, R) * binom(y, i, R)  for all e, i  if e - i >= theta)
```
- `binom(n, k, p)` = the chance of exactly k successes in n tries. Each connection lands on a lit point with chance R.
- Loop over every possible number of lit excitatory points `e` and inhibitory points `i`, and add up the chances of the cases where the unit fires.

### 3.2 `Pc`: Eq. (2)
It's the same idea with 4 more loops. Starting from a unit that fired for S1 (`e`, `i` lit), it counts how many of its lit points go dark in S2 (`le`, `li`, each with chance **L**) and how many dark points light up (`ge`, `gi`, each with chance **G**). Then:
```
Pc = P(fires for S1 AND S2) / P(fires for S1)
```

### 3.3 `overlap_to_LG`
The paper's Figure 6 uses "overlap C". This converts it to L and G for two stimuli of the same size: `L = 1 − C`, `G = R(1 − C)/(1 − R)`.

### 3.4 How we know the formulas are right
`test_Pa_and_Pc_match_simulation` builds a **real** random retina of 20,000 points and 20,000 random A-units, lights up random stimuli, and **counts** how many units fire. The formula must match that count to within 0.01–0.02. It does, for three different settings.

---

## 4. `experiments.py`: reproducing the paper

| Function | Reproduces | What it shows |
|---|---|---|
| `fig4_data`, `fig5_data`, `fig6_data` | Figures 4–6 | Pa and Pc curves, from the formulas |
| `fig7_ideal` | Figure 7, conclusions 1–3 | random stimuli: recall falls, P_g = 0.5; μ vs Σ |
| `fig10_variable` | Figure 10 | γ beats α when responses get unequal training |
| `fig11_differentiated` | Figure 11, conclusions 4–5 | real classes: P_r and P_g meet; more A-units → higher |
| `generalization_condition` | page 400 | Pc12 < Pa < Pc11 measured on 4 environments |
| `distributed_memory` | conclusion 9 | remove A-units → gradual, general loss |
| `bivalent` | conclusion 7 | trial-and-error mistakes go down |

**The environments:**
- `ideal_environment`: random dots. No classes.
- `prototype_classes`: each class is a random "prototype" picture, and members are copies with 30% of points flipped. Similar within a class, different between classes.
- `shape_classes`: squares vs circles, optionally shifted by a few pixels.

**`pick_theta`:** the paper says Pa must be near a good value for the stimulus size. Small shapes light only ~12% of the retina, so they need a lower θ than half-lit random patterns. This picks the θ whose **theoretical** Pa is closest to 0.05.

---

## 5. `test_rosenblatt.py`: what the 13 tests prove

| Test | Proves |
|---|---|
| `test_Pa_and_Pc_match_simulation` (×3) | Eqs. (1)–(2) match a real simulation |
| `test_Pa_falls_with_threshold_and_inhibition` | Figure 4's trends |
| `test_Pc_is_one_for_identical...` | Figures 5–6: Pc = 1 for identical stimuli, > 0 for separate ones |
| `test_Pc_min_is_reached...` | Eq. (3) |
| `test_gamma_system_keeps_source_set_value_constant` | Table 1: γ conserves value |
| `test_ideal_environment_recall_falls...` | conclusions 1–3 |
| `test_mean_beats_sum_in_alpha_but_not_in_gamma` | pages 394–398 |
| `test_gamma_beats_alpha_with_unequal_training` | Figure 10 |
| `test_differentiated_environment_generalizes` | conclusions 4–5 |
| `test_memory_is_distributed` | conclusion 9 |
| `test_bivalent_trial_and_error_learning` | conclusion 7 |

---

## 6. Paper → code map

| In the paper | In the code |
|---|---|
| S-points, retina | the columns of a stimulus array `S` |
| A-unit origin points (x excitatory, y inhibitory) | `exc`, `inh` |
| A-unit threshold θ | `theta`; firing rule in `activate` |
| Source-set of a response | `source == r` |
| Value V of an A-unit | `V` |
| α, γ systems (Table 1) | `reinforce(system=...)` |
| Σ-system / μ-system | `strengths(mode="sigma"/"mu")` |
| Forced learning series | `train_forced` |
| Bivalent (reward/punishment) | `train_bivalent` |
| P_r, P_g | `p_correct` on training / new stimuli |
| Eq. (1) Pa, Eq. (2) Pc, Eq. (3) Pc_min | `theory.Pa`, `theory.Pc`, `theory.Pc_min` |
| Eq. (4) learning-curve law | `theory.learning_curve` |
| Removing part of the association system | `alive[...] = False` |

**Not implemented:** the β system (Table 1 is too damaged in the scan to read its rule reliably), the formulas for c1…c4 (Eqs. 5–11, also damaged), two-layer A-systems with a projection area, and time sequences.

---

## 7. Try it yourself

1. In `demo.py`, change `theta=3` to `1` and `5`. How do Pa and learning change?
2. Run `fig7_ideal` with `NA=500` instead of 2000. Does memory fill up sooner?
3. Make `shape_classes` work with shifts: give each A-unit **nearby** retina points instead of random ones (the paper's "projection area", page 389). Does P_g improve?
4. Add the **β system**: each reinforcement gives the source-set a fixed total of K, shared among its active units.
5. Try **spontaneous learning** (page 404): reinforce whatever response wins, with no teacher, and let values decay slowly. Do the two classes separate by themselves?
