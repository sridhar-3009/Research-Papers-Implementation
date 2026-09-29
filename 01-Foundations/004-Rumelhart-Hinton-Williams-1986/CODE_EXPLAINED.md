# The code, explained simply

How the code in this folder implements Rumelhart, Hinton & Williams (1986).
Read [EXPLAINED.md](EXPLAINED.md) first for the paper itself.

---

## 1. The files at a glance

| File | What it is | Paper part |
|---|---|---|
| `backprop.py` | The layered network, the **backward pass**, training with momentum, and the recurrent net | Eqs. (1)–(9), Figure 5 |
| `tasks.py` | The data: mirror symmetry, the two family trees, the family-tree network shape | Figures 1–3 |
| `experiments.py` | Reproduces every experiment; writes `results.md` and `figures/` | Figures 1, 4, 5 |
| `demo.py` | A guided tour, including one backward pass worked by hand | everything |
| `test_backprop.py` | **14 tests** | everything |

**Run it** (from `01-Foundations/004-Rumelhart-Hinton-Williams-1986`):
```
python3 demo.py              # the tour (~1 s)
python3 experiments.py       # every result + figures (~1 min)
python3 -m pytest -q         # 14 tests (~1 s)
```
Needs `numpy`; the figures need `matplotlib`.

**Suggested reading order:**
1. `backprop.py`: `forward`, then `gradients`
2. `demo.py` section 1 (one backward pass by hand)
3. `tasks.py`
4. `experiments.py`

---

## 2. `backprop.py`

### 2.1 Describing a network
```python
LayeredNet([("in", 6, []), ("hidden", 2, ["in"]), ("out", 1, ["hidden"])])
```
Each layer is **(name, number of units, the layers it receives from)**. A layer with no sources is an input layer, and the last layer is the output.

Because each layer lists its sources, one class handles all the paper's shapes:
- the **family-tree** net (two input groups, each with its own hidden group);
- **skip connections** ("connections can skip intermediate layers", page 533).

Storage:
- `W[(src, dst)]`: the weights from layer `src` to layer `dst`, shape (units in src, units in dst).
- `b[dst]`: the biases, which are weights from an always-on unit.

### 2.2 `forward`: Eqs. (1)–(2)
```python
x = self.b[name] + sum(y[s] @ self.W[(s, name)] for s in src)    # Eq. (1)
y[name] = logistic(x)                                            # Eq. (2)
```
It goes through the layers **in order**, so each layer's sources are already computed.
`logistic` is written as `0.5·(1 + tanh(x/2))`. That's the same function, but it can't overflow for large x.

### 2.3 `gradients`: the backward pass, Eqs. (4)–(7)
```python
dE_dy[out] = y_out - d                              # Eq. (4)
for each layer, from the TOP down:
    dE_dx = dE_dy[name] * y * (1 - y)               # Eq. (5)
    grad b = dE_dx summed over cases                # bias: input is always 1
    for each source layer s:
        grad W[(s, name)] = y[s].T @ dE_dx          # Eq. (6)
        dE_dy[s] += dE_dx @ W[(s, name)].T          # Eq. (7): error sent DOWN
```
The only trick is the **order**. By the time a layer is processed, every layer above it has already added its share to that layer's `dE_dy` (Eq. 7 sums over all the units it feeds). Going through the spec in reverse guarantees this.

`@` is matrix multiplication. It does Eqs. (6)–(7) for **all cases at once** and sums over them, which is exactly the paper's "accumulate ∂E/∂w over all input-output cases".

### 2.4 `output_error` and the margin rule
Eq. (4) is `y − d`. With `margin=(0.2, 0.8)` it becomes 0 when an "on" unit is already above 0.8 or an "off" unit is below 0.2 (the family-tree rule from Figure 4's caption).

### 2.5 `train`: Eqs. (8)–(9)
```python
velocity = -eps * gradient + alpha * velocity       # Eq. (9)
weight  += velocity
weight  *= (1 - decay)                              # optional weight decay
```
- One loop iteration = **one sweep** through all cases, then one weight change.
- `schedule` lets ε and α change over time (the family trees use one setting for the first 20 sweeps and another after that).
- `stop` ends training early (e.g. once all 64 symmetry patterns are right).

### 2.6 `IterativeNet`: Figure 5
- One weight matrix `W` connects every unit to every unit, from step t to step t+1.
- `run` applies it 3 times, which is the same as a 3-layer net with **tied weights**.
- `gradients` is **backprop through time**: go back through the steps and **add up** each weight's gradient from every copy. (The paper averages; averaging only changes the effective step size.)

---

## 3. `tasks.py`

### `symmetry_data`
All 2⁶ = 64 six-bit vectors, with target 1 for the 8 that read the same backwards.

### The family trees
- `ENGLISH` and `ITALIAN` list the 12 people **in matching order**: Italian person i is English person i's twin.
- The English tree is described once, as couples, children and who is male. `_english_facts` **derives** every relationship from that:
  - spouses → husband/wife
  - parents ↔ son/daughter
  - shared parents → brother/sister
  - a parent's sibling, **and that sibling's spouse** → uncle/aunt, reversed into nephew/niece
- `family_facts` makes the Italian tree by renaming.
- `family_cases` groups the facts by (person 1, relation): **104 cases**, matching the paper's "104 possible triples". The target turns on **every** correct person 2.

Tests check that Colin's aunts are exactly Jennifer and Margaret (as in Figure 3), and that the Italian facts are an exact renamed copy of the English ones.

---

## 4. `experiments.py`

| Function | What it does |
|---|---|
| `symmetry_runs` | trains the symmetry net from 20 random starts and records the sweeps needed |
| `weight_structure` | measures Figure 1's claims: mirror error, the correlation between the twin hidden units, the 1:2:4 ratio |
| `no_hidden_layer_symmetry` | shows symmetry can't be learned without hidden units |
| `family_run` | trains on 100 facts, tests on 4 held-out ones |
| `family_settings` | the paper's exact recipe vs what had to change (the vanishing-gradient finding) |
| `feature_analysis` | Figure 4's claims: which code units track nationality, generation, branch; how close the twins are |
| `bptt_check` | Figure 5: BPTT vs finite differences, plus a small memory task |
| `save_figures` | `fig1_symmetry_weights.png`, `fig4_person_codes.png` |

---

## 5. `test_backprop.py`: what the 14 tests prove

| Test | Proves |
|---|---|
| `test_backprop_matches_finite_differences` (×2) | Eqs. (1)–(7) give the **true** gradient, with skip connections, two input groups, and the margin rule |
| `test_error_is_half_sum_of_squares` | Eq. (3) |
| `test_margin_rule_ignores_close_enough_outputs` | Figure 4's 0.2/0.8 rule |
| `test_momentum_and_weight_decay` | decay keeps weights smaller |
| `test_symmetry_data` | 64 patterns, 8 symmetric |
| `test_learns_symmetry_with_the_papers_solution` | Figure 1: solved in < 2000 sweeps, mirror weights, sign-flipped twins, 1 : 2 : 4, bias signs |
| `test_no_hidden_layer_cannot_do_symmetry` | a hidden layer is needed |
| `test_smaller_steps_avoid_the_plateau` | ε = 0.03 lets bigger nets solve it |
| `test_family_trees_have_104_cases`, `test_trees_are_isomorphic` | the Figure 2 data is right |
| `test_family_net_learns_training_cases` | > 95% of training facts |
| `test_papers_exact_family_recipe_stalls` | the reproduction finding (vanishing gradients) |
| `test_backprop_through_time` | Figure 5 |

**Finite differences** (used in the first test): nudge one weight up by 0.000001, then down, measure E each time, and divide the change by the nudge. That's the slope measured **directly**. Backprop must agree, and it does to about 10⁻¹⁰.

---

## 6. Paper → code map

| In the paper | In the code |
|---|---|
| Eq. (1) x_j = Σ y_i w_ji | `forward`: `y[s] @ W[(s, name)]` |
| Eq. (2) logistic | `logistic` |
| bias = weight from an always-on unit | `b[name]` |
| Eq. (3) E | `error` |
| Eq. (4) ∂E/∂y = y − d | `output_error` |
| Eq. (5) · y(1 − y) | `gradients`: `dE_dx` |
| Eq. (6) ∂E/∂w = ∂E/∂x · y_i | `gradients`: `y[s].T @ dE_dx` |
| Eq. (7) error sent down | `gradients`: `dE_dx @ W.T` |
| Eq. (8)–(9) with momentum | `train` |
| weight decay 0.2% | `train(decay=0.002)` |
| 0.8 / 0.2 rule | `margin=(0.2, 0.8)` |
| layered net with skip connections | `LayeredNet(spec)` |
| Figure 1 net | `[("in",6,[]), ("hidden",2,["in"]), ("out",1,["hidden"])]` |
| Figure 3 net | `tasks.FAMILY_NET` |
| Figure 5 iterative net | `IterativeNet` |

---

## 7. Try it yourself

1. In `demo.py`, set `alpha=0` (no momentum) for symmetry. How many more sweeps does it take?
2. Change the symmetry net to update **after every case** instead of once per sweep. Is it faster?
3. Swap `logistic` for `tanh` (outputs −1 to 1; its slope is 1 − y²). Does the family-tree net learn with ±0.3 initial weights now?
4. Try `init_range=0.3` with ε = 0.05 on the family trees. Is initialization or step size the bigger problem?
5. Add a **skip connection** from `person_code` straight to `out`. Does generalization change?
