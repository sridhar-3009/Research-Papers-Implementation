# The code, explained simply

How the code in this folder implements the Introduction to *Perceptrons* (1969).
Read [EXPLAINED.md](EXPLAINED.md) first for the book itself.

---

## 1. The files at a glance

| File | What it is | Book part |
|---|---|---|
| `minsky.py` | Perceptrons, partial predicates, convexity, connectedness, both impossibility proofs | Sections 0.5–0.8 |
| `demo.py` | A guided tour, with pictures printed in the terminal | everything |
| `test_minsky.py` | **26 tests**: definitions, examples, theorems | everything |

**Run it** (from `01-Foundations/003-Minsky-Papert-1969`):
```
python3 demo.py          # the tour (~1 s)
python3 -m pytest -q     # 26 tests (<1 s)
```
Needs `numpy`. The parity extension also needs `scipy` (for linear programming).

**Figures** are NumPy boolean arrays: `X[row, col] = True` means that point is black.

---

## 2. Partial predicates and perceptrons (Sections 0.5, 0.8)

### 2.1 `Predicate`
```python
Predicate(support=((0, 0), (0, 3)), rule=lambda v: v[0] and not v[1])
```
- `support`: the points it looks at. **Nothing else can affect it.**
- `rule`: what it computes from those points' 0/1 values.
- `order` = number of support points; `diameter` = widest distance between them.

Helpers:
- `point_predicate(p)`: "is p black?"
- `mask_predicate(A)`: "are all of A's points black?"

### 2.2 `Perceptron`
```python
psi(X) = 1  if  sum(alpha * phi(X) for each phi) > theta
```
That's the definition from Section 0.8, word for word. `order` and `diameter` are the maximum over its φ's.

**Compare with the other papers:**

| Paper | The φ's | What's learned |
|---|---|---|
| 1. McCulloch & Pitts | the inputs themselves | nothing (hand-designed) |
| 2. Rosenblatt | **random** A-units | the values V |
| **3. Minsky & Papert** | **any** local tests you like | the weights α; the question is **what's possible at all** |
| 4–5. Rumelhart; Du et al. | a hidden layer | the φ's **and** the weights, by backprop |

### 2.3 `conjunctive`: "all must say yes" as a vote
It flips every test (so it fires on **failure**), gives each weight −1, and uses θ = −1. The sum is > −1 only if **no** failure test fires. This is the book's Example in Section 0.8.

---

## 3. The seesaw: `seesaw_perceptron` (Figure 0.3)
7 point predicates, weights `i − 4`, threshold 0. `tips_right` does the physics (total torque > 0) separately, and a test checks they agree on all 2⁷ = 128 arrangements.

---

## 4. Convexity, order 3 (Section 0.6)

### `points_between(p, r)`
The grid points **strictly inside** the segment from p to r. It uses the **gcd** of the steps: from (0,0) to (4,2), gcd = 2, so there's one point in between, (2,1).

### `convexity_perceptron(shape)`
For every pair (p, r) and every q between them, it makes one φ that looks at **exactly 3 points** and fires when **p black, q white, r black** (a dent). The perceptron says "convex" if none fires.

### `is_line_convex(X)`
Checks the same definition directly, for comparison.

*(On a grid, "convex" means: every grid point between two black points is black. The book works in the continuous plane.)*

---

## 5. Connectedness

### `is_connected(X)`
A **breadth-first search**: start at one black point, visit every black neighbour, and see whether that reaches **all** black points. `diagonal=True` counts diagonal neighbours too. The theorems hold either way, and a test checks both.

### Theorem 0.6.1: `ladder` and `fooling_figure`
- `ladder(n)`: two rails, 5 rows high, with room for `n` rungs. `ladder(n, {j})` adds rung j.
- `fooling_figure(support, n)`: the **proof as code**.
  1. Work out which rungs the test's points touch.
  2. Pick a rung it **doesn't** touch.
  3. Return the ladder with only that rung.

  The test sees exactly what it saw on Y0 (no rungs) and says "no", but the figure is connected.

The test tries **100 random tests** for each k = 1, 2, 5, 10, 20 and always finds the fooling figure.

### Theorem 0.8: `four_figures`, `locality_identity_holds`, `window_features`

**`four_figures(length)`** builds X00, X01, X10, X11:
```
4 horizontal strands (rows 0, 2, 4, 6), closed at each end in one of two ways:
  type 0:  strands 1-2 joined, 3-4 joined     (two U-turns)
  type 1:  strands 1-4 joined, 2-3 joined     (nested U-turns)
  00 → two loops (not connected)   01, 10 → one big loop   11 → two loops
```

**`locality_identity_holds(figs, d)`:** the key step of the proof.
- Slide a d×d window over **every** position.
- In each one, check that the window **can't see both ends**: the left end looks the same in X0b and X1b, **or** the right end looks the same in Xa0 and Xa1.
- If that's true everywhere, then **every** predicate inside any window satisfies
  `φ(X11) − φ(X10) − φ(X01) + φ(X00) = 0`, whatever its rule, and so does any weighted sum.

This checks the theorem for **all** diameter-limited predicates at once, not just a sample.

**`window_features(figs, d)`:** every local feature that could possibly help. For each window and each pattern seen in it, one feature = "does X look exactly like this in the window?" Any diameter-limited predicate is a combination of these.

**`train_perceptron(F, y)`:** Rosenblatt's error-correction rule (the same rule as Paper 5's `perceptron_train`). Given **all** the local features, it **never** gets all 4 figures right, which is the theorem seen in action.

---

## 6. Extension: the order of parity (Chapter 3 of the book)

`representable(n, k, target)` asks: can a perceptron whose φ's are masks of at most k of the n points compute `target`?
- The unknowns are the weights α_A.
- The requirements are "sum ≥ 1 when target = 1" and "sum ≤ −1 when target = 0".
- That's a **linear program**, and `scipy.optimize.linprog` finds whether any solution exists.
- Using only masks loses nothing, because any predicate on k points is a weighted sum of masks.

`order_of(n, target)` is the smallest k that works. Results: **parity has order n**, and **AND has order 1**.

---

## 7. `test_minsky.py`: what the 26 tests prove

| Tests | Prove |
|---|---|
| `test_perceptron_is_a_weighted_vote`, `test_order_and_diameter` | the definitions |
| `test_seesaw_matches_physics...` | Figure 0.3, all 128 arrangements |
| `test_conjunctive_predicate_as_a_perceptron` | the Example of Section 0.8 |
| `test_convexity_perceptron_has_order_3...`, `test_convexity_examples` | Section 0.6 |
| `test_any_k_point_test_is_fooled` (×5) | Theorem 0.6.1 |
| `test_the_four_figures` | X00 and X11 disconnected, X01 and X10 connected |
| `test_locality_identity_for_every_local_window` (×5) | Theorem 0.8's key step |
| `test_identity_fails_once_a_window_sees_both_ends` | the limit really is locality |
| `test_identity_holds_for_random_weighted_sums...` | the identity carries over to any weights |
| `test_perceptron_learning_fails_with_local_features` (×4) | learning can't beat the theorem |
| `test_parity_has_order_n...` (×3) | Chapter 3's parity result |

---

## 8. Paper → code map

| In the book | In the code |
|---|---|
| figure X on the retina R | boolean array `X` |
| partial predicate φ, its support | `Predicate`, `.support` |
| φ_p, φ_A ("A ⊆ X") | `point_predicate`, `mask_predicate` |
| ψ linear in Φ, weights α, threshold θ | `Perceptron` |
| order, diameter | `.order`, `.diameter` |
| conjunctively local | `conjunctive` |
| seesaw (Figure 0.3) | `seesaw_perceptron`, `tips_right` |
| ψ_CONVEX of order 3 | `convexity_perceptron` |
| ψ_CONNECTED | `is_connected` |
| Theorem 0.6.1 figures Y0, Y1, Y2 | `ladder`, `fooling_figure` |
| Theorem 0.8 figures X00…X11 | `four_figures` |
| "group 1 / group 2 / group 3" argument | `locality_identity_holds` |
| learning "by feedback devices" (0.9) | `train_perceptron` |

---

## 9. Try it yourself

1. Change `four_figures(30)` to `four_figures(8)`. At what window size does learning start to succeed?
2. Write `is_circle` and try to build a low-order perceptron for it. How many points must each φ see?
3. Use `representable` to find the order of **majority** ("more than half the points are black") on 4 points.
4. Make the seesaw's pivot movable: which weights and threshold give "tips right" for a pivot at position 3?
5. Build a **Gamba perceptron** (Section 0.8): each φ is itself a perceptron over the whole retina. Can a Gamba perceptron learn the four figures? (This is a 2-layer network, the idea Paper 4 builds on.)
