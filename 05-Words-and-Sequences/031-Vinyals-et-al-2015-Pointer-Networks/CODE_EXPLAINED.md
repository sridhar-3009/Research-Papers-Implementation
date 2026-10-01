# The code, explained simply

How the code in this folder implements Pointer Networks (Vinyals, Fortunato & Jaitly 2015).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `ptrnet.py` | the Ptr-Net and the two baselines (seq2seq, + attention), teacher-forced log-likelihood, beam search with the valid-tour constraint |
| `tasks.py` | convex hull, Delaunay and TSP: the exact / approximate solvers, the target formats, and the paper's metrics |
| `experiments.py` | Table 1, the Delaunay results and Table 2 at the paper's scale (heavy, not run here) |
| `demo.py` | sorting with length generalization, convex hull at n = 5, TSP heuristics (about 20 seconds) |
| `test_ptrnet.py` | 8 quick tests (about 3 seconds) |

**Run it** (from `05-Words-and-Sequences/031-Vinyals-et-al-2015-Pointer-Networks`):
```
python3 -m pytest -q             # ~3 seconds
python3 demo.py                  # ~20 seconds
python3 experiments.py --quick   # ~1 hour
```

---

## 2. `ptrnet.py`

### `PtrNet(hidden, mode, n_max, in_dim=2, embed=True)`
- **`mode`:**
  - `"ptr"`: the paper's model;
  - `"seq2seq"`: Section 2.1, a softmax over a fixed n_max + 1 classes;
  - `"attention"`: Section 2.2, the same fixed softmax over [d ; d′], with input feeding.
- **`embed`:** a learned linear map from the coordinates to `hidden` dimensions (our addition; `embed=False` is the paper's literal setup).
- **Init:** every parameter is uniform in [−0.08, 0.08], as in Section 4.1.

### The pieces
- **`encode(P)`:** runs `nn.LSTMCell` over the points, collecting e_1 … e_n. It **prepends `e0`**, a learned vector that is the "stop" target, so pointer index 0 means ⇐ and indices 1 … n are the points.
- **`scores(E, W1E, d)`:** u = vᵀ tanh(W₁e_j + W₂d). W₁E is computed once per input set.
- **`start(P)`:** embeds, encodes, and sets the first decoder input to the learned `go` vector (⇒).
- **`step(S, mask)`:** one decoder step.
  - In `ptr` mode, the scores themselves go through `log_softmax` (n + 1 choices).
  - A `mask` sets forbidden choices to −∞ (probability 0).
- **`feed(S, idx)`:** the next decoder input is **the chosen point's (embedded) coordinates**. "Stop" feeds zeros.
- **`log_prob(P, C)`:** teacher forcing. At each step it adds log p(correct index) and then feeds the correct point. A −1 pad is ignored.

### `targets(seqs)`
Turns lists of 1-based indices into a tensor: each list gets END (0) appended and is padded with −1.

### `beam_search(model, P, beam, constraint)`
- Keeps `beam` partial outputs and expands each one with its top choices.
- Finished outputs (those that pointed at END) are set aside.
- It stops when the best finished score beats every live one.
- With `constraint="tour"`, visited cities and an early END are masked out, so every output is a valid tour (Section 4.4).

---

## 3. `tasks.py`

| Function | What it does |
|---|---|
| `convex_hull(P)` | Andrew's monotone chain using the cross product; lowest index first, counter-clockwise, closed |
| `polygon_area`, `is_simple_polygon`, `same_polygon` | the shoelace area, a self-crossing check, equality up to rotation |
| `hull_metrics` | Table 1's accuracy and area coverage ("FAIL" if > 1% of outputs aren't simple) |
| `delaunay(P)` | scipy's triangulation, as increasing triples sorted by incenter |
| `delaunay_metrics` | exact accuracy and triangle coverage |
| `held_karp(P)` | exact TSP, O(2ⁿn²); the inner minimum is vectorized over the previous city |
| `nearest_neighbour`, `greedy_edge`, `two_opt`, `christofides` | the classic heuristics |
| `A1`, `A2`, `A3` | our stand-ins: greedy edge; NN + 2-opt; Christofides + 2-opt |
| `tour_length`, `is_valid_tour`, `brute_force_tsp` | the TSP helpers |

---

## 4. `experiments.py`

- **`make_data`:** builds a fixed training set (default 1M pairs) with n drawn from a list of sizes.
- **`batches`:** groups examples with **equal n**, so the inputs need no padding.
- **`train`:** SGD lr 1.0, batch 128, clip 2.0 (`--quick` switches to Adam 1e-2, which our demo needed).

| Function | Reproduces |
|---|---|
| `e1` | Table 1: the three models at n = 50; LSTM at n = 5 and 10; Ptr-Net trained on 5–50 and tested on 5 … 500 |
| `e2` | Delaunay at n = 5, 10, 50 (outputs read back in groups of 3) |
| `e3` | Table 2: optimal-trained n = 5 and 10; A1- and A3-trained n = 50; the 5–20 model tested on 5 … 50 with constrained beam 10 |

---

## 5. The tests

| Test | Proves |
|---|---|
| `test_convex_hull_target_format` | matches scipy; lowest index first; counter-clockwise; closed |
| `test_hull_metrics_polygon_equality_and_simplicity` | rotation-equality, crossing detection, area, FAIL rule |
| `test_delaunay_target_is_empty_circle_and_ordered` | the empty-circumcircle property; the metrics |
| `test_held_karp_is_optimal_and_heuristics_are_valid` | DP = brute force; valid heuristics; Christofides ≤ 1.5·OPT; 2-opt never hurts |
| `test_pointer_dictionary_grows_with_the_input` | n + 1 outputs for any n (Ptr-Net) vs n_max + 1 (baselines) |
| `test_log_prob_is_the_chain_rule_with_copied_inputs` | Eq. 1 by hand; the decoder input = the previously chosen point; padding ignored |
| `test_constrained_beam_search_returns_valid_tours` | the Section 4.4 constraint, even for an untrained model |
| `test_ptr_net_learns_to_sort` | ≥ 30/50 lists of 5 sorted exactly after 300 steps |

---

## 6. Try it yourself

1. Run the sorting demo with `embed=False`. How much slower is learning?
2. Train the convex hull on points **sorted by x** before encoding. Does it learn faster? (Preview of Paper 032.)
3. Replace the hull's "lowest index first" rule with "a random starting vertex". What happens to the loss?
4. In the TSP demo, train a Ptr-Net on n = 6 with `held_karp` labels and decode with and without `constraint="tour"`. Count the invalid tours.
