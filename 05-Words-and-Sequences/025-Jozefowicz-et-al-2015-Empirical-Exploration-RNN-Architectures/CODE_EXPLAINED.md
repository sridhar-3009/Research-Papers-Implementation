# The code, explained simply

How the code in this folder implements Jozefowicz, Zaremba & Sutskever (2015).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `architectures.py` | the 10 hand-written cells (Tanh, LSTM and its 4 variants, GRU, MUT1–3), a sequence model, architectures as graphs, the 6 mutations, Eq. 1, the search loop |
| `tasks.py` | memorization, arithmetic-with-distractors and XML generators |
| `experiments.py` | Table 1, the forget-bias study, a small search (heavy, not run here) |
| `demo.py` | A light tour (under a second) |
| `test_architectures.py` | 12 quick tests (under a second) |

**Run it** (from `05-Words-and-Sequences/025-Jozefowicz-et-al-2015-Empirical-Exploration-RNN-Architectures`):
```
python3 -m pytest -q             # < 1 second
python3 demo.py                  # < 1 second
python3 experiments.py --quick   # tiny budgets, no PTB: ~20-40 minutes
```

---

## 2. `architectures.py`: hand-written cells

- Every cell has `init_state(N)` and `forward(x, state) -> (h, new_state)`. The state is a tuple: (h,) or (h, c).
- **`LSTM(n, no_forget, no_input, no_output, forget_bias)`:**
  - one `nn.Linear(2n, 4n)` on `[x, h]`, split into i, j, f, o (paper order);
  - the variants replace a gate with ones;
  - `forget_bias` is added to the f slice of the bias.
- **`GRU(n)`** and **`MUT(n, which)`** follow the paper's equations line by line.
- **`CELLS`** maps the names in Table 1 to constructors.
- **`SequenceModel`:**
  - embedding → cells → linear output;
  - initialized U[−x, x] with x = scale/√units (Section 3.7);
  - then the LSTM-b forget bias is added back, since a uniform re-init would erase it.

## 3. `architectures.py`: the search

### `Graph`
- **`nodes`** is a list of `{op, inputs}` in topological order: a node only reads earlier nodes.
  - **ops:** `h`, `c`, `x` (inputs); `tanh`, `sigmoid`, `relu`; `lin<a>` (Linear(a, ·)); `add`, `mul`, `sub`.
- **`outputs`** are the node ids of the new state.
- **Helpers:**
  - `ancestors`, `consumers`;
  - `prune` drops nodes no output needs;
  - `is_valid` checks there are no cycles, every op has the right number of inputs, and #outputs = #state inputs.

### Other pieces
- **`lstm_graph()` / `gru_graph()`** build the two starting cells as graphs. The GRU's z·h + (1 − z)·h̃ is written as h̃ + z·(h − h̃), which needs no constant nodes.
- **`GraphCell(graph, n)`** runs a graph as a cell. Each `lin<a>` node gets an `nn.Linear(n, n)` with a added to its diagonal.
- **`mutate(graph, rng)`** applies 1–3 of the six transformations at random nodes (renumbering after an insertion; choosing operands from earlier nodes so no cycle appears), prunes, and retries until valid.
- **`fitness(best, gru)`** is Eq. 1.
- **`Search(evaluate, passes_memorization, tasks, gru_acc)`:**
  - **`step()`:** half the time it re-evaluates a pool member on new settings. Otherwise it mutates one, applies the memorization filter, then the tasks one by one (stopping below 90% of the GRU), and inserts the survivor into the sorted top-100 list.
  - The evaluator is passed in, so the tests can use a fake one.

## 4. `tasks.py`
| Function | Generates |
|---|---|
| `memorization()` | `abcde=abcde.`, scored on the answer part |
| `arithmetic()` | two signed numbers (up to 8 digits), an operator, distractor letters after each character, then `=answer.` |
| `xml()` | nested random tags with the paper's open/close rule; `xml_is_well_formed` checks the output |

`VOCAB` holds letters, digits, `+ - = .`, and `< > /` for XML.

---

## 5. `experiments.py`

- **`char_stream`** concatenates examples into one stream, with a mask that scores only the answer part (arithmetic, memorization) or everything (XML).
- **`run`** does one epoch of truncated BPTT with SGD and clipping, and returns the next-step accuracy on masked positions.
- **`train_eval`** applies the search schedule: after 3 epochs without improvement, halve the learning rate for 4 epochs, then stop.
- **`random_hp`** draws from Section 3.7's ranges.

| Function | Reproduces |
|---|---|
| `e1` | Table 1: every cell × (Arithmetic, XML, PTB); the best of `--settings` random hyperparameter draws |
| `e2` | LSTM vs LSTM-b with fixed hyperparameters over several seeds |
| `e3` | a real (small) search: GRU baselines, a memorization filter at 95%, mutations, Eq. 1 |

---

## 6. The tests

| Test | Proves |
|---|---|
| `test_every_cell_runs` | all 10 cells |
| `test_lstm_without_forget_gate_never_forgets` | LSTM-f: c' = c + i·j |
| `test_forget_bias_of_one_opens_the_forget_gate` | f ≈ 0.5 vs 0.731 at init |
| `test_forget_bias_survives_sequence_model_init` | LSTM-b really has the bias |
| `test_gru_update_rule` | the GRU equations |
| `test_mut1_update_formula` | MUT1's equations (z ignores h) |
| `test_lstm_graph_equals_the_hand_written_lstm` | the graph form = the hand-written LSTM |
| `test_gru_graph_equals_the_hand_written_gru` | the graph form = the hand-written GRU |
| `test_mutations_always_give_valid_runnable_graphs` | 200 mutations: always valid and runnable |
| `test_fitness_is_the_worst_task_relative_to_gru` | Eq. 1 |
| `test_search_adds_only_architectures_that_pass_every_stage` | the staged filter and top-k list |
| `test_task_strings` | all three generators |

---

## 7. Try it yourself
1. Run `e2` on the adding problem from Paper 021 with T = 100: how much does the forget bias of 1 help there?
2. Set the forget bias to 2 or 3. Is more better?
3. Let `e3` run longer and print the best graph it finds. Does it look like a GRU?
4. Add a "peephole" option (the gates also see c) to `LSTM` and compare it on Arithmetic.
