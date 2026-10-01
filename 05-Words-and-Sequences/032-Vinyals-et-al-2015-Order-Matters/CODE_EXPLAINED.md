# The code, explained simply

How the code in this folder implements "Order Matters" (Vinyals, Bengio & Kudlur 2016).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `orderset.py` | the Read–Process–Write model and its Ptr-Net baseline; the set language model and Eq. 9 (exact max, ancestral sampling, uniform pretraining); word-order transforms; depth- and breadth-first tree linearizations; star graphical models |
| `experiments.py` | E1–E6: Table 1, the PTB orderings, parse orders, sorting as a set, the graphical-model grid, Table 2 (heavy, not run here) |
| `demo.py` | permutation invariance, Table 1 in miniature, the star model, Figure 2, Eq. 9 (~13 seconds) |
| `test_orderset.py` | 12 quick tests (~3 seconds) |

**Run it** (from `05-Words-and-Sequences/032-Vinyals-et-al-2015-Order-Matters`):
```
python3 -m pytest -q             # ~3 seconds
python3 demo.py                  # ~13 seconds
python3 experiments.py --quick   # ~1-2 hours
```

---

## 2. `orderset.py`: input sets

### `ReadProcessWrite(in_dim, d, steps, glimpses, encoder="set" | "lstm", mask_repeats=False)`
- **`read`:** a 2-layer MLP applied to every element separately, giving the memories m_i.
- **`process(M)`:** Eqs. 3–7.
  - The LSTM's only input is q*_{t−1}.
  - The score is a dot product m_i·q_t, followed by a softmax and a weighted sum r_t.
  - Returns q*_T and every step's attention.
  - With `steps = 0`, q*_T is all zeros: the "blind" writer.
- **`encode(X)`:**
  - **`"set"` mode:** the memories, plus a writer start state from q*_T. Order-free.
  - **`"lstm"` mode:** the Table 1 baseline. An LSTM reads the memories in order, and the pointer targets are its states.
- **`pointer_logits(M, h)`:**
  - **`glimpses` attention reads:** each **adds** its readout to the query;
  - then the Paper 031 pointer, vᵀtanh(W₁m_i + W₂·query);
  - an optional mask stops it pointing twice at the same element.
- **`log_prob(X, order)`:** teacher-forced chain rule. The decoder's next input is the chosen element's memory.
- **`greedy(X)`:** decodes with no repeats.

### `sort_batch(N, n)`
Random numbers in [0, 1] and their `argsort`, the target pointer sequence.

---

## 3. `orderset.py`: output sets and Eq. 9

- **`SetLM(vocab, d)`:** an LSTM language model over tokens (Eq. 8's chain rule), with an extra start symbol.
  - `seq_log_prob` gives log p of a token sequence.
- **`order_log_prob(model, items, order)`:** log p(Y_π), the set `items` emitted in the order π.
- **`best_order`:** Eq. 9's inner max, **exactly**, by trying all n! orders (small n only).
- **`sample_order`:** ancestral sampling.
  - At each step, choose among the **remaining** elements in proportion to the model's probabilities.
  - Returns the order and its exact sampling log-probability (see EXPLAINED.md section 4 for when this equals p(Y_π)/Σp).
- **`eq9_step(model, opt, items, mode)`:** one update. `mode` is one of:
  - `"given"` (the order in `items`);
  - `"uniform"` (a random order: the pretraining phase);
  - `"max"` (exact max);
  - `"sample"` (Section 5.2).

---

## 4. `orderset.py`: the ordering tasks

| Function | What it does |
|---|---|
| `reverse_words`, `three_word_reversal` | Section 5.1.1's transforms ("a is This <pad> . sentence") |
| `dfs_linearize`, `bfs_linearize`, `bfs_delinearize` | Figure 2's two tree orders (breadth-first: LEV starts a level, PAR separates parents, trailing empty groups dropped) and the inverse |
| `star_model`, `star_sample`, `star_log_prob` | a star graphical model (head + conditional tables), sampling, and its exact log-likelihood |
| `star_tokens(values, head_first)` | "variable j has value v" tokens (j·K + v), head first or last |

---

## 5. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Table 1: Ptr-Net and P ∈ {0, 1, 5, 10}, glimpses 0/1, N ∈ {5, 10, 15}, 10,000 updates |
| `e2` | PTB perplexity for natural / reversed / 3-word-reversed text (the Paper 024 "medium" LSTM) |
| `e3` | depth- vs breadth-first parse trees with Paper 030's LSTM+A on NLTK's WSJ sample; F1 and % valid trees |
| `e4` | sorting as a set of (index, rank) pairs: fixed increasing order vs a random one of n! |
| `e5` | head-first vs head-last over variables × data size × peakiness, with the true model's NLL as a floor |
| `e6` | Table 2 on PTB 5-grams: fixed (1,2,3,4,5), fixed (5,1,3,4,2), easy (Eq. 9 over 2 orders), hard (Eq. 9 over 120) |

---

## 6. The tests

| Test | Proves |
|---|---|
| `test_read_process_is_permutation_invariant_but_the_lstm_encoder_is_not` | the core property of Section 4 |
| `test_process_attention_is_a_distribution_and_reads_a_convex_combination` | Eqs. 5–6 |
| `test_zero_process_steps_leave_the_writer_blind` | P = 0 |
| `test_log_prob_is_the_chain_rule_and_greedy_returns_a_permutation` | the writer |
| `test_figure_2_linearizations` | both strings, character for character |
| `test_breadth_first_linearization_is_invertible` | 50 random trees round-trip |
| `test_word_order_transforms` | 3-word reversal and reversal |
| `test_star_model_is_a_normalized_distribution` | the probabilities sum to 1; token layout |
| `test_eq9_max_is_the_exact_best_order` | equals brute force |
| `test_ancestral_sampling_matches_its_stated_probability` | Monte-Carlo frequencies = the stated q(π) |
| `test_eq9_training_steps_use_valid_orders_and_reduce_the_loss` | all four modes |
| `test_read_process_write_learns_to_sort` | > 60% of 5-number lists sorted after 450 updates |

---

## 7. Try it yourself

1. In `demo.py`, replace the glimpse's `query + readout` with just `readout`. Does the set model still learn?
2. Raise the process steps to 10 for N = 10. Does accuracy keep improving, as in Table 1?
3. Make the star model's tables very peaky (peaky = 8). Does the head-first advantage disappear, as the paper says?
4. Run Eq. 9 with `"max"` from the very first step, with no uniform pretraining. Does the model lock into one random order?
