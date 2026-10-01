# The code, explained simply

How the code in this folder implements Hochreiter & Schmidhuber (1997).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `lstm.py` | the squashing functions, the error-scaling analysis, the 1997 LSTM in NumPy with the paper's truncated learning rule, the same network in PyTorch, a plain RNN baseline |
| `tasks.py` | the benchmark tasks: embedded Reber grammar, task 2a, adding, multiplication, temporal order |
| `experiments.py` | Experiments 1, 2a, 4, 5, 6a, plus a modern-LSTM comparison (heavy, not run here) |
| `demo.py` | A light tour, with no training (about a second) |
| `test_lstm.py` | 16 quick tests (about a second) |

**Run it** (from `05-Words-and-Sequences/021-Hochreiter-Schmidhuber-1997-LSTM`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~1 second
python3 experiments.py --quick   # small budgets: ~20-40 minutes
```

---

## 2. `lstm.py`

### Squashing functions (Appendix A.1)
`f` (logistic), `g = 4f − 2`, `h = 2f − 1`, with their derivatives `df`, `dg`, `dh`.

### `error_scaling(factors)` and `max_sigmoid_factor(w)`
The product of |f′·w| along a path (Eq. 2), and the best case 0.25·|w| for a logistic unit.

### `LSTM1997` (NumPy)
**Weights:**
- `W_in` and `W_out`: one row per block (the gates);
- `W_c`: one row per cell;
- `W_k`: the output units.

Each row reads `z = [x(t), y_c(t−1), 1]`; the last column is the bias.

**`forward(xs)`:**
```python
y_in, y_out = f(W_in z), f(W_out z)                 # one per block
s  = s + repeat(y_in) * g(W_c z)                    # the CEC
yc = repeat(y_out) * h(s)
y  = f(W_k [yc, 1])
```
`repeat` copies a block's gate value to each of its S cells.

**`truncated_gradient(xs, targets, mask)`:** Appendix A.1, forward in time.
1. **Two running sensitivity matrices** (C × Z):
   ```python
   dS_c  += (g'(net_c) * y_in) ⊗ z      # how s moves if W_c changes
   dS_in += (g(net_c) * f'(net_in)) ⊗ z # how s moves if the block's W_in changes
   ```
   They only ever **add** up. That's the CEC's factor 1.0, written in derivative form.
2. **At steps with a target (`mask`):**
   ```python
   e_k   = f'(net_k) * (y - target)
   e_c   = W_k^T e_k                                   # error at each cell output
   e_out = f'(net_out) * (h(s) * e_c) summed per block
   e_s   = y_out * h'(s) * e_c                         # error entering the CEC
   grad W_c  += e_s * dS_c
   grad W_in += (e_s * dS_in) summed per block
   grad W_out += e_out ⊗ z                             # not carried through time
   ```
   Nothing is ever propagated back through y_c(t−1): that's the truncation.

`sgd_step` = gradient + update.

### `LSTM1997Torch`
- **The same equations and the same initial weights** (it builds a `LSTM1997` and copies its arrays), batched over N sequences.
- **`truncate=True`:** uses `yc.detach()` where y_c(t−1) enters z. **Autograd then computes exactly the paper's truncated gradient**, and the test checks this against the NumPy rule.
- `return_states=True` also returns s at every step. `s0` lets you start from a given state (used to measure ∂s(T)/∂s(0)).
- Gate biases are set with `in_gate_bias` and `out_gate_bias`.

### `VanillaRNN`
h(t) = σ(W·[x(t), h(t−1), 1]), trained with full BPTT. It's the baseline that Section 3 predicts will fail.

---

## 3. `tasks.py`

| Function | Task |
|---|---|
| `embedded_reber(rng)` | B, T/P, (B + a walk through `REBER_GRAPH` + E), the same T/P, E. Returns the string, one-hot inputs and multi-hot "legal next symbols" targets. `reber_is_valid` checks a string. |
| `noise_free_2a(p, rng)` | (x or y, a₁ … a_{p−1}, the same x or y), with next-symbol targets |
| `adding_problem(T, rng)` | length T to T + T/10; value + marker pairs; markers at one of the first 10 and one of the first T/2 − 1; −1 at the ends; target 0.5 + (X₁ + X₂)/4 |
| `multiplication(T, rng)` | values in [0, 1]; target X₁·X₂ |
| `temporal_order(rng)` | length 100–110; E … B; X/Y at t₁ ∈ [10, 20] and t₂ ∈ [50, 60]; 4 classes |

---

## 4. `experiments.py`

| Function | Experiment |
|---|---|
| `e1` | Reber: LSTM (4 blocks of 1 cell, output-gate biases −1 … −4) vs RNN; "solved" when every legal-next prediction is correct on 256 test strings |
| `e2` | task 2a for p = 4, 10, 100; success = max error < 0.25 on the final prediction |
| `e4` | adding, T = 100: LSTM (2 blocks of 2, input-gate biases −3, −6, learning rate 0.5) vs RNN; wrong out of 2560 at tolerance 0.04 |
| `e5` | multiplication |
| `e6` | temporal order: classification error on 2560 sequences |
| `e7` | adding with `torch.nn.LSTM` (forget gate) + Adam, for comparison |

Errors are only given at sequence ends (except Reber and 2a), like the paper. `--batch 1` is the paper's on-line learning.

---

## 5. The tests

| Test | Proves |
|---|---|
| `test_squashing_function_ranges` | g ∈ [−2, 2], h ∈ [−1, 1], f′ ≤ 0.25 |
| `test_logistic_errors_vanish_when_weights_are_below_4` | Section 3.1 (and blow-up for |w| > 4) |
| `test_error_vanishes_exponentially_with_the_time_lag` | real f′ values: 10⁻⁶⁰ after 100 steps |
| `test_constant_error_carrousel_has_factor_one` | f′·w = 1 forever |
| `test_closed_input_gate_keeps_the_state_constant_for_1000_steps` | the CEC stores a value |
| `test_error_flows_back_through_the_cec_unchanged` | ∂s(500)/∂s(0) = 1 |
| `test_numpy_and_torch_forward_agree` | the two implementations match |
| `test_paper_truncated_gradient_equals_autograd_with_detached_recurrence` | **Appendix A.1's algorithm = autograd with detach** |
| `test_truncated_gradient_differs_from_full_bptt` | truncation really changes the gradient |
| `test_sgd_with_the_paper_rule_reduces_the_error` | the learning rule learns |
| `test_vanilla_rnn_runs` | the baseline |
| `test_embedded_reber_strings_are_valid_and_targets_are_legal` | Experiment 1's data |
| `test_task_2a_needs_the_first_symbol` | Experiment 2a's data |
| `test_adding_problem_structure` | Experiment 4's data |
| `test_multiplication_target` | Experiment 5's data |
| `test_temporal_order_class` | Experiment 6a's data |

---

## 6. Try it yourself

1. Add a **forget gate**: s = f_t·s(t−1) + y_in·g(net_c), with f_t = σ(W_f·z). Rerun demo part 2: does state drift disappear even without negative input-gate biases?
2. Train `LSTM1997Torch(truncate=False)` (full BPTT) on the adding problem and compare it with the truncated version. The paper says there is "no significant difference".
3. Build the **strongly delayed XOR** task and check that the truncated LSTM can't learn it.
4. Replace g and h with tanh (as modern LSTMs do) and compare.
