# The code, explained simply

How the code in this folder implements Zaremba, Sutskever & Vinyals (2014).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `reg_lstm.py` | the LSTM cell, the deep LSTM with three dropout modes, the dropout path count, stateful truncated-BPTT training, ensemble perplexity |
| `experiments.py` | Table 1, the dropout-placement ablation, ensembles and samples on Penn Treebank (heavy, not run here) |
| `demo.py` | A light tour (about a second) |
| `test_reg_lstm.py` | 10 quick tests (about a second) |

**Run it** (from `05-Words-and-Sequences/024-Zaremba-et-al-2014-RNN-Regularization`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~1 second
python3 experiments.py --quick   # downloads PTB (~5 MB), 1 epoch per model: ~20-40 minutes
```

---

## 2. `reg_lstm.py`

### `LSTMCell(n_in, n)`
One `nn.Linear(n_in + n, 4n)` = T_{2n,4n}, applied to `cat([x, h])`. The output is split into i, f, o, g.

### `DeepLSTM(vocab, n, layers, dropout, mode, init)`
For each time step and each layer:
```python
vertical  = D(below)      if mode != "none"   else below     # from the layer below (or the embedding)
recurrent = D(h_prev)     if mode == "naive"  else h_prev    # own previous output
h, c = cell(vertical, recurrent, c)
```
Then `D(top)` before the softmax layer.
- `D` = `F.dropout`: inverted dropout, scaling the kept units by 1/(1 − p) during training. That's equivalent to the paper's test-time scaling.
- `record=True` saves (vertical, recurrent, h_prev) for every cell call, so tests can check which inputs got dropout.
- Every parameter is initialized U[−init, init], as in the paper.

### `dropouts_on_path(L, k, mode)`
Figure 3's count: L + 1 for the paper's mode, plus k for naive dropout.

### Training: `batchify` and `run_epoch`
- `batchify(ids, 20)` cuts the token stream into 20 parallel contiguous streams.
- `run_epoch(model, data, steps, lr, clip)`:
  - walks through `data` in chunks of `steps` tokens (the unroll length), carrying the hidden state across chunks with `detach()` (truncated BPTT);
  - the loss is the mean cross-entropy × steps (sum over time, mean over the batch: "normalized by minibatch size");
  - clips the gradient norm, then applies plain SGD;
  - returns the perplexity exp(mean cross-entropy).

### `ensemble_perplexity(models, data)`
Averages the models' **probabilities** at each step, then computes the perplexity.

---

## 3. `experiments.py`

- `RECIPES` holds the paper's settings for small, medium and large: units, dropout, init range, unroll length, epochs, when and how much to decay the learning rate, and the clip value.
- `load_ptb()` downloads `ptb.{train,valid,test}.txt` from the paper's repository, replaces line ends with `<eos>`, and builds the 10k vocabulary.

| Function | Reproduces |
|---|---|
| `e1` | Table 1's single models (valid / test perplexity); saves the medium model |
| `e2` | the medium model with no dropout / the paper's dropout / naive dropout |
| `e3` | averaging 1 … K medium models |
| `e4` | samples after "the meaning of life is", with `<unk>`, N and $ banned (as in Figure 4) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_cell_matches_torch_lstm` | the LSTM equations |
| `test_model_sizes_of_section_4_1` | medium ≈ 20M, large ≈ 66M |
| `test_nonrecurrent_mode_drops_only_vertical_connections` | **the recipe**: recurrent inputs untouched, ~50% of vertical inputs zeroed |
| `test_naive_mode_also_drops_recurrent_connections` | the contrast |
| `test_no_dropout_at_test_time` | eval is deterministic |
| `test_information_is_corrupted_L_plus_1_times_whatever_the_lag` | Figure 3 |
| `test_batchify_makes_parallel_streams` | the data layout |
| `test_untrained_model_perplexity_is_about_the_vocab_size` | perplexity = V for uniform predictions |
| `test_training_reduces_perplexity_on_a_repeating_stream` | training works |
| `test_ensemble_of_one_equals_the_model` | model averaging |

---

## 5. Try it yourself

1. Implement **variational dropout** (Gal & Ghahramani 2016): one mask per sequence, reused at every time step, also on h_{t−1}. Compare it with the paper's recipe in `e2`.
2. Sweep the dropout rate of the medium model (0.2 … 0.7) and plot validation perplexity.
3. Turn off dropout on the **embedding** only, or the **softmax input** only. Which matters more?
4. Check how much of the medium model's gain comes from regularization and how much from size: train a 650-unit model **without** dropout and watch the validation perplexity.
