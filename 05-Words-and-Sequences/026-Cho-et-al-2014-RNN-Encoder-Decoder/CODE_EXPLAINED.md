# The code, explained simply

How the code in this folder implements Cho et al. (2014).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `encdec.py` | the GRU cell, a tanh cell, the full encoder–decoder (Appendix A), orthogonal init, BLEU, the log-linear score |
| `experiments.py` | translation-pair scoring, BLEU, phrase representations, log-linear rescoring on Multi30k (heavy, not run here) |
| `demo.py` | A light tour (about a second) |
| `test_encdec.py` | 12 quick tests (about a second) |

**Run it** (from `05-Words-and-Sequences/026-Cho-et-al-2014-RNN-Encoder-Decoder`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~2 seconds
python3 experiments.py --quick   # downloads Multi30k (~3 MB): ~20-40 minutes
```

---

## 2. `encdec.py`

### `GRUCell(n_in, n)`
`W` stacks W, W_z and W_r (applied to x); `U_zr` stacks U_z and U_r; `U` is the candidate's recurrent matrix.
```python
z = sigmoid(W_z x + U_z h);  r = sigmoid(W_r x + U_r h)
h_tilde = tanh(W x + U (r * h))
return z * h + (1 - z) * h_tilde
```
`TanhCell` is the ungated baseline.

### `EncoderDecoder(src_vocab, tgt_vocab, n, emb, maxout, rank, unit)`
- **`encode(src)`:** runs the cell over the source. **Padded positions keep the old state**, so a padded batch gives the same c as unpadded sentences. Returns c = tanh(V·h_N).
- **`decode_step(y_prev, h, c, cc)`:** Appendix A.1.1.
  - The gates use W′·e(y), U′·h and C·c; the candidate is tanh(W′e + r ⊙ (U′h + Cc)).
  - The output is s′ = O_h·h + O_y·e + O_c·c, then **maxout** (`view(..., -1, 2).max(-1)`), then the factored softmax layer G_l(G_r(s)).
  - `cc` = (C c, C_z c, C_r c) is computed once per sentence.
- **`forward(src, tgt_in)`:** teacher forcing (the decoder is fed the true previous word).
- **`log_prob(src, tgt)`:** log p(y | x) per pair, summed over real tokens only, with BOS prepended to the decoder input.
- **`generate(src, max_len, sample)`:** greedy or sampled decoding until EOS.
- **Initialization:** every weight ~ N(0, 0.01²); the recurrent matrices are set by `orthogonal_` (left singular vectors of a Gaussian matrix).

### `bleu(hypotheses, references, max_n=4)`
Corpus BLEU:
- clip each n-gram count by its count in the reference;
- take the geometric mean of the precisions for n = 1..4;
- multiply by the brevity penalty exp(1 − r/c) if the output is shorter than the reference.

### `loglinear(features, weights)`
Eq. 9: a weighted sum of feature values.

---

## 3. `experiments.py`

- **Data:** Multi30k English → German (lowercased; punctuation split off), sentences ≤ 20 tokens, vocabularies of the most frequent words with `<unk>`.
- **`train`:** AdaDelta (ρ = 0.95, ε = 10⁻⁶), batches of 64, gradient clipping at 5.

| Function | What |
|---|---|
| `e1` | the GRU and tanh models; for each test source, its true translation plus 9 random others are scored by log p(y\|x); accuracy of ranking the true one first |
| `e2` | greedy translations of the test set, corpus BLEU, 5 examples |
| `e3` | phrase vectors c for test sentences; each sentence's nearest neighbour by cosine |
| `e4` | rank candidates by w₁·log p + w₂·length; w₂ chosen on the dev set (a toy MERT) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_gru_equations_5_to_8` | the GRU equations |
| `test_update_gate_near_one_copies_the_state` | z ≈ 1 → h unchanged |
| `test_reset_gate_near_zero_ignores_the_past` | r ≈ 0 → the candidate doesn't depend on h |
| `test_orthogonal_init` | WWᵀ = I |
| `test_encoder_ignores_padding` | padded = unpadded |
| `test_log_prob_sums_the_target_tokens_only` | batched = single, padding ignored |
| `test_first_word_distribution_sums_to_one` | a real distribution |
| `test_maxout_halves_the_output_layer` | the maxout layer |
| `test_training_learns_to_copy` | end-to-end learning |
| `test_generation_stops_at_eos` | decoding |
| `test_bleu` | BLEU on hand-computed cases |
| `test_loglinear_is_a_weighted_sum` | Eq. 9 |

---

## 5. Try it yourself

1. Make the toy task harder: reverse phrases of length 10, 20 and 40. Watch the single vector c become a bottleneck. (Paper 028 fixes this.)
2. Compare the GRU with the LSTM from Paper 024 as the encoder and decoder cell.
3. Plot the reset- and update-gate activations of a trained model on one sentence. Do some units mostly reset while others mostly update, as the paper suggests?
4. Use `generate(sample=True)` 50 times for one source (like Table 3) and list the 5 most probable distinct translations.
