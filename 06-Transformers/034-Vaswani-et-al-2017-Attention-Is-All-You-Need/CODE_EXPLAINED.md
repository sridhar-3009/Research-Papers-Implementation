# The code, explained simply

How the code in this folder implements the Transformer (Vaswani et al. 2017).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `transformer.py` | attention, multi-head attention, the FFN, sinusoidal positions, encoder/decoder layers, the full Transformer with tied embeddings, the Noam schedule, label-smoothed loss, checkpoint averaging, beam search with length penalty, BPE |
| `experiments.py` | E1–E5 on Multi30k (+ NLTK WSJ for parsing): base BLEU with checkpoint averaging, Table 3 variations, speed vs RNNsearch, head plots, parsing (heavy, not run here) |
| `demo.py` | attention by hand, why √d_k, relative positions, Table 1 costs, a tiny Transformer learning reversal with an anti-diagonal head, the lr schedule (~7 seconds) |
| `test_transformer.py` | 11 quick tests (~2 seconds) |

**Run it** (from `06-Transformers/034-Vaswani-et-al-2017-Attention-Is-All-You-Need`):
```
python3 -m pytest -q             # ~2 seconds
python3 demo.py                  # ~7 seconds
python3 experiments.py --quick   # ~1-2 hours
```

---

## 2. `transformer.py`

### Attention
- **`attention(Q, K, V, mask)`:** Eq. 1. The mask is True where attending is **allowed**; elsewhere the score becomes −∞, so the weight is exactly 0. Returns the output and the weights.
- **`MultiHeadAttention(d_model, h)`:**
  - four d_model × d_model linear maps;
  - `split` reshapes (N, T, d_model) → (N, h, T, d_k), so all heads run as one batched matrix multiply;
  - the heads are concatenated back, then W^O;
  - the last weights are kept in `.weights` for plots.

### The other pieces
- **`FeedForward`:** Eq. 2.
- **`sinusoidal_positions(max_len, d_model)`:** the PE table.
- **`EncoderLayer`, `DecoderLayer`:** post-LN, `LN(x + Dropout(Sublayer(x)))`. The decoder runs masked self-attention, then cross-attention (queries from the decoder, keys and values from `memory`), then the FFN.
- **`causal_mask(T)`:** a lower-triangular True matrix.

### `Transformer(src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout, shared_vocab, positions)`
- **Tied weights:** `shared_vocab=True` uses one embedding for source, target and output (`y @ tgt_emb.weightᵀ + bias`).
- **`embed`:** emb·√d_model + positions, then dropout. `positions="learned"` gives Table 3 row (E).
- **`encode`:** pad mask, then N encoder layers.
- **`decode`:** causal ∧ pad mask, then N decoder layers, then output logits.
- **Init:** Xavier-uniform for all matrices.

### Training and decoding helpers
| Function | What it does |
|---|---|
| `noam_lr(step, d_model, warmup)` | Eq. 3 |
| `label_smoothed_loss(logits, target, eps)` | (1 − ε)·NLL + ε·(mean −log p over classes), ignoring PAD |
| `average_checkpoints(state_dicts)` | element-wise mean |
| `length_penalty(len, α)` | ((5 + len)/6)^α |
| `beam_search(model, src, beam, alpha, extra_len)` | one sentence; max length = |src| + 50; ranks finished candidates by log p / lp |
| `learn_bpe(word_counts, n_merges)`, `apply_bpe(word, merges)` | Sennrich-style byte-pair encoding with an end-of-word marker |
| `count_params` | the number of parameters |

---

## 3. `experiments.py`

- **Data:** `multi30k()` tokenizes lowercase words and punctuation; `BPEVocab` learns one **shared** BPE vocabulary on source + target.
- **`token_batches`:** groups sentences of similar length into ~`batch_tokens`-token batches (Section 5.1).
- **`train`:** Adam (0.9, 0.98, 1e-9), Eq. 3 via `LambdaLR`, label smoothing, residual dropout; keeps the last 5 epoch snapshots.
- **`shrink_keys(m, d_k)`:** Table 3 row (B), which shrinks only the query/key projections.

| Function | Reproduces |
|---|---|
| `e1` | base model test BLEU (beam 4, α 0.6), single vs averaged checkpoints |
| `e2` | Table 3 rows A–E: dev perplexity and BLEU per variation |
| `e3` | training throughput: Transformer vs Paper 028's RNNsearch |
| `e4` | encoder self-attention heads of one sentence, plotted |
| `e5` | Table 4: a 4-layer Transformer parser on NLTK's WSJ sample (Paper 030's linearization and EVALB) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_scaled_dot_product_attention_and_masking` | Eq. 1 by hand; masked weights are exactly 0 |
| `test_why_divide_by_sqrt_dk` | Var(q·k) = d_k, and 1 after scaling |
| `test_multi_head_with_one_head_is_projected_attention` | the multi-head plumbing |
| `test_sinusoidal_positions_formula_and_relative_offsets` | the formula; PE(pos + k) = M_k PE(pos) |
| `test_decoder_cannot_see_future_tokens_and_padding_is_ignored` | the causal and pad masks |
| `test_model_sizes_match_table_3` | base ≈ 65M, big ≈ 213M |
| `test_noam_schedule` | peak value; 1/√step decay |
| `test_label_smoothing` | the formula; minimum = entropy of the smoothed target |
| `test_checkpoint_averaging_and_length_penalty` | the helpers |
| `test_bpe_merges_frequent_pairs` | the most frequent pair merges first; unseen words still split |
| `test_transformer_learns_to_reverse` | ≥ 16/20 reversed exactly after 300 steps |

---

## 5. Try it yourself

1. In `demo.py`, set `h=1` and retrain. Is the anti-diagonal still learned? How fast?
2. Remove the √d_model embedding scale. What happens to training with sinusoidal positions?
3. Switch to `positions="learned"` and train on length 8, then test on length 12. Which kind of position extrapolates better?
4. Implement pre-LN (`x + Sublayer(LN(x))`) and train without warm-up at a higher learning rate. Compare with post-LN.
