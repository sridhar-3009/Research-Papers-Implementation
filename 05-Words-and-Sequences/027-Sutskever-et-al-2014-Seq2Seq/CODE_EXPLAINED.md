# The code, explained simply

How the code in this folder implements Sutskever, Vinyals & Le (2014).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `seq2seq.py` | the Seq2Seq model, the paper's init, reversal and time lags, beam search (with ensembles), n-best rescoring, clipping, the learning-rate schedule, length-bucketed batches, BLEU |
| `experiments.py` | reversal, beam/ensemble, length analysis, sentence representations, rescoring on Multi30k (heavy, not run here) |
| `demo.py` | A light tour (a few seconds) |
| `test_seq2seq.py` | 11 quick tests (about a second) |

**Run it** (from `05-Words-and-Sequences/027-Sutskever-et-al-2014-Seq2Seq`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~3 seconds
python3 experiments.py --quick   # small LSTMs, 2 epochs: ~20-40 minutes
```

---

## 2. `seq2seq.py`

### `Seq2Seq(src_vocab, tgt_vocab, emb, hidden, layers)`
- **Two separate `nn.LSTM`s** (encoder and decoder) with the same number of layers. The LSTM is Graves' formulation, which PyTorch implements.
- `encode(src, lengths)` returns the final (h, c) of every layer. With `lengths` it packs the padded batch, so padding doesn't change the result.
- `forward(src, tgt_in)` starts the decoder from the encoder's (h, c) and returns logits with teacher forcing.
- `step(y_prev, state)` does one decoder step (for beam search) and returns log-probabilities.

### `paper_init_`, `reverse_source`, `time_lags`
- U[−0.08, 0.08] for every parameter.
- Reverse a token list.
- `time_lags(n, reverse)` gives the distance between each source word and its translation under a monotone alignment.

### `beam_search(models, src, beam, max_len)`
1. Encode with every model in the ensemble.
2. Each step, run all live hypotheses through every model together, and **average the probabilities** over the ensemble.
3. Add the step's log-probability to each hypothesis's score; keep the global top-B (word, hypothesis) pairs.
4. A hypothesis that ends in EOS goes to `finished`.
5. Stop when B hypotheses are finished or `max_len` is reached. Return them sorted.

### Training helpers
- `clip_grad_(model, 5)`: the norm of the whole gradient; rescale if it's above 5.
- `lr_schedule(epoch_fraction)`: 0.7 until epoch 5, then halved every half epoch.
- `length_batches(pairs, B)`: sort by length (with random tie-breaks), cut into batches, shuffle the batches.
- `rescore_nbest(nbest, lstm)`: 0.5 × baseline score + 0.5 × LSTM score.
- `bleu`: corpus BLEU-4.

---

## 3. `experiments.py`

- **Data:** Multi30k (English → German), lowercased, with 10k-word vocabularies.
- **`train`** follows Section 3.4:
  - plain SGD whose learning rate follows `lr_schedule`, with our epochs mapped onto the paper's 7.5;
  - the loss summed over tokens and divided by the batch size;
  - clipping at 5.

| Function | Reproduces |
|---|---|
| `e1` | forward vs reversed source: test perplexity and beam-12 BLEU |
| `e2` | beams 1/2/12 × (1 model, an ensemble of K) |
| `e3` | BLEU per source-length bucket (Figure 3) |
| `e4` | 2-D PCA of the top layer's (h, c) for word-order and voice variants (Figure 2) |
| `e5` | a 12-best list from one model, re-ranked by the average with a second model's log-probability |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_the_paper_model_has_384M_parameters` | 384M total, 32M for the encoder LSTM |
| `test_sentence_representation_is_8000_numbers` | 4 × (h + c) × 1000 |
| `test_paper_init_range` | U[−0.08, 0.08] |
| `test_reversal_keeps_the_average_lag_but_shrinks_the_minimal_lag` | Section 3.3's argument |
| `test_lr_schedule` | the halving schedule |
| `test_clipping_scales_the_gradient_to_norm_5` | the clipping rule |
| `test_length_batches_group_similar_lengths` | bucketing |
| `test_greedy_can_miss_the_best_sequence_but_a_wide_beam_finds_it` | beam search = exact search when wide enough |
| `test_ensemble_of_identical_models_equals_one_model` | ensemble averaging |
| `test_beam_results_are_sorted_and_scores_are_log_probs` | beam scores = log p(y\|x) |
| `test_rescoring_is_an_even_average` | the rescoring rule |

---

## 5. Try it yourself

1. Add **length normalization** to `beam_search` (divide each score by its length, or use (5 + L)⁶/6⁶ as in Wu et al. 2016) and rerun demo part 4.
2. Train with momentum or Adam instead of plain SGD. Does it still need the halving schedule?
3. Reverse **both** source and target, or **neither**, and compare BLEU.
4. Give the decoder the encoder's state at every step (as Cho et al. do, Paper 026). Does it help?
