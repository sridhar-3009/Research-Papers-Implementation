# The code, explained simply

How the code in this folder implements Grammar as a Foreign Language (Vinyals et al. 2015).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `parser.py` | trees (read, clean, linearize, balance, delinearize), EVALB F1, the LSTM of Section 2, LSTM+A (and the no-attention baseline), beam search with ensembles, vocabularies and batching |
| `grammar.py` | a toy probabilistic grammar (S, NP, VP, PP with recursion) giving Penn-style trees |
| `experiments.py` | E1–E6 on NLTK's WSJ sample (heavy, not run here) |
| `demo.py` | linearization, LSTM+A vs baseline, the Figure 4 attention statistics (about 9 seconds) |
| `test_parser.py` | 13 quick tests (about 2 seconds) |

**Run it** (from `05-Words-and-Sequences/030-Vinyals-et-al-2015-Grammar-as-Foreign-Language`):
```
python3 -m pytest -q             # ~2 seconds
python3 demo.py                  # ~9 seconds
python3 experiments.py --quick   # ~30-60 minutes
```

---

## 2. `parser.py`: trees

A tree is `(label, [children])`. A preterminal is `(POS, [word])`.

- **`parse_tree`:** reads bracketed text and drops the unlabeled outer `( … )` of PTB files.
- **`clean_ptb`:** removes `-NONE-` empty elements (and the nodes left empty) and strips function tags (`NP-SBJ-1` → `NP`).
- **`linearize(t, normalize_pos=True)`:** depth-first, `(L … )L`, with POS → `XX`.
- **`balance`:** the paper's repair. Missing `)L` are added at the end, and missing `(L` at the start.
- **`delinearize(seq, sent)`:**
  - balances the output first;
  - each non-bracket symbol takes the next word;
  - `)` closes the innermost open node;
  - extra symbols are dropped, and left-over words go under the root.
- **`evalb(golds, preds)`:**
  - removes punctuation (found from the **gold** tags) from both trees;
  - counts labeled spans (excluding preterminals), with ADVP = PRT;
  - returns corpus P / R / F1.

---

## 3. `parser.py`: the model

- **`Cell`:** Section 2's LSTM. The memory is m = m ⊙ f + i ⊙ i′ and the output is h = m ⊙ o (with `cell_tanh=True`, h = tanh(m) ⊙ o).
- **`Deep`:** a stack of cells, with dropout applied **between** layers only.
- **`LSTMA(n_words, n_labels, d=256, emb=512, layers=3, attention=True, dropout)`:**
  - **`encode`:** runs the encoder over the (already reversed) input. Padded steps keep the state unchanged. Returns the top-layer states H, the mask and the final states of every layer, which start the decoder.
  - **`attend(d_t, H, W1H, mask)`:** u = vᵀ tanh(W1′h_i + W2′d_t), masked softmax, d′ = Σ a h. W1′H is computed once.
  - **`step`:**
    - the input is [E_out(previous symbol) ; previous [d ; d′]] (input feeding);
    - it runs the deep LSTM, then the attention, then the softmax over [d ; d′];
    - with `attention=False` it is a plain seq2seq baseline: no feeding, and the softmax uses d alone.
  - **`log_prob(src, tgt)`:** log P(B | A) with teacher forcing.
- **`beam_search(models, src, beam=10)`:** works on one sentence. With a list of models it averages their log-probabilities (an ensemble). It also returns the attention row of every output step.
- **`batch_pairs(pairs, reverse=True)`:** reverses each sentence (not the tree), pads, and appends EOS to the targets.

---

## 4. `experiments.py`

- **Data:** NLTK's WSJ sample. Files 0001–0159 are training, 0160–0179 dev, 0180–0199 test.
- **Training:** SGD lr 0.5, batch 64, clipping at 5. The epoch with the best dev F1 is kept (greedy decoding for speed).

| Function | Reproduces |
|---|---|
| `e1` | Table 1 (small data): baseline LSTM+D, LSTM+A+D, an ensemble of 5, LSTM+A without dropout; malformed %, speed, F1 by length (Figure 3) |
| `e3` | beam 1 / 2 / 10 |
| `e4` | XX vs real POS tags in the output |
| `e5` | reversed vs in-order input |
| `e6` | Figure 4: an attention heatmap |

---

## 5. The tests

| Test | Proves |
|---|---|
| `test_linearization_is_figure_2` | Figure 2, with and without XX |
| `test_linearization_is_invertible_given_the_words` | tree → sequence → tree is exact |
| `test_treebank_cleaning` | `-NONE-` and function tags are removed |
| `test_malformed_outputs_are_balanced_at_the_ends` | the repair rule |
| `test_delinearize_keeps_every_word_even_with_the_wrong_number_of_terminals` | robustness |
| `test_evalb_ignores_punctuation_and_counts_labeled_spans` | P = R = 75 on a hand example; ADVP = PRT |
| `test_lstm_cell_is_section_2` | h = m ⊙ o |
| `test_dropout_only_between_layers` | the LSTM+A+D placement |
| `test_attention_uses_the_current_decoder_state_and_ignores_padding` | u_ti depends on d_t; input feeding |
| `test_padding_does_not_change_log_prob` | masking |
| `test_input_is_reversed_but_not_the_tree` | Section 2.3 |
| `test_greedy_and_ensemble_of_copies` | beam 1 = greedy; an ensemble of identical models = one model |
| `test_lstm_with_attention_learns_to_parse_the_toy_grammar` | F1 > 90 on short toy sentences after 200 steps |

---

## 6. Try it yourself

1. In the demo, use `layers=3` and train longer. Does the attention become as sharp as Figure 4?
2. Train with `normalize_pos=False` on the toy grammar. Here the tags are a deterministic function of the word, so does XX still help?
3. Remove input reversal in `batch_pairs`. What happens to the F1 on long sentences, with and without attention?
4. Make the toy grammar's PP attachment ambiguous (the same words, different trees) and see what F1 is still possible.
