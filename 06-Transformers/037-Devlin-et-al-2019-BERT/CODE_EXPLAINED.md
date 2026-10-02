# The code, explained simply

How the code in this folder implements BERT (Devlin et al. 2019).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `bert.py` | WordPiece (building a vocabulary, greedy tokenizing), input packing, masking, NSP pairs, the BERT encoder (with an optional causal mode for the LTR ablation), pre-training loss, fine-tuning heads (classification, SQuAD spans, multiple choice), feature extraction, the warm-up schedule and optimizer |
| `experiments.py` | mini-BERT on Gutenberg books, then SST-2 / RTE: pre-trained vs scratch, Tables 5–6, masking mixes, feature-based (heavy, not run here) |
| `demo.py` | WordPiece, Figure 2's input, masking statistics, bidirectional vs left-to-right, a toy NSP, sizes (~5 seconds) |
| `test_bert.py` | 11 quick tests (~2 seconds) |

**Run it** (from `06-Transformers/037-Devlin-et-al-2019-BERT`):
```
python3 -m pytest -q             # ~2 seconds
python3 demo.py                  # ~5 seconds
python3 experiments.py --quick   # ~1-2 hours
```

---

## 2. `bert.py`

### Text
- **`build_wordpiece_vocab(word_counts, size)`:** pieces start as characters (word-initial "a", inner "##a"); the most frequent adjacent pair is merged until there are `size` pieces.
- **`wordpiece_tokenize(word, vocab)`:** greedy longest-match-first; [UNK] if impossible.
- **`pack_pair(a, b, max_len)`:** builds [CLS] a [SEP] b [SEP] and the segment ids, truncating the longer side.
- **`collate`:** pads, and returns ids, segments and the attention mask.

### Pre-training data
- **`mask_tokens(ids, V, rate=0.15, probs=(0.8, 0.1, 0.1))`:** returns the corrupted inputs and labels (IGNORE where nothing is predicted). Special tokens are never chosen.
- **`nsp_pairs(documents, n, rng)`:** returns (A, B, label) with label 0 = IsNext, 1 = NotNext.

### The model
- **`Layer`:** post-LN encoder layer with GELU. Padding is masked; `causal=True` adds a lower-triangular mask.
- **`Bert(vocab, H, L, A, max_len, dropout, causal)`:**
  - **embeddings:** token + segment + position, then LayerNorm and dropout;
  - **layers;** then a **pooler** (tanh) on [CLS] that gives C;
  - **MLM head:** dense → GELU → LN, then the **tied** token-embedding matrix + bias;
  - **NSP head:** 2-way on C;
  - **init:** N(0, 0.02).
  - **`forward(..., all_layers)`** returns T, C and optionally every layer's output.
  - **`pretrain_loss`** = MLM + NSP.

### Fine-tuning
| Class / function | What it does |
|---|---|
| `Classifier(bert, K)` | C Wᵀ |
| `SpanQA(bert)` | start/end scores S·T_i and E·T_j; `best_span(..., null_threshold)` implements SQuAD 1.1 and 2.0 decoding |
| `MultipleChoice(bert)` | scores v·C for each choice |
| `feature_based(bert, ..., how)` | frozen features: last layer, sum of the top 4, or concatenation of the top 4 |
| `warmup_linear`, `optimizer` | Appendix A.2's schedule; AdamW with no weight decay on 1-D parameters |

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | pre-train mini-BERT (MLM + NSP), fine-tune SST-2 and RTE with the paper's lr grid, compare with from-scratch training; held-out MLM perplexity |
| `e2` | Table 5: MLM + NSP / no NSP / left-to-right without NSP |
| `e3` | Table 6: four model sizes, MLM perplexity and dev accuracy |
| `e4` | Appendix C.2: six (MASK, SAME, RND) mixes, fine-tuned and feature-based |
| `e5` | Section 5.3: linear probes on frozen features vs fine-tuning (SST-2) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_wordpiece_greedy_longest_match` | the tokenizer |
| `test_input_packing` | Figure 2's format and truncation |
| `test_masking_rates` | 15% and 80/10/10; no special tokens |
| `test_next_sentence_pairs` | 50/50 and correct next/random |
| `test_model_sizes` | 110M / 340M |
| `test_bidirectional_vs_left_to_right_context` | which positions can see a later token |
| `test_padding_is_ignored` | the attention mask |
| `test_pretraining_loss_and_tied_output` | MLM + NSP; tied output weights |
| `test_fine_tuning_heads` | classification, span (+ no answer), multiple choice, four-layer features |
| `test_schedule_and_optimizer_groups` | warm-up + linear decay; weight-decay groups |
| `test_masked_lm_uses_right_context` | > 90% of right-determined tokens recovered by the MLM |

---

## 5. Try it yourself

1. In `demo.py`, train the marker task with `probs=(1.0, 0.0, 0.0)`, then evaluate with markers left **unmasked but corrupted**. Which mix is more robust?
2. Fine-tune the toy NSP model as a classifier on another pair task and compare with training from scratch.
3. Replace the pooler output C with T₀ (the raw [CLS] state) in `Classifier`. Does it matter?
4. Build a 2-layer `SpanQA` toy where the answer is the token after a marker, and test `best_span`.
