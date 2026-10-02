# The code, explained simply

How the code in this folder implements GPT (Radford et al. 2018).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `gpt.py` | the decoder-only Transformer with learned positions and a tied output layer, the LM loss, special tokens, the four input transformations, `FineTuner` (W_y on ⟨e⟩, L₃ = L₂ + λL₁), layer transfer, zero-shot scoring helpers, AdamW with decay groups, schedules; BPE is reused from paper 034 |
| `experiments.py` | pre-train on Gutenberg books with BPE; fine-tune SST-2 / RTE / MRPC / COPA; layer transfer; zero-shot over checkpoints; LSTM baseline (heavy, not run here) |
| `demo.py` | size, input formats, a toy pre-train / zero-shot / fine-tune / layer-transfer study (~11 seconds) |
| `test_gpt.py` | 8 quick tests (~1 second) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/048-Radford-et-al-2018-GPT`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~11 seconds
python3 experiments.py --quick
```

---

## 2. `gpt.py`

### The model
- **`Block`:** post-LN decoder block, x = LN(x + attention(x)), then x = LN(x + MLP(x)), with a GELU MLP and dropout.
- **`GPT(vocab, n_ctx, d, layers, heads, ff, dropout)`:**
  - initialised from N(0, 0.02);
  - **`hidden(ids, n_layers)`** computes h₀ = W_e[ids] + W_p, then the causal blocks;
  - **`logits(h)`** = h W_eᵀ (tied);
  - **`lm_loss(ids, mask)`** is next-token cross-entropy, averaged over real tokens;
  - **`extend_vocab(n)`** appends n new token embeddings (⟨s⟩, $, ⟨e⟩) and returns their ids.

### Fine-tuning
- **`Specials`** holds the start, delimiter and extract ids.
- **Input builders:** `classification_input`, `entailment_input`, `similarity_inputs` (both orders), `multiple_choice_inputs` (one per answer).
- **`pad_batch`** pads sequences and returns a mask.
- **`FineTuner(gpt, n_classes)`:**
  - **`features`** takes h at the last real token (⟨e⟩);
  - **`forward`** applies dropout, then W_y;
  - **`loss(..., lam)`** = cross-entropy + λ × LM cross-entropy on the same inputs.
- **`transfer_layers(src, dst, n)`** copies the embeddings and the first n blocks (Figure 2, left).

### Zero-shot helpers
| Function | What it computes |
|---|---|
| `next_token_logprobs` | log P(next \| ids) |
| `avg_logprob(gpt, context, continuation)` | the RACE / DPRD heuristic |
| `zero_shot_sentiment(gpt, ids, very, pos, neg)` | the SST-2 heuristic |

### Optimisation helpers
| Function | What it does |
|---|---|
| `optimizer` | AdamW with weight decay 0.01 on matrices, 0 on biases and LN gains |
| `warmup_cosine` | the pre-training schedule |
| `warmup_linear` | the fine-tuning schedule |

---

## 3. `experiments.py`

**Data and tokenizer:**
- **Corpus:** 20 Gutenberg books, as long contiguous text.
- **`BPETok`:** learns merges with paper 034's `learn_bpe` and caches word pieces.

**Tasks** (SST-2 / RTE / MRPC from GLUE, COPA from SuperGLUE): `encode` builds Figure 1's sequences, and `batch_loss` handles each type:
- **MRPC:** the two orders' features are summed;
- **COPA:** one score per choice, with a softmax across the two choices.

| Function | Reproduces |
|---|---|
| `e1` | pre-training (warm-up + cosine, contiguous windows), held-out perplexity |
| `e2` | full vs no auxiliary LM vs no pre-training (Table 5) |
| `e3` | layers transferred, 0 … L (Figure 2, left) |
| `e4` | zero-shot SST-2 / CoLA / COPA over pre-training checkpoints (Figure 2, right) |
| `e5` | an LSTM language model in the same pipeline (Table 5's LSTM row) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_size_matches_117M` | 116.5M parameters with the paper's settings |
| `test_output_layer_is_tied_and_causal` | logits = h W_eᵀ; a later token never changes earlier states |
| `test_lm_loss_is_next_token_cross_entropy_and_masks_padding` | Eq. 1 as cross-entropy; padding ignored |
| `test_input_transformations_figure_1` | exact token layouts for all four tasks |
| `test_finetuner_reads_the_extract_token_and_combines_losses` | the features come from the ⟨e⟩ position; λ = 0 gives L₂ and λ > 0 adds the LM term; new vocab ids |
| `test_layer_transfer_and_zero_shot_scoring` | the copied vs fresh blocks; probabilities sum to 1; average log-prob by hand |
| `test_schedules_and_weight_decay_groups` | the schedule endpoints; the decay groups |
| `test_bpe_reused_from_034` | the imported BPE works |

---

## 5. Try it yourself

1. In the demo, pre-train for 2000 steps. Does the fine-tuned accuracy with 40 labels pass the zero-shot score?
2. Give fine-tuning 400 labels instead of 40. Does λ = 0.5 now help, as the paper finds for larger datasets?
3. Remove the final "overall it was very good/bad" sentence from the pre-training corpus. Is zero-shot still possible? Does pre-training still help fine-tuning?
4. Swap `Block` to pre-LN (LayerNorm before attention and MLP, as GPT-2 does) and compare pre-training stability.
