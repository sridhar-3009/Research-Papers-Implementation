# The code, explained simply

How the code in this folder implements GPT-2 (Radford et al. 2019).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `gpt2.py` | GPT-2's pre-tokenizer regex and a byte-level BPE (learn, encode, decode), the pre-LN model with a final LN and 1/√N residual init, the exact size formula and Table 2 shapes, generation (greedy / top-k / temperature), sequence log-probability, per-unit perplexity, LAMBADA stop-word prediction, cloze scoring, prompt builders, the 8-gram Bloom filter and overlap |
| `experiments.py` | the tokenizer comparison, four shape-matched model sizes, LAMBADA, TL;DR summarisation, data overlap, pre-LN vs post-LN (heavy, not run here) |
| `demo.py` | BPE, sizes, init depth stability, zero-shot task transfer, the Bloom filter (~4 seconds) |
| `test_gpt2.py` | 9 quick tests (~0.8 seconds) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/049-Radford-et-al-2019-GPT-2`):
```
python3 -m pytest -q             # ~0.8 seconds
python3 demo.py                  # ~4 seconds
python3 experiments.py --quick
```

---

## 2. `gpt2.py`

### Byte-level BPE
- **`PAT`, `pretokenize`:** contractions, optional space + letters, optional space + digits, optional space + symbols, then whitespace.
- **`ByteBPE(text, n_merges)`:**
  - ids 0–255 are the raw bytes;
  - merges are learned *within* pre-tokenized pieces, weighted by piece frequency;
  - **`encode`** repeatedly applies the earliest-learned merge available;
  - **`decode`** concatenates the byte strings.

### The model
- **`Block`:** x + attention(LN(x)), then x + proj(GELU(fc(LN(x)))).
- **`GPT2(vocab, n_ctx, d, layers, heads, dropout, scaled_init)`:**
  - **init:** N(0, 0.02), and the residual output layers (`attn.out_proj`, `proj`) use 0.02/√(2·layers);
  - **output:** the final LN, then the tied output layer;
  - **methods:** `loss`, `generate` (greedy, top-k, temperature, stop token), `sequence_logprob`.
- **Sizes:**
  - `SIZES` holds the four (layers, d_model) pairs of Table 2;
  - `model_size` = vocab·d + n_ctx·d + layers·(12d² + 13d) + 2d.

### Evaluation helpers
| Function | What it does |
|---|---|
| `per_unit` | perplexity and bits per canonical unit from a total NLL |
| `lambada_predict` | the most probable candidate word, optionally skipping stop words |
| `cloze_choice` | the CBT-style best candidate by full-sentence score |
| `translation_prompt`, `tldr_prompt`, `qa_prompt` | Section 3's prompt formats |

### Overlap
- **`normalize`** keeps lower-case alphanumeric words; **`ngrams`** builds the 8-grams.
- **`BloomFilter(m, k)`:** double hashing with SHA-256, `add`, membership test, and the false-positive estimate.
- **`overlap_fraction`** gives the share of a text's 8-grams found in the filter.

---

## 3. `experiments.py`

**Data:** OpenWebText via the `datasets` package if installed, else Gutenberg. `SHAPES` keeps Table 2's depth with 1/8 of the width.

| Function | Reproduces |
|---|---|
| `e1` | tokens per byte and out-of-vocabulary rates (byte-BPE vs words vs characters) |
| `e2` | train all four sizes; zero-shot bits per byte on held-out and other-domain text |
| `e3` | LAMBADA accuracy vs size, with and without the stop-word filter |
| `e4` | ROUGE-1 of TL;DR vs no hint vs random 3 sentences |
| `e5` | Bloom-filter overlap of the test sets with the training corpus |
| `e6` | 48-layer pre-LN (GPT-2) vs post-LN (GPT) training curves and gradient norms |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_pretokenizer_keeps_categories_apart` | the exact pieces of a mixed string |
| `test_byte_bpe_is_invertible_on_any_unicode_and_never_merges_across_categories` | exact round trips (emoji, CJK, whitespace, empty string); no letter+punctuation tokens; compression |
| `test_model_sizes_table_2` | the module count = the formula; 124 / 355 / 774 / 1558M |
| `test_pre_ln_final_ln_and_scaled_residual_init` | the init stds; causality; the final LN exists |
| `test_generation_top_k_and_sequence_logprob` | top-k never leaves the top k; greedy = argmax; summed log-probs |
| `test_per_unit_conversions` | PPL 8 ↔ 3 bits |
| `test_lambada_stop_word_filter_and_cloze` | the filter skips "the"; cloze picks the best-scored candidate |
| `test_prompts` | the exact prompt strings |
| `test_bloom_filter_overlap` | no false negatives; zero overlap for unrelated text; a tiny false-positive rate |

---

## 5. Try it yourself

1. Train `ByteBPE` with 2,000 merges on a book from paper 048's Gutenberg list. How many bytes per token does it reach?
2. Remove the category rule (use `text.split(" ")` instead of `pretokenize`) and look for "dog."-style tokens.
3. In demo section 4, put the Q/A demonstrations in a different format for some countries ("question : … answer :"). Does a prompt in one format still work for the other?
4. Set `scaled_init=False` and train a 24-layer model in the demo's toy. Is early training less stable?
