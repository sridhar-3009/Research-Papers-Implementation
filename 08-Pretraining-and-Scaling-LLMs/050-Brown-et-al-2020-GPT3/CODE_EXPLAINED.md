# The code, explained simply

How the code in this folder implements GPT-3's ideas (Brown et al. 2020).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `gpt3.py` | Table 2.1 and the parameter estimate, 6ND compute, alternating dense / banded attention masks, the Table 2.2 mixture (epochs and sampling), the Pareto quality filter, MinHash fuzzy dedup, n-gram contamination, zero / one / few-shot prompts, the three multiple-choice scoring rules, arithmetic and word-scramble generators, the in-context Dirichlet task with its Bayes-optimal predictor |
| `experiments.py` | a family of small models on a synthetic web of tasks: loss vs compute, K-shot accuracy vs size, in-context curves, a data pipeline on real text, contamination (heavy, not run here) |
| `demo.py` | sizes and compute, mixture epochs, in-context learning vs Bayes, prompts and scoring, data hygiene (~5 seconds) |
| `test_gpt3.py` | 10 quick tests (~0.4 seconds) |

The models reuse paper 049's `GPT2` class, loaded by file path.

**Run it** (from `08-Pretraining-and-Scaling-LLMs/050-Brown-et-al-2020-GPT3`):
```
python3 -m pytest -q             # ~0.4 seconds
python3 demo.py                  # ~5 seconds
python3 experiments.py --quick
```

---

## 2. `gpt3.py`

### Models and compute
| Name | What it is |
|---|---|
| `TABLE_2_1` | the eight models' sizes and hyperparameters |
| `param_estimate(L, d)` | 12 L d² + (vocab + n_ctx) d |
| `training_flops(N, D)` | 6 N D |
| `PFS_DAY` | 8.64·10¹⁹ FLOPs |
| `layer_mask(layer, T, window)` | even layers dense-causal; odd layers attend only within a band of `window` previous positions |

### Data
- **`TABLE_2_2`, `epochs_elapsed`, `sample_source`:** the mixture weights, the passes per dataset, and drawing a source by weight.
- **`pareto_keep(score, rng, alpha)`:** a Lomax sample (inverse CDF: u^(−1/α) − 1) compared with 1 − score.
- **Deduplication:**
  - **`shingles`** builds 5-word shingles;
  - **`minhash`** takes, for each of n hash functions (md5 of "i|shingle"), the minimum over the shingles;
  - **`estimated_jaccard`** is the fraction of agreeing slots;
  - **`dedup`** keeps a document unless it is too similar to one already kept.
- **`ngram_set`, `is_dirty`:** Section 4's contamination test.

### Prompting and scoring
- **`build_prompt(description, demos, query, k, sep, arrow)`** makes zero-, one- or few-shot prompts.
- **`choose(logprob_fn, context, completions, mode)`** scores answers by "sum", "per_token" or "unconditional".

### Tasks
| Function | What it generates |
|---|---|
| `arithmetic_example` | "Q: What is a plus b?" / "A: …" |
| `scramble(word, kind)` | CL, A1, A2, RI and RW word manipulations |
| `dirichlet_sequences`, `bayes_optimal_nll` | the in-context estimation task and its optimal (count + α)/(t + Vα) predictor |

---

## 3. `experiments.py`

**The synthetic web:**
- `web_document` mixes filler sentences with runs of task demonstrations;
- `CharTok` is a character-level tokenizer;
- `train_family` trains six GPT-2-style sizes and records validation loss against compute.

| Function | Reproduces |
|---|---|
| `e1` | a power-law fit of loss vs compute (Figure 3.1) |
| `e2` | zero / one / K-shot accuracy per task per size (`k_shot_accuracy` with greedy decoding) |
| `e3` | in-context loss curves vs size against Bayes |
| `e4` | quality classifier (logistic regression on hashed words) → Pareto filter → MinHash dedup, on a Pride and Prejudice pool with word-salad junk and near-duplicates |
| `e5` | clean vs dirty zero-shot accuracy |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_table_2_1_parameter_counts` | the estimates are within 2%; heads × d_head = d_model except the paper's two inconsistent rows |
| `test_compute_175b` | 3.15·10²³ FLOPs ≈ 3,640 PF-days |
| `test_alternating_dense_and_banded_attention` | the dense and banded masks |
| `test_data_mixture` | 0.44 epochs for Common Crawl and Books2; weights sum to 1.01; sampling frequencies |
| `test_pareto_filter_keeps_by_quality` | P(keep) = (2 − score)⁻⁹ |
| `test_minhash_estimates_jaccard_and_dedups` | the estimate is close to the true Jaccard; the near-duplicate is removed |
| `test_contamination_ngrams` | a 13-gram match counts as dirty; 12 words can't be |
| `test_prompts_and_multiple_choice_scoring` | the exact prompt strings; the three rules give different answers |
| `test_synthetic_tasks` | the arithmetic answer; the scramble invariants |
| `test_bayes_optimal_predictor_is_the_floor` | uniform at the start; better with more context |

---

## 5. Try it yourself

1. In demo section 3, train a 2-layer model. Does it get closer to the Bayes curve?
2. Change α from 0.3 to 3. The distributions become flatter. How much can context help now?
3. In `experiments.py`, double the rate of task demonstrations in `web_document` and see whether few-shot accuracy rises at small sizes.
4. Increase `n_hashes` in `minhash` from 10 to 100 and watch the Jaccard estimate's error shrink.
