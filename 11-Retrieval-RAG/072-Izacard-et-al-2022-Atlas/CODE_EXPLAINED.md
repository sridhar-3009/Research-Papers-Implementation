# The code, explained simply

How the code in this folder implements Atlas (Izacard et al. 2022).
Read [EXPLAINED.md](EXPLAINED.md) first.

The code re-uses paper 070's `dpr.py` (names, templates and values) through `importlib`.

---

## 1. The files

| File | What it is |
|---|---|
| `atlas.py` | corpus with every fact written twice; masked-LM and QA examples; vocabulary; retriever; crop training; index; FiD reader; the four retriever losses; joint training; evaluation; recall; product quantisation |
| `experiments.py` | the loss ablation from two starts, fine-tuning strategies (incl. re-ranking), shots curve, k, partial index swaps, PQ grid, and real Contriever + FLAN-T5 FiD |
| `demo.py` | corpus and retriever, 64-shot table, reader teaching a weak retriever, the targets on one example, index swap, PQ (~24 seconds) |
| `test_atlas.py` | 3 quick tests (~2 seconds) |

**Run it** (from `11-Retrieval-RAG/072-Izacard-et-al-2022-Atlas`):
```
python3 -m pytest -q             # ~2 seconds
python3 demo.py                  # ~24 seconds
python3 experiments.py --quick
```

---

## 2. `atlas.py`

### Data
| Name | What it does |
|---|---|
| `make_world` | `gold[(person, relation)]` lists the two passage indices; `fact_of[i]` maps a fact passage back to its fact |
| `mlm_examples` | (passage with the value replaced by `<mask>`, value, the passage's own index, which is excluded from retrieval) |
| `qa_examples` | questions in the overlap or paraphrase style |
| `Vocab` | word ids with padding; every possible answer word is included |

### Retriever
- **`Retriever`:** separate query and document towers, each embedding → softmax-weighted pooling with a learned per-word weight → linear. The towers start identical.
  - `init_idf` sets the weights to 0.8·idf, with 0 for words absent from the corpus such as `<mask>`. Without that, `<mask>` dominated every query.
  - `q_params` / `d_params` select a tower's parameters.
- **`contriever_pretrain`:** in-batch contrastive loss on two random crops of each passage (temperature 0.1).
- **`Index`:** stores document vectors. `refresh` re-encodes them (Atlas's re-indexing); `topk` can exclude one index per query.

### Reader
- **`FiDReader`:**
  - each document token gets [token embedding + position; query vector] → MLP (the per-document encoding);
  - the decoder vector u comes from the query;
  - the attention α is a softmax over **all** documents' tokens together;
  - the output is logits from [Σ α·v; u].
  - `doc_keep` masks out documents, which PDist (one document) and LOOP (all but one) use.
  - With K = 0 it is closed-book.
- **`answer_logp`:** log p(answer) plus α and ‖v‖.

### Retriever training
- **`retriever_target(loss)`:** computed without gradient.
  - ADist: Σ_tokens α‖v‖, normalised;
  - PDist: softmax of single-document log-likelihoods;
  - LOOP: softmax of minus the leave-one-out log-likelihoods.
- **`retriever_loss`:** KL(target ‖ softmax(s/θ)) over the K retrieved documents. For EMDR² it is −log Σ_k p_LM·p_retr, with p_LM held fixed. `train_docs` decides whether gradients reach the document tower.
- **`train_atlas`:**
  - the reader's NLL plus the retriever loss;
  - separate learning rates;
  - `train_q` / `train_d` choose which towers learn (query-side = `train_q` only);
  - `reindex_every` refreshes the index;
  - `loss=None` means a fixed retriever.
- **`evaluate`, `recall`:** exact match, and whether a passage stating the asked fact (other than the query passage) is in the top k.

### Compression
- **`kmeans`, `product_quantize(D, m, bits)`:** returns the reconstructed index and bytes per vector.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | 7 settings × 2 retriever starts × 3 seeds: masked-LM accuracy, recall, 64-shot EM |
| `e2` | fixed, query-side, full + re-index, full with a stale index, and top-20 re-ranking (`rerank_topk`) |
| `e3` | 16 / 64 / 256 / 1,024 shots, Atlas vs closed-book |
| `e4` | k ∈ {1, 3, 5, 10, 20}: EM and time |
| `e5` | 10 / 50 / 100% of people changed; matched vs mismatched index |
| `e6` | PQ grid: recall and EM |
| `e7` | `facebook/contriever` (mean pooling) over a simple-Wikipedia slice, FLAN-T5 FiD (per-passage encoding, concatenated states), 64 NQ examples, PDist on the query encoder |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_world_mlm_and_fid_shapes` | facts are written twice; the masked example's answer is in its excluded passage; FiD attention is one softmax across all documents; masked documents get zero attention |
| `test_retriever_targets_are_distributions_and_emdr_formula` | ADist / PDist / LOOP targets are distributions; PDist equals the softmax of single-document log-likelihoods; the EMDR² loss equals −logsumexp(log p_LM + log p_retr) |
| `test_pq_and_pretraining_learns` | more PQ bits give less reconstruction error; masked-LM pre-training raises accuracy by > 30 points |

---

## 5. Try it yourself

1. Start from `Retriever(V)` with no idf (all weights 0). Can joint pre-training rescue it, and with which loss?
2. Use `train_d=True, reindex_every=0`, i.e. a stale index. How quickly does recall drop?
3. Raise the temperature θ in `retriever_loss` to 5. Does PDist from the weak start begin to help?
4. Fine-tune on 1,024 examples instead of 64. Does the gap to closed-book shrink?
5. Pre-train on only one template per fact, so there is no redundancy. What happens to masked-LM accuracy?
