# The code, explained simply

How the code in this folder implements RAG (Lewis et al. 2020).
Read [EXPLAINED.md](EXPLAINED.md) first.

The code re-uses paper 070's `dpr.py` (toy Wikipedia, BM25, dual encoder) through `importlib`.

---

## 1. The files

| File | What it is |
|---|---|
| `rag.py` | "describe" questions; changed world for hot-swaps; the tiny generator; a fixed reader; dense and BM25 retrievers; both marginalisations; training; RAG-Token and RAG-Sequence (thorough/fast) decoding; exact match; retrieval recall; token posteriors; DPR pre-training |
| `experiments.py` | Table 6 sweep, test-time k, warm-up fragility, trained two-fact task, partial hot-swaps, real Hugging Face RAG on NQ |
| `demo.py` | numeric example, QA table, k sweep, hot-swap, RAG-Token vs RAG-Sequence on two facts (~8 seconds) |
| `test_rag.py` | 3 quick tests (~1 second) |

**Run it** (from `11-Retrieval-RAG/071-Lewis-et-al-2020-RAG`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~8 seconds
python3 experiments.py --quick
```

---

## 2. `rag.py`

### Data
- **`describe_questions`:** "where was X born and what does X work as ?" → "city job".
- **`changed_world`:** a copy of the world where a fraction of people get a new job, instrument and team, with their passages rewritten.

### The generator
- **`Generator`:** `step_logits` = MLP([mean emb(document words); mean emb(question words); emb(previous token); emb(position)]).
- **`token_logps`:** teacher-forced log p(y_i | x, z, y_<i), shape (N, L).
- **`FixedReader`:** no parameters. If the document states the relation needed for answer position i, it puts 0.9 on that value; otherwise it is uniform over that relation's values. It is used to compare the marginalisations cleanly.

### Retrievers
- **`DenseRetriever`:**
  - copies DPR's question encoder (trainable) and pre-computes the frozen document matrix D;
  - `topk` selects with no gradient but returns the scores **with** gradient;
  - `swap_index` re-encodes a new corpus.
- **`BM25Retriever`:** BM25 scores used as logits, as in the paper's ablation. They are cached, since they never change.

### RAG
| Function | What it does |
|---|---|
| `marginal_logp(mode)` | log_softmax over the top-k scores gives log p(z\|x); the generator's token log-probs are (B, k, L). Sequence: logsumexp_k(log p(z) + Σ_L). Token: Σ_L logsumexp_k(log p(z) + tok). `retriever=None` is closed-book. |
| `train_rag` | Adam on −log p(y\|x). Separate learning rates for the generator and the question encoder; the first `warmup` steps leave the retriever untouched. |
| `decode` | Token: greedy on the per-step mixture. Sequence: one greedy candidate per document, then thorough re-scoring under all documents, or fast (only under its own). |
| `exact_match`, `retrieval_recall` | answer accuracy, and whether the gold passage(s) are in the top k |
| `token_posteriors` | p(z \| x, y_i, y_<i) per answer token (Figure 2) |
| `pretrain_dpr` | DPR trained on the overlap-style version of every training fact |

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | 4 retrievers × 2 training modes × 3 decoders, 3 seeds |
| `e2` | test-time k ∈ {1, …, 20} for training k ∈ {5, 10} |
| `e3` | warm-up {0, 100, 200, 400} × retriever lr {3e-3, 1e-3, 3e-4} |
| `e4` | two-fact task with a trained generator in a 2,000-person world |
| `e5` | 10% / 50% / 100% of people changed; matched vs mismatched index |
| `e6` | Hugging Face `facebook/rag-sequence-nq` and `rag-token-nq` with the dummy index on `nq_open` |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_marginalisations_match_formulas` | `marginal_logp` equals the hand-computed RAG-Sequence and RAG-Token sums, and Token > Sequence when each document supports one token |
| `test_retriever_gradients_flow_and_index_swap` | the generator loss gives non-zero gradients to the question encoder; `swap_index` and `changed_world` behave (birthplaces fixed, other facts changed) |
| `test_fixed_reader_rag_token_beats_rag_sequence_on_two_facts` | with the birth and job passages retrieved, RAG-Token decodes every answer and RAG-Sequence fewer than half |

---

## 5. Try it yourself

1. Set `warmup=0` in `train_rag` and watch retrieval recall during training.
2. Train with `mode="sequence"` and decode with `mode="token"` (and vice versa). Does it matter for single-token answers?
3. Lower `FixedReader`'s confidence to 0.5. How does the RAG-Token vs RAG-Sequence gap change?
4. Change only 10% of people in `changed_world`. How does the old index do on unchanged people?
5. Give the closed-book generator twice the hidden size. Does it memorise more training facts? Does it ever get held-out facts?
