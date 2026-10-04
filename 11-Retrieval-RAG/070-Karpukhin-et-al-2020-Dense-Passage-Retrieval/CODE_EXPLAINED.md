# The code, explained simply

How the code in this folder implements Dense Passage Retrieval (Karpukhin et al. 2020).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `dpr.py` | toy Wikipedia and questions; Okapi BM25; vocabulary and bag-of-words encoders; dual encoder; in-batch loss; BM25 hard negatives; training with each negative scheme; dense index; top-k accuracy; hybrid; λ tuning; rule-based reader; exact match |
| `experiments.py` | Table 3 with equal steps, sample efficiency, similarity functions, initialisation, real BERT DPR on Natural Questions |
| `demo.py` | corpus, in-batch example, retrieval table, training schemes, sample efficiency, end-to-end exact match, speed (~13 seconds) |
| `test_dpr.py` | 3 quick tests (~1 second) |

**Run it** (from `11-Retrieval-RAG/070-Karpukhin-et-al-2020-Dense-Passage-Retrieval`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~13 seconds
python3 experiments.py --quick
```

---

## 2. `dpr.py`

### The corpus
- **`make_world`:**
  - 500 people with three-syllable invented names;
  - 4 facts each (values drawn from `VALUES`), each written with one of two `PASSAGE_TEMPLATES`;
  - 1,500 `FILLER` passages mentioning two people and a city.
  - It records `gold[(person, relation)]`, the index of the passage that states that fact.
- **`make_questions`:**
  - 80% of people are "seen"; 70% of their facts are used for training, each with a random question style;
  - the test sets are held-out facts and unseen people, each in the `OVERLAP_Q` and `PARAPHRASE_Q` styles.

### BM25
- **`BM25(docs, k1=0.9, b=0.4)`:** pre-computes term frequencies, idf and an inverted index.
- **`scores(query)`:** adds the BM25 term for each query word over the passages that contain it.

### The dual encoder
| Name | What it does |
|---|---|
| `Vocab.encode` | flat token ids plus offsets for `nn.EmbeddingBag` |
| `Encoder` | mean of word embeddings, then a linear layer |
| `DPR` | **two independent encoders** `eq` and `ep`; `same_init=True` gives them the same starting weights (like two copies of one pre-trained BERT) |
| `in_batch_loss(q, p_pos, p_extra)` | cross-entropy over S = Q [P_pos; P_extra]ᵀ with targets 0 … B−1 |
| `bm25_negative` | the top BM25 passage without the answer |

### `train_dpr(negatives=...)`
- **"gold":** in-batch only.
- **"gold+bm25":** in-batch plus `n_extra` BM25 negatives per question, shared across the batch. This is the paper's best.
- **"random" / "bm25":** each question gets `n_extra` negatives of that type and no in-batch negatives.
- **Defaults:** batch 128, 40 epochs, Adam lr 1e-2 (higher than the paper's 1e-5 because our encoders are tiny and start from scratch), d = 128.

### Search and evaluation
| Name | What it does |
|---|---|
| `DenseIndex` | encodes all passages once; `scores` is a single matrix product (exact maximum inner product search, which FAISS approximates at scale) |
| `top_k_accuracy` | does any top-k passage contain the answer word? |
| `hybrid`, `tune_lambda` | BM25 + λ·DPR, with λ picked on development questions |
| `rule_reader`, `exact_match` | finds the person (the word that isn't template vocabulary) and the relation (by its keywords), then reads the value from the first retrieved passage that is about that person and states that relation |

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | Table 3 schemes with an equal number of training steps, 3 seeds |
| `e2` | 50 … 1,120 training questions (equal steps) vs BM25 |
| `e3` | dot product vs cosine vs −L2 similarity in the training loss and at search time |
| `e4` | shared vs unrelated initialisation of the two encoders |
| `e5` | two `bert-base-uncased` encoders ([CLS]), in-batch + 1 BM25 negative, Adam 1e-5, on Natural Questions pairs; top-1/5/20/100 vs BM25 on a 10k-passage sub-corpus |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_bm25_formula` | BM25 matches a hand computation (idf, tf saturation, length normalisation) |
| `test_in_batch_loss_is_cross_entropy_over_shared_negatives` | the loss equals manual log-softmax over [gold; extra] columns; aligned vectors give ~0 loss |
| `test_world_reader_and_training_beats_bm25_on_paraphrase` | the reader parses passages; a trained DPR beats BM25 top-1 on paraphrased questions by > 20 points |

---

## 5. Try it yourself

1. Set `same_init=False`. How much worse is DPR on held-out facts, and on unseen people?
2. Make filler passages copy the question wording ("X's hometown was discussed at a party"). Does BM25 get worse on paraphrases?
3. Give the passage templates the same words as the paraphrase questions. Does DPR's advantage vanish?
4. Change BM25's k1 and b. How sensitive is it in this corpus?
5. Run `experiments.py --only e1`. With equal steps, does batch 128 now beat batch 32?
