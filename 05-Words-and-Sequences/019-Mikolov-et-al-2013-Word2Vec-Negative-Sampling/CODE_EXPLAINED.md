# The code, explained simply

How the code in this folder implements Mikolov et al. (2013).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `word2vec.py` | vocabulary, subsampling, noise distribution, skip-gram pairs, the Skip-gram model with full softmax / hierarchical softmax / negative sampling / NCE (all gradients by hand, NumPy), Huffman tree, phrases, analogies |
| `experiments.py` | Table 1 on text8 + the paper's analogy questions, phrases, vector addition, Figure 2 (heavy, not run here) |
| `demo.py` | A light tour on a tiny made-up corpus (under a second) |
| `test_word2vec.py` | 15 quick tests (under a second) |

**Run it** (from `05-Words-and-Sequences/019-Mikolov-et-al-2013-Word2Vec-Negative-Sampling`):
```
python3 -m pytest -q                 # < 1 second
python3 demo.py                      # < 1 second
python3 experiments.py --quick       # 2M words, NEG/NCE only: ~5-15 minutes (downloads 31 MB)
python3 experiments.py --only e1     # full text8, all 8 models (HS is slow)
```

---

## 2. `word2vec.py`

### Data
- `Vocab(tokens, min_count=5)`: words sorted by frequency, `index` (word → id), `counts`, and `encode(tokens)` → ids (unknown words dropped).
- `keep_probability(counts, t)`: `min(1, sqrt(t / f))` (Eq. 5). `subsample` keeps each occurrence with that probability.
- `noise_distribution(counts)`: `counts ** 0.75`, normalized.
- `skipgram_pairs(ids, c)`: all (center, context) pairs.
  - It loops over distances j = 1..c and both directions, so it is vectorized.
  - With `dynamic=True`, each center word gets its own window size, uniform in 1..c, so near words are used more.

### `SkipGram(vocab_size, dim, counts)`
| Array | Meaning |
|---|---|
| `v_in` (W × d) | input vectors v_w: the word vectors. Small random start. |
| `v_out` (W × d) | output vectors v′_w (softmax, NEG, NCE). Start at zero, as in the C code. |
| `v_node` (W−1 × d) | one vector per inner node of the Huffman tree (HS) |

**Negative sampling: `neg_step(w_in, w_out, negatives, lr)`**
```python
targets = [w_out, n_1, ..., n_k];  labels = [1, 0, ..., 0]
s = sigmoid(v_out[targets] @ v)                  # predicted "is this a real context word?"
g = labels - s                                   # how wrong each prediction is
v_in[w_in]      += lr * (g @ u)                  # move the word vector
v_out[targets]  += lr * g[:, None] * v           # move the output vectors
```
That's the gradient of Eq. 4 written out. `np.add.at` handles a negative that appears twice. `neg_step_batch` does the same for a batch of B pairs with `einsum`; `shift=log(k·P_n)` turns it into NCE.

**Hierarchical softmax: `hs_log_prob` / `hs_step`**
For the target word's path (`paths[w]` = inner-node ids, `codes[w]` = 0/1 branches): `sign = 1 − 2·bit`, `p = Π σ(sign · v_node[n] · v)`. The update has the same form as NEG: g = (1 − σ)·sign.

**`nce_step`:** like `neg_step`, but the logit is `u·v − log(k·P_n(w))`.

### `huffman_codes(counts)`
The classic algorithm with a heap: repeatedly merge the two least frequent nodes. Then walk up from every leaf to the root, recording the inner nodes and the branch bits, and reverse to get root → leaf order.

### Phrases
- `phrase_scores(tokens, delta)`: Eq. 6 for every adjacent pair.
- `merge_phrases(tokens, threshold)`: a left-to-right pass that joins high-scoring pairs into `a_b`. Run it again for longer phrases.

### Analogies
- `nearest(V, query, k, exclude)`: cosine similarity on normalized vectors.
- `analogy(V, a, b, c)`: `nearest(V[b] − V[a] + V[c])`, excluding a, b and c (on normalized vectors, like the original evaluation code).

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Table 1: NEG-5, NEG-15, HS, NCE-5 × (no subsampling, 10⁻⁵); training minutes; syntactic / semantic / total accuracy on `questions-words.txt` (the questions whose 4 words are in the vocabulary). Saves the NEG-15 + subsampling vectors. |
| `e2` | 2 passes of phrase merging on text8, train NEG-15, show the most frequent phrases and their neighbours |
| `e3` | Table 5: nearest words to vec(a) + vec(b) |
| `e4` | Figure 2: PCA of 11 country/capital pairs |

The learning rate decays linearly from 0.025 to about 0, over one epoch, as in the C code. NEG and NCE are batched (1024 pairs); HS goes one pair at a time (slow; `--skip-hs` skips it).

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_full_softmax_sums_to_one` | Eq. 2 |
| `test_hierarchical_softmax_sums_to_one` | Eq. 3 ("it can be verified") |
| `test_huffman_tree_gives_short_codes_to_frequent_words` | short codes, prefix-free, ≤ log W + 1 |
| `test_negative_sampling_step_is_gradient_descent_on_eq_4` | the NEG update = −lr × numerical gradient |
| `test_hierarchical_softmax_step_is_gradient_descent` | the same for HS |
| `test_batched_negative_sampling_equals_single_steps` | batch = single |
| `test_batched_nce_equals_single_nce_step` | NCE via the shift |
| `test_subsampling_formula` | Eq. 5 |
| `test_three_quarter_power_flattens_the_unigram` | 100 → 31.6 |
| `test_skipgram_pairs_fixed_window` | the exact pairs |
| `test_dynamic_window_prefers_near_words` | dynamic window |
| `test_phrase_detection_eq_6` | "new york" merged, random pairs not |
| `test_analogy_with_linear_structure` | man : woman :: king : queen |
| `test_negative_sampling_learns_co_occurrence` | words of the same topic end up close |
| `test_vocab_min_count` | min_count |

---

## 5. Try it yourself

1. Implement **CBOW** (the other model from the earlier paper): predict the center word from the **average** of its context vectors. Compare it with skip-gram on `e1`.
2. Train with the plain unigram (power 1.0) and the uniform distribution (power 0) as noise, and check the paper's claim that 3/4 is best.
3. Verify **Levy & Goldberg**: on a small corpus, compare `v_in @ v_out.T` after training with the PMI matrix minus log k.
4. Use `v_in + v_out` as the word vector (as GloVe does) and see if analogy accuracy improves.
