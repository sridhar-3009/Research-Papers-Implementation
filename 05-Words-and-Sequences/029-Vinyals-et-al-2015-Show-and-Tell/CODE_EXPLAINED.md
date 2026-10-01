# The code, explained simply

How the code in this folder implements Show and Tell (Vinyals et al. 2015).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `nic.py` | the LSTM of Eqs. 4–8, the NIC model, sampling, beam search with N-best lists, ranking (R@k, median rank), BLEU-n, human BLEU, CIDEr-D, novelty, embedding neighbours |
| `toy.py` | a toy image world: 24 kinds of object (size × colour × shape), fake "CNN features", 5 caption styles per image |
| `experiments.py` | E1–E5 on Flickr8k with fixed GoogLeNet features (heavy, not run here) |
| `demo.py` | trains on the toy world and shows generation, metrics, ranking and embeddings (about 2 seconds) |
| `test_nic.py` | 12 quick tests (about 1 second) |

**Run it** (from `05-Words-and-Sequences/029-Vinyals-et-al-2015-Show-and-Tell`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~2 seconds
python3 experiments.py --quick   # ~20-40 minutes (downloads Flickr8k, ~1 GB)
```

---

## 2. `nic.py`

### `LSTMCell(n_in, n_hid, cell_tanh=False)`
- **Eqs. 4–8 as printed:** one linear map for x and one for m, giving i, f, o, g.
  - c = f ⊙ c + i ⊙ tanh(g);
  - **m = o ⊙ c**.
- `cell_tanh=True` gives the usual m = o ⊙ tanh(c).

### `NIC(vocab, feat_dim, d=512, dropout, image_every_step=False, cell_tanh=False)`
- **Layers:**
  - `img`: the CNN's top layer, mapping the fixed CNN features into the 512-d word space;
  - `We`: the word embeddings;
  - `out`: the softmax layer.
- **`start(feats)`:** feeds x₋₁ = img(feats) from a zero state. The output is thrown away, so p₀ is not scored. Returns the state and the image vector.
- **`step(word, state, v)`:** feeds one word and returns log p(next word). `v` is only used when `image_every_step=True` (the m-RNN-style ablation the paper rejects).
- **`forward(feats, words)`:** teacher forcing over S₀ … S_{N−1}.
- **`log_prob(feats, caps)`:** log p(S | I) per caption, summed over t = 1 … N (Eq. 13), with padding masked out.

### Inference
- **`sample`:** draws a word each step until EOS. It takes a `generator` for reproducibility.
- **`beam_search(model, feat, beam=20)`:** keeps `beam` live hypotheses and moves finished ones aside.
  - It stops once the beam-th best finished score beats every live score. This is safe because log-probabilities only go down.
  - It returns the whole sorted N-best list. beam = 1 is greedy.

### Ranking
- **`score_matrix`:** S[i, j] = log p(caption j | image i).
- **`normalize_for_annotation`:** subtracts log of the mean over images of p(S|I′). This is our reading of "normalized similar to [21]".
- **`ranks`:**
  - annotation: for each image, the rank of its best true caption;
  - search: for each caption, the rank of its image.
- **`recall_report`:** R@k and median rank.

### Metrics
- **`bleu`:** corpus BLEU-n with multiple references (clipped counts, closest-reference brevity penalty).
- **`human_bleu`:** each reference scored against the others (Table 2's "Human").
- **`cider_d`:**
  - TF-IDF n-gram vectors (document frequency over the reference sets);
  - clipped cosine with a Gaussian length penalty (σ = 6);
  - ×10, then ×100 for the scale of Table 1.
- **`novelty`:** the share of generated sentences not seen in training.
- **`nearest_words`:** cosine neighbours in W_e.

---

## 3. `experiments.py`

- **Data:**
  - Flickr8k: 6000 / 1000 / 1000 images, 5 captions each;
  - **fixed GoogLeNet features**, cached in `data/flickr8k/feats_googlenet.pt`;
  - vocabulary: words seen at least 5 times.
- **Training:**
  - SGD with a fixed lr and no momentum (lr 2.0 and clip 5, from im2txt), dropout 0.3, batch 32;
  - the epoch with the best dev **perplexity** is kept.

| Function | Reproduces |
|---|---|
| `e1` | Table 2 (Flickr8k), Table 1's metrics, beam 20 vs greedy |
| `e2` | Table 4: annotation (normalized) and search R@1/5/10, median rank on 1000 × 5000 |
| `e3` | image once vs every step |
| `e4` | Section 4.3.4: best caption in the training set (%), novel share of the 15-best, n-best BLEU agreement |
| `e5` | Table 6: neighbours of car / boy / street / horse / computer |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_lstm_is_eqs_4_to_8_with_m_equal_o_times_c` | the cell, as printed and with `cell_tanh` |
| `test_image_is_fed_once_at_t_minus_1` | v is unused after t = −1, but used in the ablation |
| `test_loss_scores_words_1_to_N_and_ignores_padding` | Eq. 13 by hand; padding has no effect |
| `test_beam_of_one_is_greedy_and_nbest_is_sorted_and_distinct` | beam search |
| `test_sampling_is_reproducible_and_never_emits_eos` | sampling |
| `test_ranking_recall_and_median_rank` | R@k and ranks on a hand-made matrix |
| `test_normalization_changes_annotation_but_not_search` | the normalization only reorders captions |
| `test_bleu_and_human_bleu` | perfect match, clipping, zero overlap, human leave-one-out |
| `test_cider_d_prefers_the_right_caption` | CIDEr-D |
| `test_novelty`, `test_nearest_words_skip_special_tokens` | helpers |
| `test_nic_learns_to_caption_unseen_combinations` | correct captions for held-out (size, colour, shape) combinations |

---

## 5. Try it yourself

1. In the demo, set `image_every_step=True`. Does it overfit the toy world more, or less? (The toy features have little noise to exploit.)
2. Compare `cell_tanh=False` and `cell_tanh=True` on the toy world. How do the cell values grow?
3. Add more caption styles per image. What happens to human BLEU and to the model's BLEU?
4. In `experiments.py`, ensemble 3 models by averaging their log-probabilities inside beam search, as the paper did.
