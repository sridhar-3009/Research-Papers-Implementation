# The code, explained simply

How the code in this folder implements CLIP (Radford et al. 2021).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `clip.py` | encoders (causal text Transformer with the [EOS] feature, bag-of-words text encoder, small CNN with optional attention pooling), the CLIP model (linear projections, L2 normalisation, learned clipped logit scale), the symmetric loss, zero-shot weights with prompt ensembling, linear probe, effective robustness, the bag-of-words predictive loss |
| `experiments.py` | CIFAR with noisy synthetic captions: objective comparison, prompts, zero-shot vs few-shot probes, robustness under corruptions, ablations (heavy, not run here) |
| `demo.py` | coloured-MNIST CLIP, zero-shot digits and colours, held-out combinations, contrastive vs predictive, the paper's robustness table (~5 seconds) |
| `test_clip.py` | 8 quick tests (~1 second) |

**Run it** (from `08-Pretraining-and-Scaling-LLMs/053-Radford-et-al-2021-CLIP`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~5 seconds
python3 experiments.py --quick
```

---

## 2. `clip.py`

### Encoders
| Name | What it is |
|---|---|
| `TextTransformer(vocab, d, layers, heads, ctx)` | token + learned position embeddings, pre-LN causal Transformer, final LN; returns the state at `eos_pos` (the paper's defaults: 512 wide, 12 layers, 8 heads, 76 tokens) |
| `CBOW` | mean of word embeddings, padding ignored (Figure 2's bag-of-words encoder; also what the demo uses because it is fast) |
| `AttentionPool` | one multi-head attention whose query is the mean feature and whose keys/values are [mean; every position] (the modified ResNet's head) |
| `SmallConvNet` | two stride-2 convs, then mean pooling or `AttentionPool` |

### The model
`CLIP(image_encoder, text_encoder, d_embed)` holds:
- `W_i`, `W_t`: linear projections with no bias;
- `t`: the log of the logit scale, initialised to log(1/0.07).

| Method | What it does |
|---|---|
| `logit_scale()` | exp(t), clamped at 100 |
| `encode_image`, `encode_text` | project, then L2-normalise |
| `forward(x, ids, eos)` | the N×N matrix of scaled cosines |
| `clip_loss(logits)` | (CE over rows + CE over columns) / 2, diagonal targets |

### Zero-shot and evaluation
- **`zero_shot_weights(model, encode_fn, classnames, templates)`:**
  1. formats every template with the class name and embeds it;
  2. averages the normalised embeddings, then renormalises;
  3. returns one row per class.
- **`zero_shot_predict(model, images, W)`:** argmax of image embedding · Wᵀ.
- **`linear_probe(...)`:** L2-regularised logistic regression with L-BFGS, as the paper does with scikit-learn.
- **`effective_robustness(baseline_pairs, id_acc, ood_acc)`:**
  1. fits logit(OOD) = w·logit(ID) + c by least squares;
  2. returns (OOD − predicted, predicted).
- **`bag_of_words_loss(features, head, ids, vocab)`:** cross-entropy between softmax(head(features)) and the caption's normalised word-count distribution.

---

## 3. `experiments.py`

**Captions:**
- a random template from `TEMPLATES`;
- a synonym of the class name 30% of the time;
- 0–3 distractor words;
- sometimes a colour word computed from the image.

The goal is noisy, web-like text.

**Pieces:**
- **`Tok`:** a word-level tokenizer with ids for pad, <s> and <eos>; it returns (ids, eos positions).
- **`Captioner`:** the image feature is a prefix token for a causal Transformer that predicts the caption. For zero-shot it picks the class whose prompt has the lowest NLL.
- **`build(kind, ...)`:** `captioner`, `bow-predict`, `bow-contrastive` (CBOW + CLIP loss) or `clip` (Transformer + CLIP loss). Options: a nonlinear projection, or a fixed temperature.

| Function | Reproduces |
|---|---|
| `e1` | zero-shot accuracy vs images seen, for all four objectives (Figure 2) |
| `e2` | bare name / one prompt / 12-template ensemble |
| `e3` | zero-shot vs 1/2/4/8/16-shot and full linear probes |
| `e4` | a same-size supervised CNN vs CLIP linear probe vs zero-shot CLIP, under blur, noise, greyscale and sketch; drop from clean |
| `e5` | learned vs fixed T = 0.07 / 1.0, nonlinear projection, batch 64 vs 512 at equal images seen |

`trained_clip` caches the CLIP checkpoint, so E2–E4 train it only once.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_logits_are_scaled_cosines_and_temperature_is_clipped` | unit-norm embeddings, logits = cos/0.07, clamp at 100 |
| `test_symmetric_loss_matches_the_pseudocode` | equals Figure 3; log N at chance; ≈ 0 when perfect |
| `test_zero_shot_prompt_ensemble_is_mean_of_normalised_embeddings` | the ensemble rule, unit-norm class weights |
| `test_text_transformer_uses_eos_and_is_causal` | tokens after [EOS] don't affect the feature |
| `test_attention_pool_and_encoders` | shapes; CBOW ignores padding |
| `test_linear_probe_and_effective_robustness` | the probe separates a simple problem; the trend fit predicts ~0.49 and finds positive effective robustness |
| `test_bag_of_words_loss` | matches a hand-computed value with repeated words |
| `test_training_learns_to_match_pairs` | loss falls from log 10 to under 0.3 |

---

## 5. Try it yourself

1. In `demo.py`, swap `CBOW` for `TextTransformer` (with `eos_pos`). Does the bare class name still do as well as the ensemble?
2. Fix the temperature at 1 (`model.t.requires_grad_(False)` after filling it with 0). How far does zero-shot accuracy fall?
3. Shrink the batch from 128 to 16. Fewer negatives: does zero-shot accuracy fall at the same number of images seen?
4. Make the held-out combinations bigger, e.g. hold out every red digit except 0. Can the model still name the colour of a red 5?
5. Add a linear probe on `model.encode_image` features with 1, 4 and 16 labelled examples per digit, and find where it overtakes zero-shot.
