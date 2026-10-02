# The code, explained simply

How the code in this folder implements DALL·E (Ramesh et al. 2021).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `dalle.py` | pixel mapping ϕ, the logit-Laplace likelihood, the dVAE (ResNet encoder/decoder, Gumbel-softmax ELB, KL to uniform, cosine schedules, codebook perplexity), row / column / conv / dense attention masks and the layer schedule, the DALL·E transformer (text + learned padding + row/column image embeddings, the 1/8 + 7/8 loss, generation with an optional image prefix), reranking, the parameter-count formula, the PowerSGD compression rate and a PowerSGD compressor with error feedback, an fp16 underflow measure |
| `experiments.py` | a synthetic captioned-shapes world; the dVAE β ablation, zero-shot composition, sparse vs dense, loss weights, padding, CLIP-style reranking, PowerSGD rank (heavy, not run here) |
| `demo.py` | sizes, logit-Laplace vs Laplace, tiny dVAE β study, text→image composition and completion, reranking, fp16 scaling and PowerSGD (~21 seconds) |
| `test_dalle.py` | 9 quick tests (~0.5 seconds) |

**Run it** (from `07-Generative-Models/047-Ramesh-et-al-2021-DALL-E`):
```
python3 -m pytest -q             # ~0.5 seconds
python3 demo.py                  # ~21 seconds
python3 experiments.py --quick
```

---

## 2. `dalle.py`

### Pixels
| Function | What it does |
|---|---|
| `phi`, `phi_inv` | Eq. 3: [0, 255] ↔ (0.1, 0.9) |
| `logit_laplace_log_prob(x, μ, ln b)` | Eq. 2 in log form |

### dVAE
- **`ResBlock`:** bottleneck resblock whose residual branch is multiplied by `post_gain = 0.1`.
- **`DVAE(K, C, width, down)`:**
  - **encoder:** 7×7 conv → [resblock, max-pool] × log₂(down) → resblock → 1×1 conv to K logits;
  - **decoder:** 1×1 conv from K channels → [resblock, nearest upsample] → resblock → 1×1 conv to 6 maps (μ and ln b per channel);
  - **`tokens`** returns argmax codes;
  - **`decode_tokens`** decodes one-hot codes;
  - **`elbo_terms(x, τ)`** returns (logit-Laplace log-likelihood, KL to uniform, the relaxed codes from `F.gumbel_softmax`);
  - **`reconstruct`** returns sigmoid(μ), the ϕ-space image (apply `phi_inv` for 0–255).
- **Helpers:**

  | Function | What it computes |
  |---|---|
  | `kl_to_uniform` | Σ q log q + log K per position |
  | `dvae_loss(model, x, τ, β)` | the negative relaxed ELB per pixel value |
  | `cosine_schedule` | the cosine annealing used for β, τ and the learning rate |
  | `codebook_perplexity` | exp(entropy) of code usage |

### Attention masks
- **`image_mask(kind, H, W)`:**
  - **row:** causal and within W positions back;
  - **column:** causal and in the same column;
  - **conv:** causal and within a k×k neighbourhood;
  - **dense:** causal.
- **`full_mask`** adds causal text, and image-to-all-text.
- **`layer_kinds(n)`:** column when (i − 2) mod 4 = 0, otherwise row; conv last.

### The transformer
**`DALLE(text_vocab, image_vocab, T, H, W, d, layers, heads, sparse)`:**
- **`embed`:**
  - text embeddings, with positions past `text_len` replaced by the per-position learned padding;
  - plus text positions;
  - the image input is shifted (`[start-of-image, z₁, …]`) and gets row + column embeddings.
- **`forward`** runs the pre-LN blocks, each with its mask from `kinds`, and returns the text logits (predicting text[1:]) and the image logits (predicting every image token).
- **`loss`:**
  - text cross-entropy over real (non-padding) targets, normalised by their count;
  - image cross-entropy;
  - weighted 1/8 and 7/8.
- **`generate(text, len, prefix, temperature, top_k)`:** samples image tokens one at a time. A `prefix` gives the first image tokens (completion and image-to-image).

### Scale helpers
| Function / class | What it does |
|---|---|
| `rerank` | top-k by score |
| `transformer_params` | ≈ 12 d² · layers + embeddings |
| `powersgd_compression_rate` | 1 − 5r/(8d) |
| `PowerSGD(shape, r)` | fixed random Q; P = qr(M Q), Q' = Mᵀ P; returns P Q'ᵀ and keeps M − P Q'ᵀ as the error for the next step |
| `fp16_underflow_fraction` | the share of nonzero values that become 0 in fp16 |

---

## 3. `experiments.py`

**The synthetic world:**
- `draw` makes 32×32 images of 1–2 coloured shapes in quadrants;
- `caption` writes "a red circle at top left and …";
- `HELD_OUT` lists colour–shape pairs never seen in training;
- `scene_correct` checks a generated image pixel by pixel.

| Function | Reproduces |
|---|---|
| `e1` | dVAE with β ∈ {0, 1, 6.6}: pixel error and perplexity |
| `e2` | prior on dVAE tokens: accuracy on seen vs held-out combinations |
| `e3` | sparse vs dense attention (validation loss) |
| `e4` | text loss weight 1/8 vs 1/2 |
| `e5` | learned vs fixed-zero padding on 4-object captions (never seen) |
| `e6` | `TinyCLIP` contrastive reranking, N = 1…64 |
| `e7` | PowerSGD ranks vs uncompressed |
| `e8` | an MS-COCO hook (needs a local copy) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_logit_laplace_is_a_density_on_the_unit_interval` | integrates to 1; equals the sigmoid-of-Laplace change of variables |
| `test_phi_mapping` | 0 / 127.5 / 255 → 0.1 / 0.5 / 0.9, and back |
| `test_kl_to_uniform_and_perplexity` | 0 for uniform, log K for one-hot; perplexity K or 1 |
| `test_gumbel_softmax_hardens_as_tau_falls` | τ = 1/16 is nearly one-hot; frequencies follow softmax |
| `test_dvae_shapes_and_loss` | token grid shape; relaxed codes sum to 1; finite loss; schedule endpoints |
| `test_attention_masks_match_figure_11` | row / column / conv sets for token 9; causal; text rules; the layer schedule |
| `test_transformer_is_causal_and_uses_padding_tokens` | logits ignore their targets; tokens past the caption are ignored; the loss weighting; generation shape |
| `test_scale_numbers_from_the_paper` | ≈ 12B parameters; 192×; Table 1 rates |
| `test_powersgd_error_feedback_and_underflow` | 512 numbers sent instead of 4096; sent = true − final error; error feedback beats none; the underflow count; rerank |

---

## 5. Try it yourself

1. In demo section 4, train with `sparse=False` (dense attention). Does zero-shot composition change?
2. Add a "both objects" caption ("a red X at top left and a blue Y at bottom right") to the toy, and test variable binding: does the colour stick to the right place?
3. In demo section 3, try β = 1 (the "correct" ELB). Where does it fall between 0 and 6.6?
4. Change `PowerSGD` to recompute Q each step (power iteration) and compare the error to the fixed-Q version.
