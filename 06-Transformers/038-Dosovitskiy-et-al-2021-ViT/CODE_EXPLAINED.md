# The code, explained simply

How the code in this folder implements the Vision Transformer (Dosovitskiy et al. 2021).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `vit.py` | patch cutting, the pre-LN encoder block, the ViT (with an optional CNN stem for the hybrid), zero-init fine-tuning head, 2-D position interpolation, Figure 7 analysis tools, masked-patch-prediction targets and corruption |
| `experiments.py` | CIFAR-10 versions of the paper's claims: ViT vs ResNet as data grows, patch size, hybrid, higher-resolution fine-tuning, Figure 7, self-supervision (heavy, not run here) |
| `demo.py` | patch counts, model sizes, CNN vs ViT on a toy task, position similarity, attention distance, mean-colour targets (~6 seconds) |
| `test_vit.py` | 8 quick tests (~1 second) |

**Run it** (from `06-Transformers/038-Dosovitskiy-et-al-2021-ViT`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~6 seconds
python3 experiments.py --quick   # ~1 hour on CPU
```

---

## 2. `vit.py`

### Patches
- **`patchify(img, P)`:** turns (B, C, H, W) into (B, N, P·P·C).
  - The reshape and permute put the patches in row-major order, each flattened as (row, column, channel).
- **`unpatchify`** is the exact inverse (tested).

### The model
- **`Block`:** pre-LN layer, `z + MSA(LN(z))` then `z + MLP(LN(z))` (Eqs. 2–3).
  - The MLP is Linear → GELU → Linear.
  - With `keep_attention=True`, it stores the per-head attention matrix in `last_attention`.
- **`CNNStem`:** stride-2 conv layers whose output grid is read as 1×1 patches (the hybrid).
- **`ViT(image, patch, C, D, layers, heads, mlp, classes, dropout, head, hybrid_downsample)`:**
  - **`tokens`:** Eq. 1 (patches → E, prepend [class], add positions);
  - **`forward`:** the blocks, a final LN, and the head on token 0 (Eq. 4);
  - **`head="mlp"`:** the pre-training head (tanh hidden layer);
  - **`head="linear"`:** zero-initialised;
  - **`new_head(K)`:** swaps in a zero-init D × K layer for fine-tuning;
  - **`resize_positions(g)`:** bicubic 2-D interpolation of the patch position grid to g × g, keeping the [class] position.

### Analysis (Figure 7)

| Function | What it computes |
|---|---|
| `position_similarity` | the cosine-similarity matrix of the patch position embeddings |
| `embedding_filters_pca` | SVD of the centred embedding weights; returns the top principal directions (each reshapes to P × P × C) |
| `mean_attention_distance` | per layer and head, Σ_j A_ij · pixel distance(i, j), averaged over queries and images ([class] excluded, weights renormalized) |

### Self-supervision
- **`mean_colour_targets(img, P, bits=3)`:** each patch's mean RGB, quantised to 8 levels per channel, gives one of 512 classes.
- **`corrupt_patches(x, rate, probs)`:** BERT-style corruption.
  - It returns the corrupted patches (masked ones set to 0 for the caller to fill with a learnable [mask]), which patches were chosen, and which were masked.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figures 3–4: ViT vs a GroupNorm ResNet trained from scratch on 5/20/50/100% of CIFAR-10 |
| `e2` | patch 8 vs 4 (16 vs 64 tokens): accuracy and time |
| `e3` | Figure 5's hybrid vs pure ViT |
| `e4` | Section 3.2: fine-tune at 48×48 with interpolated positions vs at 32×32 |
| `e5` | Figure 7 plots from E1's model |
| `e6` | Section 4.6: masked patch prediction, then fine-tuning, vs from scratch |

Training uses AdamW (0.9, 0.999), weight decay 0.1, linear warm-up and decay, gradient clipping, crops and flips.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_patchify_is_a_reshape_of_the_image` | patch order (row-major, (row, col, channel)) and the exact inverse |
| `test_model_sizes_match_table_1` | B/16, L/16, H/14 parameter counts |
| `test_tokens_and_equations` | Eqs. 1–4 recomputed by hand match the model; the zero head gives 0 logits |
| `test_higher_resolution_fine_tuning` | resizing to the same grid changes nothing; a bigger grid keeps the [class] position; the model runs at the new resolution |
| `test_hybrid_uses_cnn_feature_map_as_1x1_patches` | the CNN stem gives the right number of tokens |
| `test_analysis_tools` | the similarity diagonal is 1; PCA singular values are sorted; attention distances lie in the valid range |
| `test_masked_patch_prediction_targets_and_corruption` | pure red → 448; 50% chosen; 80/10/10 |
| `test_vit_learns_a_simple_global_task` | a tiny ViT learns a simple global task |

---

## 5. Try it yourself

1. In `demo.py`, train the ViT on the bright-square task 10× longer. Does it catch up with the CNN? Does the position similarity start showing rows and columns?
2. Replace the learnable positions with **no** positions (set `pos` to zeros and freeze it). What happens on a task where position matters?
3. Use mean pooling over patch tokens instead of the [class] token in `forward`. The paper (Appendix D.3) says both work if the learning rate is tuned.
4. Change `bits=3` to `bits=1` in `mean_colour_targets`. How many classes are there now?
