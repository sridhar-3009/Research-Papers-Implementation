# The code, explained simply

How the code in this folder implements Sparse Transformers (Child et al. 2019).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `sparse.py` | strided and fixed patterns, the two-hop validity check, masked attention, blocked kernels (block-local and strided column), the Sparse Transformer block (pre-LN, GELU, 1/√(2N) init) and LM (attention embeddings, zero-initialized logits, recomputation) |
| `experiments.py` | E1–E4: enwik8 and CIFAR-10 dense vs fixed vs strided, context at evaluation, time/memory vs length (heavy, not run here) |
| `demo.py` | Figure 3's patterns as ASCII, pair counts, blocked kernels vs masks, the periodic copy task (~5 seconds) |
| `test_sparse.py` | 9 quick tests (~2 seconds) |

**Run it** (from `06-Transformers/036-Child-et-al-2019-Sparse-Transformers`):
```
python3 -m pytest -q             # ~2 seconds
python3 demo.py                  # ~5 seconds
python3 experiments.py --quick   # ~1 hour
```

---

## 2. `sparse.py`

### Patterns
- **`causal(n)`:** j ≤ i.
- **`strided_patterns(n, l)`:**
  - A1 = the previous l positions;
  - A2 = (i − j) mod l = 0.
- **`fixed_patterns(n, l, c)`:**
  - A1 = the same block of length l;
  - A2 = the last c cells of every block, **plus i itself** (so no row is empty).
- **`reachable_in_two(A_first, A_second)`:** a boolean matrix product. j reaches i if it is in A_first of some a that is in A_second of i (or reaches i directly).
- **`entries(mask)`:** the number of computed pairs.

### Attention
- **`masked_attention(q, k, v, mask)`:** dense scores with −∞ outside the mask.
- **`block_local_attention(q, k, v, l)`:** reshape (n) → (n/l, l) and attend causally inside each block. No n × n matrix.
- **`strided_column_attention(q, k, v, l)`:** reshape into an (n/l) × l grid, transpose, and attend causally down the columns (the paper's "transpose, then local window").

### The model
- **`gelu_approx(x)`:** x·σ(1.702x).
- **`SparseAttention(d, heads, patterns, mode, layer_index, n_layers)`:**
  - `mode` is one of "dense", "interleave" (pattern = layer index mod p), "merged" (union) or "multihead" (heads split across patterns);
  - W_p's init is scaled by 1/√(2N).
- **`ResBlock`:**
  - a = attention(LN(H));
  - b = W₂·GELU(W₁·LN(H + a));
  - returns a + b, with W₂'s init scaled by 1/√(2N).
- **`SparseTransformerLM(vocab, n_ctx, d, layers, heads, pattern, stride, c, mode, dropout, recompute)`:**
  - **embeddings:** token embedding + row and column embeddings of the stride grid (Eq. 15);
  - **blocks:** `H = H + block(H)`, optionally through `torch.utils.checkpoint` (recomputation);
  - **output:** final LN, then the output layer, initialized to zero.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Table 2 on enwik8: dense / fixed / strided, bits per byte and seconds per iteration (AdamW, warm-up + cosine, clip 1.0, weight decay 0.01) |
| `e2` | Table 2 on CIFAR-10 as 3,072-byte raster sequences (stride = one image row); `--quick` uses 16×16 crops |
| `e3` | Table 3: score only the last tokens of each window, so every scored token has at least a given amount of context |
| `e4` | time and peak GPU memory of one forward/backward pass: dense vs blocked strided kernels |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_patterns_match_the_definitions` | hand examples; nothing from the future |
| `test_every_earlier_position_is_reachable_in_two_steps` | strided in both orders; fixed only "block, then summary" |
| `test_cost_grows_like_n_sqrt_n` | strided pair count < 2n√n |
| `test_efficient_blocked_and_strided_kernels_equal_the_masked_ones` | the kernels are exact |
| `test_head_modes` | interleave / merged / multi-head masks |
| `test_initialisation_and_residual_block` | uniform start, the 1/√(2N) scaling, Eqs. 12–14 |
| `test_gelu_approximation` | within 0.03 of exact GELU |
| `test_recomputation_gives_the_same_gradients` | checkpointing changes nothing |
| `test_strided_model_learns_to_copy_the_row_above` | loss < 0.1 after 150 updates |

---

## 5. Try it yourself

1. Raise c in the fixed pattern from 2 to 8 on the periodic task. How close does it get to strided?
2. Make a "text-like" task without periodic structure (e.g. copy a token from a random earlier position marked by a pointer) and compare strided vs fixed.
3. Time `block_local_attention + strided_column_attention` vs `masked_attention` for n = 4,096 and 16,384.
4. Train with 16 layers, with and without the 1/√(2N) init. Watch the loss in the first 100 steps.
