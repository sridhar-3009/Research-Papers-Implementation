# The code, explained simply

How the code in this folder implements FlashAttention (Dao et al. 2022).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `flash.py` | HBM traffic counter, standard attention (Alg. 0), online softmax, block sizes, FlashAttention forward (Alg. 1, with optional block-sparse mask), recomputing backward (Alg. 4), standard backward traffic |
| `experiments.py` | exponent fit of the IO complexity, backward traffic and FLOPs, block size, block-sparse patterns, real GPU SDPA benchmark |
| `demo.py` | online softmax by hand, exactness, traffic tables, memory, block-sparse, CPU timing note (~1 second) |
| `test_flash.py` | 3 quick tests (~1 second) |

**Run it** (from `13-Systems-Inference-and-Training/079-Dao-et-al-2022-FlashAttention`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~1 second
python3 experiments.py --quick
```

---

## 2. `flash.py`

### Counting and the baseline
- **`HBM`:** `read(x)` adds x.numel() to the reads and returns a copy (the "SRAM" copy); `write(dst, idx, x)` adds to the writes and stores the value.
- **`standard_attention`:** reads Q, K and writes S; reads S and writes P; reads P, V and writes O.
- **`online_softmax(x, block)`:** a running max and sum over blocks of a vector.

### FlashAttention
- **`block_sizes(M, d)`:** B_c = ⌈M/4d⌉, B_r = min(B_c, d).
- **`flash_attention(Q, K, V, M, block_mask)`:** Algorithm 1.
  - The outer loop reads K_j, V_j.
  - The inner loop reads Q_i, O_i, ℓ_i, m_i and computes S_ij, the block max and P̃_ij = exp(S_ij − m̃) and its row sums.
  - It merges with the running (m, ℓ), writes O_i = (ℓ_i e^(m_i − m_new) O_i + e^(m̃ − m_new) P̃_ij V_j)/ℓ_new, and writes m and ℓ.
  - `block_mask[i, j] = False` skips the block.
- **`flash_backward`:** D = rowsum(dO ∘ O). For each key block j and query block i:
  - recompute S_ij, then P_ij = exp(S_ij − m_i)/ℓ_i;
  - dV_j += P_ijᵀ dO_i;
  - dP = dO_i V_jᵀ, dS = P ∘ (dP − D_i);
  - dQ_i += dS K_j · scale; dK_j += dSᵀ Q_i · scale.
- **`standard_backward_io(N, d)`:** the traffic of the textbook backward pass, which reads and writes the N×N matrices.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | grid over N, d, M; least-squares fit of log(traffic) on log N, log d, log M |
| `e2` | fwd + bwd traffic, flash vs standard, and FLOP counts including recomputation |
| `e3` | traffic for fixed (B_c, B_r) pairs (`block_sizes` overridden) |
| `e4` | local-window + global-block masks: kept fraction, traffic, error vs masked dense attention |
| `e5` | CUDA: `scaled_dot_product_attention` with the FLASH vs MATH backend, causal, fp16, fwd + bwd time and peak memory |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_online_softmax_matches` | block-wise softmax equals `torch.softmax` |
| `test_flash_forward_backward_exact` | forward equals standard attention; backward equals autograd for dQ, dK, dV |
| `test_io_scaling_and_block_sparsity` | block sizes; traffic ~N² and ~1/M; a one-block mask halves the traffic and equals attention over the first 64 keys |

---

## 5. Try it yourself

1. Add a causal mask: skip blocks entirely above the diagonal. How much traffic is saved?
2. Swap the loop order (Q blocks outside, as in FlashAttention-2). How does the traffic change?
3. Store P in the forward pass and use it in the backward pass. Count the traffic difference.
4. Use float16 for Q, K, V. Does the online softmax stay exact enough?
5. Run E5 on a GPU and plot time vs N for the two backends.
