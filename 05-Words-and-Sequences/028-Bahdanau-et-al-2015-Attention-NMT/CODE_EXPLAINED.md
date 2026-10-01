# The code, explained simply

How the code in this folder implements Bahdanau, Cho & Bengio (2015).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `attention.py` | the GRU (this paper's convention), the bidirectional encoder, the alignment model, RNNsearch and the RNNencdec baseline, the paper's init, beam search (returns alignments too), BLEU |
| `experiments.py` | Table 1, Figure 2, Figure 3 on Multi30k (heavy, not run here) |
| `demo.py` | uniform attention at init; encdec vs search on short and long toy sequences; an ASCII alignment matrix (about 10 seconds) |
| `test_attention.py` | 9 quick tests (a couple of seconds) |

**Run it** (from `05-Words-and-Sequences/028-Bahdanau-et-al-2015-Attention-NMT`):
```
python3 -m pytest -q             # ~2 seconds
python3 demo.py                  # ~10 seconds (tiny training)
python3 experiments.py --quick   # ~30-60 minutes
```

---

## 2. `attention.py`

### `GRU(m, n, ctx)`
s = (1 − z)·s + z·tanh(W e + U(r ⊙ s) + C c). With `ctx > 0` the gates also read the context c (decoder); with 0 they don't (encoder).

### `RNNsearch(src_vocab, tgt_vocab, m, n, l, n_align, mode)`
- **`encode(src)`:**
  - runs `fwd` left-to-right and `bwd` right-to-left over the shared source embeddings;
  - **padded steps don't change the state** (`torch.where(mask, new, old)`);
  - returns the annotations `ann = [fwd ; bwd]` (Tx, N, 2n), the mask, and s₀ = tanh(W_s ←h₁).
- **`attend(s_prev, ann, Ua_h, mask)`:**
  ```python
  e = v_a(tanh(W_a(s_prev) + Ua_h))      # (Tx, N): one score per source position
  e[padding] = -inf                       # padding gets exactly 0 weight after the softmax
  alpha = softmax(e, over Tx)
  c = sum(alpha * ann)                    # Eq. (5)
  ```
- **`fixed_context`:** the RNNencdec baseline's single context [→h_last ; ←h₁].
- **`step(y_prev, s_prev, ...)`:**
  1. compute the context (attention or fixed);
  2. compute the output t̃ = U_o s_{prev} + V_o e + C_o c → maxout → log-softmax;
  3. update the decoder state with the GRU.
- **`forward(src, tgt_in, return_alpha)`:** teacher forcing. `Ua_h = U_a(ann)` is computed once. It can also return all α's.
- **`log_prob(src, tgt)`:** log p(y | x) per pair.

### `paper_init_`
- Biases 0; weights N(0, 0.01²).
- The GRUs' U, U_z, U_r are made orthogonal (via QR).
- Wₐ, Uₐ ~ N(0, 0.001²); **vₐ = 0**.

### `beam_search(model, src, beam, max_len)`
Like Paper 027's, but it carries one decoder state per hypothesis and **records the α of every step**, so alignments can be drawn.

---

## 3. `experiments.py`

- **`batches`:** Appendix B.2's trick: 1600 pairs → sort by length → 20 minibatches of 80.
- **`train(mode, max_len)`:** AdaDelta (ρ = 0.95, ε = 10⁻⁶), clipping at 1, the paper's init, only pairs up to `max_len`.

| Function | Reproduces |
|---|---|
| `e1` | Table 1: RNNencdec / RNNsearch × short / long cutoff; BLEU on all and on UNK-free sentences |
| `e2` | Figure 2: BLEU by source-length bucket |
| `e3` | Figure 3: heatmaps of α for 4 test sentences |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_gru_update_convention` | s = (1 − z)s + z·s̃ |
| `test_annotations_concatenate_forward_and_backward_states` | h₁'s backward half sees the last word |
| `test_attention_weights_are_a_distribution_over_real_words` | Σα = 1, α = 0 on padding, Eq. 5 |
| `test_zero_v_a_gives_uniform_attention_at_the_start` | the paper's init |
| `test_precomputing_U_a_h_changes_nothing` | the A.1.2 optimization |
| `test_encdec_baseline_uses_one_fixed_context` | the baseline |
| `test_padding_does_not_change_the_translation_probability` | masking is right end to end |
| `test_paper_initialization` | orthogonal U, small Wₐ, zero vₐ |
| `test_attention_model_learns_to_reverse_and_aligns_sharply` | learned anti-diagonal alignment and a correct beam output |

---

## 5. Try it yourself

1. Replace the additive score vₐᵀ·tanh(Wₐs + Uₐh) with **dot-product** attention sᵀ(Wh), as Luong et al. 2015 and the Transformer do. Same results on the toy task?
2. Train on reversal **and** on copy (identity order). Do the alignment matrices change from anti-diagonal to diagonal?
3. Remove the backward RNN (forward annotations only). What happens to the alignments for early source words?
4. In `e3`, find sentences whose alignment is **not** monotonic and check whether they involve word-order changes between English and German.
