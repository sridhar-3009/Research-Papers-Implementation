# The code, explained simply

How the code in this folder implements ByteNet (Kalchbrenner et al. 2016).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `bytenet.py` | masked / centred dilated convolutions, channel-wise layer norm, the ReLU and Multiplicative-Unit residual blocks, the dilation schedule and receptive field, the ByteNet language model, the full ByteNet translator with dynamic unfolding, greedy decoding, gradient saliency |
| `experiments.py` | E1–E5: enwik8 bits/byte, char-level Multi30k translation (beam 12), length correlation, saliency figures, time vs length against Paper 028 (heavy, not run here) |
| `demo.py` | dilation and the receptive field, a lag task dilated vs undilated, a translation toy, saliency, linear time (~11 seconds) |
| `test_bytenet.py` | 9 quick tests (~2 seconds) |

**Run it** (from `05-Words-and-Sequences/033-Kalchbrenner-et-al-2016-ByteNet`):
```
python3 -m pytest -q             # ~2 seconds
python3 demo.py                  # ~11 seconds
python3 experiments.py --quick   # ~1-2 hours
```

---

## 2. `bytenet.py`

### The layers
- **`Conv1d(c_in, c_out, k, dilation, causal)`:**
  - **causal = "masked":** pad (k − 1)·r zeros on the **left**, so output i sees inputs i, i − r, …, i − (k − 1)r;
  - **otherwise centred:** padding is split between both sides.
- **`ChannelNorm`:** layer norm over the channels at each time step. It never mixes positions, so it is safe in a causal decoder.
- **`ReLUBlock(d, k, r, causal)`:** Figure 3 left, LN–ReLU–1×1(2d→d)–LN–ReLU–k-conv(d→d)–LN–ReLU–1×1(d→2d), plus the shortcut.
- **`MU`:** one convolution produces four maps, g₁, g₂, g₃ (sigmoids) and u (tanh); the output is g₁ ⊙ tanh(g₂ ⊙ h + g₃ ⊙ u).
- **`MUBlock`:** Figure 3 right: LN–ReLU–1×1–LN, then MU (1×1) and MU (k-wide, dilated), then LN–ReLU–1×1, plus the shortcut.

### The schedule
- **`dilations(n)`:** 1, 2, 4, 8, 16, 1, 2, …
- **`receptive_field(k, rates)`:** 1 + Σ(k − 1)·r.

### The models
- **`ByteNetLM(vocab, d, n_blocks, k, block, dropout, max_rate)`:** embedding (2d channels) → causal stack → 1×1 → ReLU → dropout → 1×1 → logits. `max_rate=1` switches dilation off.
- **`unfold_length(|s|, a, b)`:** Eq. 2, round(a|s| + b).
- **`ByteNet(src_vocab, tgt_vocab, d, n_blocks, k, a, b)`:**
  - **`encode`:** pads the source to |t̂|, then runs the centred stack.
  - **`decode(rep, tgt_in)`:** target embedding + encoder output at the same position (zeros past |t̂|), then the causal stack and the head.
  - **`log_prob`:** teacher-forced, ignores padding.
  - **`greedy`:** dynamic unfolding, generating until EOS.
- **`saliency(model, src, tgt)`:** |∂ log p(t_i)/∂ embeddings| summed over channels, for the source and the previous target characters (Figure 6).

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Table 3: 30 MU blocks, d = 512, Adam 3e-4, wd 1e-4, dropout 0.1, 500-byte windows predicting the last 400; bits/byte |
| `e2` | Tables 2/4 on Multi30k: char vocabularies, padding to multiples of 50 (+20% for the source), length bucketing, beam 12 without length normalization, bits/char and BLEU |
| `e3` | Figure 5: Pearson ρ of English/German character lengths and the fitted slope |
| `e4` | Figure 6: saliency heatmaps from the E2 model |
| `e5` | forward + backward time vs length: ByteNet vs Paper 028's RNNsearch |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_masked_convolution_is_causal_and_unmasked_is_centred` | exact input positions of one output |
| `test_dilation_schedule_and_receptive_field` | 1, 2, 4, 8, 16, 1…; RF 63 / 11 / 373 |
| `test_receptive_field_equals_the_gradient_support` | exactly RF positions, none in the future |
| `test_mu_block_formula` | the MU equation |
| `test_the_full_decoder_cannot_see_the_future_but_the_encoder_sees_everything` | causality on the target side, full context on the source side |
| `test_dynamic_unfolding_lengths` | encoder length round(1.2·|s|); decoding past it works |
| `test_log_prob_ignores_padding` | masking |
| `test_saliency_shapes_and_causality` | the target saliency is lower-triangular |
| `test_dilation_lets_a_shallow_stack_learn_a_long_lag` | 3 dilated blocks learn a lag of 10 (undilated would see only 7) |

---

## 5. Try it yourself

1. In `demo.py`, try lags of 14 and 15 with 3 dilated blocks (receptive field 15). Where does learning stop?
2. Replace `+` with concatenation (and double the decoder's input channels) in `ByteNet.decode`. Any difference on the toy?
3. Swap `ChannelNorm` for `nn.BatchNorm1d` in a causal stack, and check whether the decoder still passes the causality test in training mode.
4. Time `ByteNet` vs Paper 028's `RNNsearch` for lengths 50–400 with `experiments.py --only e5 --quick`.
