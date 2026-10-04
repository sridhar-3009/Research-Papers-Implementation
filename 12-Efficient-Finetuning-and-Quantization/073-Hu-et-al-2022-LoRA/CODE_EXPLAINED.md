# The code, explained simply

How the code in this folder implements LoRA (Hu et al. 2022).
Read [EXPLAINED.md](EXPLAINED.md) first.

The code re-uses the tiny LLaMA of paper 056 through `importlib`.

---

## 1. The files

| File | What it is |
|---|---|
| `lora.py` | `LoRALinear`; injection into attention matrices; adapter on/off; parameter counting; GPT-3 arithmetic; toy tasks; pre-training and fine-tuning; subspace similarity; amplification |
| `experiments.py` | rank × weight-type grid, budget splits, cross-seed subspace similarity, amplification per layer, small data, α/learning rate, real GPT-2 + LoRA on SST-2 |
| `demo.py` | GPT-3 arithmetic, layer checks, rank sweep, Table 5 analogue, task switching, Figure 3 / Table 7 analogues (~26 seconds) |
| `test_lora.py` | 3 quick tests (~1 second) |

**Run it** (from `12-Efficient-Finetuning-and-Quantization/073-Hu-et-al-2022-LoRA`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~26 seconds
python3 experiments.py --quick
```

---

## 2. `lora.py`

### The layer
- **`LoRALinear(base, r, alpha)`:**
  - freezes `base`;
  - A is r×k Gaussian (scaled 1/√k) and B is d×r zeros; `scale` = α/r, with α = r by default;
  - `forward` = base(x) + scale·(x Aᵀ) Bᵀ, unless `merged` or not `enabled`;
  - `merge` / `unmerge` add or subtract ΔW in place.

### Using it
| Function | What it does |
|---|---|
| `add_lora(model, targets, r)` | freezes everything, then wraps `blocks[i].attn.<target>`. Each layer's A gets its own seed, which also depends on r, so the r = 8 and r = 64 runs start from different A's. |
| `set_adapters(model, on)` | turns every adapter on or off; off is the exact pre-trained model |
| `trainable`, `lora_params_gpt3` | parameter counting |

### Toy tasks
- **`make_batch`:** sequences `[task] 6 digits [SEP] 6 output digits`. `desc` uses the **same token as sort** but expects descending order.
- **`seq_loss`:** loss only on the output digits.
- **`accuracy`:** exact match on a fixed evaluation set (seed 999).
- **`pretrain`:** COPY / REVERSE / SORT, 800 steps.
- **`finetune`:** Adam on whatever requires grad; `train_data` allows a fixed small dataset.

### Analyses
- **`subspace_similarity(A1, A2, i, j)`:** right singular vectors of each A; ‖U1[:, :i]ᵀ U2[:, :j]‖²_F / min(i, j).
- **`amplification(W, dW, r)`:** ‖UᵀWVᵀ‖ on ΔW's top-r directions, on W's own and on random directions; ‖ΔW‖; and their ratio.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | r ∈ {1, …, 64} × {Wq}, {Wq, Wv}, all four; full fine-tuning; best learning rate; 3 seeds |
| `e2` | budgets 4/8/16 (single-matrix r) split over 7 target sets |
| `e3` | φ between r = 8 and r = 64 (different seeds) and between two r = 64 seeds |
| `e4` | amplification for every adapter at r ∈ {1, 4, 16, 64} |
| `e5` | 32 … 2,048 training examples, full vs LoRA r = 8 |
| `e6` | best learning rate for r ∈ {2, 8, 32} with α fixed at 8 |
| `e7` | GPT-2 c_attn wrapped with LoRA r = 8 vs full fine-tuning on SST-2 (prompted LM classification) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_lora_layer_zero_start_merge_and_rank` | identical output at start; scale α/r; rank of ΔW = r; merge gives the same outputs; unmerge restores W₀ |
| `test_injection_freezes_and_counts` | parameter count = layers × matrices × r(d + k); GPT-3 formula; only A and B change in training; adapters off equals the base model |
| `test_analyses` | φ = 1 for identical subspaces and for a subspace inside a bigger one; W's own directions give the largest projection |

---

## 5. Try it yourself

1. Add LoRA to the FFN matrices (`w1`, `w2`, `w3`) too. Does r = 2 on everything beat r = 8 on attention?
2. Initialise B randomly instead of zero. What happens to accuracy at step 0, and to training?
3. Change α at a fixed r and learning rate. Is it the same as changing the learning rate?
4. Make the new task unrelated (e.g. "add 3 to every digit"). Does the needed rank grow?
5. Merge two adapters (desc and another task) at once. What breaks?
