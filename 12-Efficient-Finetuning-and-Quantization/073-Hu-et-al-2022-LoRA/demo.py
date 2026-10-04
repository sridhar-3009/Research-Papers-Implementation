"""LoRA in ~30 seconds: GPT-3 parameter arithmetic, the LoRA layer (zero start, merging, no extra latency), adapting a
pre-trained tiny LLaMA to a new behaviour with full fine-tuning vs LoRA at different ranks and on different weight
types (Tables 5 and 6), switching tasks by swapping adapters, and the paper's analyses of what Delta W learns
(Figure 3 subspace similarity, Table 7 amplification)."""

import copy
import time

import numpy as np
import torch

from lora import (REPORTED, LoRALinear, accuracy, add_lora, amplification, finetune, lora_layers, lora_params_gpt3,
                  new_model, pretrain, set_adapters, subspace_similarity, trainable)

T0 = time.time()
rng = np.random.default_rng(0)


def section(t):
    print(f"\n=== {t} ===")


section("1. Parameter arithmetic for GPT-3 175B (d_model 12288, 96 layers)")
for r, mats in ((1, 2), (4, 2), (8, 1), (2, 4)):
    n = lora_params_gpt3(r, mats)
    print(f"    r = {r}, {mats} matrices per layer: {n / 1e6:5.1f}M trainable = {n * 2 / 1e6:4.0f} MB in fp16  "
          f"({175e9 / n:,.0f}x fewer than 175B)")
print("  -> r = 4 on Wq and Wv is the paper's 18M budget (~35 MB); r = 1 on Wq, Wv is its 4.7M row in Table 4.")
print("     100 adapted models: 350 GB + 100 x 35 MB ~ 354 GB, instead of 100 x 350 GB = 35 TB.")

section("2. The LoRA layer: starts as the identity change, merges into one matrix")
base = torch.nn.Linear(64, 64, bias=False)
lin = LoRALinear(copy.deepcopy(base), r=4)
x = torch.randn(512, 64)
print(f"  B = 0 at start, so LoRA(x) == W0 x exactly: max |diff| = {(lin(x) - base(x)).abs().max().item():.1e}")
with torch.no_grad():
    lin.B.normal_()
y_unmerged = lin(x)
lin.merge()
print(f"  after training (random B here): merged W0 + (alpha/r) BA gives the same output: max |diff| = "
      f"{(lin(x) - y_unmerged).abs().max().item():.1e}")
print(f"  rank of the update: {torch.linalg.matrix_rank(lin.delta()).item()} (r = 4); merged inference is one plain matmul")

section("3. Pre-train a tiny LLaMA, then adapt it to a NEW behaviour")
t = time.time()
model = pretrain(new_model(), rng)
print(f"  pre-trained on COPY / REVERSE / SORT of 6 digits ({time.time() - t:.0f}s): sort accuracy "
      f"{accuracy(model, 'sort'):.1%}; the SORT prompt sorting DESCENDING: {accuracy(model, 'desc'):.1%}")
print("  downstream: make the SORT prompt sort in descending order; 300 steps of Adam (lr 3e-3 full, 1e-2 LoRA)")
print("    method                              trainable params   descending-sort accuracy")
full = finetune(copy.deepcopy(model), "desc", rng, lr=3e-3)
print(f"    full fine-tuning                         {trainable(full):7,d}          {accuracy(full, 'desc'):6.1%}")
ranked = {}
for r in (1, 2, 4, 8, 16, 64):
    m = finetune(add_lora(copy.deepcopy(model), ("wq", "wv"), r), "desc", rng, lr=1e-2)
    ranked[r] = m
    print(f"    LoRA on Wq, Wv, r = {r:2d}                   {trainable(m):7,d}          {accuracy(m, 'desc'):6.1%}")
print(f"  -> r = 8 ({trainable(ranked[8]):,} parameters, 33x fewer) gets within a few points of full fine-tuning, and r = 64 (full")
print(f"     rank here, since d = 64) is no better than r = 8-16. Very small ranks are not enough in our toy (r = 1:")
print(f"     {accuracy(ranked[1], 'desc'):.0%}); the paper found r = 1 enough for GPT-3, a far richer pre-trained model.")

section("4. Same budget (2,048 parameters), different weight types (Table 5)")
budget = {}
for label, targets, r in (("Wq, r = 8", ("wq",), 8), ("Wk, r = 8", ("wk",), 8), ("Wv, r = 8", ("wv",), 8),
                          ("Wo, r = 8", ("wo",), 8), ("Wq, Wk, r = 4", ("wq", "wk"), 4),
                          ("Wq, Wv, r = 4", ("wq", "wv"), 4), ("Wq, Wk, Wv, Wo, r = 2", ("wq", "wk", "wv", "wo"), 2)):
    m = ranked[4] if targets == ("wq", "wv") and r == 4 else finetune(add_lora(copy.deepcopy(model), targets, r), "desc", rng, lr=1e-2)
    budget[label] = accuracy(m, "desc")
    print(f"    {label:24s} {trainable(m):6,d} params   accuracy {budget[label]:6.1%}")
best = max(budget, key=budget.get)
singles = {k: v for k, v in budget.items() if k.count("W") == 1}
print(f"  -> best: {best} ({budget[best]:.1%}); the same parameters in ONE matrix at r = 8 give "
      f"{min(singles.values()):.1%} to {max(singles.values()):.1%}.")
print("     As in the paper, spreading a fixed budget over several matrices at a low rank works best. Unlike GPT-3")
print("     (where Wq or Wk alone were worst and Wq + Wv was as good as all four), our toy ranks the single matrices")
print(f"     differently (Wo alone {budget['Wo, r = 8']:.1%}, Wq + Wv {budget['Wq, Wv, r = 4']:.1%}).")

section("5. Task switching: the frozen base model is untouched")
m = ranked[8]
print(f"    full fine-tuning: descending {accuracy(full, 'desc'):.1%}, ascending sort {accuracy(full, 'sort'):.1%}"
      "  (the old skill is overwritten)")
print(f"    LoRA adapter ON:  descending {accuracy(m, 'desc'):.1%}")
set_adapters(m, False)
print(f"    LoRA adapter OFF: ascending sort {accuracy(m, 'sort'):.1%}  (subtract BA -> the original model; keep one")
print("    copy of the 175B weights and swap 35 MB adapters per task)")
set_adapters(m, True)

section("6. What does Delta W look like? (Figure 3, Table 7)")
lv8 = [l for l in lora_layers(ranked[8]) if l.A.shape[0] == 8][1::2]        # the Wv adapters
lv64 = [l for l in lora_layers(ranked[64]) if l.A.shape[0] == 64][1::2]
print("  subspace similarity phi(A_r=8, A_r=64, i, j) = ||U8[:, :i]^T U64[:, :j]||^2 / min(i, j), last layer's Wv")
print("  (1 = the top-i directions of one lie inside the top-j of the other; a random pair scores about max(i,j)/64):")
A8, A64 = lv8[-1].A.detach(), lv64[-1].A.detach()
for i in (1, 2, 4, 8):
    print(f"    i = {i}: " + "  ".join(f"j={j}: {subspace_similarity(A8, A64, i, j):.2f}" for j in (1, 2, 4, 8)))
g = torch.Generator().manual_seed(1)
rand = torch.randn(8, 64, generator=g)
print(f"  a random 8-dim subspace vs A_64's top 8: {subspace_similarity(rand, A64, 8, 8):.2f}; vs A_8: "
      f"{subspace_similarity(rand, A8, 8, 8):.2f}")
p18 = subspace_similarity(A8, A64, 1, 8)
print(f"  -> above chance: e.g. phi(1, 8) = {p18:.2f} vs 8/64 = 0.12 -- the top direction found with r = 8 lies largely inside")
print("     the top directions found with r = 64 (the paper's Figure 3 shows the same, with a stronger overlap).")
lq = [l for l in lora_layers(ranked[4])][-2]                                  # last layer's Wq adapter, r = 4
amp = amplification(lq.base.weight.detach(), lq.delta().detach(), 4)
print("  Table 7 analogue (last layer's Wq, r = 4):")
for k, v in amp.items():
    print(f"    {k:40s} {v:.3f}")
on_dw, on_w = amp["||U^T W V^T|| on dW's directions"], amp["on W's own top directions"]
print(f"  -> W projected on Delta W's directions ({on_dw:.2f}) is larger than on random directions "
      f"({amp['on random directions']:.2f}) but")
print(f"     much smaller than on W's own top directions ({on_w:.2f}): Delta W amplifies features that W")
print(f"     contains but does not emphasise, by ~{amp['amplification ||dW|| / ||U^T W V^T||']:.0f}x -- the paper's pattern (GPT-3 layer 48: 0.32 vs")
print("     21.67 on W's own and 0.02 on random directions; amplification 21.5).")

section("7. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
