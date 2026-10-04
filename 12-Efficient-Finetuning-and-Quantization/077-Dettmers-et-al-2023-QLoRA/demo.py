"""QLoRA in ~30 seconds: building 4-bit NormalFloat from normal quantiles (and checking it against the paper's
Appendix E), NF4 vs Int4 vs FP4 on normally distributed weights (Table 2's ranking), double quantization of the block
constants, and QLoRA fine-tuning of a tiny LLaMA whose frozen base is stored in 4 bits (vs 16-bit LoRA and full
fine-tuning; LoRA on all layers vs only q, v), plus the 65B memory budget."""

import copy
import time

import numpy as np
import torch

from qlora import (DATA_TYPES, DATA_TYPES_3BIT, REPORTED, L, add_lora_everywhere, bits_per_parameter, double_quantize,
                   fake_quant, memory_budget_gb, nf4_values, quantize_blockwise, quantize_model)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Building NF4 (Eq. 4)")
nf4 = nf4_values()
paper = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                      -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                      0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                      0.7229568362236023, 1.0])
print("  8 positive + 7 negative quantiles of N(0, 1) (averaged neighbours), plus an exact 0, scaled to [-1, 1]:")
print(f"    {[round(v, 4) for v in nf4.tolist()]}")
print(f"  max |difference| from the paper's Appendix E values: {(nf4 - paper).abs().max():.1e}")
for k in ("Int4", "FP4 (E2M1)", "FP4 (E3M0)"):
    print(f"    {k:11s} {[round(v, 3) for v in DATA_TYPES[k].tolist()]}")
print("  -> NF4 packs its levels where normal weights are dense (near 0) and uses all 16 codes; Int4 spaces them evenly")
print("     and wastes one code; FP4 E3M0 crowds them near 0 and leaves big gaps near the block's absmax")

section("2. Quantizing normally distributed weights, blocks of 64 (why NF4 wins in Table 2)")
torch.manual_seed(0)
W = torch.randn(1024, 64) * 0.02
print("    data type      relative MSE   code usage (entropy, max 4 bits)")
for k, v in DATA_TYPES.items():
    q = fake_quant(W, v)
    codes, _ = quantize_blockwise(W, v)
    p = torch.bincount(codes.flatten(), minlength=len(v)).float()
    p = p / p.sum()
    ent = float(-(p[p > 0] * p[p > 0].log2()).sum())
    print(f"    {k:12s}    {float(((q - W) ** 2).mean() / W.var()):.4f}         {ent:.2f}")
print("  -> NF4 has the lowest error and uses its codes most evenly (each bin gets a similar share of the weights), as in")
print("     the paper's Table 2 (Int4 34.34, FP4 E2M1 31.07, E3M0 29.48, NF4 + DQ 27.41 ppl). The FP4 variants rank")
print("     differently here: on purely Gaussian weights E3M0 is worst, while on real models it beat E2M1.")

section("3. Double quantization of the block constants")
codes, c = quantize_blockwise(W, nf4)
c_dq = double_quantize(c)
print(f"  one FP32 absmax per 64 weights: {bits_per_parameter(dq=False) - 4:.3f} extra bits per parameter; after 8-bit "
      f"quantization of the constants in blocks of 256: {bits_per_parameter(dq=True) - 4:.3f}")
print(f"  (= 8/64 + 32/(64*256); saves {bits_per_parameter() - bits_per_parameter(dq=True):.3f} bits per parameter, "
      f"~{65e9 * (bits_per_parameter() - bits_per_parameter(dq=True)) / 8 / 1e9:.1f} GB on a 65B model)")
print(f"  relative error of the constants: {float((c_dq - c).abs().max() / c.abs().max()):.4f}; weight MSE with vs without DQ: "
      f"{float(((nf4[codes] * c_dq[:, None]).flatten() - W.flatten()).pow(2).mean() / W.var()):.4f} vs "
      f"{float(((nf4[codes] * c[:, None]).flatten() - W.flatten()).pow(2).mean() / W.var()):.4f}")

section("4. A tiny LLaMA stored in 4 / 3 bits (no fine-tuning yet)")
t = time.time()
base = L.pretrain(L.new_model(), np.random.default_rng(0))
seq = torch.cat([L.make_batch(task, 200, np.random.default_rng(9)) for task in ("copy", "reverse", "sort")])
def loss(m):
    with torch.no_grad():
        return float(L.seq_loss(m, seq))
print(f"  paper 073's tiny LLaMA ({time.time() - t:.0f}s), loss {loss(base):.4f}; every block weight quantized in blocks of 64:")
ql = {}
for k, v in list(DATA_TYPES.items()) + list(DATA_TYPES_3BIT.items()):
    ql[k] = loss(quantize_model(base, v))
    print(f"    {k:12s} loss {ql[k]:.4f}  (x{ql[k] / loss(base):.1f})")
print(f"  -> at 4 bits NF4, Int4 and E2M1 barely move the loss on this small, easy model, E3M0 multiplies it by "
      f"{ql['FP4 (E3M0)'] / loss(base):.1f}; at 3 bits")
print(f"     Int3 is worst (x{ql['Int3'] / loss(base):.1f}) and NF3 / FP3 cost about x2.")

section("5. QLoRA: fine-tune through a FROZEN 4-bit base (new task: the SORT prompt sorts descending)")
rng = np.random.default_rng(1)
print("    method                                     stored base   trainable params   accuracy")
res = {}
for name, mk, bits in (("full fine-tuning (16-bit)", lambda: copy.deepcopy(base), 16),
                       ("LoRA r=8 on all linear layers (16-bit)", lambda: add_lora_everywhere(copy.deepcopy(base), 8), 16),
                       ("LoRA r=8 on q, v only (16-bit)", lambda: add_lora_everywhere(copy.deepcopy(base), 8, True), 16),
                       ("QLoRA: NF4 + DQ base, LoRA on all layers", lambda: add_lora_everywhere(quantize_model(base, DATA_TYPES["NF4"], dq=True), 8), 4),
                       ("QLoRA with an FP4 (E3M0) base", lambda: add_lora_everywhere(quantize_model(base, DATA_TYPES["FP4 (E3M0)"]), 8), 4)):
    m = mk()
    n = L.trainable(m)
    L.finetune(m, "desc", rng, steps=300, lr=3e-3 if name.startswith("full") else 1e-2)
    res[name] = L.accuracy(m, "desc")
    print(f"    {name:42s}   {bits:2d}-bit       {n:7,d}          {res[name]:6.1%}")
print("  -> QLoRA matches 16-bit LoRA and full fine-tuning (the paper: 'QLoRA replicates 16-bit full fine-tuning'),")
print("     and adapters on ALL linear layers matter more than the data type: q, v only falls short, as in the paper.")
print("     (On this tiny model even the FP4 base is recovered by the adapters; the paper's MMLU shows FP4 ~1 point")
print("     behind: BF16 53.0, FP4 52.2, NF4 + DQ 53.1.)")

section("6. Memory for fine-tuning a 65B model (weights + trained-parameter gradients + Adam states + ~6 GB activations)")
for method in ("16-bit full fine-tuning", "16-bit LoRA", "QLoRA (NF4 + DQ)"):
    print(f"    {method:24s} ~{memory_budget_gb(65e9, method) / 1e9:5.0f} GB")
print("  -> the paper: > 780 GB for 16-bit full fine-tuning vs < 48 GB with QLoRA (one GPU); paged optimizers absorb")
print("     the remaining memory spikes by paging optimizer states to CPU RAM.")

section("7. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
