"""LLaMA in a few seconds: the three architecture changes (RMSNorm, SwiGLU, RoPE) checked by hand, memory-efficient
causal attention, Table 2 parameter counts, the 21-day and carbon arithmetic, the inference-budget argument under
Chinchilla's fitted law (paper 052), a tiny GPT-2-vs-LLaMA training race on this repository's own text, and the
paper's benchmark numbers."""

import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama import (COMMON_SENSE, MMLU, TABLE_1, TABLE_2, TABLE_15_GPU_HOURS, LLaMA, RMSNorm, apply_rope, carbon,
                   ffn_hidden, memory_efficient_causal_attention, param_count, rope_frequencies, split_digits,
                   training_days)

HERE = Path(__file__).resolve().parent
torch.manual_seed(0); torch.set_num_threads(1)
T0 = time.time()


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------------------------------- 1
section("1. RMSNorm vs LayerNorm (pre-normalisation)")
x = torch.tensor([[1.0, 2.0, 3.0, 6.0]])
print(f"  x = {x.tolist()[0]}: mean {x.mean():.2f}, RMS {x.pow(2).mean().sqrt():.3f}")
print(f"  LayerNorm: {[round(v, 3) for v in F.layer_norm(x, (4,)).tolist()[0]]}  (centred AND scaled)")
print(f"  RMSNorm:   {[round(v, 3) for v in RMSNorm(4, eps=0)(x).tolist()[0]]}  (only scaled: x / RMS)")

section("2. SwiGLU: three matrices, hidden 2/3 * 4d (rounded up to a multiple of 256)")
for d in (4096, 5120, 6656, 8192):
    h = ffn_hidden(d)
    print(f"  d = {d}: hidden {h} ({h / d:.3f} d); SwiGLU {3 * d * h / 1e6:.0f}M params vs a 4d GELU MLP {8 * d * d / 1e6:.0f}M")

section("3. RoPE: rotate query/key pairs by position-dependent angles")
cos, sin = rope_frequencies(8, 128)
q, k = torch.randn(8), torch.randn(8)
at = lambda v, m: apply_rope(v.expand(128, 8), cos, sin)[m]
for m, n in ((5, 2), (50, 47), (120, 117), (9, 2)):
    print(f"  q at position {m:3d}, k at {n:3d} (offset {m - n}): score {float(at(q, m) @ at(k, n)):+.4f}")
print("  -> the score depends only on the OFFSET m - n (rows 1-3 are identical), not on where the pair sits")

section("4. Memory-efficient causal attention (Section 2.4, xformers)")
qq, kk, vv = (torch.randn(1, 4, 1024, 32) for _ in range(3))
t = time.time(); ref = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
eff = memory_efficient_causal_attention(qq, kk, vv, block=128)
print(f"  T = 1024: max |difference| vs standard attention {float((ref - eff).abs().max()):.1e}")
blocks = 1024 // 128
print(f"  score matrix never stored: largest temporary {128}x{128} per head instead of 1024x1024 ({(128 / 1024) ** 2:.1%});")
print(f"  key blocks computed: {blocks * (blocks + 1) // 2} of {blocks * blocks} (future blocks skipped: "
      f"{1 - (blocks + 1) / (2 * blocks):.0%} saved)")

# ---------------------------------------------------------------------------------------------------- 5
section("5. Table 2 sizes, the 21-day run, and Table 15's carbon")
for name, (rep, d, h, L, lr, tok) in TABLE_2.items():
    n = param_count(d, L)
    print(f"  {name:4s} d={d} heads={h} layers={L}: {n / 1e9:6.2f}B (paper {rep / 1e9}B), lr {lr}, {tok / 1e12}T tokens, "
          f"{tok / n:5.0f} tokens/param")
print(f"  65B on 1.4T tokens at 380 tokens/s/GPU on 2048 GPUs: {training_days(1.4e12):.1f} days (paper: ~21)")
flops = 6 * param_count(8192, 80) * 1.4e12
print(f"  6ND = {flops:.2e} FLOPs; at 312 TFLOP/s bf16 peak per A100 that is "
      f"{flops / (2048 * 312e12 * training_days(1.4e12) * 86400):.0%} utilisation")
for k in ("7B", "65B", "OPT-175B", "BLOOM-175B"):
    mwh, t = carbon(TABLE_15_GPU_HOURS[k])
    print(f"  {k:10s} {TABLE_15_GPU_HOURS[k]:>9,} GPU-hours -> {mwh:6.1f} MWh -> {t:5.1f} tCO2eq")
print("  Table 1 epochs: " + ", ".join(f"{k} {v[1]}" for k, v in TABLE_1.items()))

# ---------------------------------------------------------------------------------------------------- 6
section("6. Why train past compute-optimal: the inference budget (Introduction), using paper 052's law")
ch = load("chinchilla_052", "052-Hoffmann-et-al-2022-Chinchilla/chinchilla.py")
C = 6 * 10e9 * 200e9                                                              # the budget of '10B on 200B'
N_opt, D_opt = ch.frontier(C)
L_ref = ch.loss(N_opt, D_opt)
print(f"  budget C = 6 x 10B x 200B = {C:.1e} FLOPs. Under the fitted law the compute-optimal model is "
      f"{N_opt / 1e9:.1f}B on {D_opt / 1e9:.0f}B tokens (loss {L_ref:.4f});")
print(f"  (10B on 200B, Chinchilla's ~20 tokens/param rule, gives {ch.loss(10e9, 200e9):.4f}: the Approach-3 law itself prefers")
print("   more tokens per parameter, the inconsistency noted in paper 052)")
N_small = N_opt / 2
lo, hi = 1e9, 1e15
for _ in range(200):
    mid = math.sqrt(lo * hi)
    lo, hi = (mid, hi) if ch.loss(N_small, mid) > L_ref else (lo, mid)
D_small = hi
extra = 6 * N_small * D_small - C
saved = 2 * (N_opt - N_small)
print(f"  a model HALF that size ({N_small / 1e9:.1f}B) reaches the same loss with {D_small / 1e9:.0f}B tokens "
      f"({D_small / N_small:.0f} tokens/param): {6 * N_small * D_small / C:.2f}x the training compute")
print(f"  but every generated token costs 2N FLOPs: it saves {saved:.1e} FLOPs per token, so the extra {extra:.1e} training")
print(f"  FLOPs are repaid after {extra / saved:.1e} generated tokens; a widely used model serves far more than that")
print(f"  LLaMA-7B on 1T tokens: predicted loss {ch.loss(7e9, 1e12):.4f}, lower than the optimum above, at {6 * 7e9 * 1e12 / C:.1f}x its compute")
print("  (the fitted law is an extrapolation; the paper simply observes 7B 'continues to improve even after 1T tokens')")

# ---------------------------------------------------------------------------------------------------- 7
section("7. A tiny race on this repository's text: GPT-2 block vs LLaMA block (same width, depth, steps)")
gpt2 = load("gpt2_049", "049-Radford-et-al-2019-GPT-2/gpt2.py")
text = "\n".join(p.read_text(errors="ignore") for p in sorted(HERE.parents[1].rglob("EXPLAINED.md")))[:400_000]
data = torch.tensor(list(text.encode()), dtype=torch.long)
cut = int(0.95 * len(data)); tr, va = data[:cut], data[cut:]
CTX, B, STEPS = 128, 16, 250


def batch(src, g):
    i = torch.randint(0, len(src) - CTX - 1, (B,), generator=g)
    return torch.stack([src[j:j + CTX] for j in i])


def race(model):
    opt = torch.optim.AdamW(model.parameters(), 3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(0)
    for s in range(STEPS):
        loss = model.loss(batch(tr, g))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    gv = torch.Generator().manual_seed(1)
    with torch.no_grad():
        return sum(model.loss(batch(va, gv)).item() for _ in range(10)) / 10


results = {}
for name, make in (("GPT-2 (LayerNorm, GELU 4d, learned positions, tied)",
                    lambda: gpt2.GPT2(vocab=256, n_ctx=CTX, d=128, layers=3, heads=4)),
                   ("LLaMA (RMSNorm, SwiGLU, RoPE, untied)",
                    lambda: LLaMA(vocab=256, d=128, layers=3, heads=4, ctx=CTX, hidden=ffn_hidden(128, 32)))):
    torch.manual_seed(0)
    m = make()
    t = time.time()
    results[name] = race(m)
    print(f"  {name:52s} {sum(p.numel() for p in m.parameters()):>8,} params  val loss {results[name]:.3f} nats/byte "
          f"({time.time() - t:.1f}s)")
a, b = results.values()
print(f"  -> difference {a - b:+.3f} nats/byte after {STEPS} steps of {B}x{CTX} bytes; one seed, tiny scale: treat a")
print("     difference this size as suggestive, not as a measurement. Part of the gap is probably optimisation speed")
print("     at this tiny budget (tied embeddings, learned positions), not final quality; experiments.py ablates each")
print("     change separately, with seeds and longer training.")

# ---------------------------------------------------------------------------------------------------- 8
section("8. The paper's numbers")
print("  zero-shot common sense (Table 3): BoolQ PIQA HellaSwag WinoGrande ARC-e ARC-c OBQA")
for k, v in COMMON_SENSE.items():
    print(f"    {k:11s} " + " ".join(f"{x:5.1f}" for x in v))
wins = sum(x > y for x, y in zip(COMMON_SENSE["LLaMA-13B"], COMMON_SENSE["GPT-3 175B"]))
print(f"  LLaMA-13B beats GPT-3 175B on {wins} of 7 (13x fewer parameters)")
print("  MMLU 5-shot (Tables 9-10): " + ", ".join(f"{k} {v}" for k, v in MMLU.items()))
print(f"  numbers become digit sequences: {split_digits('pay 1250 by 2023')}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
