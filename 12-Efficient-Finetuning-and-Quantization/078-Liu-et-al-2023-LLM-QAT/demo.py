"""LLM-QAT in ~30 seconds: fake quantization with the straight-through estimator, a tiny LLaMA that models whole
sequences and can therefore generate its own training data, post-training quantization at several W-A-KV settings,
and quantization-aware training from real vs self-generated (data-free) sequences with logits or hard-label
distillation (Tables 1 and 3)."""

import time
from collections import Counter

import numpy as np
import torch

from llmqat import (REPORTED, TASK, accuracy, diversity, fake_quant, generate, new_model, pretrain, quantize,
                    real_batch, train_qat)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Symmetric MinMax fake quantization with the straight-through estimator")
x = torch.tensor([[0.3, -1.2, 0.05, 2.0]], requires_grad=True)
q = fake_quant(x, 4, -1)
q.sum().backward()
print(f"  x = [0.3, -1.2, 0.05, 2.0], 4 bits per token: alpha = 2.0 / 7 = {2 / 7:.4f}; forward -> "
      f"{[round(v, 3) for v in q.tolist()[0]]}")
print(f"  backward: d(sum)/dx = {x.grad.tolist()[0]} -- round() has zero gradient almost everywhere, so QAT pretends")
print("  it is the identity (STE) and the full-precision weights keep learning through the quantized forward pass")

section("2. A tiny LLaMA that models WHOLE sequences, so it can write its own training data")
t = time.time()
teacher = pretrain(new_model(), np.random.default_rng(0))
print(f"  sequences '<BOS> task x1..x6 SEP y1..y6', next-token loss on every position ({time.time() - t:.0f}s): answer "
      f"accuracy {accuracy(teacher):.1%}")
inv = {v: k for k, v in TASK.items()}
gen = {}
print("    generation mode               distinct sequences   valid task instances   task mix")
for mode, label in (("top1", "top-1 always"), ("sample", "sample always"), ("hybrid", "hybrid (top-1 for 3 tokens)")):
    gen[mode] = generate(teacher, 3000, mode, seed=1)
    d, v = diversity(gen[mode])
    mix = Counter(inv.get(t, "?") for t in gen[mode][:, 1].tolist())
    print(f"    {label:30s}      {d:6.1%}              {v:6.1%}          {dict(mix)}")
print("  -> top-1 repeats one sequence; sampling gives diverse, mostly valid data. Honest note: the paper's hybrid")
print("     scheme fails HERE -- our first generated token picks the task, so making it deterministic leaves only")
print("     'sort'. In natural text the first few tokens matter less; the paper found sampled data far better than")
print("     top-1 and uses hybrid sampling as its default.")

section("3. Post-training quantization (round-to-nearest), W-A-KV bits")
for cfg in ((8, 8, 8), (4, 8, 8), (4, 8, 4), (4, 6, 4), (3, 8, 8), (3, 8, 3), (2, 8, 8)):
    print(f"    {'-'.join(map(str, cfg)):8s} accuracy {accuracy(quantize(teacher, *cfg)):6.1%}")
print("  -> 8-bit everything is free; 4-bit weights + 4-bit KV cache lose a little; 3 bits hurt and 2-bit weights collapse")
print("     (the paper: at 4-8-4 'all PTQ methods produce poor results' on LLaMA)")

section("4. Quantization-aware training at W3-A8-KV3 (300 steps, the FP model as teacher)")
real = real_batch(3000, np.random.default_rng(5))
print(f"    PTQ (no training)                                   accuracy {accuracy(quantize(teacher, 3, 8, 3)):6.1%}")
res = {}
for label, data, loss in (("QAT, REAL data, logits distillation", real, "logits"),
                          ("QAT, generated (sampled), logits distillation", gen["sample"], "logits"),
                          ("QAT, generated (sampled), hard labels", gen["sample"], "hard"),
                          ("QAT, generated (hybrid), logits", gen["hybrid"], "logits"),
                          ("QAT, generated (top-1), logits", gen["top1"], "logits")):
    s = train_qat(quantize(teacher, 3, 8, 3), teacher, data, steps=300, loss=loss)
    res[label] = accuracy(s)
    print(f"    {label:50s}  accuracy {res[label]:6.1%}")
print("  -> DATA-FREE QAT works: training on the model's own sampled generations recovers as much as real data.")
print("     The data's diversity is what matters: top-1 data does not help at all and our sort-only hybrid data much less")
print("     than sampled data. Honest note:")
print("     soft (logits) and hard labels did equally well here; the paper found soft labels better.")

section("5. Extreme case: 2-bit weights (W2-A8-KV8)")
print(f"    PTQ                                    accuracy {accuracy(quantize(teacher, 2, 8, 8)):6.1%}")
for label, data in (("QAT on real data", real), ("QAT on generated (sampled) data", gen["sample"])):
    s = train_qat(quantize(teacher, 2, 8, 8), teacher, data, steps=300)
    print(f"    {label:38s} accuracy {accuracy(s):6.1%}")
print("  -> where PTQ collapses, QAT recovers almost everything -- with no access to the original data")

section("6. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
