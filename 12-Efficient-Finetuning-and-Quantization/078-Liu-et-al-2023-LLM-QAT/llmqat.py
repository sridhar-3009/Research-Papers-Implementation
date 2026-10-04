"""LLM-QAT (Liu et al. 2023): quantization-AWARE training of LLMs without their training data. The full-precision
model generates its own training text, and a quantized copy is trained (with straight-through gradients) to match the
teacher's next-token distributions -- reaching 4-bit weights AND a 4-bit KV cache where post-training methods fail.

  quantizer        symmetric MinMax, no clipping (LLM outliers matter; clipping blew perplexity past 10,000):
                     X_q = alpha * round(X / alpha),  alpha = max|X| / (2^(N-1) - 1)
                   weights per output channel, activations per token, KV cache per token
  training         forward with fake-quantized weights / activations / KV; backward with the straight-through
                   estimator (round treated as identity); AdamW, lr 2e-5, cosine decay (the paper)
  data-free        sequences sampled from the FP model starting from <BOS>; 'hybrid' sampling = top-1 for the first
                   3-5 tokens, then sample from the distribution (pure top-1 repeats itself; pure sampling is noisy)
  loss             logits distillation: cross-entropy of the student against the teacher's full next-token
                   distribution (soft labels), sum_c p_T(c) log p_S(c)                                      (Eq. 4)

This file builds a tiny LLaMA (paper 056) that models WHOLE sequences [<BOS> task digits SEP answer] -- so it can
generate its own data -- plus the quantizers with STE, per-token KV-cache quantization (applied to the key / value
projections' outputs, i.e. before RoPE -- a simplification), the three generation modes, real-data and data-free QAT
with hard-label or logits distillation, and round-to-nearest PTQ as the baseline.
"""

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "llama056", Path(__file__).resolve().parents[2] / "08-Pretraining-and-Scaling-LLMs" / "056-Touvron-et-al-2023-LLaMA" / "llama.py")
llama = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llama)

# ----------------------------------------------------------------------------------------------- the toy language

DIGITS, N = 10, 6
TASK = {"copy": 10, "reverse": 11, "sort": 12}
SEP, BOS, VOCAB = 14, 15, 16
LEN = 2 * N + 3                                                      # <BOS> task x1..x6 SEP y1..y6


def make_seq(task, n, rng):
    x = rng.integers(0, DIGITS, size=(n, N))
    y = {"copy": x, "reverse": x[:, ::-1], "sort": np.sort(x, 1)}[task]
    return torch.tensor(np.concatenate([np.full((n, 1), BOS), np.full((n, 1), TASK[task]), x, np.full((n, 1), SEP), y], 1))


def real_batch(n, rng):
    return torch.cat([make_seq(t, n // 3, rng) for t in TASK])


def new_model(seed=0, d=64, layers=2, heads=4):
    torch.manual_seed(seed)
    return llama.LLaMA(vocab=VOCAB, d=d, layers=layers, heads=heads, ctx=LEN + 1, hidden=4 * d)


def lm_loss(model, seq):
    """Next-token loss on EVERY position: the model learns the whole sequence distribution, prompts included."""
    lg = model(seq[:, :-1])
    return F.cross_entropy(lg.reshape(-1, VOCAB), seq[:, 1:].reshape(-1))


def pretrain(model, rng, steps=900, batch=126, lr=5e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        loss = lm_loss(model, real_batch(batch, rng))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def accuracy(model, task=None, n=600):
    """Exact match of the 6 answer digits (teacher forcing on the answer, i.e. greedy correctness)."""
    rng = np.random.default_rng(999)
    seq = real_batch(n, rng) if task is None else make_seq(task, n, rng)
    pred = model(seq[:, :-1])[:, N + 2:].argmax(-1)
    return float((pred == seq[:, N + 3:]).all(1).float().mean())


@torch.no_grad()
def eval_loss(model, n=600):
    return float(lm_loss(model, real_batch(n, np.random.default_rng(998))))

# ----------------------------------------------------------------------------------------------- quantization with STE

def fake_quant(x, bits, dim):
    """Symmetric MinMax quantization along `dim` (per row of a weight = per output channel; last dim of an activation
    = per token), with the straight-through estimator: forward rounds, backward passes the gradient unchanged."""
    if bits is None or bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    alpha = x.detach().abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / alpha), -qmax, qmax) * alpha
    return x + (q - x).detach()                                          # STE


class QLinear(nn.Module):
    """A linear layer with fake-quantized weight (per output channel), input activation (per token) and, for the key /
    value projections, output (per token: the values that would be written to the KV cache)."""

    def __init__(self, lin, w_bits, a_bits, kv_bits=None):
        super().__init__()
        self.weight = lin.weight
        self.w_bits, self.a_bits, self.kv_bits = w_bits, a_bits, kv_bits

    def forward(self, x):
        y = F.linear(fake_quant(x, self.a_bits, -1), fake_quant(self.weight, self.w_bits, 1))
        return fake_quant(y, self.kv_bits, -1) if self.kv_bits else y


def quantize(model, w_bits, a_bits, kv_bits):
    """W-A-KV fake-quantized copy (every block linear; embeddings and head stay in full precision)."""
    m = copy.deepcopy(model)
    for b in m.blocks:
        for name in ("wq", "wk", "wv", "wo"):
            setattr(b.attn, name, QLinear(getattr(b.attn, name), w_bits, a_bits, kv_bits if name in ("wk", "wv") else None))
        for name in ("w1", "w2", "w3"):
            setattr(b.ffn, name, QLinear(getattr(b.ffn, name), w_bits, a_bits))
    return m

# ----------------------------------------------------------------------------------------------- data-free generation

@torch.no_grad()
def generate(teacher, n, mode="hybrid", k_det=3, temperature=1.0, seed=0):
    """Sample n sequences of length LEN from the teacher, starting from <BOS>. mode: 'top1' (always argmax),
    'sample' (always sample), 'hybrid' (argmax for the first k_det tokens, then sample)."""
    g = torch.Generator().manual_seed(seed)
    seq = torch.full((n, 1), BOS)
    for i in range(LEN - 1):
        logits = teacher(seq)[:, -1] / temperature
        if mode == "top1" or (mode == "hybrid" and i < k_det):
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).squeeze(1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return seq


def diversity(seqs):
    """Share of distinct sequences, and how often the generated sequence is a VALID task instance."""
    valid = 0
    inv_task = {v: k for k, v in TASK.items()}
    for s in seqs.tolist():
        t = inv_task.get(s[1])
        x, y = np.array(s[2:2 + N]), np.array(s[N + 3:])
        if t and s[N + 2] == SEP and (x < DIGITS).all():
            want = {"copy": x, "reverse": x[::-1], "sort": np.sort(x)}[t]
            valid += int((y == want).all())
    return len({tuple(s) for s in seqs.tolist()}) / len(seqs), valid / len(seqs)

# ----------------------------------------------------------------------------------------------- QAT

def train_qat(student, teacher, data, steps=400, batch=64, lr=1e-3, loss="logits", seed=0):
    """Train the fake-quantized student on `data` (a tensor of sequences). loss='logits': cross-entropy against the
    teacher's next-token distribution (soft labels); loss='hard': cross-entropy against the sampled tokens."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        seq = data[torch.randint(len(data), (batch,), generator=g)]
        s_logits = student(seq[:, :-1])
        if loss == "logits":
            with torch.no_grad():
                p_t = torch.softmax(teacher(seq[:, :-1]), -1)
            l = -(p_t * torch.log_softmax(s_logits, -1)).sum(-1).mean()
        else:
            l = F.cross_entropy(s_logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        opt.zero_grad()
        l.backward()
        opt.step()
        sched.step()
    return student


REPORTED = {
    "Table 1 (LLaMA-30B zero-shot avg, W-A-KV)": "8-8-4: LLM-QAT 69.7 vs SmoothQuant 50.7; 4-8-8: LLM-QAT beats the best "
                                                 "PTQ (RTN) by 1.4; 4-8-4: all PTQ poor, LLM-QAT 69.9 (FP16 - 1.5)",
    "data": "100k sequences generated by LLaMA-7B (max length 1024), hybrid sampling: top-1 for the first 3-5 tokens",
    "quantizer": "symmetric MinMax, no clipping (clipping -> perplexity > 10,000); per-channel weights, per-token "
                 "activations, per-token KV cache",
    "training": "logits distillation (soft labels); attention or hidden-state distillation hurt; AdamW, lr 2e-5, "
                "cosine, batch 1 per GPU",
    "Table 3 (data ablation, W4-A6)": "generated data generalises better than fine-tuning on real WikiText / C4 (which "
                                      "over-fit or transfer poorly to zero-shot tasks); sampled data, being more diverse, "
                                      "is much better than top-1; the hybrid scheme is the paper's default",
}
