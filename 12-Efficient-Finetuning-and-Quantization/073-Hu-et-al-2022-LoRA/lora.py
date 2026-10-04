"""LoRA: Low-Rank Adaptation of Large Language Models (Hu et al. 2022).

Freeze the pre-trained weight W0 (d x k) and learn only a low-rank update:
    h = W0 x + (alpha / r) * B A x,      B: d x r (initialised to ZERO), A: r x k (random Gaussian), r << min(d, k)
so the update Delta W = (alpha/r) B A starts at zero and has rank <= r. After training, W = W0 + Delta W can be MERGED
into one matrix, so inference costs exactly what the original model costs (unlike adapters).

This file has:
  * LoRALinear (with merge / unmerge for task switching) and helpers to inject it into chosen attention matrices of the
    tiny LLaMA from paper 056, freeze everything else, and count trainable parameters;
  * GPT-3 175B parameter / checkpoint arithmetic (18M trainable parameters at r = 4 on Wq, Wv; 350 GB -> 35 MB);
  * a toy transfer setting: pre-train on COPY / REVERSE / SORT of digit strings, then adapt to a NEW behaviour (the
    SORT prompt should now sort in DESCENDING order) with full fine-tuning or LoRA -- the downstream task is written
    with existing tokens, as in the paper, because LoRA leaves the embeddings frozen;
  * the paper's analyses: subspace similarity between the r = 8 and r = 64 solutions (Figure 3), and how much Delta W
    amplifies directions of W that W itself under-emphasises (Table 7).
"""

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

# ----------------------------------------------------------------------------------------------- the LoRA layer

class LoRALinear(nn.Module):
    """y = W0 x + (alpha/r) B A x with W0 frozen. merge() folds the update into W0 (zero extra inference cost)."""

    def __init__(self, base: nn.Linear, r, alpha=None, seed=0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        g = torch.Generator().manual_seed(seed)
        self.r, self.scale = r, (alpha if alpha is not None else r) / r          # the paper sets alpha = first r tried
        self.A = nn.Parameter(torch.randn(r, d_in, generator=g) / math.sqrt(d_in))   # random Gaussian
        self.B = nn.Parameter(torch.zeros(d_out, r))                                  # zero: Delta W = 0 at start
        self.merged, self.enabled = False, True

    def delta(self):
        return self.scale * self.B @ self.A

    def forward(self, x):
        y = self.base(x)
        if self.enabled and not self.merged:
            y = y + self.scale * (x @ self.A.T) @ self.B.T
        return y

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta()
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= self.delta()
            self.merged = False


def add_lora(model, targets=("wq", "wv"), r=4, alpha=None):
    """Freeze the whole model, then wrap the chosen attention matrices of every block with LoRA."""
    for p in model.parameters():
        p.requires_grad_(False)
    for i, b in enumerate(model.blocks):
        for j, name in enumerate(targets):
            setattr(b.attn, name, LoRALinear(getattr(b.attn, name), r, alpha, seed=10_000 * r + 100 * i + j))
    return model


def set_adapters(model, enabled):
    """Switch every (unmerged) LoRA adapter on or off: off = exactly the frozen pre-trained model."""
    for layer in lora_layers(model):
        layer.enabled = enabled


def lora_layers(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_params_gpt3(r, n_matrices, d_model=12288, layers=96):
    """Each adapted d x d matrix adds r (d + d) parameters per layer."""
    return layers * n_matrices * r * 2 * d_model

# ----------------------------------------------------------------------------------------------- toy tasks

DIGITS = 10
TASKS = {"copy": 10, "reverse": 11, "sort": 12, "desc": 12}     # the NEW task re-uses the SORT prompt, new behaviour
SEP, VOCAB, N = 14, 16, 6


def make_batch(task, n, rng):
    x = rng.integers(0, DIGITS, size=(n, N))
    y = {"copy": x, "reverse": x[:, ::-1], "sort": np.sort(x, 1), "desc": -np.sort(-x, 1)}[task]
    seq = np.concatenate([np.full((n, 1), TASKS[task]), x, np.full((n, 1), SEP), y], 1)
    return torch.tensor(seq)


def seq_loss(model, seq):
    """Next-token loss on the OUTPUT digits only."""
    lg = model(seq)[:, N + 1:-1]
    return F.cross_entropy(lg.reshape(-1, VOCAB), seq[:, N + 2:].reshape(-1))


@torch.no_grad()
def accuracy(model, task, rng=None, n=1000):
    """Whole-sequence exact match with teacher forcing (equals greedy decoding: every step must be argmax-correct).
    Uses a FIXED evaluation set (seed 999) unless an rng is given."""
    seq = make_batch(task, n, rng if rng is not None else np.random.default_rng(999))
    pred = model(seq)[:, N + 1:-1].argmax(-1)
    return float((pred == seq[:, N + 2:]).all(1).float().mean())


def new_model(seed=0, d=64, layers=2, heads=4):
    torch.manual_seed(seed)
    return llama.LLaMA(vocab=VOCAB, d=d, layers=layers, heads=heads, ctx=2 * N + 2, hidden=4 * d)


def pretrain(model, rng, steps=800, batch=128, lr=5e-3, tasks=("copy", "reverse", "sort")):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for s in range(steps):
        loss = sum(seq_loss(model, make_batch(t, batch // len(tasks), rng)) for t in tasks) / len(tasks)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def finetune(model, task, rng, steps=300, batch=64, lr=3e-3, train_data=None):
    """Adam on whatever parameters require grad. train_data: an optional fixed tensor of examples (small-data regime)."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        seq = make_batch(task, batch, rng) if train_data is None else train_data[torch.randint(len(train_data), (batch,))]
        loss = seq_loss(model, seq)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

# ----------------------------------------------------------------------------------------------- analyses

def subspace_similarity(A1, A2, i, j):
    """phi(A1, A2, i, j) = ||U1[:, :i]^T U2[:, :j]||_F^2 / min(i, j), U = right singular vectors of A (Section 7.2)."""
    U1 = torch.linalg.svd(A1, full_matrices=False).Vh.T
    U2 = torch.linalg.svd(A2, full_matrices=False).Vh.T
    return float((U1[:, :i].T @ U2[:, :j]).pow(2).sum() / min(i, j))


def amplification(W, dW, r):
    """Table 7: project W onto the top-r singular directions of dW: ||U^T W V^T||_F, compared with ||dW||_F,
    and with projections onto W's own top-r directions and onto random directions."""
    def proj(M, U, V):
        return float((U.T @ M @ V).norm())
    U, _, Vh = torch.linalg.svd(dW)
    Uw, _, Vhw = torch.linalg.svd(W)
    g = torch.Generator().manual_seed(0)
    Ur = torch.linalg.qr(torch.randn(W.shape[0], r, generator=g)).Q
    Vr = torch.linalg.qr(torch.randn(W.shape[1], r, generator=g)).Q
    on_dW = proj(W, U[:, :r], Vh[:r].T)
    return {"||U^T W V^T|| on dW's directions": on_dW, "on W's own top directions": proj(W, Uw[:, :r], Vhw[:r].T),
            "on random directions": proj(W, Ur, Vr), "||dW||_F": float(dW.norm()),
            "amplification ||dW|| / ||U^T W V^T||": float(dW.norm()) / max(on_dW, 1e-12)}


REPORTED = {
    "GPT-3 175B": "trainable parameters cut 10,000x, GPU memory 3x (1.2 TB -> 350 GB VRAM), checkpoint 350 GB -> 35 MB "
                  "(r = 4 on Wq, Wv), 25% faster training, no extra inference latency",
    "Table 4 (GPT-3, WikiSQL / MNLI-m / SAMSum R1)": "full FT 73.8 / 89.5 / 52.0; LoRA 4.7M params 73.4 / 91.7 / 53.8; "
                                                    "LoRA 37.7M 74.0 / 91.6 / 53.4",
    "Table 5 (18M budget, WikiSQL / MNLI)": "Wq r8 70.4/91.0, Wk r8 70.0/90.8, Wv r8 73.0/91.0, Wo r8 73.2/91.3, "
                                           "Wq+Wk r4 71.4/91.3, Wq+Wv r4 73.7/91.3, all four r2 73.7/91.7",
    "Table 6": "r = 1 already suffices for Wq + Wv (WikiSQL 73.4, MNLI 91.3); Wq alone needs larger r",
    "Figure 3": "the top directions of the r = 8 and r = 64 solutions overlap strongly; the rest is mostly noise",
    "Table 7": "Delta W amplifies directions NOT emphasised in W; factor ~21.5 = 6.91 / 0.32 for r = 4",
}
