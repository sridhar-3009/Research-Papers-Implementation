"""LLaMA: Open and Efficient Foundation Language Models (Touvron, Lavril, Izacard, Martinet, Lachaux, Lacroix, Rozière,
Goyal, Hambro, Azhar, Rodriguez, Joulin, Grave & Lample, Meta AI 2023).

  Goal: the best model for a given INFERENCE budget, not training budget: train smaller models on more tokens than
    Chinchilla's compute-optimal ~20 per parameter (7B on 1T tokens; Chinchilla would suggest 10B on 200B).
  Data (Table 1), public sources only, 1.4T tokens: CommonCrawl/CCNet 67%, C4 15%, GitHub 4.5%, Wikipedia 4.5%,
    Gutenberg+Books3 4.5%, ArXiv 2.5%, StackExchange 2%; Wikipedia and books seen ~2 epochs, the rest ~1.
  Tokenizer: SentencePiece BPE, 32k vocabulary, numbers split into single digits, byte fallback for unknown UTF-8.
  Architecture (Section 2.2), a GPT-style decoder with:
    pre-normalisation with RMSNorm [GPT-3]: x / sqrt(mean(x^2) + eps) * g (no mean subtraction, no bias);
    SwiGLU feed-forward [PaLM]: W2 (silu(W1 x) * W3 x), hidden size 2/3 * 4d (rounded up to a multiple of 256);
    rotary position embeddings (RoPE) [GPT-Neo] applied to queries and keys in every layer, no absolute positions.
  Table 2: 7B (d 4096, 32 heads, 32 layers), 13B (5120, 40, 40), 33B (6656, 52, 60), 65B (8192, 64, 80); context 2048;
    AdamW(0.9, 0.95), weight decay 0.1, grad clip 1.0, 2000 warm-up steps, cosine to 10% of the peak LR
    (3e-4 for 7B/13B, 1.5e-4 for 33B/65B), batch 4M tokens; 1.0T tokens (7B, 13B) or 1.4T (33B, 65B).
  Efficiency: causal attention that never stores the attention matrix nor computes masked scores (xformers),
    activation checkpointing that saves expensive activations; 65B: ~380 tokens/s/GPU on 2048 A100-80GB -> ~21 days.
  Results: LLaMA-13B beats GPT-3 (175B) on most benchmarks; LLaMA-65B is competitive with Chinchilla-70B and
    PaLM-540B; MMLU 5-shot 63.4 (65B), 68.9 after brief instruction tuning (LLaMA-I).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------- layers
class RMSNorm(nn.Module):
    """Zhang & Sennrich 2019: rescale by the root-mean-square, with a learned gain; no centring, no bias."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def ffn_hidden(d, multiple_of=256):
    """2/3 * 4d, rounded UP to a multiple of 256 (the released code): 4096 -> 11008, 8192 -> 22016."""
    h = int(2 * 4 * d / 3)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """Shazeer 2020: a gated linear unit with SiLU (swish) gate. Three matrices instead of two, so the hidden size is
    shrunk by 2/3 to keep the parameter count of a 4d MLP."""

    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or ffn_hidden(d)
        self.w1, self.w3 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rope_frequencies(head_dim, length, base=10000.0):
    """Angles m * theta_i for positions m and pair index i, theta_i = base^(-2i / head_dim); returned as cos, sin."""
    theta = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    ang = torch.arange(length).float()[:, None] * theta[None]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """Rotate each consecutive pair (x_{2i}, x_{2i+1}) by angle m * theta_i. x: (..., T, head_dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[-2]], sin[: x.shape[-2]]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1)
    return out.flatten(-2)


def memory_efficient_causal_attention(q, k, v, block=64):
    """Causal attention in key blocks with an online softmax: never materialises the T x T matrix and skips blocks
    that are entirely in the future (the two savings described in Section 2.4). q, k, v: (B, H, T, Dh)."""
    B, H, T, Dh = q.shape
    out = torch.zeros_like(q)
    for qs in range(0, T, block):
        qb = q[:, :, qs:qs + block] / math.sqrt(Dh)
        qpos = torch.arange(qs, min(qs + block, T), device=q.device)
        m = torch.full(qb.shape[:-1], -float("inf"), device=q.device)          # running max
        l = torch.zeros(qb.shape[:-1], device=q.device)                         # running normaliser
        acc = torch.zeros_like(qb)
        for ks in range(0, min(qs + block, T), block):                          # future blocks are never touched
            kb, vb = k[:, :, ks:ks + block], v[:, :, ks:ks + block]
            s = qb @ kb.transpose(-1, -2)
            kpos = torch.arange(ks, min(ks + block, T), device=q.device)
            s = s.masked_fill(kpos[None] > qpos[:, None], -float("inf"))
            m_new = torch.maximum(m, s.amax(-1))
            p = torch.exp(s - m_new[..., None])
            scale = torch.exp(m - m_new)
            l = l * scale + p.sum(-1)
            acc = acc * scale[..., None] + p @ vb
            m = m_new
        out[:, :, qs:qs + block] = acc / l[..., None]
    return out


class Attention(nn.Module):
    def __init__(self, d, heads, efficient=False):
        super().__init__()
        self.h, self.dh, self.efficient = heads, d // heads, efficient
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = (w(x).view(B, T, self.h, self.dh).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)                   # positions enter ONLY here
        if self.efficient:
            o = memory_efficient_causal_attention(q, k, v)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, d, heads, hidden=None, efficient=False):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(d), Attention(d, heads, efficient)
        self.ffn_norm, self.ffn = RMSNorm(d), SwiGLU(d, hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)                          # pre-normalisation
        return x + self.ffn(self.ffn_norm(x))


class LLaMA(nn.Module):
    def __init__(self, vocab=32000, d=4096, layers=32, heads=32, ctx=2048, hidden=None, efficient=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, hidden, efficient) for _ in range(layers))
        self.norm = RMSNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)                                # NOT tied to the input embedding
        cos, sin = rope_frequencies(d // heads, ctx)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids):
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return self.out(self.norm(x))

    def loss(self, ids):
        lg = self(ids)[:, :-1]
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))


# ---------------------------------------------------------------------------------------------------- counting
TABLE_2 = {  # name: (params reported, d, heads, layers, lr, tokens)
    "7B": (6.7e9, 4096, 32, 32, 3.0e-4, 1.0e12), "13B": (13.0e9, 5120, 40, 40, 3.0e-4, 1.0e12),
    "33B": (32.5e9, 6656, 52, 60, 1.5e-4, 1.4e12), "65B": (65.2e9, 8192, 64, 80, 1.5e-4, 1.4e12)}


def param_count(d, layers, vocab=32000):
    """Embedding + output (untied) + per layer (4 d^2 attention + 3 d h SwiGLU + 2 d RMSNorm gains) + final norm."""
    h = ffn_hidden(d)
    return 2 * vocab * d + layers * (4 * d * d + 3 * d * h + 2 * d) + d


def training_days(tokens, tokens_per_sec_per_gpu=380, gpus=2048):
    return tokens / (tokens_per_sec_per_gpu * gpus) / 86400


def carbon(gpu_hours, watts=400, pue=1.1, kg_per_kwh=0.385):
    """Table 15: MWh = GPU-h x power x PUE; tCO2eq = MWh x 0.385 (US national average)."""
    mwh = gpu_hours * watts * pue / 1e6
    return mwh, mwh * kg_per_kwh


TABLE_15_GPU_HOURS = {"7B": 82_432, "13B": 135_168, "33B": 530_432, "65B": 1_022_362, "OPT-175B": 809_472,
                      "BLOOM-175B": 1_082_880}
TABLE_1 = {  # source: (sampling proportion, epochs at 1.4T tokens, disk size GB)
    "CommonCrawl": (0.670, 1.10, 3300), "C4": (0.150, 1.06, 783), "Github": (0.045, 0.64, 328),
    "Wikipedia": (0.045, 2.45, 83), "Books": (0.045, 2.23, 85), "ArXiv": (0.025, 1.06, 92),
    "StackExchange": (0.020, 1.03, 78)}
COMMON_SENSE = {  # Table 3, zero-shot: BoolQ PIQA HellaSwag WinoGrande ARC-e ARC-c OBQA (SIQA omitted)
    "GPT-3 175B": (60.5, 81.0, 78.9, 70.2, 68.8, 51.4, 57.6),
    "LLaMA-7B": (76.5, 79.8, 76.1, 70.1, 72.8, 47.6, 57.2), "LLaMA-13B": (78.1, 80.1, 79.2, 73.0, 74.8, 52.7, 56.4),
    "LLaMA-33B": (83.1, 82.3, 82.8, 76.0, 80.0, 57.8, 58.6), "LLaMA-65B": (85.3, 82.8, 84.2, 77.0, 78.9, 56.0, 60.2)}
MMLU = {"GPT-3 175B": 43.9, "Gopher 280B": 60.0, "Chinchilla 70B": 67.5, "PaLM 540B": 69.3, "LLaMA-7B": 35.1,
        "LLaMA-13B": 46.9, "LLaMA-33B": 57.8, "LLaMA-65B": 63.4, "LLaMA-I 65B": 68.9}


# ---------------------------------------------------------------------------------------------------- tokenizer bits
def split_digits(text):
    """LLaMA's number handling: every digit is its own piece, so '2023' -> '2', '0', '2', '3' (consistent arithmetic)."""
    out, cur = [], ""
    for ch in text:
        if ch.isdigit():
            if cur:
                out.append(cur); cur = ""
            out.append(ch)
        else:
            cur += ch
    return out + ([cur] if cur else [])


def byte_fallback(text, vocab):
    """Characters not in the vocabulary become their UTF-8 bytes as <0xNN> tokens, so nothing is ever <unk>."""
    out = []
    for ch in text:
        out += [ch] if ch in vocab else [f"<0x{b:02X}>" for b in ch.encode("utf-8")]
    return out


def lr_schedule(step, peak, warmup=2000, total=250_000, final_frac=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def normalised_choice_score(logp_completion_given_context, logp_completion_given_answer_prompt):
    """Section 3: for some tasks, score log P(completion | context) - log P(completion | 'Answer:')."""
    return logp_completion_given_context - logp_completion_given_answer_prompt
