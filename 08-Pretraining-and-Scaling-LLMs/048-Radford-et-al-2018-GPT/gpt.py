"""Improving Language Understanding by Generative Pre-Training (GPT; Radford, Narasimhan, Salimans & Sutskever, 2018).

  Stage 1, unsupervised pre-training (Eq. 1):  L1(U) = sum_i log P(u_i | u_{i-k}, ..., u_{i-1}; Theta)
      h_0 = U W_e + W_p,  h_l = transformer_block(h_{l-1}),  P(u) = softmax(h_n W_e^T)                   (Eq. 2)
      12 decoder-only blocks, d = 768, 12 heads, FFN 3072, GELU, learned positions, context 512, dropout 0.1,
      Adam lr 2.5e-4 (2000-step warm-up, cosine to 0), N(0, 0.02) init, weight decay 0.01 on non-bias/gain
      weights (decoupled 'modified L2'), BPE with 40,000 merges  -> ~117M parameters.
  Stage 2, supervised fine-tuning (Eqs. 3-5):  P(y | x_1..x_m) = softmax(h_l^m W_y),
      L3(C) = L2(C) + lambda L1(C), lambda = 0.5 (language modelling kept as an auxiliary objective).
  Task-specific input transformations (Section 3.3, Figure 1), with new start <s>, delimiter $ and extract <e>
  tokens: classification [<s> text <e>]; entailment [<s> premise $ hypothesis <e>]; similarity both orders, the two
  final states added; multiple choice [<s> context $ answer_k <e>] for each k, softmax over k.
  Zero-shot heuristics (Section 5): e.g. sentiment by appending 'very' and comparing P('positive') vs P('negative').
"""

import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "transformer034",
    Path(__file__).resolve().parents[2] / "06-Transformers" / "034-Vaswani-et-al-2017-Attention-Is-All-You-Need" / "transformer.py")
transformer034 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(transformer034)
learn_bpe, apply_bpe = transformer034.learn_bpe, transformer034.apply_bpe


# ---------------------------------------------------------------------------------------------------- model
class Block(nn.Module):
    """A transformer decoder block as in the original Transformer (post-layer-norm): x + attn, LN, x + MLP, LN."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal):
        x = self.ln1(x + self.drop(self.attn(x, x, x, attn_mask=causal, need_weights=False)[0]))
        return self.ln2(x + self.mlp(x))


class GPT(nn.Module):
    def __init__(self, vocab, n_ctx=512, d=768, layers=12, heads=12, ff=3072, dropout=0.1):
        super().__init__()
        self.n_ctx = n_ctx
        self.tok = nn.Embedding(vocab, d)                                          # W_e
        self.pos = nn.Embedding(n_ctx, d)                                           # W_p (learned, not sinusoidal)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, heads, ff, dropout) for _ in range(layers)])
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)                                    # 'a simple N(0, 0.02)'
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def hidden(self, ids, n_layers=None):
        """Final (or first n_layers) block outputs h_l for ids (B, T)."""
        T = ids.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
        h = self.drop(self.tok(ids) + self.pos(torch.arange(T, device=ids.device)))   # h_0 = U W_e + W_p
        for blk in self.blocks[:n_layers]:
            h = blk(h, causal)
        return h

    def logits(self, h):
        return h @ self.tok.weight.T                                                # softmax(h_n W_e^T): tied

    def lm_loss(self, ids, mask=None):
        """-L1 / (number of predicted tokens): next-token cross-entropy. mask (B, T) marks real tokens."""
        lg = self.logits(self.hidden(ids))[:, :-1]
        tgt = ids[:, 1:]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
        if mask is None:
            return ce.mean()
        m = mask[:, 1:].float()
        return (ce * m).sum() / m.sum().clamp(min=1)

    def extend_vocab(self, n_new):
        """Add embeddings for new special tokens (<s>, $, <e>): the only new parameters besides W_y."""
        old = self.tok.weight.data
        self.tok = nn.Embedding(old.shape[0] + n_new, old.shape[1])
        nn.init.normal_(self.tok.weight, 0.0, 0.02)
        self.tok.weight.data[:old.shape[0]] = old
        return list(range(old.shape[0], old.shape[0] + n_new))


# ---------------------------------------------------------------------------------------------------- fine-tuning
class Specials:
    def __init__(self, start, delim, extract):
        self.start, self.delim, self.extract = start, delim, extract


def classification_input(x, sp):
    return [sp.start] + x + [sp.extract]


def entailment_input(premise, hypothesis, sp):
    return [sp.start] + premise + [sp.delim] + hypothesis + [sp.extract]


def similarity_inputs(a, b, sp):
    """No natural order: both orders are processed and their final states added."""
    return [[sp.start] + a + [sp.delim] + b + [sp.extract], [sp.start] + b + [sp.delim] + a + [sp.extract]]


def multiple_choice_inputs(context, answers, sp):
    return [[sp.start] + context + [sp.delim] + ans + [sp.extract] for ans in answers]


def pad_batch(seqs, pad=0):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = True
    return ids, mask


class FineTuner(nn.Module):
    """Wraps a pre-trained GPT with W_y on the final hidden state at the <e> token (Eq. 3). mode:
    'classify' -> logits over classes; 'similarity' -> pairs of sequences summed; 'choice' -> one score per
    candidate (softmax across candidates)."""

    def __init__(self, gpt, n_classes, dropout=0.1):
        super().__init__()
        self.gpt = gpt
        d = gpt.tok.weight.shape[1]
        self.drop = nn.Dropout(dropout)
        self.Wy = nn.Linear(d, n_classes)
        nn.init.normal_(self.Wy.weight, 0, 0.02); nn.init.zeros_(self.Wy.bias)

    def features(self, ids, mask):
        h = self.gpt.hidden(ids)
        last = mask.sum(1) - 1                                                     # position of <e>
        return h[torch.arange(len(ids)), last], h

    def forward(self, ids, mask):
        f, _ = self.features(ids, mask)
        return self.Wy(self.drop(f))

    def loss(self, ids, mask, y, lam=0.5):
        """L3 = L2 + lambda L1 written as a loss to minimise: CE(y) + lambda * LM cross-entropy on the inputs."""
        f, h = self.features(ids, mask)
        l2 = F.cross_entropy(self.Wy(self.drop(f)), y)
        if lam == 0:
            return l2
        lg = self.gpt.logits(h)[:, :-1]
        tgt = ids[:, 1:]
        m = mask[:, 1:].float()
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
        return l2 + lam * (ce * m).sum() / m.sum()


def transfer_layers(src, dst, n_layers):
    """Figure 2 (left): copy the embeddings and the first n_layers blocks of a pre-trained model into a fresh one."""
    dst.tok.load_state_dict(src.tok.state_dict())
    dst.pos.load_state_dict(src.pos.state_dict())
    for i in range(n_layers):
        dst.blocks[i].load_state_dict(src.blocks[i].state_dict())
    return dst


# ---------------------------------------------------------------------------------------------------- zero-shot
@torch.no_grad()
def next_token_logprobs(gpt, ids):
    """log P(next token | ids) for a single sequence (list of ids)."""
    gpt.eval()
    x = torch.tensor([ids[-gpt.n_ctx:]])
    return F.log_softmax(gpt.logits(gpt.hidden(x))[0, -1], -1)


@torch.no_grad()
def avg_logprob(gpt, context, continuation):
    """Average log-probability of `continuation` given `context` (Section 5's RACE / DPRD heuristics)."""
    gpt.eval()
    ids = context + continuation
    x = torch.tensor([ids[-gpt.n_ctx:]])
    lp = F.log_softmax(gpt.logits(gpt.hidden(x))[0], -1)
    start = len(ids[-gpt.n_ctx:]) - len(continuation)
    return torch.stack([lp[start + i - 1, t] for i, t in enumerate(continuation)]).mean().item()


def zero_shot_sentiment(gpt, ids, very, pos, neg):
    """SST-2 heuristic: append 'very', compare P(positive word) and P(negative word)."""
    lp = next_token_logprobs(gpt, ids + [very])
    return int(lp[pos] > lp[neg])


# ---------------------------------------------------------------------------------------------------- training helpers
def optimizer(model, lr, weight_decay=0.01):
    """Adam with decoupled weight decay on matrices only (biases and LayerNorm gains excluded)."""
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    rest = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}],
                             lr=lr)


def warmup_cosine(step, warmup, total):
    """Pre-training schedule: linear warm-up from 0, then cosine annealing to 0."""
    if step < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def warmup_linear(step, warmup, total):
    """Fine-tuning schedule: warm-up over 0.2% of training, then linear decay."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
