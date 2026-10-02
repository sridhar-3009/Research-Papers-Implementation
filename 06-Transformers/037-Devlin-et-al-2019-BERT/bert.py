"""BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin, Chang, Lee &
Toutanova, NAACL 2019).

  Model: a Transformer ENCODER (Paper 034) with L layers, hidden size H, A heads, feed-forward 4H, GELU.
         BERT-base L=12 H=768 A=12 (110M), BERT-large L=24 H=1024 A=16 (340M).
  Input: [CLS] sentence A [SEP] sentence B [SEP]; embedding = token + segment (A/B) + learned position.
  Pre-training (Section 3.1), loss = mean masked-LM loss + mean next-sentence loss:
    Masked LM: pick 15% of the tokens; replace 80% of those by [MASK], 10% by a random token, keep 10%;
               predict the originals from the final hidden states (bidirectional context).
    NSP: B is the real next sentence 50% of the time, a random one otherwise; classify from C (the [CLS] state).
  Fine-tuning (Section 3.2/4): add one small output layer and train everything:
    classification log softmax(C W^T); SQuAD spans with start/end vectors S, E; SWAG scores v . C per choice;
    token tagging from T_i.
  Optimizer (Appendix A.2): Adam lr 1e-4, betas (0.9, 0.999), weight decay 0.01, 10k-step warm-up, linear decay.
"""

import collections
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD, UNK, CLS, SEP, MASK = range(5)
IGNORE = -100                                                   # label for positions that are not predicted


# ---------------------------------------------------------------------------------------------------- WordPiece
def build_wordpiece_vocab(word_counts, size):
    """A small WordPiece-style vocabulary: start from characters (word-initial 'a', inner '##a') and greedily merge
    the most frequent adjacent pair until `size` pieces exist. (Google's WordPiece picks merges by likelihood gain;
    frequency is the usual simple stand-in.)"""
    words = {tuple([w[0]] + ["##" + c for c in w[1:]]): c for w, c in word_counts.items()}
    vocab = set(SPECIALS) | {p for w in words for p in w}
    while len(vocab) < size:
        pairs = collections.Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[a, b] += c
        if not pairs:
            break
        a, b = max(pairs, key=lambda p: (pairs[p], p))
        merged = a + b[2:]
        vocab.add(merged)
        new = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == (a, b):
                    out.append(merged); i += 2
                else:
                    out.append(w[i]); i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + c
        words = new
    return SPECIALS + sorted(vocab - set(SPECIALS))


def wordpiece_tokenize(word, vocab, max_chars=100):
    """BERT's greedy longest-match-first: 'playing' -> ['play', '##ing']; [UNK] if no split exists."""
    if len(word) > max_chars:
        return ["[UNK]"]
    pieces, start = [], 0
    while start < len(word):
        end, cur = len(word), None
        while start < end:
            sub = word[start:end] if start == 0 else "##" + word[start:end]
            if sub in vocab:
                cur = sub
                break
            end -= 1
        if cur is None:
            return ["[UNK]"]
        pieces.append(cur)
        start = end
    return pieces


# ---------------------------------------------------------------------------------------------------- inputs
def pack_pair(a, b=None, max_len=None):
    """[CLS] a [SEP] (b [SEP]); returns token ids and segment ids (0 for A, 1 for B). Truncates the longer
    sentence first if max_len is given."""
    a, b = list(a), list(b) if b is not None else None
    if max_len is not None:
        budget = max_len - (3 if b is not None else 2)
        while len(a) + (len(b) if b else 0) > budget:
            (a if b is None or len(a) >= len(b) else b).pop()
    ids = [CLS] + a + [SEP] + ((b + [SEP]) if b is not None else [])
    seg = [0] * (len(a) + 2) + ([1] * (len(b) + 1) if b is not None else [])
    return ids, seg


def mask_tokens(ids, vocab_size, rate=0.15, probs=(0.8, 0.1, 0.1), generator=None):
    """Section 3.1. ids (N, T). Choose `rate` of the non-special tokens; of those, `probs` = (MASK, random, keep).
    Returns (inputs, labels) where labels hold the original id at chosen positions and IGNORE elsewhere."""
    g = generator
    special = ids < len(SPECIALS)
    chosen = (torch.rand(ids.shape, generator=g) < rate) & ~special
    labels = torch.where(chosen, ids, torch.full_like(ids, IGNORE))
    r = torch.rand(ids.shape, generator=g)
    to_mask = chosen & (r < probs[0])
    to_rand = chosen & (r >= probs[0]) & (r < probs[0] + probs[1])
    inputs = ids.clone()
    inputs[to_mask] = MASK
    inputs[to_rand] = torch.randint(len(SPECIALS), vocab_size, (int(to_rand.sum()),), generator=g)
    return inputs, labels


def nsp_pairs(documents, n, rng):
    """Appendix A.2: sample sentence A from a document; 50% of the time B is the next sentence (IsNext = 0),
    otherwise a random sentence from a different document (NotNext = 1)."""
    out = []
    docs = [d for d in documents if len(d) >= 2]
    for _ in range(n):
        d = rng.randrange(len(docs))
        i = rng.randrange(len(docs[d]) - 1)
        if rng.random() < 0.5:
            out.append((docs[d][i], docs[d][i + 1], 0))
        else:
            other = rng.choice([k for k in range(len(docs)) if k != d])
            out.append((docs[d][i], rng.choice(docs[other]), 1))
    return out


def collate(pairs_or_seqs):
    """List of (ids, seg) -> padded tensors ids, seg, attention mask."""
    T = max(len(i) for i, _ in pairs_or_seqs)
    ids = torch.tensor([i + [PAD] * (T - len(i)) for i, _ in pairs_or_seqs])
    seg = torch.tensor([s + [0] * (T - len(s)) for _, s in pairs_or_seqs])
    return ids, seg, ids != PAD


# ---------------------------------------------------------------------------------------------------- the model
class Layer(nn.Module):
    """One Transformer encoder layer (post-LN, as in Vaswani et al. and the BERT code), GELU feed-forward."""

    def __init__(self, H, A, dropout):
        super().__init__()
        self.A, self.dh = A, H // A
        self.qkv, self.o = nn.Linear(H, 3 * H), nn.Linear(H, H)
        self.ln1, self.ln2 = nn.LayerNorm(H, eps=1e-12), nn.LayerNorm(H, eps=1e-12)
        self.ff1, self.ff2 = nn.Linear(H, 4 * H), nn.Linear(4 * H, H)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, causal=False):
        N, T, H = x.shape
        q, k, v = self.qkv(x).view(N, T, 3, self.A, self.dh).permute(2, 0, 3, 1, 4)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))         # no attending to padding
        if causal:                                                         # the 'LTR' ablation (Table 5)
            s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        a = (self.drop(torch.softmax(s, -1)) @ v).transpose(1, 2).reshape(N, T, H)
        x = self.ln1(x + self.drop(self.o(a)))
        return self.ln2(x + self.drop(self.ff2(F.gelu(self.ff1(x)))))


class Bert(nn.Module):
    def __init__(self, vocab, H=768, L=12, A=12, max_len=512, dropout=0.1, causal=False):
        """causal=True gives the left-to-right model of the 'LTR & No NSP' ablation (like OpenAI GPT)."""
        super().__init__()
        self.causal = causal
        self.tok, self.seg, self.pos = nn.Embedding(vocab, H), nn.Embedding(2, H), nn.Embedding(max_len, H)
        self.ln, self.drop = nn.LayerNorm(H, eps=1e-12), nn.Dropout(dropout)
        self.layers = nn.ModuleList([Layer(H, A, dropout) for _ in range(L)])
        self.pool = nn.Linear(H, H)                                        # tanh 'pooler' on [CLS]
        # pre-training heads
        self.mlm_dense, self.mlm_ln = nn.Linear(H, H), nn.LayerNorm(H, eps=1e-12)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab))                   # output weights tied to self.tok
        self.nsp = nn.Linear(H, 2)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)                             # BERT's truncated-normal(0.02) init
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, ids, seg, mask, all_layers=False):
        """Returns (T: final hidden states (N, T, H), C: pooled [CLS] (N, H)) and optionally every layer."""
        x = self.tok(ids) + self.seg(seg) + self.pos(torch.arange(ids.shape[1]))
        x = self.drop(self.ln(x))
        hs = []
        for layer in self.layers:
            x = layer(x, mask, self.causal)
            hs.append(x)
        C = torch.tanh(self.pool(x[:, 0]))
        return (x, C, hs) if all_layers else (x, C)

    def mlm_logits(self, T):
        h = self.mlm_ln(F.gelu(self.mlm_dense(T)))
        return h @ self.tok.weight.T + self.mlm_bias

    def pretrain_loss(self, ids, seg, mask, mlm_labels, nsp_labels):
        """'the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood'."""
        T, C = self(ids, seg, mask)
        mlm = F.cross_entropy(self.mlm_logits(T).reshape(-1, self.tok.num_embeddings), mlm_labels.reshape(-1),
                              ignore_index=IGNORE)
        nsp = F.cross_entropy(self.nsp(C), nsp_labels)
        return mlm + nsp, mlm, nsp


# ---------------------------------------------------------------------------------------------------- fine-tuning heads
class Classifier(nn.Module):
    """GLUE: log softmax(C W^T); W is the only new parameter matrix."""

    def __init__(self, bert, K):
        super().__init__()
        self.bert, self.W = bert, nn.Linear(bert.pool.out_features, K)

    def forward(self, ids, seg, mask):
        return self.W(self.bert(ids, seg, mask)[1])


class SpanQA(nn.Module):
    """SQuAD: P(start = i) = softmax_i(S . T_i), P(end = j) = softmax_j(E . T_j); score(i, j) = S.T_i + E.T_j.
    SQuAD 2.0: 'no answer' is the span at [CLS]: s_null = S.C' + E.C' (C' = T_0)."""

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        H = bert.pool.out_features
        self.S, self.E = nn.Parameter(torch.randn(H) * 0.02), nn.Parameter(torch.randn(H) * 0.02)

    def forward(self, ids, seg, mask):
        T, _ = self.bert(ids, seg, mask)
        start, end = T @ self.S, T @ self.E
        neg = ~mask
        return start.masked_fill(neg, float("-inf")), end.masked_fill(neg, float("-inf"))

    @staticmethod
    def best_span(start, end, min_pos=1, max_len=30, null_threshold=None):
        """max_{j >= i} S.T_i + E.T_j over context positions; with null_threshold tau, predict 'no answer' unless
        the best span beats s_null + tau."""
        best, arg = -float("inf"), (0, 0)
        n = start.shape[0]
        for i in range(min_pos, n):
            for j in range(i, min(n, i + max_len)):
                s = (start[i] + end[j]).item()
                if s > best:
                    best, arg = s, (i, j)
        if null_threshold is not None and best <= (start[0] + end[0]).item() + null_threshold:
            return (0, 0)
        return arg


class MultipleChoice(nn.Module):
    """SWAG: one sequence per choice ([CLS] context [SEP] ending [SEP]); score = v . C; softmax over choices."""

    def __init__(self, bert):
        super().__init__()
        self.bert, self.v = bert, nn.Linear(bert.pool.out_features, 1)

    def forward(self, ids, seg, mask):                    # (N, choices, T)
        N, K, T = ids.shape
        C = self.bert(ids.reshape(N * K, T), seg.reshape(N * K, T), mask.reshape(N * K, T))[1]
        return self.v(C).view(N, K)


def feature_based(bert, ids, seg, mask, how="concat_last_four"):
    """Section 5.3 (Table 7): fixed features from the frozen model, e.g. the concatenation of the top four layers."""
    with torch.no_grad():
        _, _, hs = bert(ids, seg, mask, all_layers=True)
    if how == "concat_last_four":
        return torch.cat(hs[-4:], -1)
    if how == "sum_last_four":
        return torch.stack(hs[-4:]).sum(0)
    return hs[-1]


def warmup_linear(step, warmup, total):
    """Appendix A.2: linear warm-up, then linear decay to 0."""
    return step / max(1, warmup) if step < warmup else max(0.0, (total - step) / max(1, total - warmup))


def optimizer(model, lr=1e-4, weight_decay=0.01):
    """Adam with decoupled weight decay 0.01, not applied to biases and LayerNorm weights (as in the BERT code)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.dim() == 1 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999), eps=1e-6)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
