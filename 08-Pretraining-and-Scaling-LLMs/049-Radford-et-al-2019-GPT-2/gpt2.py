"""Language Models are Unsupervised Multitask Learners (GPT-2; Radford, Wu, Child, Luan, Amodei & Sutskever, 2019).

  p(x) = prod_n p(s_n | s_1, ..., s_{n-1})                                                             (Eq. 1)
  a task is p(output | input, task); language can SPECIFY the task inside the sequence
    ('translate to french, english text, french text', 'TL;DR:', 'Q: ... A:'), so a good enough language model
    trained on diverse text (WebText: 8M documents, 40 GB, outbound Reddit links with >= 3 karma) performs tasks
    zero-shot, with no parameter updates.
  Input: byte-level BPE (base vocabulary 256 bytes, 50,257 tokens) that never merges across character categories,
    except that a leading space may join a word -> any Unicode string has a probability, and tokenization is
    invertible.
  Model changes vs GPT: layer norm moved to the input of each sub-block (pre-LN), an extra layer norm after the last
    block, residual-layer weights scaled by 1/sqrt(N) at init (N = number of residual layers), context 1024, batch
    512, vocabulary 50,257. Sizes (Table 2): 12 x 768, 24 x 1024, 36 x 1280, 48 x 1600.
  Evaluation helpers: per-canonical-unit perplexity / bits, top-k sampling (k = 2 for 'TL;DR:' summaries),
    LAMBADA last-word accuracy with a stop-word filter, cloze scoring (CBT), and Section 4's 8-gram Bloom-filter
    overlap analysis.
"""

import hashlib
import math
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------- byte-level BPE
# GPT-2's pre-tokenizer: contractions, optional-space + letters, optional-space + digits, optional-space + other
# symbols, then whitespace. BPE runs INSIDE each piece, so 'dog' never merges with '.', '!' or '?'.
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


def pretokenize(text):
    return PAT.findall(text)


class ByteBPE:
    """Learn merges over UTF-8 bytes of pre-tokenized pieces. Token ids 0..255 are the raw bytes."""

    def __init__(self, text, n_merges):
        pieces = Counter(tuple(p.encode("utf-8")) for p in pretokenize(text))
        words = {k: list(k) for k in pieces}
        self.merges, self.vocab = {}, {i: bytes([i]) for i in range(256)}
        for _ in range(n_merges):
            pairs = Counter()
            for k, syms in words.items():
                for a, b in zip(syms, syms[1:]):
                    pairs[(a, b)] += pieces[k]
            if not pairs:
                break
            (a, b), _ = pairs.most_common(1)[0]
            new = 256 + len(self.merges)
            self.merges[(a, b)] = new
            self.vocab[new] = self.vocab[a] + self.vocab[b]
            for k, syms in words.items():
                words[k] = self._merge(syms, a, b, new)

    @staticmethod
    def _merge(syms, a, b, new):
        out, i = [], 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                out.append(new); i += 2
            else:
                out.append(syms[i]); i += 1
        return out

    def encode(self, text):
        ids = []
        for p in pretokenize(text):
            syms = list(p.encode("utf-8"))
            while len(syms) > 1:                                                   # apply merges in learned order
                cands = [(self.merges[(a, b)], j) for j, (a, b) in enumerate(zip(syms, syms[1:])) if (a, b) in self.merges]
                if not cands:
                    break
                new, j = min(cands)
                syms = syms[:j] + [new] + syms[j + 2:]
            ids += syms
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def __len__(self):
        return len(self.vocab)


# ---------------------------------------------------------------------------------------------------- model
class Block(nn.Module):
    """Pre-LN block: x + attn(LN(x)), then x + MLP(LN(x)) (Section 2.3)."""

    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.fc, self.proj = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, attn_mask=causal, need_weights=False)[0])
        return x + self.drop(self.proj(F.gelu(self.fc(self.ln2(x)))))


class GPT2(nn.Module):
    def __init__(self, vocab=50257, n_ctx=1024, d=768, layers=12, heads=12, dropout=0.0, scaled_init=True):
        super().__init__()
        self.n_ctx = n_ctx
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(n_ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)                                                # the extra final layer norm
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0, 0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
        for b in self.blocks:
            nn.init.normal_(b.attn.in_proj_weight, 0, 0.02); nn.init.zeros_(b.attn.in_proj_bias)
            if scaled_init:                                                         # residual layers: 0.02 / sqrt(N)
                std = 0.02 / math.sqrt(2 * layers)
                nn.init.normal_(b.attn.out_proj.weight, 0, std)
                nn.init.normal_(b.proj.weight, 0, std)

    def forward(self, ids):
        T = ids.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
        x = self.wte(ids) + self.wpe(torch.arange(T, device=ids.device))
        for b in self.blocks:
            x = b(x, causal)
        return self.ln_f(x) @ self.wte.weight.T                                     # tied output

    def loss(self, ids):
        lg = self(ids)[:, :-1]
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))

    @torch.no_grad()
    def generate(self, ids, n, top_k=None, temperature=1.0, greedy=False, stop=None):
        out = list(ids)
        for _ in range(n):
            lg = self(torch.tensor([out[-self.n_ctx:]]))[0, -1] / max(temperature, 1e-6)
            if greedy:
                nxt = int(lg.argmax())
            else:
                if top_k:
                    v, _ = lg.topk(top_k)
                    lg = lg.masked_fill(lg < v[-1], -float("inf"))
                nxt = int(torch.multinomial(F.softmax(lg, -1), 1))
            out.append(nxt)
            if stop is not None and nxt == stop:
                break
        return out[len(ids):]

    @torch.no_grad()
    def sequence_logprob(self, ids, start=1):
        """sum_{t >= start} log p(ids[t] | ids[:t])."""
        lp = F.log_softmax(self(torch.tensor([ids[-self.n_ctx:]]))[0], -1)
        return sum(lp[t - 1, ids[t]].item() for t in range(start, len(ids)))


SIZES = {"117M": (12, 768), "345M": (24, 1024), "762M": (36, 1280), "1542M": (48, 1600)}       # Table 2


def model_size(layers, d, vocab=50257, n_ctx=1024):
    """Exact parameter count of this architecture (tied embeddings): embeddings + 12 d^2 + 13 d per block + LN."""
    per_block = 12 * d * d + 13 * d
    return vocab * d + n_ctx * d + layers * per_block + 2 * d


# ---------------------------------------------------------------------------------------------------- evaluation
def per_unit(total_nll_nats, n_units):
    """Section 3.1: the average negative log-probability per canonical unit (word, character, byte), reported as
    perplexity exp(nll / units) or as bits per unit nll / (units ln 2)."""
    avg = total_nll_nats / n_units
    return {"perplexity": math.exp(avg), "bits per unit": avg / math.log(2)}


STOP_WORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "was", "he", "she", "it", "they",
              "i", "you", "his", "her", "that", "this", "with", "for", "as", "but", "not", "be", "by", ".", ","}


def lambada_predict(candidates_with_logprob, stop_filter=True):
    """Pick the most probable final word, optionally skipping stop words (Section 3.3: 52.66% -> 63.24%)."""
    ranked = sorted(candidates_with_logprob, key=lambda t: -t[1])
    for w, _ in ranked:
        if not stop_filter or w.lower() not in STOP_WORDS:
            return w
    return ranked[0][0]


def cloze_choice(score_fn, prefix, candidates, suffix):
    """CBT (Section 3.2): score 'prefix + candidate + rest of sentence' with the LM; keep the best candidate."""
    scores = [score_fn(prefix + c + suffix) for c in candidates]
    return candidates[max(range(len(candidates)), key=lambda i: scores[i])]


def translation_prompt(pairs, query):
    """Section 3.7: 'english sentence = french sentence' examples, then 'query ='."""
    return "\n".join(f"{e} = {f}" for e, f in pairs) + f"\n{query} ="


def tldr_prompt(article):
    """Section 3.6: append 'TL;DR:' to induce a summary (sampled with top-k, k = 2)."""
    return article + "\nTL;DR:"


def qa_prompt(examples, question):
    """Section 3.8: seed with example question-answer pairs, then 'Q: ... A:'."""
    return "".join(f"Q: {q}\nA: {a}\n" for q, a in examples) + f"Q: {question}\nA:"


# ---------------------------------------------------------------------------------------------------- overlap
def normalize(text):
    """Section 4: lower-cased alphanumeric words joined by single spaces."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def ngrams(words, n=8):
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


class BloomFilter:
    """m bits, k hash functions: no false negatives; false-positive rate ~ (1 - exp(-k n / m))^k."""

    def __init__(self, m, k):
        self.m, self.k = m, k
        self.bits = bytearray((m + 7) // 8)
        self.n = 0

    def _hashes(self, s):
        h = hashlib.sha256(s.encode()).digest()
        a, b = int.from_bytes(h[:8], "little"), int.from_bytes(h[8:16], "little") | 1
        return [(a + i * b) % self.m for i in range(self.k)]                         # double hashing

    def add(self, s):
        for i in self._hashes(s):
            self.bits[i // 8] |= 1 << (i % 8)
        self.n += 1

    def __contains__(self, s):
        return all(self.bits[i // 8] >> (i % 8) & 1 for i in self._hashes(s))

    def false_positive_rate(self):
        return (1 - math.exp(-self.k * self.n / self.m)) ** self.k


def overlap_fraction(bloom, text, n=8):
    """Percentage of a dataset's normalized 8-grams found in the training-set filter (Table 6)."""
    grams = ngrams(normalize(text).split(), n)
    return sum(g in bloom for g in grams) / max(1, len(grams))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
