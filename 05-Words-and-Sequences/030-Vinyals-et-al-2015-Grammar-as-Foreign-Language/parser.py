"""Grammar as a Foreign Language (Vinyals, Kaiser, Koo, Petrov, Sutskever & Hinton, NIPS 2015).

Parsing as translation: sentence -> linearized parse tree, with an attention-enhanced seq2seq model (LSTM+A).

  Linearization (Sec. 2.2, Figure 2): depth-first, POS tags replaced by 'XX':
      (S (NP NNP )NP (VP VBZ (NP DT NN )NP )VP . )S   ->   (S (NP XX )NP (VP XX (NP XX XX )NP )VP XX )S
  LSTM (Sec. 2):  i = s(W1 x + W2 h), i' = tanh(W3 x + W4 h), f = s(W5 x + W6 h), o = s(W7 x + W8 h)
                  m = m f + i i',  h = m o            (as printed: no tanh on the memory m)
  Attention (Sec. 2.1), over the TOP encoder states h_i and the CURRENT decoder state d_t:
                  u_ti = v^T tanh(W1' h_i + W2' d_t),  a_t = softmax(u_t),  d'_t = sum_i a_ti h_i
                  [d_t ; d'_t] is used for the prediction 'and is fed to the next time step'.
  Model: 3 layers x 256 units, separate encoder / decoder LSTMs, input reversed, dropout between layers 1-2 and
         2-3 for the small WSJ set (LSTM+A+D), beam 10. Malformed outputs are fixed by adding brackets at the
         beginning or the end.

Sequences are time-major: (T, N).
"""

import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, GO, EOS, UNK = 0, 1, 2, 3


# ---------------------------------------------------------------------------------------------------- trees
# A tree is (label, [children]); a leaf is a word (str); a preterminal is (POS, [word]).

def parse_tree(s):
    """Read a bracketed (Penn Treebank style) tree. An unlabeled outer bracket '( (S ...) )' is dropped."""
    toks = re.findall(r"\(|\)|[^\s()]+", s)
    pos = 0

    def node():
        nonlocal pos
        assert toks[pos] == "("
        pos += 1
        label = ""
        if toks[pos] not in "()":
            label = toks[pos]; pos += 1
        kids = []
        while toks[pos] != ")":
            if toks[pos] == "(":
                kids.append(node())
            else:
                kids.append(toks[pos]); pos += 1
        pos += 1
        return (label, kids)

    t = node()
    while t[0] in ("", "ROOT", "TOP") and len(t[1]) == 1 and not isinstance(t[1][0], str):
        t = t[1][0]
    return t


def is_pre(t):
    return len(t[1]) == 1 and isinstance(t[1][0], str)


def words(t):
    return [t[1][0]] if is_pre(t) else [w for k in t[1] for w in words(k)]


def tags(t):
    return [t[0]] if is_pre(t) else [p for k in t[1] for p in tags(k)]


def to_string(t):
    if isinstance(t, str):
        return t
    return "(" + t[0] + " " + " ".join(to_string(k) for k in t[1]) + ")"


def clean_ptb(t):
    """Treebank preprocessing usual for EVALB parsing: drop empty elements (-NONE-) and the nodes they leave
    empty, and strip function tags / indices (NP-SBJ-1 -> NP)."""
    if is_pre(t):
        return None if t[0] == "-NONE-" else t
    kids = [k for k in (clean_ptb(k) for k in t[1]) if k is not None]
    if not kids:
        return None
    label = t[0] if t[0] in ("-LRB-", "-RRB-") else re.split(r"[-=]", t[0])[0] or t[0]
    return (label, kids)


def linearize(t, normalize_pos=True):
    """Figure 2: depth-first, '(L' ... ')L', POS tags -> 'XX' (Sec. 2.3, 'POS-tag normalization')."""
    if is_pre(t):
        return ["XX" if normalize_pos else t[0]]
    return ["(" + t[0]] + [s for k in t[1] for s in linearize(k, normalize_pos)] + [")" + t[0]]


def balance(seq):
    """Sec. 3.2: 'we simply add brackets to either the beginning or the end of the tree to make it balanced'.
    Unclosed '(L' get a ')L' appended; stray ')L' get a matching '(L' prepended."""
    stack, prefix = [], []
    for s in seq:
        if s.startswith("("):
            stack.append(s[1:])
        elif s.startswith(")"):
            if stack:
                stack.pop()
            else:
                prefix.insert(0, "(" + s[1:])
    return prefix + list(seq) + [")" + l for l in reversed(stack)]


def delinearize(seq, sent):
    """Linearized output + the sentence's words -> a tree. Every non-bracket symbol is a preterminal that takes the
    next word. A ')' closes the innermost open node whatever its label. Extra preterminals are dropped; words left
    over are attached as 'XX' to the root (the paper does not say what it does here; this is our choice)."""
    seq = balance(seq)
    root = ("", [])
    stack, i = [root], 0
    for s in seq:
        if s.startswith("("):
            n = (s[1:], [])
            stack[-1][1].append(n); stack.append(n)
        elif s.startswith(")"):
            if len(stack) > 1:
                stack.pop()
        elif i < len(sent):
            stack[-1][1].append((s, [sent[i]])); i += 1
    target = root[1][0] if len(root[1]) == 1 else root
    for w in sent[i:]:
        target[1].append(("XX", [w]))

    def prune(t):                                         # drop constituents that ended up with no words
        if is_pre(t):
            return t
        kids = [k for k in (prune(k) for k in t[1]) if k is not None]
        return (t[0], kids) if kids else None

    root = prune(root) or ("", [])
    return root[1][0] if len(root[1]) == 1 else ("S", root[1])


def is_well_formed(seq):
    depth = 0
    for s in seq:
        depth += s.startswith("(") - s.startswith(")")
        if depth < 0:
            return False
    return depth == 0 and len(seq) > 0


# ---------------------------------------------------------------------------------------------------- EVALB
PUNCT = {",", ":", "``", "''", ".", "-NONE-"}             # COLLINS.prm DELETE_LABEL_FOR_LENGTH
EQUIV = {"PRT": "ADVP"}                                    # COLLINS.prm EQ_LABEL ADVP PRT


def drop_words(t, drop, k=None):
    """Remove the preterminals at word positions in `drop` (and constituents left empty)."""
    k = k if k is not None else [0]
    if is_pre(t):
        pos = k[0]; k[0] += 1
        return None if pos in drop else t
    kids = [c for c in (drop_words(c, drop, k) for c in t[1]) if c is not None]
    return (t[0], kids) if kids else None


def constituents(t):
    """Labeled spans (label, start, end); preterminals and empty (dummy root) labels are excluded."""
    out, k = Counter(), [0]

    def walk(n):
        if is_pre(n):
            k[0] += 1
            return
        s = k[0]
        for c in n[1]:
            walk(c)
        if n[0]:
            out[(EQUIV.get(n[0], n[0]), s, k[0])] += 1

    walk(t)
    return out


def evalb(golds, preds):
    """Corpus labeled bracketing precision / recall / F1 (EVALB with COLLINS.prm-style punctuation deletion).
    Punctuation is identified from the GOLD tags (our predictions only have 'XX')."""
    match = ng = np_ = 0
    for g, p in zip(golds, preds):
        drop = {i for i, tag in enumerate(tags(g)) if tag in PUNCT}
        g2, p2 = drop_words(g, drop), drop_words(p, drop)
        cg = constituents(g2) if g2 else Counter()
        cp = constituents(p2) if p2 else Counter()
        match += sum((cg & cp).values()); ng += sum(cg.values()); np_ += sum(cp.values())
    P, R = match / max(np_, 1), match / max(ng, 1)
    return {"P": 100 * P, "R": 100 * R, "F1": 100 * 2 * P * R / max(P + R, 1e-12)}


# ---------------------------------------------------------------------------------------------------- the model
class Cell(nn.Module):
    """The LSTM of Sec. 2 (h = m * o). `cell_tanh=True` gives the usual h = tanh(m) * o."""

    def __init__(self, n_in, n, cell_tanh=False):
        super().__init__()
        self.Wx, self.Wh, self.cell_tanh = nn.Linear(n_in, 4 * n), nn.Linear(n, 4 * n, bias=False), cell_tanh

    def forward(self, x, state):
        h, m = state
        i, i2, f, o = (self.Wx(x) + self.Wh(h)).chunk(4, -1)
        m = m * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(i2)
        h = (torch.tanh(m) if self.cell_tanh else m) * torch.sigmoid(o)
        return h, m


class Deep(nn.Module):
    """A stack of Cells; dropout (when > 0) between consecutive layers (LSTM1-LSTM2, LSTM2-LSTM3)."""

    def __init__(self, n_in, n, layers, dropout, cell_tanh):
        super().__init__()
        self.cells = nn.ModuleList([Cell(n_in if k == 0 else n, n, cell_tanh) for k in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, states):
        new = []
        for k, (cell, st) in enumerate(zip(self.cells, states)):
            h, m = cell(x if k == 0 else self.drop(x), st)
            new.append((h, m)); x = h
        return x, new


class LSTMA(nn.Module):
    def __init__(self, n_words, n_labels, d=256, emb=512, layers=3, attention=True, dropout=0.0, cell_tanh=False):
        super().__init__()
        self.d, self.L, self.attention = d, layers, attention
        self.E_in = nn.Embedding(n_words, emb, padding_idx=PAD)
        self.E_out = nn.Embedding(n_labels, emb, padding_idx=PAD)
        self.enc = Deep(emb, d, layers, dropout, cell_tanh)
        feed = 2 * d if attention else 0                   # [d_t ; d'_t] fed to the next time step
        self.dec = Deep(emb + feed, d, layers, dropout, cell_tanh)
        self.W1 = nn.Linear(d, d, bias=False)              # W1' (square: same size at encoder and decoder)
        self.W2 = nn.Linear(d, d, bias=False)              # W2'
        self.v = nn.Linear(d, 1, bias=False)
        self.Wo = nn.Linear(2 * d if attention else d, n_labels)

    def encode(self, src):
        """src (T, N), already reversed and padded at the end. Returns the top-layer states, mask, final states."""
        T, N = src.shape
        z = src.new_zeros(N, self.d, dtype=torch.float)
        states = [(z, z) for _ in range(self.L)]
        mask = src != PAD
        tops, x = [], self.E_in(src)
        for t in range(T):
            top, new = self.enc(x[t], states)
            keep = mask[t, :, None]
            states = [(torch.where(keep, h, h0), torch.where(keep, m, m0)) for (h, m), (h0, m0) in zip(new, states)]
            tops.append(states[-1][0])
        return torch.stack(tops), mask, states

    def attend(self, d_t, H, W1H, mask):
        u = self.v(torch.tanh(W1H + self.W2(d_t)[None]))[..., 0].masked_fill(~mask, float("-inf"))
        a = torch.softmax(u, 0)                            # (T_A, N)
        return (a[..., None] * H).sum(0), a

    def start(self, src):
        H, mask, states = self.encode(src)
        N = src.shape[1]
        feed = H.new_zeros(N, 2 * self.d) if self.attention else None
        return {"H": H, "W1H": self.W1(H), "mask": mask, "states": states, "feed": feed}

    def step(self, y_prev, S):
        """One decoder step. Returns log p over output symbols, the attention vector, and the new decoder state."""
        x = self.E_out(y_prev)
        if self.attention:
            x = torch.cat([x, S["feed"]], -1)
        d_t, states = self.dec(x, S["states"])
        a = None
        if self.attention:
            c, a = self.attend(d_t, S["H"], S["W1H"], S["mask"])
            d_t = torch.cat([d_t, c], -1)
        S2 = dict(S, states=states, feed=d_t if self.attention else None)
        return F.log_softmax(self.Wo(d_t), -1), a, S2

    def forward(self, src, tgt_in, return_attention=False):
        S = self.start(src)
        lps, atts = [], []
        for t in range(tgt_in.shape[0]):
            lp, a, S = self.step(tgt_in[t], S)
            lps.append(lp); atts.append(a)
        return (torch.stack(lps), torch.stack(atts) if self.attention else None) if return_attention else torch.stack(lps)

    def log_prob(self, src, tgt):
        """tgt (T, N) ends with EOS then PAD. Returns log P(B | A) per pair."""
        tgt_in = torch.cat([torch.full_like(tgt[:1], GO), tgt[:-1]])
        lp = self(src, tgt_in).gather(-1, tgt[..., None])[..., 0]
        return (lp * (tgt != PAD)).sum(0)


@torch.no_grad()
def beam_search(models, src, beam=10, max_len=None):
    """Beam search for ONE sentence (src (T, 1), reversed), averaging log-probabilities over an ensemble.
    Returns (score, symbols without EOS, attention rows)."""
    models = models if isinstance(models, (list, tuple)) else [models]
    max_len = max_len or 4 * src.shape[0] + 10
    starts = [m.start(src) for m in models]
    hyps, done = [(0.0, [], starts, [])], []
    for _ in range(max_len):
        cand = []
        for score, toks, Ss, att in hyps:
            y = torch.tensor([toks[-1] if toks else GO])
            outs = [m.step(y, S) for m, S in zip(models, Ss)]
            lp = torch.stack([o[0][0] for o in outs]).mean(0)
            a = outs[0][1][:, 0].tolist() if outs[0][1] is not None else None
            top = lp.topk(min(beam, lp.shape[-1]))
            for s, w in zip(top.values.tolist(), top.indices.tolist()):
                cand.append((score + s, toks + [w], [o[2] for o in outs], att + [a]))
        cand.sort(key=lambda h: -h[0])
        hyps = []
        for h in cand:
            (done if h[1][-1] == EOS else hyps).append(h)
            if len(hyps) == beam:
                break
        if not hyps or (done and max(d[0] for d in done) >= hyps[0][0]):
            break
    best = max(done or hyps, key=lambda h: h[0])
    toks = best[1][:-1] if best[1] and best[1][-1] == EOS else best[1]
    return best[0], toks, best[3][:len(toks)]


class Vocab:
    def __init__(self, items, min_count=1, specials=("<pad>", "<go>", "</s>", "<unk>")):
        c = Counter(items)
        self.itos = list(specials) + sorted(w for w, k in c.items() if k >= min_count and w not in specials)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, seq):
        return [self.stoi.get(w, UNK) for w in seq]

    def __len__(self):
        return len(self.itos)


def batch_pairs(pairs, reverse=True):
    """pairs of (word ids, label ids) -> src (reversed, padded at the end) and tgt (labels + EOS, padded)."""
    srcs = [list(reversed(s)) if reverse else list(s) for s, _ in pairs]
    tgts = [list(t) + [EOS] for _, t in pairs]
    S, T = max(map(len, srcs)), max(map(len, tgts))
    src = torch.tensor([s + [PAD] * (S - len(s)) for s in srcs]).T
    tgt = torch.tensor([t + [PAD] * (T - len(t)) for t in tgts]).T
    return src, tgt
