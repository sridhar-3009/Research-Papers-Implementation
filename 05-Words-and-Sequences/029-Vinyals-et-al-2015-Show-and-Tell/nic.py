"""Show and Tell: the Neural Image Caption generator (Vinyals, Toshev, Bengio & Erhan, CVPR 2015).

The model (Section 3, Figure 3):
    x_{-1} = CNN(I)                      the image, projected into the word-embedding space, fed ONCE
    x_t    = W_e S_t,  t = 0 .. N-1      word embeddings (S_0 = start word)
    p_{t+1} = LSTM(x_t)                  softmax over the vocabulary
    loss   = - sum_{t=1..N} log p_t(S_t)  (Eq. 13)

The LSTM is written exactly as Eqs. (4)-(8):
    i = s(W_ix x + W_im m),  f = s(W_fx x + W_fm m),  o = s(W_ox x + W_om m)
    c = f * c + i * tanh(W_cx x + W_cm m)
    m = o * c                            <- no tanh on c (the usual LSTM has m = o * tanh(c)); `cell_tanh=True`
                                            switches to the usual form.

Everything here works on precomputed CNN features (the paper keeps the CNN fixed, Sec. 4.3.1), so `feats` is
an (N, F) tensor. Sequences are time-major: (T, N).
"""

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


class LSTMCell(nn.Module):
    """Eqs. (4)-(8). One matrix for the input part, one for the recurrent part (the paper shows no biases;
    we keep one bias per gate on the input side, which is standard and harmless)."""

    def __init__(self, n_in, n_hid, cell_tanh=False):
        super().__init__()
        self.Wx = nn.Linear(n_in, 4 * n_hid)
        self.Wm = nn.Linear(n_hid, 4 * n_hid, bias=False)
        self.cell_tanh = cell_tanh

    def forward(self, x, state):
        m, c = state
        i, f, o, g = (self.Wx(x) + self.Wm(m)).chunk(4, -1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)              # Eq. (7)
        m = torch.sigmoid(o) * (torch.tanh(c) if self.cell_tanh else c)          # Eq. (8)
        return m, c


class NIC(nn.Module):
    def __init__(self, vocab, feat_dim, d=512, dropout=0.0, image_every_step=False, cell_tanh=False):
        super().__init__()
        self.d, self.every = d, image_every_step
        self.img = nn.Linear(feat_dim, d)                    # the CNN's top layer, mapping into word space
        self.We = nn.Embedding(vocab, d, padding_idx=PAD)
        self.lstm = LSTMCell(2 * d if image_every_step else d, d, cell_tanh)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, vocab)                       # Eq. (9): p = Softmax(W m)

    def _in(self, x, v):
        return torch.cat([x, v], -1) if self.every else x

    def start(self, feats):
        """Feed the image at t = -1. Its output p_0 is not scored. Returns (state, image vector)."""
        v = self.img(feats)
        z = feats.new_zeros(feats.shape[0], self.d)
        return self.lstm(self._in(v, v), (z, z)), v

    def step(self, word, state, v):
        """Feed one word; return (log p over the next word, new state)."""
        state = self.lstm(self._in(self.drop(self.We(word)), v), state)
        return F.log_softmax(self.out(self.drop(state[0])), -1), state

    def forward(self, feats, words):
        """words (T, N) = S_0 .. S_{T-1} -> log p for S_1 .. S_T, shape (T, N, vocab)."""
        state, v = self.start(feats)
        out = []
        for t in range(words.shape[0]):
            lp, state = self.step(words[t], state, v)
            out.append(lp)
        return torch.stack(out)

    def log_prob(self, feats, caps):
        """caps (T, N): S_0 = BOS, then words, EOS, PAD... -> log p(S | I) per caption (Eq. 13 negated)."""
        lp = self(feats, caps[:-1]).gather(-1, caps[1:, :, None])[..., 0]
        return (lp * (caps[1:] != PAD)).sum(0)


# ---------------------------------------------------------------------------------------------------- inference
@torch.no_grad()
def sample(model, feats, max_len=20, temperature=1.0, generator=None):
    """'Sampling': draw S_1 from p_1, feed it, draw S_2, ... until EOS. Returns a list of token lists."""
    state, v = model.start(feats)
    w = torch.full((feats.shape[0],), BOS, dtype=torch.long)
    outs, done = [[] for _ in range(feats.shape[0])], torch.zeros(feats.shape[0], dtype=torch.bool)
    for _ in range(max_len):
        lp, state = model.step(w, state, v)
        w = torch.multinomial((lp / temperature).softmax(-1), 1, generator=generator)[:, 0]
        for k in range(len(outs)):
            if not done[k] and w[k] != EOS:
                outs[k].append(w[k].item())
        done |= w == EOS
        if done.all():
            break
    return outs


@torch.no_grad()
def beam_search(model, feat, beam=20, max_len=20):
    """'BeamSearch' for ONE image (feat: (F,)): keep the k best partial sentences. beam = 1 is greedy.
    Returns all finished hypotheses (the N-best list), best first, as (log p, tokens without EOS)."""
    state, v = model.start(feat[None])
    hyps = [(0.0, [], state)]
    done = []
    for _ in range(max_len):
        words = torch.tensor([h[1][-1] if h[1] else BOS for h in hyps])
        st = (torch.cat([h[2][0] for h in hyps]), torch.cat([h[2][1] for h in hyps]))
        lp, st = model.step(words, st, v.expand(len(hyps), -1))
        cand = []
        for k, (score, toks, _) in enumerate(hyps):
            top = lp[k].topk(min(beam, lp.shape[-1]))
            for s, w in zip(top.values.tolist(), top.indices.tolist()):
                cand.append((score + s, toks + [w], (st[0][k:k + 1], st[1][k:k + 1])))
        cand.sort(key=lambda h: -h[0])
        hyps = []
        for h in cand:
            if h[1][-1] == EOS:
                done.append((h[0], h[1][:-1]))
            else:
                hyps.append(h)
            if len(hyps) == beam:
                break
        # scores only go down, so once the beam-th best finished sentence beats every live one, stop
        if not hyps or (len(done) >= beam and sorted(d[0] for d in done)[-beam] >= hyps[0][0]):
            break
    done += [(h[0], h[1]) for h in hyps]                     # unfinished ones, if max_len was hit
    return sorted(done, key=lambda h: -h[0])


# ---------------------------------------------------------------------------------------------------- ranking
@torch.no_grad()
def score_matrix(model, feats, caps):
    """S[i, j] = log p(caption j | image i). feats (I, F), caps (T, J)."""
    rows = []
    for i in range(feats.shape[0]):
        rows.append(model.log_prob(feats[i:i + 1].expand(caps.shape[1], -1), caps))
    return torch.stack(rows)


def normalize_for_annotation(S):
    """Sec. 4.3.5: 'we normalized our scores similar to [21]'. Our reading: divide p(S|I) by the caption's
    prior p(S) ~ mean over images of p(S|I'), so that generic, high-probability captions do not win for every
    image."""
    return S - (torch.logsumexp(S, 0, keepdim=True) - math.log(S.shape[0]))


def ranks(S, gt):
    """Tables 4-5. S (I, J) scores, gt[j] = the image of caption j.
    Image annotation: for each image, the rank (1-based) of its best-ranked true caption.
    Image search: for each caption, the rank of its image."""
    gt = torch.as_tensor(gt)
    ann = []
    for i in range(S.shape[0]):
        order = S[i].argsort(descending=True)
        ann.append(int((gt[order] == i).nonzero()[0]) + 1)
    search = []
    for j in range(S.shape[1]):
        order = S[:, j].argsort(descending=True)
        search.append(int((order == gt[j]).nonzero()[0]) + 1)
    return ann, search


def recall_report(r, ks=(1, 5, 10)):
    t = torch.tensor(r, dtype=torch.float)
    return {**{f"R@{k}": 100 * (t <= k).float().mean().item() for k in ks}, "Med r": t.median().item()}


# ---------------------------------------------------------------------------------------------------- metrics
def ngrams(s, n):
    return Counter(tuple(s[i:i + n]) for i in range(len(s) - n + 1))


def bleu(hyps, refs_list, n=4):
    """Corpus BLEU-n with several references per hypothesis (clipped counts against the max over references,
    brevity penalty against the closest reference length). BLEU-1 is what Table 2 reports."""
    match, total = [0] * n, [0] * n
    hl = rl = 0
    for h, refs in zip(hyps, refs_list):
        hl += len(h)
        rl += min((abs(len(r) - len(h)), len(r)) for r in refs)[1]
        for k in range(1, n + 1):
            hc = ngrams(h, k)
            mx = Counter()
            for r in refs:
                mx |= ngrams(r, k)
            match[k - 1] += sum(min(c, mx[g]) for g, c in hc.items())
            total[k - 1] += max(len(h) - k + 1, 0)
    if min(match) == 0:
        return 0.0
    logp = sum(math.log(m / t) for m, t in zip(match, total)) / n
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(hl, 1))
    return 100 * bp * math.exp(logp)


def human_bleu(refs_list, n=4):
    """Table 2's 'Human': score each of the references against the others, average over which one is held out.
    (The paper then adds back the average gain of having 5 rather than 4 references; we report it raw.)"""
    k = min(len(r) for r in refs_list)
    return sum(bleu([r[j] for r in refs_list], [r[:j] + r[j + 1:] for r in refs_list], n) for j in range(k)) / k


def cider_d(hyps, refs_list, n=4, sigma=6.0):
    """CIDEr-D (Vedantam et al. 2015), as in the MSCOCO toolkit: TF-IDF n-gram vectors (document frequency over
    the reference sets), cosine similarity with clipped hypothesis counts and a Gaussian length penalty,
    averaged over n = 1..4 and references, times 10."""
    N = len(refs_list)
    df = Counter()
    for refs in refs_list:
        for g in set(g for r in refs for k in range(1, n + 1) for g in ngrams(r, k)):
            df[g] += 1

    def vec(s, k):
        c = ngrams(s, k)
        return {g: tf * math.log(N / max(1.0, df[g])) for g, tf in c.items()}

    def norm(v):
        return math.sqrt(sum(x * x for x in v.values()))

    total = 0.0
    for h, refs in zip(hyps, refs_list):
        score = 0.0
        for k in range(1, n + 1):
            vh = vec(h, k)
            for r in refs:
                vr = vec(r, k)
                dot = sum(min(x, vr[g]) * vr[g] for g, x in vh.items() if g in vr)
                nh, nr = norm(vh), norm(vr)
                if nh and nr:
                    score += dot / (nh * nr) * math.exp(-((len(h) - len(r)) ** 2) / (2 * sigma ** 2))
        total += 10 * score / (n * len(refs))
    return 100 * total / len(hyps)                        # x100 as reported in Table 1 (85.5)


def novelty(captions, train_set):
    """Sec. 4.3.4: the fraction of generated captions that do NOT appear verbatim in the training set."""
    return sum(tuple(c) not in train_set for c in captions) / max(1, len(captions))


def nearest_words(model, word, k=5):
    """Table 6: nearest neighbours in the learned embedding W_e (cosine)."""
    E = F.normalize(model.We.weight.detach(), dim=-1)
    sims = E @ E[word]
    sims[word] = -2
    sims[:3] = -2                                         # skip PAD/BOS/EOS
    return sims.topk(k).indices.tolist()


def pad_captions(caps):
    """list of token lists (no BOS/EOS) -> (T, N) with BOS ... EOS PAD."""
    seqs = [[BOS] + c + [EOS] for c in caps]
    L = max(map(len, seqs))
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs]).T
