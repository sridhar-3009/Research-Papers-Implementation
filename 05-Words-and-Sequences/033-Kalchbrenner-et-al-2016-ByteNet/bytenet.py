"""ByteNet: Neural Machine Translation in Linear Time (Kalchbrenner, Espeholt, Simonyan, van den Oord, Graves &
Kavukcuoglu, 2016).

  - Encoder and decoder are stacks of 1-D convolutions (no recurrence): every position is computed in parallel.
  - The decoder's convolutions are MASKED (causal): position i only sees positions <= i (Section 3.4).
  - DILATION doubles every layer, 1, 2, 4, 8, 16, then starts again (Section 3.5): the receptive field grows
    exponentially with depth, so any two tokens are linked by a path of a few layers.
  - Each layer sits in a RESIDUAL BLOCK with 1x1 convolutions and layer normalization (Figure 3): the ReLU block
    (used for translation) or the Multiplicative-Unit block (used for language modelling).
  - The decoder is STACKED on the encoder's output, position by position (resolution preserving), and DYNAMIC
    UNFOLDING handles different lengths: the encoder output is given length |t^| = a|s| + b, and the decoder runs
    over it until it emits EOS, seeing zeros past |t^| (Section 3.2).

Tensors are (batch, channels, time) inside the convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


class Conv1d(nn.Module):
    """1 x k convolution with dilation r. causal=True: left padding only, so output i sees inputs i-(k-1)r .. i
    (the 'masked' convolution). causal=False: centred, sees i-(k-1)r/2 .. i+(k-1)r/2."""

    def __init__(self, c_in, c_out, k=1, dilation=1, causal=False):
        super().__init__()
        self.k, self.r, self.causal = k, dilation, causal
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):
        total = (self.k - 1) * self.r
        pad = (total, 0) if self.causal else (total // 2, total - total // 2)
        return self.conv(F.pad(x, pad))


class ChannelNorm(nn.Module):
    """Layer normalization over the channels of each time step (statistics never mix positions, so it is safe for
    causal decoders, unlike batch norm over time)."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ReLUBlock(nn.Module):
    """Figure 3 left: 2d -> LN, ReLU, 1x1 (2d->d) -> LN, ReLU, 1xk dilated (d->d) -> LN, ReLU, 1x1 (d->2d) -> + x."""

    def __init__(self, d, k, dilation, causal):
        super().__init__()
        self.net = nn.Sequential(ChannelNorm(2 * d), nn.ReLU(), Conv1d(2 * d, d),
                                 ChannelNorm(d), nn.ReLU(), Conv1d(d, d, k, dilation, causal),
                                 ChannelNorm(d), nn.ReLU(), Conv1d(d, 2 * d))

    def forward(self, x):
        return x + self.net(x)


class MU(nn.Module):
    """Multiplicative Unit (Kalchbrenner et al. 2016b): gates g1, g2, g3 = sigmoid(conv), update u = tanh(conv),
    MU(h) = g1 * tanh(g2 * h + g3 * u). The four convolutions share the 1 x k dilated (masked) shape."""

    def __init__(self, d, k, dilation, causal):
        super().__init__()
        self.conv = Conv1d(d, 4 * d, k, dilation, causal)

    def forward(self, h):
        g1, g2, g3, u = self.conv(h).chunk(4, 1)
        return torch.sigmoid(g1) * torch.tanh(torch.sigmoid(g2) * h + torch.sigmoid(g3) * torch.tanh(u))


class MUBlock(nn.Module):
    """Figure 3 right: 2d -> LN, ReLU, 1x1 (2d->d) -> LN, MU, MU (dilated, masked) -> LN, ReLU, 1x1 (d->2d) -> + x."""

    def __init__(self, d, k, dilation, causal):
        super().__init__()
        self.inp = nn.Sequential(ChannelNorm(2 * d), nn.ReLU(), Conv1d(2 * d, d), ChannelNorm(d))
        self.mu1, self.mu2 = MU(d, 1, 1, causal), MU(d, k, dilation, causal)
        self.out = nn.Sequential(ChannelNorm(d), nn.ReLU(), Conv1d(d, 2 * d))

    def forward(self, x):
        return x + self.out(self.mu2(self.mu1(self.inp(x))))


def dilations(n_blocks, max_rate=16):
    """1, 2, 4, ..., max_rate, then start again at 1 (Section 3.5)."""
    rates, r = [], 1
    for _ in range(n_blocks):
        rates.append(r)
        r = 1 if r == max_rate else 2 * r
    return rates


def receptive_field(k, rates, causal=True):
    """How many input positions one output position can see: 1 + sum_l (k - 1) r_l."""
    return 1 + sum((k - 1) * r for r in rates)


class Stack(nn.Module):
    def __init__(self, d, n_blocks, k, causal, block="relu", max_rate=16):
        super().__init__()
        B = ReLUBlock if block == "relu" else MUBlock
        self.rates = dilations(n_blocks, max_rate)
        self.blocks = nn.ModuleList([B(d, k, r, causal) for r in self.rates])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class ByteNetLM(nn.Module):
    """The ByteNet decoder alone, as a character language model (Section 5)."""

    def __init__(self, vocab, d=512, n_blocks=30, k=3, block="mu", dropout=0.1, max_rate=16):
        """max_rate=1 turns dilation off (every rate 1), for comparison."""
        super().__init__()
        self.k = k
        self.emb = nn.Embedding(vocab, 2 * d)                  # 'input embedding tensor' of size n x 2d (Sec. 3.3)
        self.stack = Stack(d, n_blocks, k, causal=True, block=block, max_rate=max_rate)
        self.head = nn.Sequential(Conv1d(2 * d, d), nn.ReLU(), nn.Dropout(dropout), Conv1d(d, vocab))

    def forward(self, x):
        """x (N, T) tokens -> logits (N, T, vocab); logits[:, i] predicts x[:, i + 1]."""
        return self.head(self.stack(self.emb(x).transpose(1, 2))).transpose(1, 2)

    @property
    def receptive_field(self):
        return receptive_field(self.k, self.stack.rates)


def unfold_length(src_len, a=1.2, b=0):
    """Eq. 2: the encoder's output length |t^| = a|s| + b (a = 1.2, b = 0 for English -> German)."""
    return int(round(a * src_len + b))


class ByteNet(nn.Module):
    """Encoder (centred dilated convs) + decoder (masked dilated convs) stacked on it (Sections 3.1-3.2)."""

    def __init__(self, src_vocab, tgt_vocab, d=800, n_blocks=30, k=3, a=1.2, b=0):
        super().__init__()
        self.a, self.b = a, b
        self.src_emb = nn.Embedding(src_vocab, 2 * d, padding_idx=PAD)
        self.enc = Stack(d, n_blocks, k, causal=False, block="relu")
        self.tgt_emb = nn.Embedding(tgt_vocab, 2 * d, padding_idx=PAD)
        self.dec = Stack(d, n_blocks, k, causal=True, block="relu")
        self.head = nn.Sequential(Conv1d(2 * d, d), nn.ReLU(), Conv1d(d, tgt_vocab))

    def encode(self, src):
        """src (N, S) padded with PAD -> encoder representation (N, 2d, |t^|). Dynamic unfolding: the source is
        padded to |t^| = a S + b positions before encoding, so the representation has the target's scale."""
        L = max(unfold_length(src.shape[1], self.a, self.b), src.shape[1])
        src = F.pad(src, (0, L - src.shape[1]), value=PAD)
        return self.enc(self.src_emb(src).transpose(1, 2))

    def decode(self, rep, tgt_in):
        """rep (N, 2d, L), tgt_in (N, T): the decoder sees, at position i, embedding(tgt_in[i]) + rep[i]
        (zeros past L: 'no representation for steps beyond the extended length')."""
        T = tgt_in.shape[1]
        if rep.shape[2] < T:
            rep = F.pad(rep, (0, T - rep.shape[2]))
        x = self.tgt_emb(tgt_in).transpose(1, 2) + rep[:, :, :T]
        return self.head(self.dec(x)).transpose(1, 2)

    def forward(self, src, tgt_in):
        return self.decode(self.encode(src), tgt_in)

    def log_prob(self, src, tgt):
        """tgt (N, T) ends with EOS and is PAD-padded; tgt_in = BOS + tgt[:-1]."""
        tgt_in = torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1]], 1)
        lp = F.log_softmax(self(src, tgt_in), -1).gather(-1, tgt[..., None])[..., 0]
        return (lp * (tgt != PAD)).sum(1)

    @torch.no_grad()
    def greedy(self, src, max_len=200):
        """Dynamic unfolding at test time: decode step by step over the encoder representation until EOS (the
        unfolding may run past |t^|). Decoding is sequential; training is parallel."""
        rep = self.encode(src)
        out = torch.full((src.shape[0], 1), BOS, dtype=torch.long)
        done = torch.zeros(src.shape[0], dtype=torch.bool)
        for _ in range(max_len):
            nxt = self.decode(rep, out)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            out = torch.cat([out, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return out[:, 1:]


def saliency(model, src, tgt):
    """Figure 6: |d log p(t_i) / d input embeddings|, summed over channels -> (T, S) for the source and (T, T) for
    the target, for ONE sentence pair (src (1, S), tgt (1, T))."""
    tgt_in = torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1]], 1)
    s_emb = model.src_emb(F.pad(src, (0, max(unfold_length(src.shape[1], model.a, model.b), src.shape[1]) - src.shape[1]), value=PAD))
    t_emb = model.tgt_emb(tgt_in)
    s_emb.retain_grad(); t_emb.retain_grad()
    rep = model.enc(s_emb.transpose(1, 2))
    T = tgt.shape[1]
    rep = F.pad(rep, (0, max(0, T - rep.shape[2])))[:, :, :T]
    logits = model.head(model.dec(t_emb.transpose(1, 2) + rep)).transpose(1, 2)
    lp = F.log_softmax(logits, -1).gather(-1, tgt[..., None])[0, :, 0]
    S_map, T_map = [], []
    for i in range(T):
        gs, gt = torch.autograd.grad(lp[i], (s_emb, t_emb), retain_graph=True)
        S_map.append(gs[0, :src.shape[1]].abs().sum(-1))
        T_map.append(gt[0].abs().sum(-1))
    return torch.stack(S_map), torch.stack(T_map)
