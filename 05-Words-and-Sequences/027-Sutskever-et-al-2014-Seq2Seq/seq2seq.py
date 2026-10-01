"""Sutskever, Vinyals & Le (2014), "Sequence to Sequence Learning with Neural Networks".

  Seq2Seq          two DIFFERENT deep LSTMs (4 layers in the paper): the encoder reads the (REVERSED)
                   source, its final (h, c) of every layer becomes the decoder's initial state (that is the
                   fixed-size sentence representation: 4 layers x (h + c) x 1000 = 8000 numbers); the
                   decoder is an LSTM language model conditioned on it (Eq. 1), with a plain softmax.
  paper_init_      uniform in [-0.08, 0.08]
  reverse_source   'instead of mapping a, b, c to alpha, beta, gamma, the LSTM is asked to map c, b, a'
  time_lags        the distances between corresponding source and target words, with and without reversal
  beam_search      Section 3.2: keep the B most probable prefixes; a hypothesis that emits <EOS> is moved
                   to the finished set. Works with an ENSEMBLE (average of the models' probabilities).
  rescore_nbest    'an even average' of a baseline score and the LSTM's log-probability
  clip_grad_       'If s > 5, we set g = 5 g / s' (g = gradient divided by the batch size)
  length_batches   minibatches of sentences of roughly the same length (a 2x speed-up)
  lr_schedule      0.7 for 5 epochs, then halved every half epoch (7.5 epochs in total)
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb=1000, hidden=1000, layers=4):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, emb)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb)
        self.encoder = nn.LSTM(emb, hidden, layers)            # one LSTM for the input ...
        self.decoder = nn.LSTM(emb, hidden, layers)            # ... and a different one for the output
        self.out = nn.Linear(hidden, tgt_vocab)                 # 'a naive softmax over 80,000 words'
        self.tgt_vocab = tgt_vocab

    def encode(self, src, lengths=None):
        """src: (S, N) ids (already reversed if wanted). Returns the final (h, c) of every layer."""
        x = self.src_emb(src)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        _, state = self.encoder(x)
        return state

    def forward(self, src, tgt_in, lengths=None):
        """Teacher forcing: logits (T, N, V) for p(y_t | v, y_1..y_{t-1})."""
        out, _ = self.decoder(self.tgt_emb(tgt_in), self.encode(src, lengths))
        return self.out(out)

    def step(self, y_prev, state):
        """One decoder step for beam search: y_prev (N,), state -> log-probs (N, V), new state."""
        out, state = self.decoder(self.tgt_emb(y_prev)[None], state)
        return F.log_softmax(self.out(out[0]), -1), state


def paper_init_(model, a=0.08):
    """'We initialized all of the LSTM's parameters with the uniform distribution between -0.08 and 0.08'."""
    for p in model.parameters():
        nn.init.uniform_(p, -a, a)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Section 3.3: reversing the source
# ---------------------------------------------------------------------------

def reverse_source(tokens):
    return list(reversed(tokens))


def time_lags(n, reverse):
    """Source of length n followed by a target of length n with word i <-> word i (a monotone alignment).
    Returns the distances (in time steps) between each corresponding pair."""
    src_pos = [n - 1 - i for i in range(n)] if reverse else list(range(n))
    tgt_pos = [n + i for i in range(n)]
    return [t - s for s, t in zip(src_pos, tgt_pos)]


# ---------------------------------------------------------------------------
# Section 3.2: beam search (single model or ensemble)
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(models, src, beam=12, max_len=50, lengths=None):
    """Translate ONE source sentence src (S, 1). models: a list (an ensemble averages the probabilities).
    Returns finished hypotheses [(log_prob, tokens)] sorted best first."""
    states = [m.encode(src, lengths) for m in models]
    beams = [(0.0, [BOS], states)]
    finished = []
    for _ in range(max_len):
        if not beams:
            break
        y_prev = torch.tensor([b[1][-1] for b in beams])
        # expand all live hypotheses at once, for every model in the ensemble
        new_states, probs = [], 0
        for k, m in enumerate(models):
            h = torch.cat([b[2][k][0] for b in beams], 1)
            c = torch.cat([b[2][k][1] for b in beams], 1)
            lp, (h2, c2) = m.step(y_prev, (h, c))
            probs = probs + lp.exp() / len(models)
            new_states.append((h2, c2))
        logp = probs.log()                                         # (B, V)
        scores = torch.tensor([b[0] for b in beams])[:, None] + logp
        top = scores.flatten().topk(min(beam, scores.numel()))
        V = logp.shape[1]
        candidates = []
        for s, idx in zip(top.values.tolist(), top.indices.tolist()):
            b, w = divmod(idx, V)
            st = [(ns[0][:, b:b + 1], ns[1][:, b:b + 1]) for ns in new_states]
            hyp = (s, beams[b][1] + [w], st)
            if w == EOS:
                finished.append((s, hyp[1][1:]))                    # 'removed from the beam' when <EOS> appears
            else:
                candidates.append(hyp)
        beams = candidates[:beam]
        if len(finished) >= beam:
            break
    if not finished:
        finished = [(b[0], b[1][1:]) for b in beams]
    return sorted(finished, key=lambda x: -x[0])


def rescore_nbest(nbest, lstm_scores):
    """'took an even average with their score and the LSTM's score'."""
    return [0.5 * base + 0.5 * lstm for (base, _), lstm in zip(nbest, lstm_scores)]


# ---------------------------------------------------------------------------
# Section 3.4: training details
# ---------------------------------------------------------------------------

def clip_grad_(model, threshold=5.0):
    """s = ||g||_2 with g already divided by the batch size; if s > 5: g <- 5 g / s."""
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
    s = g.norm().item()
    if s > threshold:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(threshold / s)
    return s


def lr_schedule(epoch_fraction, lr0=0.7):
    """0.7 for the first 5 epochs, then halved every half epoch (the paper stops at 7.5 epochs)."""
    if epoch_fraction < 5:
        return lr0
    return lr0 * 0.5 ** (math.floor((epoch_fraction - 5) / 0.5) + 1)


def length_batches(pairs, batch_size, rng=random):
    """Sort by source length, cut into batches of similar length, shuffle the batch order."""
    order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0]), rng.random()))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------------------
# Evaluation: corpus BLEU (the same function as in Paper 026's folder)
# ---------------------------------------------------------------------------

def bleu(hypotheses, references, max_n=4):
    """Corpus BLEU-4 (Papineni et al. 2002): clipped n-gram precisions, geometric mean, brevity penalty."""
    from collections import Counter
    clipped, totals = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp); ref_len += len(ref)
        for n in range(1, max_n + 1):
            h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
            r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            clipped[n - 1] += sum(min(c, r[g]) for g, c in h.items())
            totals[n - 1] += max(len(hyp) - n + 1, 0)
    if min(totals) == 0 or min(clipped) == 0:
        return 0.0
    log_p = sum(math.log(c / t) for c, t in zip(clipped, totals)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100 * bp * math.exp(log_p)
