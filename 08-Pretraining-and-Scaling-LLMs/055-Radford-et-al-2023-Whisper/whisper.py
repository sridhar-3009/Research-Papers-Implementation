"""Robust Speech Recognition via Large-Scale Weak Supervision (Whisper; Radford, Kim, Xu, Brockman, McLeavey &
Sutskever, OpenAI 2022/ICML 2023).

  Data: 680,000 hours of (audio, transcript) pairs from the internet: 438k h English ASR, 117k h in 96 other
    languages, 125k h X->English translation. Minimal processing: raw transcripts (no normalisation), heuristics to
    REMOVE machine-generated transcripts (all upper/lower case, no commas, ...), audio-language vs text-language check,
    fuzzy de-dup, 30-second segments (including no-speech segments for voice activity detection).
  Front end: 16 kHz audio -> 80-channel log-magnitude Mel spectrogram, 25 ms windows (400 samples), 10 ms stride (160);
    30 s = 3000 frames; globally scaled to about [-1, 1].
  Model: an off-the-shelf encoder-decoder Transformer. Encoder: two Conv1d (width 3, GELU; the second stride 2) ->
    sinusoidal positions -> pre-LN blocks -> final LN (1500 positions). Decoder: learned positions, tied input/output
    embeddings, self-attention + cross-attention. Same width and depth in both (Table 1: tiny 4x384 ... large 32x1280).
  Multitask format (Figure 1): [prev text] <|startoftranscript|> <|lang|> (or <|nospeech|>) <|transcribe|> or
    <|translate|> (<|notimestamps|> or interleaved timestamp tokens quantised to 20 ms) text ... <|endoftext|>;
    loss masked only on the previous-text context.
  Training: AdamW, grad clipping, 2048 warm-up updates then linear decay, batch 256 segments, 2^20 updates (2-3 epochs),
    no augmentation (Large-v2 added SpecAugment, stochastic depth, BPE dropout).
  Evaluation: ZERO-SHOT on many datasets with WER after an extensive text normaliser. Long-form decoding heuristics:
    beam 5, temperature fallback 0 -> 0.2 -> ... -> 1.0 when avg log-prob < -1 or gzip compression ratio > 2.4,
    no-speech prob > 0.6 AND avg log-prob < -1 means silence, previous-text conditioning, first timestamp in [0, 1] s.
"""

import gzip
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE, N_FFT, HOP, N_MELS, CHUNK = 16000, 400, 160, 80, 30


# ---------------------------------------------------------------------------------------------------- front end
def hz_to_mel(f):
    """Slaney-style mel scale (as librosa's default, which Whisper's filters come from): linear below 1 kHz, log above."""
    f = np.asarray(f, float)
    lin = f / (200.0 / 3)
    log = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000, log, lin)


def mel_to_hz(m):
    m = np.asarray(m, float)
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), m * 200.0 / 3)


def mel_filters(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    """Triangular filters equally spaced on the mel scale, area-normalised (Slaney)."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2))
    W = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = edges[i:i + 3]
        W[i] = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
        W[i] *= 2.0 / (hi - lo)
    return torch.tensor(W, dtype=torch.float32)


_FILTERS = {}


def log_mel(audio, n_mels=N_MELS):
    """Whisper's log_mel_spectrogram: |STFT|^2 (Hann 400, hop 160) -> mel -> log10 -> clamp to (max - 8) -> (x+4)/4."""
    if n_mels not in _FILTERS:
        _FILTERS[n_mels] = mel_filters(n_mels=n_mels)
    st = torch.stft(audio, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True)
    power = st[..., :-1].abs() ** 2                                              # drop the last frame, as Whisper does
    log = torch.clamp(_FILTERS[n_mels] @ power, min=1e-10).log10()
    log = torch.maximum(log, log.amax(dim=(-2, -1), keepdim=True) - 8.0)
    return (log + 4.0) / 4.0


def pad_or_trim(audio, length=CHUNK * SAMPLE_RATE):
    return F.pad(audio, (0, length - audio.shape[-1])) if audio.shape[-1] < length else audio[..., :length]


def add_noise(audio, noise, snr_db):
    """Scale the noise so that 10 log10(P_signal / P_noise) = snr_db (Section 3.7)."""
    ps, pn = audio.pow(2).mean(), noise.pow(2).mean()
    return audio + noise * torch.sqrt(ps / (pn * 10 ** (snr_db / 10)))


# ---------------------------------------------------------------------------------------------------- model
def sinusoids(length, d, max_timescale=10000):
    inc = math.log(max_timescale) / (d // 2 - 1)
    inv = torch.exp(-inc * torch.arange(d // 2))
    t = torch.arange(length)[:, None] * inv[None]
    return torch.cat([t.sin(), t.cos()], 1)


class Block(nn.Module):
    def __init__(self, d, heads, cross=False):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.ln2, self.xattn = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask=None, mem=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        if self.cross:
            h = self.ln2(x)
            x = x + self.xattn(h, mem, mem, need_weights=False)[0]
        return x + self.mlp(self.ln3(x))


class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_ctx, d, heads, layers):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.register_buffer("pos", sinusoids(n_ctx, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)

    def forward(self, mel):                                                       # (B, n_mels, frames)
        x = F.gelu(self.conv2(F.gelu(self.conv1(mel)))).transpose(1, 2)           # (B, frames/2, d)
        x = x + self.pos[:x.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.ln(x)


class TextDecoder(nn.Module):
    def __init__(self, vocab, n_ctx, d, heads, layers):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(n_ctx, d) * 0.01)
        self.blocks = nn.ModuleList(Block(d, heads, cross=True) for _ in range(layers))
        self.ln = nn.LayerNorm(d)

    def forward(self, ids, mem):
        T = ids.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), 1)
        x = self.tok(ids) + self.pos[:T]
        for b in self.blocks:
            x = b(x, mask, mem)
        return self.ln(x) @ self.tok.weight.T                                     # tied output embedding


class Whisper(nn.Module):
    def __init__(self, vocab=51865, n_mels=80, n_audio_ctx=1500, n_text_ctx=448, d=384, heads=6, layers=4):
        super().__init__()
        self.encoder = AudioEncoder(n_mels, n_audio_ctx, d, heads, layers)
        self.decoder = TextDecoder(vocab, n_text_ctx, d, heads, layers)

    def forward(self, mel, ids):
        return self.decoder(ids, self.encoder(mel))


SIZES = {"tiny": (4, 384, 6), "base": (6, 512, 8), "small": (12, 768, 12), "medium": (24, 1024, 16),
         "large": (32, 1280, 20)}                                                 # Table 1: layers, width, heads
PAPER_PARAMS = {"tiny": 39e6, "base": 74e6, "small": 244e6, "medium": 769e6, "large": 1550e6}


def param_count(layers, d, vocab=51865, n_mels=80, n_audio_ctx=1500, n_text_ctx=448):
    """Exact count for this architecture without building it: convs + encoder blocks + LN; token + position embeddings
    + decoder blocks (with cross-attention) + LN. (Sinusoidal encoder positions are not parameters.)"""
    attn = 4 * d * d + 4 * d                                                      # q, k, v, out (with biases)
    mlp = 8 * d * d + 5 * d
    ln = 2 * d
    enc = (n_mels * d * 3 + d) + (d * d * 3 + d) + layers * (attn + mlp + 2 * ln) + ln
    dec = vocab * d + n_text_ctx * d + layers * (2 * attn + mlp + 3 * ln) + ln
    return enc + dec


# ---------------------------------------------------------------------------------------------------- tokens
class SpecialTokens:
    """The multitask vocabulary: text tokens, then <|endoftext|>, <|startoftranscript|>, one token per language,
    <|translate|>, <|transcribe|>, <|nospeech|>, <|notimestamps|>, <|prev|>, and 1501 timestamps 0.00 ... 30.00 s."""

    def __init__(self, n_text, languages):
        names = ["endoftext", "startoftranscript"] + [f"lang:{l}" for l in languages] + \
                ["translate", "transcribe", "nospeech", "notimestamps", "prev"]
        self.id = {n: n_text + i for i, n in enumerate(names)}
        self.ts0 = n_text + len(names)
        self.n_ts = int(CHUNK / 0.02) + 1
        self.vocab = self.ts0 + self.n_ts

    def timestamp(self, seconds):
        """Quantise to the nearest 20 ms (Whisper's native time resolution)."""
        return self.ts0 + int(round(min(max(seconds, 0), CHUNK) / 0.02))

    def time_of(self, tok):
        return (tok - self.ts0) * 0.02


def build_sequence(sp, language, task, segments, timestamps=True, prev=None):
    """segments: list of (start s, end s, [text token ids]). Returns (tokens, loss_mask). Loss is masked over the
    previous-text context only (and over the first <|prev|> marker); everything else is predicted."""
    toks, mask = [], []
    if prev:
        toks += [sp.id["prev"]] + list(prev); mask += [0] * (len(prev) + 1)
    toks.append(sp.id["startoftranscript"]); mask.append(0)                      # given as input, not predicted
    if not segments:
        toks += [sp.id["nospeech"], sp.id["endoftext"]]; mask += [1, 1]
        return toks, mask
    head = [sp.id[f"lang:{language}"], sp.id[task]] + ([] if timestamps else [sp.id["notimestamps"]])
    toks += head; mask += [1] * len(head)
    for start, end, text in segments:
        part = ([sp.timestamp(start)] if timestamps else []) + list(text) + ([sp.timestamp(end)] if timestamps else [])
        toks += part; mask += [1] * len(part)
    toks.append(sp.id["endoftext"]); mask.append(1)
    return toks, mask


# ---------------------------------------------------------------------------------------------------- metrics
def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[-1]


def wer(reference, hypothesis):
    """(substitutions + deletions + insertions) / reference words."""
    r, h = reference.split(), hypothesis.split()
    return edit_distance(r, h) / max(len(r), 1)


CONTRACTIONS = {"you're": "you are", "i'm": "i am", "it's": "it is", "don't": "do not", "can't": "can not",
                "won't": "will not", "we're": "we are", "they're": "they are", "isn't": "is not", "that's": "that is"}
SPELLINGS = {"colour": "color", "favourite": "favorite", "centre": "center", "mr": "mister", "mrs": "missus"}
ONES = "zero one two three four five six seven eight nine ten eleven twelve".split()


def normalize(text):
    """A small English normaliser in the spirit of Appendix C: lower-case, drop bracketed/parenthesised asides and
    filler words, expand contractions, standardise spellings and small numbers, remove punctuation, squeeze spaces."""
    t = text.lower()
    t = re.sub(r"[\[(<][^\])>]*[\])>]", " ", t)
    t = re.sub(r"\b(uh|um|hmm|mm|ah)\b", " ", t)
    for k, v in CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t)
    t = re.sub(r"[^\w\s']", " ", t).replace("'", " ")
    words = [SPELLINGS.get(w, w) for w in t.split()]
    words = [str(ONES.index(w)) if w in ONES else w for w in words]
    return " ".join(words)


def relative_error_reduction(a, b):
    """Figure 2's RER: (WER_a - WER_b) / WER_a."""
    return (a - b) / a


def compression_ratio(text):
    """len(text bytes) / len(gzip(text bytes)): repetitive (looping) output compresses very well -> high ratio."""
    b = text.encode()
    return len(b) / max(len(gzip.compress(b)), 1)


def needs_fallback(avg_logprob, text, logprob_threshold=-1.0, compression_threshold=2.4):
    return avg_logprob < logprob_threshold or compression_ratio(text) > compression_threshold


def is_silence(no_speech_prob, avg_logprob, no_speech_threshold=0.6, logprob_threshold=-1.0):
    """Section 4.5: <|nospeech|> probability alone isn't reliable; require BOTH conditions."""
    return no_speech_prob > no_speech_threshold and avg_logprob < logprob_threshold


def decode_with_fallback(decode_fn, temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """decode_fn(T) -> (text, avg_logprob). Try T = 0 first; raise T while the output looks bad."""
    for T in temperatures:
        text, lp = decode_fn(T)
        if not needs_fallback(lp, text):
            return text, lp, T
    return text, lp, T


# ---------------------------------------------------------------------------------------------------- decoding
@torch.no_grad()
def greedy_decode(model, mel, prefix, eot, max_len=40, temperature=0.0, gen=None):
    """Batch decode from a prefix; returns generated ids (until eot) and mean log-prob per sequence."""
    mem = model.encoder(mel)
    ids = prefix.clone()
    lps = torch.zeros(len(ids)); done = torch.zeros(len(ids), dtype=torch.bool); count = torch.zeros(len(ids))
    for _ in range(max_len):
        lg = model.decoder(ids, mem)[:, -1]
        logp = F.log_softmax(lg, -1)
        nxt = lg.argmax(-1) if temperature == 0 else torch.multinomial(F.softmax(lg / temperature, -1), 1, generator=gen).squeeze(1)
        lps += torch.where(done, torch.zeros(()), logp.gather(1, nxt[:, None]).squeeze(1))
        count += (~done).float()
        nxt = torch.where(done, torch.full_like(nxt, eot), nxt)
        ids = torch.cat([ids, nxt[:, None]], 1)
        done |= nxt == eot
        if done.all():
            break
    return ids[:, prefix.shape[1]:], lps / count.clamp(min=1)


@torch.no_grad()
def beam_search(model, mel, prefix, eot, beams=5, max_len=40):
    """Beam search for ONE example; score = sum log-prob (as in Section 4.5), finished hypotheses ranked by
    sum log-prob / length."""
    mem = model.encoder(mel[:1])
    alive, finished = [(list(prefix[0].tolist()), 0.0)], []
    for _ in range(max_len):
        x = torch.tensor([s for s, _ in alive])
        logp = F.log_softmax(model.decoder(x, mem.expand(len(alive), -1, -1))[:, -1], -1)
        cand = []
        for (s, sc), lp in zip(alive, logp):
            v, i = lp.topk(beams)
            cand += [(s + [int(t)], sc + float(l)) for l, t in zip(v, i)]
        cand.sort(key=lambda c: -c[1])
        alive = []
        for s, sc in cand:
            (finished if s[-1] == eot else alive).append((s, sc))
            if len(alive) == beams:
                break
        if len(finished) >= beams or not alive:
            break
    pool = finished or alive
    n0 = prefix.shape[1]
    best = max(pool, key=lambda c: c[1] / max(len(c[0]) - n0, 1))
    return best[0][n0:], best[1] / max(len(best[0]) - n0, 1)


# ---------------------------------------------------------------------------------------------------- paper numbers
TABLE_6 = {  # hours: (English WER, multilingual WER, X->en BLEU)
    3405: (30.5, 92.4, 0.2), 6811: (19.6, 72.7, 1.7), 13621: (14.4, 56.6, 7.9), 27243: (12.3, 45.0, 13.9),
    54486: (10.9, 36.4, 19.2), 681070: (9.9, 29.2, 24.8)}
FIGURE_2 = {  # LibriSpeech-trained wav2vec 2.0 Large (no LM) vs zero-shot Whisper Large V2, WER %
    "LibriSpeech Clean": (2.7, 2.7), "Artie": (24.5, 6.2), "Common Voice": (29.9, 9.0), "Fleurs En": (14.6, 4.4),
    "Tedlium": (10.5, 4.0), "CHiME6": (65.8, 25.5), "VoxPopuli En": (17.9, 7.3), "CORAAL": (35.6, 16.2),
    "AMI IHM": (37.0, 16.9), "Switchboard": (28.3, 13.8), "CallHome": (34.8, 17.6), "WSJ": (7.7, 3.9),
    "AMI SDM1": (67.6, 36.4), "LibriSpeech Other": (6.2, 5.2)}
TABLE_7 = [  # long-form average WER as heuristics are added
    ("Greedy decoding only", 11.0), ("+ Beam search", 10.6), ("+ Temperature fallback", 10.6),
    ("+ Voice activity detection", 10.2), ("+ Previous text conditioning", 10.0), ("+ Initial timestamp constraint", 10.0)]
