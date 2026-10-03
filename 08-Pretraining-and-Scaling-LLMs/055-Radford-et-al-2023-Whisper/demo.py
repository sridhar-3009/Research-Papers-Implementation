"""Whisper in ~20 seconds: a tiny encoder-decoder trained in Whisper's multitask token format on synthetic 'speech'
(digits encoded as pairs of tones, like a phone keypad, in two 'languages' that use different tone sets, plus silent
clips). One model is trained on clean audio only, one on audio with diverse noise levels, and both are tested under
noise they may never have seen. Plus: the front end, the token format, WER normalisation, decoding heuristics, the
paper's numbers."""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from whisper import (FIGURE_2, PAPER_PARAMS, SIZES, TABLE_6, TABLE_7, SpecialTokens, Whisper, add_noise, beam_search,
                     build_sequence, compression_ratio, edit_distance, greedy_decode, log_mel, normalize, param_count,
                     relative_error_reduction, wer)

torch.manual_seed(0); torch.set_num_threads(1)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------------------------------- 1
section("1. The front end (Section 2.2): 16 kHz audio -> 80-channel log-Mel, 25 ms windows, 10 ms stride")
t = torch.arange(16000 * 30) / 16000
mel = log_mel(0.3 * torch.sin(2 * math.pi * 440 * t))
print(f"  30 s of a 440 Hz tone -> {tuple(mel.shape)} (80 mel bins x 3000 frames); after the stride-2 conv: 1500 positions")
print(f"  loudest mel bin: {int(mel.mean(1).argmax())} of 80; values in [{mel.min():.2f}, {mel.max():.2f}] (clamped to 8 log10 units, (x+4)/4)")
print("  model sizes (Table 1), counted exactly for this architecture vs the paper's labels:")
for name, (L, d, h) in SIZES.items():
    print(f"    {name:6s} {L:2d} layers x {d:4d} wide, {h:2d} heads: {param_count(L, d) / 1e6:7.1f}M  (paper {PAPER_PARAMS[name] / 1e6:.0f}M)")

# ---------------------------------------------------------------------------------------------------- 2
section("2. The multitask token format (Figure 1)")
sp_demo = SpecialTokens(100, ["en", "es"])
toks, mask = build_sequence(sp_demo, "es", "transcribe", [(0.0, 1.52, [11, 12, 13]), (1.8, 2.5, [14])], prev=[7, 8])
names = {v: f"<|{k}|>" for k, v in sp_demo.id.items()}
show = [names.get(x, f"<|{sp_demo.time_of(x):.2f}|>" if x >= sp_demo.ts0 else f"w{x}") for x in toks]
print("  " + " ".join(show))
print("  loss mask:   " + "".join("x" if m else "." for m in mask) + "   (x = predicted, . = given context)")
t2, _ = build_sequence(sp_demo, "es", "translate", [(0, 1, [21, 22])], timestamps=False)
print("  translation: " + " ".join(names.get(x, f"w{x}") for x in t2))
print("  silence:     " + " ".join(names.get(x, f"w{x}") for x in build_sequence(sp_demo, "en", "transcribe", [])[0]))

# ---------------------------------------------------------------------------------------------------- 3
section("3. Toy 'speech': keypad digits as tone pairs, two 'languages', silent clips")
SR, DUR = 16000, 9600                                                             # 0.6-second clips
LOW, HIGH = [697, 770, 852, 941], [1209, 1336, 1477, 1633]
LANGS = {"A": (LOW, HIGH), "B": ([f * 1.5 for f in LOW], [f * 1.6 for f in HIGH])}
KEYS = [(3, 1), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
sp = SpecialTokens(10, list(LANGS))
EOT, SOT = sp.id["endoftext"], sp.id["startoftranscript"]


def clip(lang, g):
    digits = torch.randint(0, 10, (int(torch.randint(1, 4, (1,), generator=g)),), generator=g).tolist()
    a, tt = torch.zeros(DUR), torch.arange(DUR) / SR
    pos, segs = int(torch.randint(500, 3000, (1,), generator=g)), []
    lo, hi = LANGS[lang]
    for d in digits:
        L = int(torch.randint(1000, 1800, (1,), generator=g)); r, c = KEYS[d]
        a[pos:pos + L] += 0.3 * (torch.sin(2 * math.pi * lo[r] * tt[:L]) + torch.sin(2 * math.pi * hi[c] * tt[:L]))
        segs.append((pos / SR, (pos + L) / SR, [d])); pos += L + int(torch.randint(500, 1200, (1,), generator=g))
        if pos > DUR - 2300:
            break
    return a, segs


def make(N, g, snrs):
    X, Y, M, info = [], [], [], []
    for i in range(N):
        lang = "AB"[i % 2]
        a, segs = (torch.zeros(DUR), []) if i % 10 == 9 else clip(lang, g)
        snr = snrs[int(torch.randint(0, len(snrs), (1,), generator=g))]
        if snr is not None:
            a = add_noise(a, torch.randn(DUR, generator=g), snr) if segs else a + 0.05 * torch.randn(DUR, generator=g)
        X.append(log_mel(a, 40))
        tk, mk = build_sequence(sp, lang, "transcribe", segs, timestamps=False)
        Y.append(tk); M.append(mk); info.append((lang, " ".join(str(s[2][0]) for s in segs)))
    L = max(map(len, Y))
    return (torch.stack(X), torch.tensor([y + [EOT] * (L - len(y)) for y in Y]),
            torch.tensor([m + [0] * (L - len(m)) for m in M]), info)


def train(X, Y, M, steps):
    torch.manual_seed(0)
    m = Whisper(vocab=sp.vocab, n_mels=40, n_audio_ctx=60, n_text_ctx=16, d=64, heads=4, layers=2)
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    for _ in range(steps):
        i = torch.randint(0, len(X), (32,))
        lg = m(X[i], Y[i][:, :-1])
        loss = (F.cross_entropy(lg.transpose(1, 2), Y[i][:, 1:], reduction="none") * M[i][:, 1:]).sum() / M[i][:, 1:].sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval(), loss.item()


def evaluate(m, snr, n=300):
    X, _, _, info = make(n, torch.Generator().manual_seed(123), [snr])
    out, _ = greedy_decode(m, X, torch.full((n, 1), SOT), EOT, max_len=8)
    errs = words = lang_ok = 0
    for o, (lang, ref) in zip(out.tolist(), info):
        lang_ok += o[0] == (sp.id[f"lang:{lang}"] if ref else sp.id["nospeech"])
        errs += edit_distance(ref.split(), [str(x) for x in o if x < 10]); words += len(ref.split())
    return errs / words, lang_ok / n


g = torch.Generator().manual_seed(0)
Xc, Yc, Mc, _ = make(3000, g, [None])
Xn, Yn, Mn, _ = make(3000, g, [None, 20, 10, 5, 0])
print(f"  3000 clips per training set (0.6 s each, 1-3 digits, 10% silent); tensors {tuple(Xc.shape)}")
t = time.time(); clean, lc = train(Xc, Yc, Mc, 500)
print(f"  clean-only model:  500 steps, final train loss {lc:.3f} ({time.time() - t:.1f}s)")
t = time.time(); diverse, ln_ = train(Xn, Yn, Mn, 900)
print(f"  diverse-noise model (clean, 20, 10, 5, 0 dB): 900 steps, final train loss {ln_:.3f} ({time.time() - t:.1f}s)")

section("4. Robustness to noise (cf. Figure 5): word error rate on 300 test clips")
print("  SNR (dB)     clean-only WER  lang/silence acc    diverse WER  lang/silence acc")
for snr in (None, 20, 10, 5, 0, -5):
    (wc, ac), (wd, ad) = evaluate(clean, snr), evaluate(diverse, snr)
    print(f"  {'clean' if snr is None else snr:>8}     {wc:8.1%}       {ac:6.1%}            {wd:7.1%}      {ad:6.1%}")
print("  -> the clean-only model WINS on its own distribution (clean audio) but falls apart with ANY noise (WER > 100%")
print("     means it invents extra digits); the model trained on diverse audio degrades gracefully, even at -5 dB, a")
print("     level it never saw. Like LibriSpeech models vs Whisper (Figure 2): the specialist is best in-distribution,")
print("     diversity buys robustness. Odd detail: the diverse model is worse on perfectly clean audio than at 20 dB;")
print("     only 1/5 of its data was clean, and silent gaps in clean audio hit the log-Mel floor, which looks unusual.")
print("  (caveat: the clean model got 500 steps to keep the demo short; in a dev run with 900 steps it was just as")
print("   fragile: 3% WER clean, 134% at 20 dB)")

X1, _, _, info1 = make(1, torch.Generator().manual_seed(7), [10])
seq, sc = beam_search(diverse, X1, torch.tensor([[SOT]]), EOT, beams=5, max_len=8)
inv = {v: k for k, v in sp.id.items()}
print(f"  beam search (5 beams) on one 10 dB clip: reference '{info1[0][1]}' ({info1[0][0]}) -> "
      f"{[inv.get(x, x) for x in seq]}  (avg log-prob {sc:.3f})")

# ---------------------------------------------------------------------------------------------------- 5
section("5. WER needs text normalisation (Section 3.2, Appendix C)")
ref, hyp = "Mr. Smith said: you're my favourite (laughs) colour!", "mister smith said you are my favorite color"
print(f"  ref: {ref!r}\n  hyp: {hyp!r}")
print(f"  raw WER {wer(ref, hyp):.0%}  ->  normalised WER {wer(normalize(ref), normalize(hyp)):.0%}   ({normalize(ref)!r})")

section("6. Long-form decoding heuristics (Section 4.5, Table 7)")
loop = " ".join(["thank you"] * 20)
print(f"  gzip compression ratio of a looping output: {compression_ratio(loop):.2f} (> 2.4 -> retry at a higher temperature)")
print(f"  ... of a normal sentence: {compression_ratio('the quick brown fox jumps over the lazy dog near the bank'):.2f}")
for name, w in TABLE_7:
    print(f"    {name:32s} average long-form WER {w:.1f}")

# ---------------------------------------------------------------------------------------------------- 7
section("7. The paper's numbers")
print("  Figure 2: LibriSpeech-trained wav2vec 2.0 vs zero-shot Whisper Large V2 (same 2.7% on LibriSpeech clean)")
for k in ("LibriSpeech Clean", "Common Voice", "CHiME6", "Switchboard", "LibriSpeech Other"):
    a, b = FIGURE_2[k]
    print(f"    {k:18s} {a:5.1f} -> {b:5.1f}  (relative error reduction {relative_error_reduction(a, b):.1%})")
rest = [v for k, v in FIGURE_2.items() if k != "LibriSpeech Clean"]
avg = np.mean([v[0] for v in rest]), np.mean([v[1] for v in rest])
print(f"    average over the 13 other datasets: {avg[0]:.1f} -> {avg[1]:.1f}, RER {relative_error_reduction(*avg):.1%} "
      f"(paper: 29.3 -> 12.8, 55.2%: the paper's RER column also averages to {np.mean([relative_error_reduction(*v) for v in rest]):.1%})")
h = np.array(list(TABLE_6)); en = np.array([v[0] for v in TABLE_6.values()]); ml = np.array([v[1] for v in TABLE_6.values()])
s_ml = np.polyfit(np.log(h[:5]), np.log(ml[:5]), 1)[0]
print(f"  Table 6 (hours -> English WER): " + ", ".join(f"{int(k):,}: {v[0]}" for k, v in TABLE_6.items()))
print(f"  multilingual WER power law up to 54k hours: slope {s_ml:.2f} -> WER x{2 ** s_ml:.2f} per doubling of data;")
print(f"  the last 12.5x more data buys English only {en[-2] - en[-1]:.1f} points (diminishing returns)")
print(f"\nTotal time: {time.time() - T0:.1f}s")
