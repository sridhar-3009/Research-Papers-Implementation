"""Whisper's experiments (Radford et al. 2022) at small scale, on real speech.

Data: LibriSpeech via torchaudio (train-clean-100 / train-other-500 / test-clean / test-other); noise: white noise and
'babble' (a mix of 5 other utterances; a stand-in for the paper's pub noise). Text: lower-case characters plus the
special tokens of whisper.py. The released OpenAI models are used in E6 if the `openai-whisper` package is installed.

  E1  Figure 5: train the same small Whisper-style model on (a) clean read speech only and (b) clean + other + noisy
      copies at random SNRs; WER on test-clean, test-other, and under white / babble noise at 40 ... -10 dB.
  E2  Table 6: train on 10, 25, 50, 100 % of the hours; WER vs hours (log-log slope; diminishing returns?).
  E3  Section 4.3-style ablation for a single language: transcription only vs + timestamp tokens + no-speech segments
      (multitask); does the extra task hurt or help WER?
  E4  Table 7: long-form transcription of concatenated chapters with 30-second windows: greedy vs beam 5 vs
      + temperature fallback (log-prob < -1 or gzip ratio > 2.4) vs + previous-text conditioning.
  E5  Section 3.2: WER with and without the text normaliser, for the E1 models and (if available) a released model.
  E6  The released models (tiny ... large): zero-shot WER on test-clean / test-other and under noise, for comparison
      with the paper's Figure 2 / Figure 5 (needs `pip install openai-whisper`).

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from whisper import (SpecialTokens, Whisper, add_noise, beam_search, build_sequence, compression_ratio,
                     edit_distance, greedy_decode, log_mel, normalize, wer)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = list(" abcdefghijklmnopqrstuvwxyz'")


# ---------------------------------------------------------------------------------------------------- data
def librispeech(split, limit=None):
    import torchaudio
    ds = torchaudio.datasets.LIBRISPEECH(DATA, url=split, download=True)
    idx = list(range(len(ds)))
    random.Random(0).shuffle(idx)
    return [ds[i] for i in idx[:limit]]                                           # (wave, sr, text, spk, chap, utt)


def text_ids(text):
    return [CHARS.index(c) for c in text.lower() if c in CHARS]


def ids_text(ids):
    return "".join(CHARS[i] for i in ids if i < len(CHARS))


def babble(pool, n, g):
    out = torch.zeros(n)
    for _ in range(5):
        w = pool[int(torch.randint(0, len(pool), (1,), generator=g))][0][0]
        w = w[:n] if len(w) >= n else F.pad(w, (0, n - len(w)))
        out += w / (w.std() + 1e-6)
    return out


def featurize(wave, a):
    w = wave[: a.chunk * 16000]
    return log_mel(F.pad(w, (0, a.chunk * 16000 - len(w))))


def make_examples(rows, sp, a, noisy=False, pool=None, timestamps=False, silence=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    ex = []
    for wave, sr, text, *_ in rows:
        w = wave[0]
        if noisy and torch.rand(1, generator=g) < 0.5:
            snr = float(torch.empty(1).uniform_(-5, 40, generator=g))
            noise = torch.randn(len(w), generator=g) if torch.rand(1, generator=g) < 0.5 else babble(pool, len(w), g)
            w = add_noise(w, noise, snr)
        segs = [(0.0, min(len(w) / 16000, a.chunk), text_ids(text))]
        if silence and torch.rand(1, generator=g) < silence:
            w, segs = 0.01 * torch.randn(len(w), generator=g), []
        toks, mask = build_sequence(sp, "en", "transcribe", segs, timestamps=timestamps)
        ex.append((featurize(w, a), toks[: a.n_text_ctx], mask[: a.n_text_ctx], text))
    return ex


# ---------------------------------------------------------------------------------------------------- model
def make_model(sp, a):
    torch.manual_seed(0)
    return Whisper(vocab=sp.vocab, n_mels=80, n_audio_ctx=a.chunk * 50, n_text_ctx=a.n_text_ctx, d=a.d, heads=a.heads,
                   layers=a.layers).to(DEV)


def train(model, examples, sp, a, steps):
    """AdamW, gradient clipping, linear warm-up then linear decay to zero (Section 2.4)."""
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.1)
    warm = max(1, min(2048, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    eot = sp.id["endoftext"]
    for s in range(steps):
        batch = [examples[i] for i in torch.randint(0, len(examples), (a.batch,))]
        L = max(len(b[1]) for b in batch)
        X = torch.stack([b[0] for b in batch]).to(DEV)
        Y = torch.tensor([b[1] + [eot] * (L - len(b[1])) for b in batch], device=DEV)
        M = torch.tensor([b[2] + [0] * (L - len(b[2])) for b in batch], device=DEV).float()
        lg = model(X, Y[:, :-1])
        loss = (F.cross_entropy(lg.transpose(1, 2), Y[:, 1:], reduction="none") * M[:, 1:]).sum() / M[:, 1:].sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if s % max(1, steps // 10) == 0:
            print(f"    step {s} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def transcribe(model, sp, mels, a, timestamps=False):
    model.eval()
    prefix = [sp.id["startoftranscript"], sp.id["lang:en"], sp.id["transcribe"]] + ([] if timestamps else [sp.id["notimestamps"]])
    out = []
    for i in range(0, len(mels), a.eval_batch):
        X = torch.stack(mels[i:i + a.eval_batch]).to(DEV)
        ids, _ = greedy_decode(model, X, torch.tensor([prefix] * len(X), device=DEV), sp.id["endoftext"], max_len=a.n_text_ctx - len(prefix))
        out += [ids_text([t for t in row if t < len(CHARS)]) for row in ids.tolist()]
    model.train()
    return out


def corpus_wer(refs, hyps, norm=True):
    f = normalize if norm else (lambda s: s)
    e = sum(edit_distance(f(r).split(), f(h).split()) for r, h in zip(refs, hyps))
    return e / max(sum(len(f(r).split()) for r in refs), 1)


def eval_sets(a):
    clean, other = librispeech("test-clean", a.n_test), librispeech("test-other", a.n_test)
    return clean, other


def noisy_eval(model, sp, rows, pool, a):
    out = {}
    g = torch.Generator().manual_seed(1)
    for kind in ("white", "babble"):
        for snr in a.snrs:
            mels = [featurize(add_noise(w[0], torch.randn(len(w[0]), generator=g) if kind == "white" else babble(pool, len(w[0]), g), snr), a)
                    for w, *_ in rows]
            out[f"{kind} {snr} dB"] = corpus_wer([r[2] for r in rows], transcribe(model, sp, mels, a))
    return out


# ---------------------------------------------------------------------------------------------------- experiments
def e1(a):
    sp = SpecialTokens(len(CHARS), ["en"])
    train_clean = librispeech("train-clean-100", a.n_train)
    test_clean, test_other = eval_sets(a)
    pool = librispeech("dev-clean", 200)
    out = {}
    for name, rows, noisy in (("clean read speech only", train_clean, False),
                              ("diverse (clean + other + noise)", train_clean + librispeech("train-other-500", a.n_train), True)):
        model = train(make_model(sp, a), make_examples(rows, sp, a, noisy=noisy, pool=pool), sp, a, a.steps)
        torch.save(model.state_dict(), HERE / f"e1_{'diverse' if noisy else 'clean'}.pt")
        res = {"test-clean": corpus_wer([r[2] for r in test_clean], transcribe(model, sp, [featurize(r[0][0], a) for r in test_clean], a)),
               "test-other": corpus_wer([r[2] for r in test_other], transcribe(model, sp, [featurize(r[0][0], a) for r in test_other], a))}
        res.update(noisy_eval(model, sp, test_clean, pool, a))
        out[name] = res
        print("  E1", name, res, flush=True)
    return out


def e2(a):
    sp = SpecialTokens(len(CHARS), ["en"])
    rows = librispeech("train-clean-100", a.n_train)
    test_clean, _ = eval_sets(a)
    hours_total = sum(r[0].shape[1] for r in rows) / 16000 / 3600
    out = {}
    for frac in a.fractions:
        sub = rows[: max(1, int(frac * len(rows)))]
        model = train(make_model(sp, a), make_examples(sub, sp, a), sp, a, a.steps)
        out[f"{frac * hours_total:.1f} h"] = corpus_wer([r[2] for r in test_clean],
                                                        transcribe(model, sp, [featurize(r[0][0], a) for r in test_clean], a))
        print("  E2", frac, out[f"{frac * hours_total:.1f} h"], flush=True)
    h = np.array([float(k.split()[0]) for k in out]); w = np.array(list(out.values()))
    out["log-log slope"] = float(np.polyfit(np.log(h), np.log(np.maximum(w, 1e-4)), 1)[0])
    return out


def e3(a):
    sp = SpecialTokens(len(CHARS), ["en"])
    rows = librispeech("train-clean-100", a.n_train)
    test_clean, _ = eval_sets(a)
    out = {}
    for name, ts, sil in (("transcription only", False, 0.0), ("+ timestamps", True, 0.0), ("+ timestamps + no-speech", True, 0.1)):
        ex = make_examples(rows, sp, a, timestamps=ts, silence=sil)
        model = train(make_model(sp, a), ex, sp, a, a.steps)
        out[name] = corpus_wer([r[2] for r in test_clean], transcribe(model, sp, [featurize(r[0][0], a) for r in test_clean], a))
        print("  E3", name, out[name], flush=True)
    return out


def e4(a):
    sp = SpecialTokens(len(CHARS), ["en"])
    f = HERE / "e1_diverse.pt"
    if not f.exists():
        return {"note": "run e1 first"}
    model = make_model(sp, a); model.load_state_dict(torch.load(f, map_location=DEV)); model.eval()
    rows = librispeech("test-clean", None)
    chapters = {}
    for w, sr, text, spk, chap, utt in rows:
        chapters.setdefault((spk, chap), []).append((utt, w[0], text))
    longs = []
    for k, utts in list(chapters.items())[: a.n_long]:
        utts.sort(key=lambda u: u[0])
        longs.append((torch.cat([u[1] for u in utts]), " ".join(u[2] for u in utts)))
    pre = [sp.id["startoftranscript"], sp.id["lang:en"], sp.id["transcribe"], sp.id["notimestamps"]]
    eot = sp.id["endoftext"]

    def window_decode(mel, mode, prev):
        prefix = ([sp.id["prev"]] + prev[-a.prev_len:] if (mode == "prev" and prev) else []) + pre
        P = torch.tensor([prefix], device=DEV)
        if mode == "greedy":
            ids, lp = greedy_decode(model, mel[None].to(DEV), P, eot, max_len=a.n_text_ctx - len(prefix))
            return ids[0].tolist(), float(lp[0])
        ids, lp = beam_search(model, mel[None].to(DEV), P, eot, beams=5, max_len=a.n_text_ctx - len(prefix))
        if mode in ("fallback", "prev"):
            for T in (0.2, 0.4, 0.6, 0.8, 1.0):
                if lp >= -1 and compression_ratio(ids_text(ids)) <= 2.4:
                    break
                g = torch.Generator().manual_seed(int(T * 10))
                out, lps = greedy_decode(model, mel[None].to(DEV), P, eot, max_len=a.n_text_ctx - len(prefix), temperature=T, gen=g)
                ids, lp = out[0].tolist(), float(lps[0])
        return ids, lp

    res = {}
    for mode in ("greedy", "beam", "fallback", "prev"):
        refs, hyps = [], []
        for wave, ref in longs:
            text, prev = [], []
            for s in range(0, len(wave), a.chunk * 16000):                         # fixed windows (no timestamp shift)
                ids, _ = window_decode(featurize(wave[s:s + a.chunk * 16000], a), mode, prev)
                ids = [t for t in ids if t < len(CHARS)]
                text.append(ids_text(ids)); prev += ids
            refs.append(ref); hyps.append(" ".join(text))
        res[mode] = corpus_wer(refs, hyps)
        print("  E4", mode, res[mode], flush=True)
    res["paper (Table 7 average)"] = {"greedy": 11.0, "+ beam": 10.6, "+ fallback": 10.6, "+ prev text": 10.0}
    return res


def e5(a):
    sp = SpecialTokens(len(CHARS), ["en"])
    test_clean, _ = eval_sets(a)
    out = {}
    for name in ("clean", "diverse"):
        f = HERE / f"e1_{name}.pt"
        if f.exists():
            m = make_model(sp, a); m.load_state_dict(torch.load(f, map_location=DEV))
            hyps = transcribe(m, sp, [featurize(r[0][0], a) for r in test_clean], a)
            refs = [r[2] for r in test_clean]
            out[name] = {"raw WER": corpus_wer(refs, hyps, norm=False), "normalised WER": corpus_wer(refs, hyps)}
    try:
        import whisper as openai_whisper                                            # noqa: F401  (pip package)
        if hasattr(openai_whisper, "load_model"):
            m = openai_whisper.load_model("base.en" if not a.quick else "tiny.en", device=DEV)
            hyps = [m.transcribe(r[0][0].numpy(), language="en")["text"] for r in test_clean]
            refs = [r[2] for r in test_clean]
            out["released model"] = {"raw WER": corpus_wer(refs, hyps, norm=False), "normalised WER": corpus_wer(refs, hyps)}
    except Exception as e:                                                          # local whisper.py shadows the package
        out["released model"] = f"unavailable: {type(e).__name__}"
    return out


def e6(a):
    import importlib
    import sys
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != HERE.resolve()]
    sys.modules.pop("whisper", None)
    try:
        ow = importlib.import_module("whisper")
        ow.load_model
    except Exception:
        return {"note": "pip install openai-whisper to run E6"}
    test_clean, test_other = eval_sets(a)
    pool = librispeech("dev-clean", 200)
    out = {}
    for size in a.released:
        m = ow.load_model(size, device=DEV)
        run = lambda waves: [m.transcribe(w.numpy(), language="en", temperature=0.0)["text"] for w in waves]
        row = {"test-clean": corpus_wer([r[2] for r in test_clean], run([r[0][0] for r in test_clean])),
               "test-other": corpus_wer([r[2] for r in test_other], run([r[0][0] for r in test_other]))}
        g = torch.Generator().manual_seed(1)
        for snr in (10, 0):
            waves = [add_noise(r[0][0], babble(pool, r[0].shape[1], g), snr) for r in test_clean]
            row[f"babble {snr} dB"] = corpus_wer([r[2] for r in test_clean], run(waves))
        out[size] = row
        print("  E6", size, row, flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        L += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 7)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.n_train, a.n_test, a.chunk, a.n_text_ctx, a.d, a.heads, a.layers = None, 500, 30, 448, 384, 6, 4
    a.lr, a.batch, a.eval_batch, a.steps, a.snrs = 1e-3, 32, 16, 30000, (40, 20, 10, 5, 0, -5, -10)
    a.fractions, a.n_long, a.prev_len, a.released = (0.1, 0.25, 0.5, 1.0), 20, 100, ("tiny.en", "base.en", "small.en")
    if a.quick:
        a.n_train, a.n_test, a.chunk, a.d, a.heads, a.layers = 40, 4, 10, 32, 2, 1
        a.steps, a.batch, a.snrs, a.fractions, a.n_long, a.released = 3, 2, (10, 0), (0.5, 1.0), 1, ("tiny.en",)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
