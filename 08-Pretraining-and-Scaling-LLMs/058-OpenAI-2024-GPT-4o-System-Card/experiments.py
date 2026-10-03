"""The GPT-4o System Card's evaluation methods, run for real at small scale (GPT-4o itself is not available).

  E1  Section 3.3.1 (voice output classifier): speaker verification on LibriSpeech. Compare the hand-made band
      embedding of gpt4o.py with a small trained speaker encoder (softmax over training speakers, cosine scoring).
      Report EER, and precision at recall 1.0 per 0.5 s chunk and per 'conversation' (10-60 chunks), for held-out
      speakers enrolled from 3 clips.
  E2  Section 3.2 (TTS-converted evaluations): speak text prompts with an offline TTS (macOS `say` or pyttsx3), add
      noise / echo, transcribe with the released Whisper (paper 055, `pip install openai-whisper`), and measure how
      often a text-level decision (a keyword filter standing in for the moderation classifier) is the same on the
      transcript as on the original text, and the WER.
  E3  Section 3.3.3 (disparate performance): WER of a released Whisper model on LibriSpeech test-clean grouped by
      speaker sex (SPEAKERS.TXT), with a speaker-level bootstrap confidence interval for the gap and a z-test.
  E4  Section 3.8 (reading capability metrics): a small LLaMA-style model (paper 056) trained on addition; single
      sample vs cons@k (majority vote) vs pass@k (any correct) for k = 1 ... 32.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt4o import consensus_at_k, two_proportion_z, voice_embedding

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def librispeech(split, limit=None):
    import torchaudio
    ds = torchaudio.datasets.LIBRISPEECH(DATA, url=split, download=True)
    idx = list(range(len(ds)))
    random.Random(0).shuffle(idx)
    return [ds[i] for i in idx[:limit]]


def released_whisper(size):
    sys.path = [p for p in sys.path if Path(p or ".").resolve() not in (HERE.resolve(), (HERE.parent / "055-Radford-et-al-2023-Whisper").resolve())]
    sys.modules.pop("whisper", None)
    import whisper as ow
    return ow.load_model(size, device=DEV)


# ---------------------------------------------------------------------------------------------------- E1
class SpeakerEncoder(nn.Module):
    """Log-Mel (paper 055's front end) -> 3 conv layers -> mean+std pooling -> 128-d embedding; trained with a softmax
    over training speakers."""

    def __init__(self, n_speakers, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(80, 256, 5, padding=2), nn.ReLU(), nn.Conv1d(256, 256, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(256, 256, 3, padding=1), nn.ReLU())
        self.emb, self.cls = nn.Linear(512, d), nn.Linear(d, n_speakers)

    def embed(self, mel):
        h = self.net(mel)
        return F.normalize(self.emb(torch.cat([h.mean(-1), h.std(-1)], -1)), dim=-1)

    def forward(self, mel):
        return self.cls(self.embed(mel))


def chunks(wave, n=8000):
    return [wave[i:i + n] for i in range(0, len(wave) - n + 1, n)]


def eer(scores_same, scores_diff):
    thr = np.sort(np.concatenate([scores_same, scores_diff]))
    best = (1, 0)
    for t in thr:
        far, frr = np.mean(scores_diff >= t), np.mean(scores_same < t)
        best = min(best, (abs(far - frr), (far + frr) / 2))
    return best[1]


def e1(a):
    wh = load("whisper_055", "055-Radford-et-al-2023-Whisper/whisper.py")
    train_rows, test_rows = librispeech("train-clean-100", a.n_train), librispeech("test-clean", a.n_test)
    spk = sorted({r[3] for r in train_rows}); sid = {s: i for i, s in enumerate(spk)}
    X = [(wh.log_mel(c), sid[r[3]]) for r in train_rows for c in chunks(r[0][0])[:4]]
    enc = SpeakerEncoder(len(spk)).to(DEV)
    opt = torch.optim.Adam(enc.parameters(), 1e-3)
    for s in range(a.steps):
        b = random.Random(s).sample(X, a.batch)
        loss = F.cross_entropy(enc(torch.stack([m for m, _ in b]).to(DEV)), torch.tensor([y for _, y in b], device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    by_spk = {}
    for r in test_rows:
        by_spk.setdefault(r[3], []).append(r[0][0])
    out = {}
    for name, emb_fn in (("band embedding (gpt4o.py)", lambda w: torch.tensor(voice_embedding(w.numpy()))),
                         ("trained speaker encoder", lambda w: enc.embed(wh.log_mel(w)[None].to(DEV))[0].detach().cpu())):
        same, diff, conv = [], [], {}
        speakers = list(by_spk)
        for s in speakers:
            enrol = torch.stack([emb_fn(c) for w in by_spk[s][:3] for c in chunks(w)[:2]]).mean(0)
            enrol = enrol / enrol.norm()
            same += [float(emb_fn(c) @ enrol) for w in by_spk[s][3:] for c in chunks(w)]
            other = speakers[(speakers.index(s) + 1) % len(speakers)]
            diff += [float(emb_fn(c) @ enrol) for w in by_spk[other][:3] for c in chunks(w)]
        same, diff = np.array(same), np.array(diff)
        thr = diff.max() + 1e-9                                                # recall 1.0 on chunk-level deviations
        fa = float(np.mean(same < thr))
        conv = {n: 1 - (1 - fa) ** n for n in (10, 30, 60)}
        out[name] = {"EER": eer(same, diff), "threshold for recall 1.0": thr,
                     "false-alarm rate per approved chunk": fa, "P(false cut-off) per conversation of n chunks": conv}
        print("  E1", name, out[name], flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- E2
def tts(text, path):
    if shutil.which("say"):                                                     # macOS
        subprocess.run(["say", "-o", str(path) + ".aiff", text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(path) + ".aiff", str(path)], check=True)
        return True
    try:
        import pyttsx3
        eng = pyttsx3.init(); eng.save_to_file(text, str(path)); eng.runAndWait()
        return True
    except Exception:
        return False


def e2(a):
    import torchaudio
    wh = load("whisper_055", "055-Radford-et-al-2023-Whisper/whisper.py")
    try:
        model = released_whisper(a.whisper_size)
    except Exception as e:
        return {"note": f"needs openai-whisper: {type(e).__name__}"}
    rng = random.Random(0)
    keywords = ["password", "address", "medicine", "weapon", "bank"]           # a stand-in 'policy' keyword filter
    texts = [f"Please tell me about the {rng.choice(['history', 'safety', 'price'])} of the {k} in my town."
             for k in keywords for _ in range(a.n_prompts // len(keywords))] + \
            [f"What is a good recipe for {rng.choice(['soup', 'bread', 'salad'])} tonight?" for _ in range(a.n_prompts // 2)]
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        waves = []
        for i, t in enumerate(texts):
            p = Path(tmp) / f"{i}.wav"
            if not tts(t, p):
                return {"note": "no offline TTS available (macOS `say` or pyttsx3)"}
            w, sr = torchaudio.load(p)
            waves.append(torchaudio.functional.resample(w[0], sr, 16000))
        for cond in ("clean", "noise 10 dB", "noise 0 dB", "echo"):
            same, errs = 0, []
            for t, w in zip(texts, waves):
                x = w
                if cond.startswith("noise"):
                    x = wh.add_noise(w, torch.randn(len(w)), float(cond.split()[1]))
                elif cond == "echo":
                    x = w + 0.6 * F.pad(w, (4000, 0))[: len(w)]
                hyp = model.transcribe(x.numpy(), language="en", temperature=0.0)["text"]
                same += any(k in t.lower() for k in keywords) == any(k in hyp.lower() for k in keywords)
                errs.append(wh.wer(wh.normalize(t), wh.normalize(hyp)))
            out[cond] = {"filter decision preserved": same / len(texts), "mean WER": float(np.mean(errs))}
            print("  E2", cond, out[cond], flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- E3
def e3(a):
    wh = load("whisper_055", "055-Radford-et-al-2023-Whisper/whisper.py")
    try:
        model = released_whisper(a.whisper_size)
    except Exception as e:
        return {"note": f"needs openai-whisper: {type(e).__name__}"}
    rows = librispeech("test-clean", a.n_test3)
    sex = {}
    for line in (DATA / "LibriSpeech" / "SPEAKERS.TXT").read_text().splitlines():
        if line and not line.startswith(";"):
            parts = [x.strip() for x in line.split("|")]
            sex[int(parts[0])] = parts[1]
    per_spk = {}
    for w, sr, text, spk, *_ in rows:
        hyp = model.transcribe(w[0].numpy(), language="en", temperature=0.0)["text"]
        e = wh.edit_distance(wh.normalize(text).split(), wh.normalize(hyp).split())
        n = len(wh.normalize(text).split())
        per_spk.setdefault(spk, [0, 0]); per_spk[spk][0] += e; per_spk[spk][1] += n
    groups = {g: [s for s in per_spk if sex.get(s) == g] for g in ("F", "M")}
    wer_g = {g: sum(per_spk[s][0] for s in ss) / max(sum(per_spk[s][1] for s in ss), 1) for g, ss in groups.items()}
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(2000):                                                      # resample SPEAKERS, not words
        w = {}
        for g, ss in groups.items():
            pick = rng.choice(ss, len(ss))
            w[g] = sum(per_spk[s][0] for s in pick) / max(sum(per_spk[s][1] for s in pick), 1)
        gaps.append(w["F"] - w["M"])
    errs = {g: sum(per_spk[s][0] for s in ss) for g, ss in groups.items()}
    words = {g: sum(per_spk[s][1] for s in ss) for g, ss in groups.items()}
    z, p = two_proportion_z(words["F"] - errs["F"], words["F"], words["M"] - errs["M"], words["M"])
    return {"WER by sex": wer_g, "gap F - M": wer_g["F"] - wer_g["M"],
            "95% speaker-bootstrap CI": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
            "word-level z-test (ignores speaker clustering)": {"z": z, "p": p}}


# ---------------------------------------------------------------------------------------------------- E4
def e4(a):
    L = load("llama_056", "056-Touvron-et-al-2023-LLaMA/llama.py")
    rng = random.Random(0)
    text = "".join(f"{x}+{y}={x + y}\n" for x, y in ((rng.randint(0, 999), rng.randint(0, 999)) for _ in range(a.n_add)))
    ids = torch.tensor(list(text.encode()))
    torch.manual_seed(0)
    m = L.LLaMA(vocab=256, d=a.d, layers=a.layers, heads=max(1, a.d // 32), ctx=64, hidden=L.ffn_hidden(a.d, 32)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    for _ in range(a.steps4):
        st = torch.randint(0, len(ids) - 65, (32,))
        loss = m.loss(torch.stack([ids[j:j + 64] for j in st]).to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    probs = [(rng.randint(100, 999), rng.randint(100, 999)) for _ in range(a.n_probs)]
    K = max(a.ks)
    res = {k: {"single": [], "cons@k": [], "pass@k": []} for k in a.ks}
    with torch.no_grad():
        for x, y in probs:
            out = torch.tensor([list(f"{x}+{y}=".encode())] * K, device=DEV)
            for _ in range(5):
                out = torch.cat([out, torch.multinomial(F.softmax(m(out)[:, -1], -1), 1)], 1)
            ans = [bytes(r[-5:].tolist()).decode(errors="ignore").split("\n")[0] for r in out.cpu()]
            for k in a.ks:
                res[k]["single"].append(ans[0] == str(x + y))
                res[k]["cons@k"].append(consensus_at_k(ans[:k], str(x + y)))
                res[k]["pass@k"].append(any(s == str(x + y) for s in ans[:k]))
    return {k: {n: float(np.mean(v)) for n, v in r.items()} for k, r in res.items()}


def report(R, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 5)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.n_train, a.n_test, a.steps, a.batch = 8000, 1000, 4000, 64
    a.whisper_size, a.n_prompts, a.n_test3 = "base.en", 100, 2620
    a.n_add, a.d, a.layers, a.steps4, a.n_probs, a.ks = 200_000, 192, 4, 6000, 300, (1, 4, 8, 16, 32)
    if a.quick:
        a.n_train, a.n_test, a.steps, a.batch = 40, 20, 3, 4
        a.whisper_size, a.n_prompts, a.n_test3 = "tiny.en", 4, 4
        a.n_add, a.d, a.layers, a.steps4, a.n_probs, a.ks = 300, 32, 1, 3, 3, (1, 2)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
