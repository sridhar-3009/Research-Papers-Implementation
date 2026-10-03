"""Reproduce GPT-2's claims (Radford et al. 2019) with scaled-down models.

WebText was never released; OpenWebText (an open re-creation) is used when the Hugging Face `datasets` package is
installed, otherwise 20 Gutenberg books. Four models keep Table 2's depth/width SHAPE at 1/8 of the width
(12 x 96, 24 x 128, 36 x 160, 48 x 200).

  E1  Byte-level BPE vs character-level vs word-level: vocabulary size, tokens per byte, out-of-vocabulary rate.
  E2  Table 3 in miniature: zero-shot bits-per-byte of each model size on held-out datasets of OTHER domains
      (WikiText-103 / PTB style text, if available) -> does held-out quality rise with size?
  E3  LAMBADA (Section 3.3): last-word accuracy vs size, with and without the stop-word filter.
  E4  Summarisation (Section 3.6): ROUGE of 'TL;DR:' + top-k (k = 2) samples vs no hint vs random 3 sentences, on
      CNN/DailyMail (needs `datasets`).
  E5  Section 4: 8-gram Bloom-filter overlap between the training corpus and each test set.
  E6  Section 2.3's architecture changes: pre-LN + final LN + 1/sqrt(N) init vs post-LN, training a 48-layer narrow
      model: loss curves and gradient norms.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

from gpt2 import GPT2, BloomFilter, ByteBPE, lambada_predict, ngrams, normalize, overlap_fraction, per_unit

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOOKS = [1342, 84, 11, 2701, 1661, 98, 1952, 345, 2600, 4300, 174, 1400, 16328, 76, 5200, 1080, 2554, 120, 768, 158]
SHAPES = {"117M-shape": (12, 96), "345M-shape": (24, 128), "762M-shape": (36, 160), "1542M-shape": (48, 200)}


# ---------------------------------------------------------------------------------------------------- data
def hf(name, *args, split="train", n=None):
    try:
        import datasets
    except ImportError:
        return None
    ds = datasets.load_dataset(name, *args, split=split, streaming=True)
    out = []
    for ex in ds:
        out.append(ex)
        if n and len(out) >= n:
            break
    return out


def training_text(a):
    rows = hf("Skylion007/openwebtext", n=a.n_docs)
    if rows:
        return "\n\n".join(r["text"] for r in rows), "OpenWebText"
    d = DATA / "gutenberg"; d.mkdir(parents=True, exist_ok=True)
    texts = []
    for b in BOOKS[:a.n_books]:
        f = d / f"{b}.txt"
        if not f.exists():
            urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt", f)
        texts.append(f.read_text(encoding="utf-8", errors="ignore").split("*** START")[-1].split("*** END")[0])
    return "\n\n".join(texts), "Gutenberg"


def test_sets(a):
    out = {}
    wiki = hf("wikitext", "wikitext-103-raw-v1", split="test")
    if wiki:
        out["WikiText-103"] = "".join(r["text"] for r in wiki)[:a.test_chars]
    ptb = hf("ptb_text_only", split="test")
    if ptb:
        out["PTB"] = "\n".join(r["sentence"] for r in ptb)[:a.test_chars]
    return out


# ---------------------------------------------------------------------------------------------------- training
def stream(bpe, text):
    return torch.tensor(bpe.encode(text))


def train_lm(ids, L, d, a, pre_ln=True):
    torch.manual_seed(0)
    m = GPT2(vocab=a.vocab, n_ctx=a.n_ctx, d=d, layers=L, heads=max(1, d // 32), dropout=0.1,
             scaled_init=pre_ln).to(DEV)
    if not pre_ln:                                                                  # post-LN variant for E6
        for b in m.blocks:
            b.forward = post_ln_forward.__get__(b)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / a.warmup))
    curve, gnorms = [], []
    for step in range(1, a.steps + 1):
        st = torch.randint(0, len(ids) - a.n_ctx - 1, (a.batch,))
        x = torch.stack([ids[s:s + a.n_ctx] for s in st]).to(DEV)
        loss = m.loss(x)
        opt.zero_grad(); loss.backward()
        gnorms.append(nn.utils.clip_grad_norm_(m.parameters(), 1e9).item())
        opt.step(); sched.step()
        if step % max(1, a.steps // 20) == 0:
            curve.append({"step": step, "loss": loss.item(), "grad norm": sum(gnorms[-50:]) / len(gnorms[-50:])})
    return m.cpu(), curve


def post_ln_forward(self, x, causal):
    x = self.ln1(x + self.drop(self.attn(x, x, x, attn_mask=causal, need_weights=False)[0]))
    return self.ln2(x + self.drop(self.proj(torch.nn.functional.gelu(self.fc(x)))))


@torch.no_grad()
def bits_per_byte(m, bpe, text, a):
    m.eval().to(DEV)
    ids = stream(bpe, text)
    nll, n = 0.0, 0
    for k in range(0, len(ids) - 1, a.n_ctx):
        x = ids[k:k + a.n_ctx + 1][None].to(DEV)
        if x.shape[1] < 2:
            break
        lg = m(x[:, :-1])
        nll += torch.nn.functional.cross_entropy(lg[0], x[0, 1:], reduction="sum").item()
        n += x.shape[1] - 1
    m.cpu()
    return per_unit(nll, len(text.encode("utf-8")))["bits per unit"]


# ---------------------------------------------------------------------------------------------------- experiments
def setup(a):
    text, source = training_text(a)
    bpe = ByteBPE(text[:a.bpe_chars], a.merges)
    a.vocab = len(bpe)
    cut = int(0.99 * len(text))
    return text[:cut], text[cut:], bpe, source


def e1(a):
    text, _, bpe, source = setup(a)
    sample = text[:200000]
    words = re.findall(r"\w+|[^\w\s]", sample)
    word_vocab = {w for w in re.findall(r"\w+|[^\w\s]", text[:a.bpe_chars])}
    return {"source": source, "byte-BPE vocab": len(bpe), "byte-BPE tokens per byte": len(bpe.encode(sample)) /
            len(sample.encode()), "character vocab": len(set(text)), "word-level vocab (BPE training slice)":
            len(word_vocab), "word-level OOV rate on later text": sum(w not in word_vocab for w in words) / len(words),
            "byte-BPE OOV rate": 0.0}


def e2(a):
    text, held, bpe, source = setup(a)
    ids = stream(bpe, text[:a.train_chars])
    tests = {"held-out " + source: held[:a.test_chars], **test_sets(a)}
    out = {}
    for name, (L, d) in SHAPES.items():
        m, curve = train_lm(ids, L, d, a)
        out[name] = {t: bits_per_byte(m, bpe, s, a) for t, s in tests.items()}
        out[name]["params"] = sum(p.numel() for p in m.parameters())
        torch.save(m.state_dict(), HERE / f"{name}.pt")
        print("  E2", name, out[name], flush=True)
    return out


def e3(a):
    rows = hf("EleutherAI/lambada_openai", "en", split="test", n=a.n_lambada)
    if not rows:
        return {"note": "needs `pip install datasets`"}
    text, _, bpe, _ = setup(a)
    out = {}
    for name, (L, d) in SHAPES.items():
        m = GPT2(vocab=len(bpe), n_ctx=a.n_ctx, d=d, layers=L, heads=max(1, d // 32))
        if (HERE / f"{name}.pt").exists():
            m.load_state_dict(torch.load(HERE / f"{name}.pt"))
        else:
            m, _ = train_lm(stream(bpe, text[:a.train_chars]), L, d, a)
        m.eval()
        res = {"no filter": 0, "stop-word filter": 0}
        for r in rows:
            ctx, target = r["text"].rsplit(" ", 1)
            ids = bpe.encode(ctx)
            cands = []
            for _ in range(a.lambada_samples):                                      # sample candidate final words
                gen = bpe.decode(m.generate(ids, 6, top_k=10))
                w = gen.strip().split(" ")[0] if gen.strip() else ""
                if w:
                    cands.append((re.sub(r"[^\w]", "", w), m.sequence_logprob(ids + bpe.encode(" " + w),
                                                                            start=len(ids))))
            if cands:
                res["no filter"] += lambada_predict(cands, False) == target
                res["stop-word filter"] += lambada_predict(cands, True) == target
        out[name] = {k: v / len(rows) for k, v in res.items()}
        print("  E3", name, out[name], flush=True)
    return out


def rouge_n(hyp, ref, n):
    h, r = ngrams(hyp.lower().split(), n), ngrams(ref.lower().split(), n)
    if not h or not r:
        return 0.0
    overlap = sum(min(h.count(g), r.count(g)) for g in set(h))
    p, rc = overlap / len(h), overlap / len(r)
    return 0.0 if overlap == 0 else 2 * p * rc / (p + rc)


def e4(a):
    rows = hf("cnn_dailymail", "3.0.0", split="test", n=a.n_summ)
    if not rows:
        return {"note": "needs `pip install datasets`"}
    text, _, bpe, _ = setup(a)
    L, d = SHAPES["1542M-shape"]
    m = GPT2(vocab=len(bpe), n_ctx=a.n_ctx, d=d, layers=L, heads=max(1, d // 32))
    if (HERE / "1542M-shape.pt").exists():
        m.load_state_dict(torch.load(HERE / "1542M-shape.pt"))
    m.eval()
    scores = {"TL;DR:": [], "no hint": [], "random 3 sentences": []}
    rng = random.Random(0)
    for r in rows:
        art, ref = r["article"][-a.n_ctx * 3:], r["highlights"]
        for name, prompt in (("TL;DR:", art + "\nTL;DR:"), ("no hint", art)):
            ids = bpe.encode(prompt)[-(a.n_ctx - 100):]
            gen = bpe.decode(m.generate(ids, 100, top_k=2))
            summ = " ".join(re.split(r"(?<=[.!?])\s+", gen.strip())[:3])
            scores[name].append(rouge_n(summ, ref, 1))
        sents = re.split(r"(?<=[.!?])\s+", r["article"])
        scores["random 3 sentences"].append(rouge_n(" ".join(rng.sample(sents, min(3, len(sents)))), ref, 1))
    return {k: sum(v) / len(v) for k, v in scores.items()}


def e5(a):
    text, held, _, _ = setup(a)
    words = normalize(text[:a.train_chars]).split()
    bloom = BloomFilter(m=max(2 ** 20, 16 * len(words)), k=7)
    for g in ngrams(words):
        bloom.add(g)
    tests = {"held-out slice of the training corpus": held[:a.test_chars], **test_sets(a)}
    return {"filter false-positive rate": bloom.false_positive_rate(),
            **{t: overlap_fraction(bloom, s) for t, s in tests.items()}}


def e6(a):
    text, _, bpe, _ = setup(a)
    ids = stream(bpe, text[:a.train_chars])
    out = {}
    for name, pre in (("pre-LN + final LN + 1/sqrt(N) init (GPT-2)", True), ("post-LN (GPT)", False)):
        _, curve = train_lm(ids, 48, 64, a, pre_ln=pre)
        out[name] = curve
        print("  E6", name, curve[-1], flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        L += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 7)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.n_docs, a.n_books, a.bpe_chars, a.merges = 200000, 20, 2_000_000, 8000
    a.train_chars, a.test_chars, a.n_ctx, a.batch = 200_000_000, 1_000_000, 256, 32
    a.lr, a.warmup, a.steps = 3e-4, 2000, 50000
    a.n_lambada, a.lambada_samples, a.n_summ = 1000, 8, 200
    if a.quick:
        a.n_docs, a.n_books, a.bpe_chars, a.merges = 200, 2, 50_000, 200
        a.train_chars, a.test_chars, a.n_ctx, a.batch, a.steps, a.warmup = 200_000, 5000, 32, 4, 20, 5
        a.n_lambada, a.lambada_samples, a.n_summ = 5, 2, 2
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
