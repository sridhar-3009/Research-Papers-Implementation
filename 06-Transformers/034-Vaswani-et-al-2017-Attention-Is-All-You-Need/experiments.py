"""Reproduce 'Attention Is All You Need' (Vaswani et al. 2017) at Multi30k scale.

The paper trains on WMT 2014 (4.5M En-De pairs, 37k shared BPE vocabulary, 8 P100 GPUs, 100k steps of ~25k source
+ 25k target tokens). Here: Multi30k English -> German (29k pairs) with a shared BPE vocabulary learned by our own
`learn_bpe`, the paper's optimizer (Adam 0.9/0.98/1e-9 with the warm-up schedule of Eq. 3), label smoothing 0.1,
residual dropout 0.1, beam 4 with length penalty 0.6, max length = input + 50, and averaging of the last 5
checkpoints.

  E1  Table 2 (base model): test BLEU with and without checkpoint averaging. Paper (WMT14 En-De): 27.3 base,
      28.4 big.
  E2  Table 3: rows (A) heads at constant compute, (B) smaller d_k, (C) model size, (D) dropout and label
      smoothing, (E) learned positions. Dev perplexity (per BPE token) and dev BLEU, as in the paper.
  E3  'Significantly more parallelizable': training throughput (target tokens per second) and time per epoch of
      the base Transformer vs Paper 028's RNNsearch at similar width.
  E4  Attention visualizations (the paper's appendix): encoder self-attention heads of one sentence.
  E5  Table 4 (small data): constituency parsing on NLTK's WSJ sample (Paper 030's linearization and EVALB).
      Paper (full WSJ, 4 layers, d_model 1024): 91.3 F1.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # small models, a few epochs, ~1-2 hours on CPU
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import collections
import gzip
import importlib.util
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import torch

from transformer import (BOS, EOS, PAD, Transformer, apply_bpe, average_checkpoints, beam_search, count_params,
                         label_smoothed_loss, learn_bpe, noam_lr)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------- data
def multi30k():
    d = DATA / "multi30k"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, name in (("train", "train"), ("valid", "val"), ("test", "test_2016_flickr")):
        pair = []
        for lang in ("en", "de"):
            p = d / f"{name}.{lang}.gz"
            if not p.exists():
                urllib.request.urlretrieve(f"https://github.com/multi30k/dataset/raw/master/data/task1/raw/{name}.{lang}.gz", p)
            with gzip.open(p, "rt", encoding="utf-8") as f:
                pair.append([tokenize(line) for line in f])
        out[split] = list(zip(*pair))
    return out


def tokenize(s):
    return re.findall(r"\w+|[^\w\s]", s.lower())


class BPEVocab:
    """A shared source/target subword vocabulary (Section 5.1)."""

    def __init__(self, sentences, n_merges):
        counts = collections.Counter(w for s in sentences for w in s)
        self.merges = learn_bpe(counts, n_merges)
        self.cache = {}
        units = collections.Counter(u for w in counts for u in self.split(w))
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted(units)
        self.stoi = {u: i for i, u in enumerate(self.itos)}

    def split(self, w):
        if w not in self.cache:
            self.cache[w] = apply_bpe(w, self.merges)
        return self.cache[w]

    def encode(self, words):
        return [self.stoi.get(u, 3) for w in words for u in self.split(w)]

    def decode(self, ids):
        text = "".join(self.itos[i] for i in ids if i > 3)
        return text.replace("</w>", " ").split()


def pad(seqs, bos=False, eos=False):
    seqs = [([BOS] if bos else []) + s + ([EOS] if eos else []) for s in seqs]
    L = max(map(len, seqs))
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])


def token_batches(pairs, max_tokens, shuffle=True, seed=0):
    """Batches of roughly `max_tokens` target tokens, pairs of similar length together (Section 5.1)."""
    order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][0]), len(pairs[i][1])))
    batches, cur, n = [], [], 0
    for i in order:
        cur.append(i); n += len(pairs[i][1]) + 2
        if n >= max_tokens:
            batches.append(cur); cur, n = [], 0
    if cur:
        batches.append(cur)
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        batches = [batches[k] for k in torch.randperm(len(batches), generator=g).tolist()]
    return batches


def bleu(hyps, refs, n=4):
    match, total, hl, rl = [0] * n, [0] * n, 0, 0
    for h, r in zip(hyps, refs):
        hl, rl = hl + len(h), rl + len(r)
        for k in range(1, n + 1):
            hc = collections.Counter(tuple(h[i:i + k]) for i in range(len(h) - k + 1))
            rc = collections.Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1))
            match[k - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            total[k - 1] += max(len(h) - k + 1, 0)
    if min(match) == 0:
        return 0.0
    return 100 * min(1.0, math.exp(1 - rl / max(hl, 1))) * math.exp(sum(math.log(m / t) for m, t in zip(match, total)) / n)


def setup(a):
    D = multi30k()
    if a.quick:
        D["train"] = D["train"][:8000]
    vocab = BPEVocab([s for s, t in D["train"]] + [t for s, t in D["train"]], a.bpe)
    enc = {k: [(vocab.encode(s), vocab.encode(t)) for s, t in v] for k, v in D.items()}
    return D, vocab, enc


# ---------------------------------------------------------------------------------------------------- training
def train(enc, vocab, a, keep_last=5, **kw):
    cfg = dict(N=a.N, d_model=a.d_model, d_ff=a.d_ff, h=a.h, dropout=0.1)
    cfg.update({k: v for k, v in kw.items() if k in ("N", "d_model", "d_ff", "h", "dropout", "positions")})
    eps = kw.get("eps", 0.1)
    torch.manual_seed(0)
    m = Transformer(len(vocab.itos), len(vocab.itos), **cfg).to(DEV)
    if "d_k" in kw:                                             # Table 3 (B): shrink only the key/query size
        shrink_keys(m, kw["d_k"])
    opt = torch.optim.Adam(m.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: noam_lr(s + 1, cfg["d_model"], a.warmup))
    snapshots, step, t0, ntok = [], 0, time.time(), 0
    for ep in range(a.epochs):
        m.train()
        for b in token_batches(enc["train"], a.batch_tokens, seed=ep):
            src = pad([enc["train"][i][0] for i in b]).to(DEV)
            tgt = pad([enc["train"][i][1] for i in b], bos=True, eos=True).to(DEV)
            loss = label_smoothed_loss(m(src, tgt[:, :-1]), tgt[:, 1:], eps=eps)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            ntok += int((tgt[:, 1:] != PAD).sum())
        snapshots = (snapshots + [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()}])[-keep_last:]
        print(f"    epoch {ep + 1}: loss {loss.item():.3f}, {ntok / (time.time() - t0):.0f} target tokens/s", flush=True)
    return m.eval(), snapshots, {"tokens/s": ntok / (time.time() - t0), "params M": count_params(m) / 1e6}


def shrink_keys(m, d_k):
    """Row (B): keep d_v = d_model / h but use smaller query/key projections of size d_k per head."""
    import torch.nn as nn
    for layer in list(m.enc) + list(m.dec):
        for att in [getattr(layer, n) for n in ("attn", "self_attn", "cross_attn") if hasattr(layer, n)]:
            d = att.W_Q.in_features
            att.W_Q, att.W_K = nn.Linear(d, att.h * d_k).to(DEV), nn.Linear(d, att.h * d_k).to(DEV)
            att.d_k_qk = d_k

            def split_qk(x, att=att):
                return x.view(x.shape[0], x.shape[1], att.h, -1).transpose(1, 2)
            att.split = split_qk


def evaluate(m, enc, D, vocab, a, split="valid", n=None):
    pairs = enc[split][:n] if n else enc[split]
    tot = cnt = 0.0
    with torch.no_grad():
        for b in token_batches(pairs, a.batch_tokens, shuffle=False):
            src = pad([pairs[i][0] for i in b]).to(DEV)
            tgt = pad([pairs[i][1] for i in b], bos=True, eos=True).to(DEV)
            lp = torch.log_softmax(m(src, tgt[:, :-1]), -1).gather(-1, tgt[:, 1:, None])[..., 0]
            keep = tgt[:, 1:] != PAD
            tot -= (lp * keep).sum().item(); cnt += keep.sum().item()
    hyps = [vocab.decode(beam_search(m, torch.tensor([s], device=DEV), beam=4, alpha=0.6)) for s, _ in pairs[:a.n_bleu]]
    refs = [t for _, t in (D[split][:n] if n else D[split])[:a.n_bleu]]
    return {"perplexity": math.exp(tot / cnt), "BLEU": bleu(hyps, refs)}


# ---------------------------------------------------------------------------------------------------- experiments
def e1(a):
    D, vocab, enc = setup(a)
    m, snaps, info = train(enc, vocab, a)
    torch.save(m.state_dict(), HERE / "base.pt")
    single = evaluate(m, enc, D, vocab, a, "test")
    m.load_state_dict(average_checkpoints(snaps))
    averaged = evaluate(m, enc, D, vocab, a, "test")
    return {"single model": single, f"average of last {len(snaps)}": averaged, **info, "bpe vocab": len(vocab.itos)}


def e2(a):
    D, vocab, enc = setup(a)
    d, h = a.d_model, a.h
    rows = {"base": {},
            "(A) h=1": dict(h=1), "(A) h=4": dict(h=4), "(A) h=16": dict(h=16), "(A) h=32": dict(h=min(32, d)),
            "(B) d_k=16": dict(d_k=16), "(B) d_k=32": dict(d_k=32),
            "(C) N=2": dict(N=2), "(C) N=4": dict(N=4), "(C) d_ff x0.5": dict(d_ff=a.d_ff // 2),
            "(C) d_ff x2": dict(d_ff=2 * a.d_ff),
            "(D) dropout 0": dict(dropout=0.0), "(D) dropout 0.2": dict(dropout=0.2),
            "(D) eps_ls 0": dict(eps=0.0), "(D) eps_ls 0.2": dict(eps=0.2),
            "(E) learned positions": dict(positions="learned")}
    out = {}
    for name, kw in rows.items():
        m, _, info = train(enc, vocab, a, **kw)
        out[name] = {**evaluate(m, enc, D, vocab, a, "valid", n=a.n_dev), "params M": info["params M"]}
        print("  E2", name, out[name], flush=True)
    return out


def e3(a):
    D, vocab, enc = setup(a)
    spec = importlib.util.spec_from_file_location("att028", HERE.parents[1] / "05-Words-and-Sequences" /
                                                  "028-Bahdanau-et-al-2015-Attention-NMT" / "attention.py")
    att = importlib.util.module_from_spec(spec); spec.loader.exec_module(att)
    out = {}
    tr = enc["train"][:2000]
    for name in ("Transformer", "RNNsearch (Paper 028)"):
        torch.manual_seed(0)
        if name == "Transformer":
            m = Transformer(len(vocab.itos), len(vocab.itos), N=a.N, d_model=a.d_model, d_ff=a.d_ff, h=a.h).to(DEV)
            run = lambda s, t: label_smoothed_loss(m(s, t[:, :-1]), t[:, 1:])
        else:
            m = att.RNNsearch(len(vocab.itos), len(vocab.itos), m=a.d_model, n=a.d_model, l=a.d_model,
                              n_align=a.d_model).to(DEV)
            run = lambda s, t: -m.log_prob(s.T, t[:, 1:].T).mean()
        opt = torch.optim.Adam(m.parameters(), 1e-4)
        t0, ntok = time.time(), 0
        for b in token_batches(tr, a.batch_tokens):
            s = pad([tr[i][0] for i in b]).to(DEV)
            t = pad([tr[i][1] for i in b], bos=True, eos=True).to(DEV)
            loss = run(s, t)
            opt.zero_grad(); loss.backward(); opt.step()
            ntok += int((t[:, 1:] != PAD).sum())
        out[name] = {"target tokens/s": ntok / (time.time() - t0), "params M": count_params(m) / 1e6}
        print("  E3", name, out[name], flush=True)
    return out


def e4(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not (HERE / "base.pt").exists():
        return {"note": "run e1 first"}
    D, vocab, enc = setup(a)
    m = Transformer(len(vocab.itos), len(vocab.itos), N=a.N, d_model=a.d_model, d_ff=a.d_ff, h=a.h).to(DEV)
    m.load_state_dict(torch.load(HERE / "base.pt", map_location=DEV)); m.eval()
    s = enc["test"][0][0]
    m.encode(torch.tensor([s], device=DEV))
    w = m.enc[-1].attn.weights[0].cpu()                        # (h, S, S)
    toks = [vocab.itos[i] for i in s]
    (HERE / "figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, (a.h + 1) // 2, figsize=(3 * ((a.h + 1) // 2), 6))
    for k, ax in enumerate(axes.flat[:a.h]):
        ax.imshow(w[k], cmap="gray_r"); ax.set_title(f"head {k}"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(" ".join(toks)[:120]); fig.tight_layout()
    f = HERE / "figures" / "e4_encoder_heads.png"
    fig.savefig(f, dpi=100); plt.close(fig)
    return {"figure": str(f.relative_to(HERE))}


def e5(a):
    gfl_dir = HERE.parents[1] / "05-Words-and-Sequences" / "030-Vinyals-et-al-2015-Grammar-as-Foreign-Language"
    sys.path.insert(0, str(gfl_dir))

    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, gfl_dir / file)
        mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
        return mod
    P = load("parser", "parser.py")
    G = load("gfl_experiments", "experiments.py")
    T = G.load()
    if a.quick:
        T = {k: v[:600] if k == "train" else v[:100] for k, v in T.items()}
    words = sorted({w.lower() for t in T["train"] for w in P.words(t)})
    labels = sorted({s for t in T["train"] for s in P.linearize(t)})
    itos = ["<pad>", "<s>", "</s>", "<unk>"] + words + labels
    stoi = {s: i for i, s in enumerate(itos)}
    encp = lambda t: ([stoi.get(w.lower(), 3) for w in P.words(t)], [stoi[s] for s in P.linearize(t) if s in stoi])
    pairs = [encp(t) for t in T["train"]]
    torch.manual_seed(0)
    m = Transformer(len(itos), len(itos), N=4, d_model=a.d_model, d_ff=a.d_ff, h=a.h, dropout=0.3).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: noam_lr(s + 1, a.d_model, a.warmup))
    for ep in range(a.parse_epochs):
        m.train()
        for b in token_batches(pairs, a.batch_tokens, seed=ep):
            s = pad([pairs[i][0] for i in b]).to(DEV)
            t = pad([pairs[i][1] for i in b], bos=True, eos=True).to(DEV)
            loss = label_smoothed_loss(m(s, t[:, :-1]), t[:, 1:])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval()
    preds = []
    for t in T["test"]:
        src = torch.tensor([encp(t)[0]], device=DEV)
        seq = [itos[i] for i in beam_search(m, src, beam=a.parse_beam, alpha=0.3, extra_len=300)]
        preds.append(P.delinearize(seq, P.words(t)))
    return {"F1": P.evalb(T["test"], preds)["F1"]}


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: Table 2 (base)"), ("e2", "E2: Table 3 variations (dev)"), ("e3", "E3: training speed"),
                 ("e4", "E4: attention heads"), ("e5", "E5: Table 4 parsing (NLTK WSJ sample)")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["e1", "e2", "e3", "e4", "e5"])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.N, a.d_model, a.d_ff, a.h, a.warmup, a.epochs, a.batch_tokens = 6, 512, 2048, 8, 4000, 30, 4096
    a.bpe, a.n_bleu, a.n_dev, a.parse_epochs, a.parse_beam = 8000, 1000, 1014, 40, 21
    if a.quick:
        a.N, a.d_model, a.d_ff, a.h, a.warmup, a.epochs, a.batch_tokens = 2, 128, 512, 8, 400, 4, 2048
        a.bpe, a.n_bleu, a.n_dev, a.parse_epochs, a.parse_beam = 2000, 100, 200, 5, 4
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
