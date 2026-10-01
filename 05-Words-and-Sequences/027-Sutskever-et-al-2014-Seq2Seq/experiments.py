"""Reproduce Sutskever, Vinyals & Le (2014) at small scale on Multi30k English -> German.

(The paper: WMT'14 English -> French, 12M sentence pairs, 4 x 1000 LSTMs, 10 days on 8 GPUs.)

  E1  Section 3.3: reversed vs forward source. Paper: test perplexity 5.8 -> 4.7, BLEU 25.9 -> 30.6.
  E2  Table 1: beam size 1 / 2 / 12, single model vs ensemble of K models (different seeds).
      Paper: single reversed 30.59; ensemble of 5, beam 1: 33.00; beam 2: 34.50; beam 12: 34.81.
  E3  Figure 3: BLEU as a function of sentence length (the paper: no degradation up to ~35 words).
  E4  Figure 2: the 2-D PCA of the encoder's sentence representation for sentences that differ in word
      order or voice (the paper: sensitive to word order, fairly invariant to active/passive).
  E5  Section 3.2: n-best rescoring. One model's beam produces a 12-best list, another model rescores it
      with 'an even average' of the two scores (a stand-in for the SMT 1000-best lists).

Training (Section 3.4, scaled down): uniform [-0.08, 0.08], plain SGD lr 0.7 halved every half epoch
after 5/7.5 of training, batches of 128 of similar lengths, gradient norm clipped at 5.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # 2 layers x 128, 2 epochs, ~20-40 minutes
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import gzip
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from seq2seq import (BOS, EOS, PAD, Seq2Seq, beam_search, bleu, clip_grad_, length_batches, lr_schedule, paper_init_,
                     rescore_nbest)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "multi30k"
URL = "https://github.com/multi30k/dataset/raw/master/data/task1/raw/{}.{}.gz"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
UNK = 3


def load(split, lang):
    DATA.mkdir(parents=True, exist_ok=True)
    name = {"train": "train", "valid": "val", "test": "test_2016_flickr"}[split]
    path = DATA / f"{name}.{lang}.gz"
    if not path.exists():
        urllib.request.urlretrieve(URL.format(name, lang), path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [line.lower().replace(".", " .").replace(",", " ,").split() for line in f]


class Vocab:
    def __init__(self, sents, size):
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + [w for w, _ in Counter(w for s in sents for w in s).most_common(size)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, s):
        return [self.stoi.get(w, UNK) for w in s]


def data(a):
    raw = {k: list(zip(load(k, "en"), load(k, "de"))) for k in ("train", "valid", "test")}
    if a.quick:
        raw["train"] = raw["train"][:8000]
    sv, tv = Vocab([s for s, _ in raw["train"]], a.vocab), Vocab([t for _, t in raw["train"]], a.vocab)
    return {k: [(sv.encode(s), tv.encode(t)) for s, t in v] for k, v in raw.items()}, sv, tv


def batch(pairs, idx, reverse):
    srcs = [list(reversed(pairs[i][0])) if reverse else pairs[i][0] for i in idx]
    tgts = [pairs[i][1] + [EOS] for i in idx]
    S, T = max(map(len, srcs)), max(map(len, tgts))
    src = torch.tensor([s + [PAD] * (S - len(s)) for s in srcs]).T
    tgt = torch.tensor([t + [PAD] * (T - len(t)) for t in tgts]).T
    tgt_in = torch.cat([torch.full((1, len(idx)), BOS), tgt[:-1]])
    return src, torch.tensor([len(s) for s in srcs]), tgt_in, tgt


def train(enc, sv, tv, a, reverse=True, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = paper_init_(Seq2Seq(len(sv.itos), len(tv.itos), a.emb, a.hidden, a.layers)).to(DEV)
    tr = enc["train"]
    steps_per_epoch = (len(tr) + a.batch - 1) // a.batch
    for ep in range(a.epochs):
        for k, idx in enumerate(length_batches(tr, a.batch, rng)):
            frac = (ep + k / steps_per_epoch) * 7.5 / a.epochs          # map our epochs onto the paper's 7.5
            src, lens, tgt_in, tgt = (t.to(DEV) for t in batch(tr, idx, reverse))
            logits = m(src, tgt_in, lens)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=PAD, reduction="sum") / len(idx)
            m.zero_grad(); loss.backward()
            clip_grad_(m, 5.0)
            with torch.no_grad():
                for p in m.parameters():
                    p -= lr_schedule(frac) * p.grad
        print(f"    {'reversed' if reverse else 'forward'} seed {seed} epoch {ep + 1}: loss {loss.item():.2f}", flush=True)
    return m


@torch.no_grad()
def perplexity(m, pairs, reverse):
    m.eval()
    tot = n = 0.0
    for i in range(0, len(pairs), 128):
        src, lens, tgt_in, tgt = (t.to(DEV) for t in batch(pairs, range(i, min(i + 128, len(pairs))), reverse))
        logits = m(src, tgt_in, lens)
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=PAD, reduction="sum").item()
        n += (tgt != PAD).sum().item()
    return float(torch.tensor(tot / n).exp())


def translate(models, pairs, reverse, beam):
    hyps, refs = [], []
    for s, t in pairs:
        src = torch.tensor(list(reversed(s)) if reverse else s)[:, None].to(DEV)
        best = beam_search(models, src, beam=beam, max_len=len(s) + 10)[0][1]
        hyps.append([w for w in best if w != EOS]); refs.append(t)
    return hyps, refs


def e1(enc, sv, tv, a):
    out = {}
    for reverse in (False, True):
        m = train(enc, sv, tv, a, reverse)
        torch.save(m.state_dict(), HERE / f"s2s_{'rev' if reverse else 'fwd'}.pt")
        hyps, refs = translate([m], enc["test"][:a.n_test], reverse, 12)
        out["reversed" if reverse else "forward"] = {"test perplexity": perplexity(m, enc["test"], reverse), "BLEU (beam 12)": bleu(hyps, refs)}
        print(f"  E1 {out}", flush=True)
    return out


def load_model(sv, tv, a, name):
    m = Seq2Seq(len(sv.itos), len(tv.itos), a.emb, a.hidden, a.layers).to(DEV)
    m.load_state_dict(torch.load(HERE / name, map_location=DEV))
    return m.eval()


def e2(enc, sv, tv, a):
    models = [load_model(sv, tv, a, "s2s_rev.pt")] + [train(enc, sv, tv, a, True, seed=s) for s in range(1, a.ensemble)]
    for k, m in enumerate(models[1:], 1):
        torch.save(m.state_dict(), HERE / f"s2s_rev_{k}.pt")
    out = {}
    for K in (1, a.ensemble):
        for beam in (1, 2, 12):
            hyps, refs = translate([m.eval() for m in models[:K]], enc["test"][:a.n_test], True, beam)
            out[f"{K} model(s), beam {beam}"] = bleu(hyps, refs)
            print(f"  E2 {K} models beam {beam}: {out[f'{K} model(s), beam {beam}']:.2f}", flush=True)
    return out


def e3(enc, sv, tv, a):
    m = load_model(sv, tv, a, "s2s_rev.pt")
    hyps, refs = translate([m], enc["test"][:a.n_test], True, 12)
    lens = [len(s) for s, _ in enc["test"][:a.n_test]]
    buckets = {}
    for lo, hi in ((1, 8), (9, 12), (13, 16), (17, 40)):
        sel = [i for i, L in enumerate(lens) if lo <= L <= hi]
        if sel:
            buckets[f"{lo}-{hi} words"] = bleu([hyps[i] for i in sel], [refs[i] for i in sel])
    return buckets


@torch.no_grad()
def e4(enc, sv, tv, a):
    m = load_model(sv, tv, a, "s2s_rev.pt")
    sents = ["a man is riding a horse", "a horse is riding a man", "a man rides a horse", "a horse is ridden by a man",
             "a woman is reading a book", "a book is reading a woman", "a woman reads a book", "a book is read by a woman"]
    vecs = []
    for s in sents:
        ids = list(reversed(sv.encode(s.split())))
        h, c = m.encode(torch.tensor(ids)[:, None].to(DEV))
        vecs.append(torch.cat([h[-1, 0], c[-1, 0]]).cpu())
    X = torch.stack(vecs)
    X = X - X.mean(0)
    _, _, V = torch.linalg.svd(X, full_matrices=False)
    Y = (X @ V[:2].T).tolist()
    return {s: [round(v, 3) for v in y] for s, y in zip(sents, Y)}


def e5(enc, sv, tv, a):
    base = load_model(sv, tv, a, "s2s_rev.pt")
    other = load_model(sv, tv, a, "s2s_rev_1.pt") if (HERE / "s2s_rev_1.pt").exists() else load_model(sv, tv, a, "s2s_fwd.pt")
    hyps0, hyps1, refs = [], [], []
    for s, t in enc["test"][:a.n_test]:
        src = torch.tensor(list(reversed(s)))[:, None].to(DEV)
        nbest = beam_search([base], src, beam=12, max_len=len(s) + 10)
        lstm = []
        for _, toks in nbest:
            tgt = torch.tensor(toks if toks[-1:] == [EOS] else toks + [EOS])[:, None].to(DEV)
            tgt_in = torch.cat([torch.tensor([[BOS]], device=DEV), tgt[:-1]])
            lstm.append(F.log_softmax(other(src, tgt_in), -1)[:, 0].gather(-1, tgt).sum().item())
        best = max(zip(rescore_nbest(nbest, lstm), nbest), key=lambda x: x[0])[1][1]
        hyps0.append([w for w in nbest[0][1] if w != EOS]); hyps1.append([w for w in best if w != EOS]); refs.append(t)
    return {"top of the 12-best": bleu(hyps0, refs), "after rescoring with a second model": bleu(hyps1, refs)}


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}


def report(R, a):
    L = ["# Results", "", f"Multi30k En->De, {a.layers} x {a.hidden} LSTMs" + (" (QUICK run)" if a.quick else ""), ""]
    titles = {"e1": "E1: reversing the source (paper: perplexity 5.8 -> 4.7, BLEU 25.9 -> 30.6)",
              "e2": "E2: beam size and ensembles (Table 1)", "e3": "E3: BLEU by source length (Figure 3)",
              "e4": "E4: 2-D PCA of sentence representations (Figure 2)", "e5": "E5: n-best rescoring"}
    for k, t in titles.items():
        if R.get(k):
            L += [f"## {t}", ""] + [f"- {name}: {v}" for name, v in R[k].items()] + [""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--emb", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.layers, a.hidden, a.emb, a.epochs, a.ensemble, a.n_test = 2, 128, 128, 2, 2, 100
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        enc, sv, tv = data(a)
        t0 = time.time()
        for name, fn in EXPS.items():
            if a.only in (None, name):
                print(name, flush=True)
                R[name] = fn(enc, sv, tv, a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
