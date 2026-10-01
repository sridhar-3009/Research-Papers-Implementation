"""Reproduce the ideas of Cho et al. (2014) at small scale.

The paper trained on 348M words of WMT'14 English-French phrase pairs and plugged the scores into the Moses
phrase-based system (BLEU 33.30 -> 33.87 test; 34.64 with a neural LM too). That pipeline is out of reach,
so we use Multi30k (29k English-German sentence pairs, github.com/multi30k/dataset):

  E1  Scoring translation pairs (Section 3.1): train the encoder-decoder, then for each test source
      sentence rank its true translation among 10 candidates (9 random other translations) by
      log p(y | x). Accuracy of picking the right one, GRU vs the plain tanh unit ('we were not able to
      get meaningful result with an oft-used tanh unit without any gating').
  E2  Generation: greedy translations of the test set; corpus BLEU (Table 3 shows sampled translations).
  E3  Section 4.4 / Figure 5: phrase representations c of test sentences, projected to 2-D (PCA);
      nearby points should be semantically/syntactically similar sentences (we print nearest neighbours).
  E4  A tiny log-linear rescorer (Eq. 9): combine the RNN score with a length feature, choose the weight
      on a dev set, report candidate-ranking accuracy on test.

Training: AdaDelta (rho 0.95, eps 1e-6), batches of 64, sentences up to 20 tokens, vocabularies of
the most frequent words (the paper: 15,000) with [UNK].

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # small model, 2 epochs, ~20-40 minutes
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

from encdec import BOS, EOS, PAD, EncoderDecoder, bleu, loglinear

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
        words = [w for w, _ in Counter(w for s in sents for w in s).most_common(size)]
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, s):
        return [self.stoi.get(w, UNK) for w in s]


def pad_batch(seqs, eos=False):
    seqs = [s + [EOS] if eos else s for s in seqs]
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs]).T


def data(a):
    src = {k: load(k, "en") for k in ("train", "valid", "test")}
    tgt = {k: load(k, "de") for k in ("train", "valid", "test")}
    keep = lambda k: [(s, t) for s, t in zip(src[k], tgt[k]) if len(s) <= a.max_len and len(t) <= a.max_len]
    pairs = {k: keep(k) for k in src}
    if a.quick:
        pairs["train"] = pairs["train"][:5000]
    sv, tv = Vocab([s for s, _ in pairs["train"]], a.vocab), Vocab([t for _, t in pairs["train"]], a.vocab)
    enc = {k: [(sv.encode(s), tv.encode(t)) for s, t in v] for k, v in pairs.items()}
    return enc, sv, tv


def train(unit, enc, sv, tv, a):
    torch.manual_seed(0)
    m = EncoderDecoder(len(sv.itos), len(tv.itos), n=a.hidden, emb=a.emb, maxout=a.maxout, rank=a.rank, unit=unit).to(DEV)
    opt = torch.optim.Adadelta(m.parameters(), rho=0.95, eps=1e-6)
    rng = random.Random(0)
    for ep in range(a.epochs):
        rng.shuffle(enc["train"])
        total = 0.0
        for i in range(0, len(enc["train"]), 64):
            b = enc["train"][i:i + 64]
            src, tgt = pad_batch([s for s, _ in b]).to(DEV), pad_batch([t for _, t in b], eos=True).to(DEV)
            loss = -m.log_prob(src, tgt).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(b)
        print(f"    {unit} epoch {ep + 1}: train NLL per sentence {total / len(enc['train']):.2f}", flush=True)
    return m


@torch.no_grad()
def candidate_scores(m, pairs, n_cand=10, seed=0):
    """For each source: log p(y|x) of its true translation (index 0) and of n_cand-1 random wrong ones."""
    rng = random.Random(seed)
    out = []
    for s, t in pairs:
        cands = [t] + [rng.choice(pairs)[1] for _ in range(n_cand - 1)]
        src = pad_batch([s] * n_cand).to(DEV)
        tgt = pad_batch(cands, eos=True).to(DEV)
        out.append((m.log_prob(src, tgt).cpu().tolist(), [len(c) for c in cands]))
    return out


def e1(enc, sv, tv, a):
    res = {}
    for unit in ("gru", "tanh"):
        m = train(unit, enc, sv, tv, a)
        if unit == "gru":
            torch.save(m.state_dict(), HERE / "encdec.pt")
        scores = candidate_scores(m, enc["test"][:a.n_test])
        res[unit] = sum(max(range(len(sc)), key=sc.__getitem__) == 0 for sc, _ in scores) / len(scores)
        print(f"  E1 {unit}: picks the true translation {100 * res[unit]:.1f}% of the time (chance 10%)", flush=True)
    return res


def load_model(sv, tv, a):
    m = EncoderDecoder(len(sv.itos), len(tv.itos), n=a.hidden, emb=a.emb, maxout=a.maxout, rank=a.rank).to(DEV)
    m.load_state_dict(torch.load(HERE / "encdec.pt", map_location=DEV))
    return m.eval()


def e2(enc, sv, tv, a):
    m = load_model(sv, tv, a)
    hyps, refs = [], []
    for i in range(0, a.n_test, 64):
        b = enc["test"][i:i + 64]
        out = m.generate(pad_batch([s for s, _ in b]).to(DEV), max_len=a.max_len + 5).T.cpu().tolist()
        for o, (_, t) in zip(out, b):
            hyps.append([w for w in o if w not in (PAD, EOS)][:a.max_len + 5]); refs.append(t)
    examples = [(" ".join(sv.itos[w] for w in s), " ".join(tv.itos[w] for w in h))
                for (s, _), h in zip(enc["test"][:5], hyps[:5])]
    return {"BLEU": bleu(hyps, refs), "examples": examples}


@torch.no_grad()
def e3(enc, sv, tv, a):
    m = load_model(sv, tv, a)
    pairs = enc["test"][:a.n_test]
    C = torch.cat([m.encode(pad_batch([s for s, _ in pairs[i:i + 64]]).to(DEV)) for i in range(0, len(pairs), 64)]).cpu()
    Cn = C / C.norm(dim=1, keepdim=True)
    sims = Cn @ Cn.T
    sims.fill_diagonal_(-1)
    show = []
    for i in range(5):
        j = sims[i].argmax().item()
        show.append((" ".join(sv.itos[w] for w in pairs[i][0]), " ".join(sv.itos[w] for w in pairs[j][0])))
    return {"nearest neighbours in c-space": show}


def e4(enc, sv, tv, a):
    m = load_model(sv, tv, a)
    dev = candidate_scores(m, enc["valid"][:a.n_test], seed=1)
    test = candidate_scores(m, enc["test"][:a.n_test], seed=2)

    def acc(scored, w):
        return sum(max(range(len(sc)), key=lambda k: loglinear([sc[k], L[k]], [1.0, w])) == 0 for sc, L in scored) / len(scored)

    best_w = max([x / 4 for x in range(-8, 9)], key=lambda w: acc(dev, w))     # 'tuned on a development set'
    return {"RNN score only": acc(test, 0.0), f"RNN + length feature (w = {best_w})": acc(test, best_w)}


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


def report(R, a):
    L = ["# Results", "", "Multi30k English -> German" + (" (QUICK run)" if a.quick else ""), ""]
    if R.get("e1"):
        L += ["## E1: choose the true translation among 10 by log p(y|x)", ""] + [f"- {k}: {100 * v:.1f}%" for k, v in R["e1"].items()] + [""]
    if R.get("e2"):
        L += [f"## E2: greedy translation, test BLEU {R['e2']['BLEU']:.2f}", ""] + [f"- {s}  ->  {h}" for s, h in R["e2"]["examples"]] + [""]
    if R.get("e3"):
        L += ["## E3: nearest neighbours of the phrase representation c", ""] + [f"- {s}  ~  {n}" for s, n in R["e3"]["nearest neighbours in c-space"]] + [""]
    if R.get("e4"):
        L += ["## E4: log-linear combination (Eq. 9)", ""] + [f"- {k}: {100 * v:.1f}%" for k, v in R["e4"].items()]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--hidden", type=int, default=1000)
    ap.add_argument("--emb", type=int, default=100)
    ap.add_argument("--maxout", type=int, default=500)
    ap.add_argument("--rank", type=int, default=100)
    ap.add_argument("--vocab", type=int, default=15000)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.hidden, a.emb, a.maxout, a.rank, a.epochs, a.n_test = 128, 64, 64, 32, 2, 200
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
