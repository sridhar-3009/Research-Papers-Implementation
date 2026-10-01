"""Reproduce Bahdanau, Cho & Bengio (2015) at small scale on Multi30k English -> German.

(The paper: WMT'14 English -> French, 348M words, 30k vocabularies, n = 1000, m = 620, l = 500,
~5 days per model.)

  E1  Table 1: RNNencdec-30/-50 vs RNNsearch-30/-50 (trained on sentences up to 30 / 50 words; here up to
      15 / 30 tokens because Multi30k sentences are short). Test BLEU on all sentences and on sentences with
      no [UNK]. Paper: encdec-30 13.93, search-30 21.50, encdec-50 17.82, search-50 26.75 (All).
  E2  Figure 2: BLEU as a function of source length. Paper: RNNencdec collapses on long sentences,
      RNNsearch-50 does not.
  E3  Figure 3: the soft alignments alpha_ij of a few test sentences, saved as heatmaps.

Training (Appendix B): AdaDelta (eps 1e-6, rho 0.95), gradient norm clipped at 1, minibatches of 80,
'before every 20-th update, we retrieved 1600 sentence pairs, sorted them according to the lengths and
split them into 20 minibatches'. Paper initialization (Appendix B.1). Beam search for decoding.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # small models, 2 epochs, ~30-60 minutes
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

from attention import BOS, EOS, PAD, RNNsearch, beam_search, bleu, paper_init_

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "multi30k"
URL = "https://github.com/multi30k/dataset/raw/master/data/task1/raw/{}.{}.gz"
DEV = "cpu"                                    # the step-by-step attention loop is CPU-friendly
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
        raw["train"] = raw["train"][:6000]
    sv, tv = Vocab([s for s, _ in raw["train"]], a.vocab), Vocab([t for _, t in raw["train"]], a.vocab)
    return {k: [(sv.encode(s), tv.encode(t)) for s, t in v] for k, v in raw.items()}, sv, tv


def pad(seqs, eos=False):
    seqs = [s + [EOS] if eos else s for s in seqs]
    L = max(map(len, seqs))
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs]).T


def batches(pairs, rng):
    """Appendix B.2: take 1600 pairs, sort by length, split into 20 minibatches of 80."""
    order = list(range(len(pairs)))
    rng.shuffle(order)
    for i in range(0, len(order), 1600):
        chunk = sorted(order[i:i + 1600], key=lambda k: len(pairs[k][0]))
        mbs = [chunk[j:j + 80] for j in range(0, len(chunk), 80)]
        rng.shuffle(mbs)
        yield from mbs


def train(mode, max_len, enc, sv, tv, a):
    torch.manual_seed(0)
    m = paper_init_(RNNsearch(len(sv.itos), len(tv.itos), a.m, a.n, a.l, a.n, mode=mode)).to(DEV)
    opt = torch.optim.Adadelta(m.parameters(), rho=0.95, eps=1e-6)
    pairs = [p for p in enc["train"] if len(p[0]) <= max_len and len(p[1]) <= max_len]
    rng = random.Random(0)
    for ep in range(a.epochs):
        for mb in batches(pairs, rng):
            src = pad([pairs[k][0] for k in mb]).to(DEV)
            tgt = pad([pairs[k][1] for k in mb], eos=True).to(DEV)
            loss = -m.log_prob(src, tgt).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)          # 'at most a predefined threshold of 1'
            opt.step()
        print(f"    {mode}-{max_len} epoch {ep + 1}: loss {loss.item():.2f}", flush=True)
    return m.eval()


def translate(m, pairs, beam):
    hyps = []
    for s, _ in pairs:
        best = beam_search(m, torch.tensor(s)[:, None], beam=beam, max_len=len(s) * 2 + 5)[0][1]
        hyps.append([w for w in best if w != EOS])
    return hyps


def e1(enc, sv, tv, a):
    test = enc["test"][:a.n_test]
    no_unk = [i for i, (s, t) in enumerate(test) if UNK not in s and UNK not in t]
    out = {}
    for mode in ("encdec", "search"):
        for L in (a.short, a.long):
            m = train(mode, L, enc, sv, tv, a)
            torch.save(m.state_dict(), HERE / f"{mode}_{L}.pt")
            hyps = translate(m, test, a.beam)
            refs = [t for _, t in test]
            out[f"RNN{mode}-{L}"] = {"All": bleu(hyps, refs), "No UNK": bleu([hyps[i] for i in no_unk], [refs[i] for i in no_unk]),
                                     "hyps": hyps}
            print(f"  E1 RNN{mode}-{L}: BLEU {out[f'RNN{mode}-{L}']['All']:.2f}", flush=True)
    return out


def e2(enc, sv, tv, a, R):
    if not R.get("e1"):
        return {"note": "run e1 first"}
    test = enc["test"][:a.n_test]
    refs = [t for _, t in test]
    out = {}
    for name, res in R["e1"].items():
        out[name] = {}
        for lo, hi in ((1, 8), (9, 12), (13, 16), (17, 60)):
            sel = [i for i, (s, _) in enumerate(test) if lo <= len(s) <= hi]
            if sel:
                out[name][f"{lo}-{hi}"] = bleu([res["hyps"][i] for i in sel], [refs[i] for i in sel])
    return out


@torch.no_grad()
def e3(enc, sv, tv, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = HERE / f"search_{a.long}.pt"
    if not path.exists():
        return {"note": "run e1 first"}
    m = RNNsearch(len(sv.itos), len(tv.itos), a.m, a.n, a.l, a.n)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    (HERE / "figures").mkdir(exist_ok=True)
    files = []
    for k, (s, _) in enumerate(enc["test"][:4]):
        _, toks, al = beam_search(m, torch.tensor(s)[:, None], beam=a.beam, max_len=len(s) * 2 + 5)[0]
        A = torch.tensor(al)                                                     # (target steps, source words)
        fig, ax = plt.subplots(figsize=(0.4 * len(s) + 2, 0.4 * len(toks) + 2))
        ax.imshow(A, cmap="gray")
        ax.set_xticks(range(len(s))); ax.set_xticklabels([sv.itos[w] for w in s], rotation=90)
        ax.set_yticks(range(len(toks))); ax.set_yticklabels([tv.itos[w] for w in toks])
        fig.tight_layout()
        f = HERE / "figures" / f"e3_alignment_{k}.png"
        fig.savefig(f, dpi=110); plt.close(fig)
        files.append(str(f.relative_to(HERE)))
    return {"figures": files}


def report(R, a):
    L = ["# Results", "", f"Multi30k En->De, n = {a.n}, m = {a.m}, l = {a.l}" + (" (QUICK run)" if a.quick else ""), ""]
    if R.get("e1"):
        L += ["## E1: Table 1 (paper, WMT'14 En->Fr, All: encdec-30 13.93, search-30 21.50, encdec-50 17.82, search-50 26.75)", "",
              "| model | All | No UNK |", "|---|---|---|"]
        L += [f"| {k} | {v['All']:.2f} | {v['No UNK']:.2f} |" for k, v in R["e1"].items()] + [""]
    if R.get("e2") and "note" not in R["e2"]:
        L += ["## E2: BLEU by source length (Figure 2)", ""] + [f"- {k}: {v}" for k, v in R["e2"].items()] + [""]
    if R.get("e3") and "figures" in R["e3"]:
        L += ["## E3: alignments (Figure 3)", ""] + [f"![]({f})" for f in R["e3"]["figures"]]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["e1", "e2", "e3"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--l", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=10000)
    ap.add_argument("--short", type=int, default=15)
    ap.add_argument("--long", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n, a.m, a.l, a.epochs, a.beam, a.n_test = 64, 32, 32, 2, 3, 100
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        enc, sv, tv = data(a)
        t0 = time.time()
        if a.only in (None, "e1"):
            R["e1"] = e1(enc, sv, tv, a); path.write_text(json.dumps(R))
        if a.only in (None, "e2"):
            R["e2"] = e2(enc, sv, tv, a, R); path.write_text(json.dumps(R))
        if a.only in (None, "e3"):
            R["e3"] = e3(enc, sv, tv, a); path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
