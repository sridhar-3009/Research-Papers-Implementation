"""Reproduce the Penn Treebank experiments of Zaremba, Sutskever & Vinyals (2014), Section 4.1.

Data: PTB from the paper's own repository (github.com/wojzaremba/lstm/data): 929k / 73k / 82k words,
10k vocabulary.

  E1  Table 1 (single models): non-regularized 2x200 (paper 120.7 / 114.5 valid / test perplexity),
      medium 2x650 with 50% dropout (86.2 / 82.7), large 2x1500 with 65% dropout (82.2 / 78.4).
  E2  Ablation of WHERE dropout goes: medium model with no dropout / the paper's non-recurrent dropout /
      naive dropout on every connection (the paper's claim: only the non-recurrent version works).
  E3  Table 1 (model averaging): average the probabilities of K medium models (paper: 2 -> 77.0, 5 -> 73.3).
  E4  Figure 4: samples from the medium model after 'the meaning of life is'.

Recipes (Section 4.1):
  small : init U[-0.1, 0.1], unroll 20, lr 1 for 4 epochs then /2 per epoch, 13 epochs, clip 5
  medium: init U[-0.05, 0.05], dropout 0.5, unroll 35, lr 1, /1.2 per epoch after epoch 6, 39 epochs, clip 5
  large : init U[-0.04, 0.04], dropout 0.65, unroll 35, lr 1, /1.15 per epoch after epoch 14, 55 epochs, clip 10
  all   : batch 20, 2 layers, hidden state carried across minibatches.

!! HEAVY. Not run on the author's laptop. The medium model took half a day on a 2014 GPU.
       python3 experiments.py --quick           # 1 epoch each, small models, ~20-40 minutes
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from reg_lstm import DeepLSTM, batchify, ensemble_perplexity, run_epoch

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "ptb"
URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{}.txt"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

RECIPES = {
    "small":  dict(n=200, dropout=0.0, init=0.1, steps=20, epochs=13, decay_start=4, decay=2.0, clip=5.0),
    "medium": dict(n=650, dropout=0.5, init=0.05, steps=35, epochs=39, decay_start=6, decay=1.2, clip=5.0),
    "large":  dict(n=1500, dropout=0.65, init=0.04, steps=35, epochs=55, decay_start=14, decay=1.15, clip=10.0),
}


def load_ptb():
    DATA.mkdir(parents=True, exist_ok=True)
    words = {}
    for split in ("train", "valid", "test"):
        path = DATA / f"ptb.{split}.txt"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(split), path)
        words[split] = path.read_text().replace("\n", " <eos> ").split()
    vocab = sorted(set(words["train"]))
    idx = {w: i for i, w in enumerate(vocab)}
    return {k: torch.tensor([idx[w] for w in v]) for k, v in words.items()}, vocab


def train(recipe, data, vocab_size, a, mode="nonrecurrent", seed=0, dropout=None):
    r = dict(RECIPES[recipe])
    epochs = a.epochs or r["epochs"]
    torch.manual_seed(seed)
    model = DeepLSTM(vocab_size, r["n"], 2, dropout=r["dropout"] if dropout is None else dropout, mode=mode,
                     init=r["init"]).to(DEV)
    tr, va = batchify(data["train"], 20), batchify(data["valid"], 20)
    lr, log = 1.0, []
    for ep in range(epochs):
        if ep >= r["decay_start"]:
            lr /= r["decay"]
        tr_ppl = run_epoch(model, tr, r["steps"], lr, r["clip"], DEV)
        va_ppl = run_epoch(model, va, r["steps"], None, device=DEV)
        log.append((tr_ppl, va_ppl))
        print(f"    {recipe} ({mode}) epoch {ep + 1}: train {tr_ppl:.1f} valid {va_ppl:.1f}", flush=True)
    te_ppl = run_epoch(model, batchify(data["test"], 20), r["steps"], None, device=DEV)
    return model, {"valid": log[-1][1], "test": te_ppl, "curve": log}


def e1(data, V, a):
    out = {}
    for recipe in (("small", "medium") if a.quick else ("small", "medium", "large")):
        model, res = train(recipe, data, V, a)
        out[recipe] = res
        if recipe == "medium":
            torch.save(model.state_dict(), HERE / "medium.pt")
    return out


def e2(data, V, a):
    return {mode: train("medium", data, V, a, mode=mode, dropout=0.0 if mode == "none" else None)[1]
            for mode in ("none", "nonrecurrent", "naive")}


def e3(data, V, a):
    models = [train("medium", data, V, a, seed=s)[0] for s in range(a.ensemble)]
    te = batchify(data["test"], 20)
    return {f"{k} models": ensemble_perplexity(models[:k], te, 35, DEV) for k in range(1, a.ensemble + 1)}


@torch.no_grad()
def e4(data, V, a, vocab=None):
    if not (HERE / "medium.pt").exists():
        return {"note": "run e1 first"}
    m = DeepLSTM(V, 650, 2).to(DEV)
    m.load_state_dict(torch.load(HERE / "medium.pt", map_location=DEV))
    m.eval()
    idx = {w: i for i, w in enumerate(vocab)}
    banned = [idx[w] for w in ("<unk>", "N", "$") if w in idx]           # 'we removed unk, N, $'
    g = torch.Generator().manual_seed(0)
    samples = []
    for _ in range(3):
        words = "the meaning of life is".split()
        state = m.init_state(1)
        logits, state = m(torch.tensor([[idx[w]] for w in words]).to(DEV), state)
        last = logits[-1, 0]
        for _ in range(40):
            last[banned] = -1e9
            w = torch.multinomial(F.softmax(last, -1).cpu(), 1, generator=g).item()
            words.append(vocab[w])
            logits, state = m(torch.tensor([[w]]).to(DEV), state)
            last = logits[-1, 0]
        samples.append(" ".join(words).replace(" <eos>", "."))
    return {"samples": samples}


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


def report(R, a):
    L = ["# Results", "", "Penn Treebank" + (" (QUICK run: 1 epoch, far from the paper)" if a.quick else ""), ""]
    paper = {"small": "120.7 / 114.5", "medium": "86.2 / 82.7", "large": "82.2 / 78.4"}
    if R.get("e1"):
        L += ["## E1: Table 1, single models (valid / test perplexity)", "", "| model | ours | paper |", "|---|---|---|"]
        L += [f"| {k} | {v['valid']:.1f} / {v['test']:.1f} | {paper[k]} |" for k, v in R["e1"].items()] + [""]
    if R.get("e2"):
        L += ["## E2: where to put dropout (medium model)", ""] + [f"- {k}: valid {v['valid']:.1f}, test {v['test']:.1f}" for k, v in R["e2"].items()] + [""]
    if R.get("e3"):
        L += ["## E3: averaging medium models (paper: 2 -> 77.0, 5 -> 73.3, 10 -> 72.0)", ""] + [f"- {k}: {v:.1f}" for k, v in R["e3"].items()] + [""]
    if R.get("e4") and "samples" in R["e4"]:
        L += ["## E4: samples", ""] + [f"> {s}" for s in R["e4"]["samples"]]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--epochs", type=int, default=None, help="override every recipe's epoch count")
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.ensemble = a.epochs or 1, 2
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        data, vocab = load_ptb()
        t0 = time.time()
        for name, fn in EXPS.items():
            if a.only in (None, name):
                print(name, flush=True)
                R[name] = fn(data, len(vocab), a, vocab) if name == "e4" else fn(data, len(vocab), a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
