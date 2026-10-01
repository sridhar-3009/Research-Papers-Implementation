"""Reproduce 'Grammar as a Foreign Language' (Vinyals et al. 2015) in the small-data setting.

The paper's WSJ (Penn Treebank, 40K sentences) and its 11M-sentence high-confidence corpus are licensed / private.
We use the freely available 10% sample of the WSJ treebank shipped with NLTK (3,914 sentences, files
wsj_0001-0199): files 0001-0159 train, 0160-0179 dev, 0180-0199 test. That is ~10x less data than the paper's
'WSJ only' row, so absolute F1 will be lower; the comparisons are what we reproduce.

  E1  Table 1 ('WSJ only' rows): baseline LSTM+D (no attention) vs LSTM+A+D vs an ensemble of 5 LSTM+A+D, and
      LSTM+A without dropout (Sec. 3.2: dropout was worth > 2 F1 on WSJ). Paper (WSJ 23): baseline 'no reasonable
      score', LSTM+A+D 88.3, ensemble 90.5. Also: malformed-tree rate (paper 1.5%) and parsing speed.
  E2  Figure 3: F1 by sentence length for LSTM+A+D and the baseline.
  E3  Beam size 1 / 2 / 10 (paper: 'almost irrelevant', beam 1 costs 0.5 F1).
  E4  POS-tag normalization: outputs with 'XX' vs with the real tags (paper: XX is ~1 F1 better).
  E5  Input reversal (paper: not reversing costs ~0.2 F1).
  E6  Figure 4: an attention matrix of the LSTM+A+D model, saved as an image.

Model (Sec. 2.3): 3 layers x 256, word embeddings 512 (random init here; the paper's word2vec init was worth
only +0.3-0.4 F1). Optimizer: the paper says SGD without details; we use SGD lr 0.5, batch 64, gradient norm
clipped at 5, and keep the epoch with the best dev F1 (early stopping).

!! HEAVY (several 3-layer models, dozens of epochs). Not run on the author's laptop.
       python3 experiments.py --quick         # small models, few epochs, ~30-60 minutes on CPU
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import torch

from parser import (PAD, LSTMA, Vocab, batch_pairs, beam_search, clean_ptb, delinearize, evalb, is_well_formed,
                    linearize, parse_tree, words)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "nltk_treebank"
URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/treebank.zip"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load():
    DATA.mkdir(parents=True, exist_ok=True)
    z = DATA / "treebank.zip"
    if not z.exists():
        urllib.request.urlretrieve(URL, z)
        zipfile.ZipFile(z).extractall(DATA)
    split = {"train": [], "dev": [], "test": []}
    for f in sorted((DATA / "treebank" / "combined").glob("wsj_*.mrg")):
        k = int(re.findall(r"\d+", f.name)[0])
        part = "train" if k < 160 else "dev" if k < 180 else "test"
        text, depth, start = f.read_text(), 0, None
        for i, ch in enumerate(text):                        # split the file into top-level bracketed trees
            if ch == "(":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    t = clean_ptb(parse_tree(text[start:i + 1]))
                    if t:
                        split[part].append(t)
    return split


def data(a, normalize_pos=True):
    D = load()
    if a.quick:
        D["train"], D["dev"], D["test"] = D["train"][:1000], D["dev"][:150], D["test"][:150]
    wv = Vocab([w.lower() for t in D["train"] for w in words(t)], min_count=1)
    lv = Vocab([s for t in D["train"] for s in linearize(t, normalize_pos)])
    enc = lambda t: (wv.encode([w.lower() for w in words(t)]), lv.encode(linearize(t, normalize_pos)))
    return D, wv, lv, enc


def parse_all(models, trees, enc, lv, beam, reverse=True):
    out, seqs, t0 = [], [], time.time()
    for t in trees:
        src, _ = batch_pairs([enc(t)], reverse)
        _, toks, att = beam_search(models, src.to(DEV), beam=beam)
        seq = [lv.itos[s] for s in toks]
        seqs.append(seq)
        out.append(delinearize(seq, words(t)))
    return out, seqs, len(trees) / (time.time() - t0)


def train(D, wv, lv, enc, a, attention=True, dropout=None, seed=0, reverse=True):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dropout = a.dropout if dropout is None else dropout
    m = LSTMA(len(wv), len(lv), d=a.d, emb=a.emb, layers=a.layers, attention=attention, dropout=dropout).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=a.lr)
    pairs = [enc(t) for t in D["train"]]
    best, state = -1, None
    for ep in range(a.epochs):
        m.train()
        rng.shuffle(pairs)
        for b in range(0, len(pairs), a.batch):
            src, tgt = batch_pairs(pairs[b:b + a.batch], reverse)
            src, tgt = src.to(DEV), tgt.to(DEV)
            loss = -m.log_prob(src, tgt).sum() / (tgt != PAD).sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        m.eval()
        f1 = evalb(D["dev"], parse_all(m, D["dev"], enc, lv, 1, reverse)[0])["F1"]
        print(f"    attention={attention} dropout={dropout} seed={seed} epoch {ep + 1}: loss {loss.item():.3f}, dev F1 {f1:.1f}",
              flush=True)
        if f1 > best:
            best, state = f1, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    return m.eval()


def by_length(golds, preds, edges=(10, 20, 30, 40, 70)):
    out, lo = {}, 1
    for hi in edges:
        idx = [i for i, t in enumerate(golds) if lo <= len(words(t)) <= hi]
        if idx:
            out[f"{lo}-{hi}"] = evalb([golds[i] for i in idx], [preds[i] for i in idx])["F1"]
        lo = hi + 1
    return out


def e1(a):
    D, wv, lv, enc = data(a)
    out = {}
    base = train(D, wv, lv, enc, a, attention=False)
    ens = [train(D, wv, lv, enc, a, seed=s) for s in range(a.ensemble)]
    torch.save(ens[0].state_dict(), HERE / "lstm_a_d.pt")
    nod = train(D, wv, lv, enc, a, dropout=0.0)
    for name, models in (("baseline LSTM+D", base), ("LSTM+A+D", ens[0]), (f"LSTM+A+D ensemble of {a.ensemble}", ens),
                         ("LSTM+A (no dropout)", nod)):
        r = {}
        for split in ("dev", "test"):
            preds, seqs, speed = parse_all(models, D[split], enc, lv, a.beam)
            r[split] = evalb(D[split], preds)["F1"]
            if split == "test":
                r["malformed %"] = 100 * sum(not is_well_formed(s) for s in seqs) / len(seqs)
                r["sentences / s (beam %d)" % a.beam] = speed
                r["by length (test)"] = by_length(D[split], preds)
        out[name] = r
        print(f"  E1 {name}: dev {r['dev']:.1f}, test {r['test']:.1f}", flush=True)
    return out


def e3(a):
    D, wv, lv, enc = data(a)
    m = LSTMA(len(wv), len(lv), d=a.d, emb=a.emb, layers=a.layers).to(DEV)
    m.load_state_dict(torch.load(HERE / "lstm_a_d.pt", map_location=DEV))
    m.eval()
    return {f"beam {b}": evalb(D["dev"], parse_all(m, D["dev"], enc, lv, b)[0])["F1"] for b in (1, 2, 10)}


def e4(a):
    out = {}
    for norm in (True, False):
        D, wv, lv, enc = data(a, normalize_pos=norm)
        m = train(D, wv, lv, enc, a)
        out["XX tags" if norm else "real POS tags"] = evalb(D["dev"], parse_all(m, D["dev"], enc, lv, a.beam)[0])["F1"]
    return out


def e5(a):
    D, wv, lv, enc = data(a)
    out = {}
    for rev in (True, False):
        m = train(D, wv, lv, enc, a, reverse=rev)
        out["reversed input" if rev else "input in order"] = evalb(D["dev"], parse_all(m, D["dev"], enc, lv, a.beam, rev)[0])["F1"]
    return out


@torch.no_grad()
def e6(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D, wv, lv, enc = data(a)
    m = LSTMA(len(wv), len(lv), d=a.d, emb=a.emb, layers=a.layers)
    m.load_state_dict(torch.load(HERE / "lstm_a_d.pt", map_location="cpu"))
    m.eval()
    t = min((t for t in D["test"] if 8 <= len(words(t)) <= 14), key=lambda t: len(words(t)))
    src, _ = batch_pairs([enc(t)])
    _, toks, att = beam_search(m, src, beam=1)
    A = torch.tensor(att).flip(1).T                        # rows: input words in reading order; columns: outputs
    fig, ax = plt.subplots(figsize=(0.35 * len(toks) + 2, 0.35 * len(words(t)) + 2))
    ax.imshow(A, cmap="gray_r", aspect="auto")
    ax.set_yticks(range(len(words(t)))); ax.set_yticklabels(words(t))
    ax.set_xticks(range(len(toks))); ax.set_xticklabels([lv.itos[s] for s in toks], rotation=90)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "e6_attention.png", dpi=110)
    plt.close(fig)
    return {"figure": "figures/e6_attention.png"}


def report(R, a):
    L = ["# Results", "", f"NLTK WSJ sample, {a.layers} x {a.d}" + (" (QUICK run)" if a.quick else ""), ""]
    for k, title in (("e1", "E1: Table 1, small data (paper WSJ 23: LSTM+A+D 88.3, ensemble 90.5, baseline 'no reasonable score')"),
                     ("e3", "E3: beam size (paper: beam 1 -0.5, beam 2 -0.2 vs beam 10)"),
                     ("e4", "E4: POS normalization (paper: XX ~ +1 F1)"),
                     ("e5", "E5: input reversal (paper: reversing ~ +0.2 F1)")):
        if R.get(k):
            L += [f"## {title}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    if R.get("e6"):
        L += ["## E6: attention (Figure 4)", "", f"![]({R['e6']['figure']})"]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["e1", "e3", "e4", "e5", "e6"])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--emb", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.d, a.emb, a.layers, a.epochs, a.beam, a.ensemble = 96, 64, 2, 8, 3, 2
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
