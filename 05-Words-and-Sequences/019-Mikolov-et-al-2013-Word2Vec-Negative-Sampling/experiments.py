"""Reproduce the experiments of Mikolov et al. (2013) at a smaller scale.

The paper trained on 1 billion words of Google News (692K-word vocabulary). We use text8
(the first 100 MB of English Wikipedia, ~17M words; http://mattmahoney.net/dc/text8.zip) and
Google's analogy questions (questions-words.txt, the test set of the paper, ~19.5K questions).

  E1  Table 1: NEG-5, NEG-15, HS-Huffman, NCE-5, each without and with 1e-5 subsampling.
      Training time and syntactic / semantic / total analogy accuracy.
      Paper (1B words, 300-d): NEG-15 + subsampling 61%, HS 47% (55% with subsampling).
  E2  Section 4: phrases. Two passes of Eq. (6), then train NEG on the phrase corpus and show the
      nearest neighbours of some phrases (Table 4 style).
  E3  Section 5 / Table 5: additive compositionality: nearest words to vec(a) + vec(b).
  E4  Figure 2: PCA of country and capital vectors.

Settings follow the released C code: 300-d (paper) or --dim, window 5 (dynamic), min_count 5,
learning rate 0.025 decaying linearly to ~0, one epoch.

!! HEAVY. Not run on the author's laptop. Downloads ~31 MB; each model takes from ~10 minutes
   (NEG, batched NumPy) to hours (HS, one pair at a time).
       python3 experiments.py --quick          # first 2M words, 100-d, ~5-15 minutes
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from word2vec import (SkipGram, Vocab, analogy, merge_phrases, nearest, noise_distribution, skipgram_pairs, subsample)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
TEXT8 = "http://mattmahoney.net/dc/text8.zip"
QUESTIONS = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"


def load_text8(n_words=None):
    DATA.mkdir(exist_ok=True)
    z = DATA / "text8.zip"
    if not z.exists():
        print("downloading text8 (31 MB)...", flush=True)
        urllib.request.urlretrieve(TEXT8, z)
    with zipfile.ZipFile(z) as f:
        tokens = f.read("text8").decode().split()
    return tokens[:n_words] if n_words else tokens


def load_questions(vocab):
    path = DATA / "questions-words.txt"
    if not path.exists():
        urllib.request.urlretrieve(QUESTIONS, path)
    qs, section = [], None
    for line in path.read_text().splitlines():
        if line.startswith(":"):
            section = line[2:]
            continue
        words = line.lower().split()
        if all(w in vocab.index for w in words):
            qs.append((section, [vocab.index[w] for w in words]))
    return qs


def evaluate(vectors, questions):
    """'gram*' sections are syntactic, the rest semantic (as in the paper's split)."""
    right = {"syntactic": [0, 0], "semantic": [0, 0]}
    for section, (a, b, c, d) in questions:
        kind = "syntactic" if section.startswith("gram") else "semantic"
        right[kind][1] += 1
        right[kind][0] += int(analogy(vectors, a, b, c)[0] == d)
    acc = {k: v[0] / max(v[1], 1) for k, v in right.items()}
    acc["total"] = sum(v[0] for v in right.values()) / max(sum(v[1] for v in right.values()), 1)
    return acc


def train(ids, vocab, method, k, sub, a, seed=0):
    """One epoch. method in {'NEG', 'NCE', 'HS'}; sub = subsampling threshold or None."""
    rng = np.random.default_rng(seed)
    if sub:
        ids = subsample(ids, vocab.counts, sub, rng)
    c, o = skipgram_pairs(ids, a.window, rng)
    order = rng.permutation(len(c))                                # shuffle pairs (the C code streams them)
    c, o = c[order], o[order]
    m = SkipGram(len(vocab), a.dim, vocab.counts if method == "HS" else None, seed=seed)
    P = noise_distribution(vocab.counts)
    shift = np.log(k * P) if method == "NCE" else None
    t0, n = time.time(), len(c)
    B = 1024
    for s in range(0, n, B):
        lr = max(0.025 * (1 - s / n), 0.025 * 1e-4)
        cb, ob = c[s:s + B], o[s:s + B]
        if method == "HS":
            for ci, oi in zip(cb, ob):
                m.hs_step(ci, oi, lr)
        else:
            negs = rng.choice(len(vocab), size=(len(cb), k), p=P)
            m.neg_step_batch(cb, ob, negs, lr, shift)
        if (s // B) % 2000 == 0:
            print(f"    {method}-{k} sub={sub}: {100 * s / n:.0f}%", flush=True)
    return m, (time.time() - t0) / 60


def e1(tokens, a):
    vocab = Vocab(tokens, 5)
    ids = vocab.encode(tokens)
    qs = load_questions(vocab)
    out = {}
    for sub in (None, 1e-5):
        for method, k in (("NEG", 5), ("NEG", 15), ("HS", 0), ("NCE", 5)):
            if method == "HS" and a.skip_hs:
                continue
            m, minutes = train(ids, vocab, method, k, sub, a)
            acc = evaluate(m.v_in, qs)
            name = f"{method}{'-' + str(k) if k else ''}" + (" + subsampling" if sub else "")
            out[name] = dict(minutes=minutes, **acc)
            print(f"  {name}: {minutes:.1f} min, total accuracy {acc['total']:.3f}", flush=True)
            if name == "NEG-15 + subsampling":
                np.save(HERE / "vectors.npy", m.v_in)
                (HERE / "vocab.txt").write_text("\n".join(vocab.words))
    out["questions used"] = len(qs)
    return out


def e2(tokens, a):
    phr = merge_phrases(tokens, threshold=a.phrase_threshold, delta=5)
    phr = merge_phrases(phr, threshold=a.phrase_threshold / 2, delta=5)           # second pass: longer phrases
    vocab = Vocab(phr, 5)
    m, _ = train(vocab.encode(phr), vocab, "NEG", 15, 1e-5, a)
    found = sorted({w for w in vocab.words if "_" in w}, key=lambda w: -vocab.counts[vocab.index[w]])[:40]
    neigh = {w: [vocab.words[i] for i in nearest(m.v_in, m.v_in[vocab.index[w]], 4, exclude=(vocab.index[w],))]
             for w in found[:15]}
    return {"phrases found (most frequent)": found, "neighbours": neigh}


def e3(_tokens, a):
    if not (HERE / "vectors.npy").exists():
        return {"note": "run e1 first"}
    V = np.load(HERE / "vectors.npy")
    words = (HERE / "vocab.txt").read_text().split("\n")
    idx = {w: i for i, w in enumerate(words)}
    out = {}
    for x, y in (("czech", "currency"), ("vietnam", "capital"), ("german", "airlines"), ("russian", "river"),
                 ("french", "actress")):
        if x in idx and y in idx:
            out[f"{x} + {y}"] = [words[i] for i in nearest(V, V[idx[x]] + V[idx[y]], 4, exclude=(idx[x], idx[y]))]
    return out


def e4(_tokens, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not (HERE / "vectors.npy").exists():
        return {"note": "run e1 first"}
    V = np.load(HERE / "vectors.npy")
    words = (HERE / "vocab.txt").read_text().split("\n")
    idx = {w: i for i, w in enumerate(words)}
    pairs = [("china", "beijing"), ("russia", "moscow"), ("japan", "tokyo"), ("turkey", "ankara"), ("poland", "warsaw"),
             ("germany", "berlin"), ("france", "paris"), ("italy", "rome"), ("greece", "athens"), ("spain", "madrid"),
             ("portugal", "lisbon")]
    pairs = [p for p in pairs if p[0] in idx and p[1] in idx]
    X = np.array([V[idx[w]] for p in pairs for w in p])
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Y = X @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (cty, cap) in enumerate(pairs):
        p, q = Y[2 * i], Y[2 * i + 1]
        ax.plot(*zip(p, q), "k-", lw=0.5)
        ax.text(*p, cty, color="C0"); ax.text(*q, cap, color="C1")
        ax.scatter(*zip(p, q), s=8)
    ax.set_title("PCA of country and capital vectors (Figure 2)")
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "e4_fig2.png", dpi=130); plt.close(fig)
    return {"figure": "figures/e4_fig2.png"}


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


def report(R, a):
    L = ["# Results", "", f"text8 ({a.words or 'all'} words), {a.dim}-d, window {a.window}" + (" (QUICK run)" if a.quick else ""), ""]
    if R.get("e1"):
        L += ["## E1: Table 1 (paper: 1B words, 300-d)", "", "| method | minutes | syntactic | semantic | total |", "|---|---|---|---|---|"]
        L += [f"| {k} | {v['minutes']:.1f} | {100 * v['syntactic']:.1f}% | {100 * v['semantic']:.1f}% | {100 * v['total']:.1f}% |"
              for k, v in R["e1"].items() if isinstance(v, dict)]
        L += ["", f"({R['e1']['questions used']} analogy questions had all four words in the vocabulary)", ""]
    if R.get("e2"):
        L += ["## E2: phrases", "", "found: " + ", ".join(R["e2"]["phrases found (most frequent)"][:20]), ""]
        L += [f"- {k}: {', '.join(v)}" for k, v in R["e2"]["neighbours"].items()] + [""]
    if R.get("e3") and "note" not in R["e3"]:
        L += ["## E3: vector addition (Table 5)", ""] + [f"- {k}: {', '.join(v)}" for k, v in R["e3"].items()] + [""]
    if R.get("e4") and "figure" in R["e4"]:
        L += ["## E4: Figure 2", "", f"![]({R['e4']['figure']})"]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--words", type=int, default=None)
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--skip-hs", action="store_true", help="skip the slow one-pair-at-a-time HS models")
    ap.add_argument("--phrase-threshold", type=float, default=1e-6)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.words, a.dim, a.skip_hs = a.words or 2_000_000, 100, True
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        tokens = load_text8(a.words)
        t0 = time.time()
        for name, fn in EXPS.items():
            if a.only in (None, name):
                print(name, flush=True)
                R[name] = fn(tokens, a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
