"""Reproduce Mikolov, Le & Sutskever (2013) on public data, English -> Spanish.

The paper used WMT11 news text (575M English / 84M Spanish words) and Google Translate for the
dictionary. We use:
  - text: the English and Spanish sides of Europarl v7 (~50M words each,
    https://www.statmt.org/europarl/v7/es-en.tgz, ~190 MB)
  - dictionary: Facebook's MUSE en-es dictionaries (en-es.0-5000.txt for training,
    en-es.5000-6500.txt for testing), like the paper's "5K most frequent for training, next 1K for test"

  E1  Table 2: Edit Distance, Word Co-occurrence, Translation Matrix, ED + TM; P@1 and P@5, EN->ES
      and ES->EN. Paper (WMT11, EN->ES): 13/24, 19/30, 33/51, 43/60 %.
  E2  Section 5.3: source vectors several times bigger than target vectors work best
      (paper: 800-d English -> 200-d Spanish). Grid over (d_source, d_target).
  E3  Table 3: confidence thresholds 0.0, 0.5, 0.6, 0.7 -> coverage and P@1/P@5.
  E4  Figure 1: PCA of numbers one..five / uno..cinco and animals, both languages.

Preprocessing as in Section 5.1: lowercase tokens only (words with capitals are dropped as named
entities), numbers -> one token, punctuation removed, vocabulary = words with count >= 5.

!! HEAVY. Not run on the author's laptop. Downloads ~190 MB; CBOW training is NumPy (minutes to
   hours per language depending on --max-words and --dim).
       python3 experiments.py --quick          # 3M words per language, 100-d, ~15-30 minutes
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import re
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from translation_matrix import (CBOW, cbow_windows, combined_scores, cooccurrence_vectors, coverage_and_precision,
                                edit_similarity, fit_translation_matrix, precision_at_k, unit)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
EUROPARL = "https://www.statmt.org/europarl/v7/es-en.tgz"
MUSE = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{}.{}.txt"


def download(url, path):
    if not path.exists():
        print(f"downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_corpus(lang, max_words):
    tgz = download(EUROPARL, DATA / "europarl-es-en.tgz")
    with tarfile.open(tgz) as t:
        member = [m for m in t.getmembers() if m.name.endswith(f".{lang}")][0]
        text = t.extractfile(member).read().decode("utf-8", errors="ignore")
    tokens = []
    for tok in re.findall(r"[^\W_]+", text):
        if tok != tok.lower():
            continue                                   # 'we discarded named entities' (words with capitals)
        tokens.append("<num>" if tok.isdigit() else tok)
        if max_words and len(tokens) >= max_words:
            break
    return tokens


def load_dictionary(pair, part):
    path = download(MUSE.format(pair, part), DATA / f"muse-{pair}.{part}.txt")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        a, b = line.split()
        out.append((a, b))
    return out


def vocab_of(tokens, min_count=5):
    counts = Counter(tokens)
    words = [w for w, c in counts.most_common() if c >= min_count]
    return words, {w: i for i, w in enumerate(words)}, np.array([counts[w] for w in words], float)


def train_cbow(tokens, a, dim, seed=0):
    words, index, counts = vocab_of(tokens)
    ids = np.array([index[t] for t in tokens if t in index])
    keep = np.minimum(1.0, np.sqrt(1e-4 / (counts / counts.sum())))         # subsampling (Paper 019)
    rng = np.random.default_rng(seed)
    P = counts ** 0.75; P /= P.sum()
    m = CBOW(len(words), dim, seed)
    for ep in range(a.epochs):
        sub = ids[rng.random(len(ids)) < keep[ids]]
        ctx, center = cbow_windows(sub, a.window)
        order = rng.permutation(len(center))
        for s in range(0, len(order), 1024):
            b = order[s:s + 1024]
            lr = 0.05 * max(1 - (ep * len(order) + s) / (a.epochs * len(order)), 1e-4)
            m.step(ctx[b], center[b], rng.choice(len(words), (len(b), 5), p=P), lr)
        print(f"    CBOW {dim}-d epoch {ep + 1} done", flush=True)
    return m.v_in, words, index, ids


def pairs_in_vocab(dictionary, src_index, tgt_index, limit=None):
    out = [(src_index[a], tgt_index[b], a, b) for a, b in dictionary if a in src_index and b in tgt_index]
    return out[:limit] if limit else out


def evaluate_all(Xs, Xt, src_words, tgt_words, train, test, ids_src, ids_tgt, size_ratio):
    """Table 2 row: ED, co-occurrence, TM, ED + TM (P@1, P@5)."""
    tr_s, tr_t = np.array([p[0] for p in train]), np.array([p[1] for p in train])
    te_s, te_t = np.array([p[0] for p in test]), np.array([p[1] for p in test])
    W = fit_translation_matrix(unit(Xs[tr_s]), unit(Xt[tr_t]), lr=0.01, epochs=20)
    tm_sims = unit(unit(Xs[te_s]) @ W.T) @ unit(Xt).T                       # (n_test, V_tgt)
    cand = np.arange(min(len(tgt_words), 20000))                             # candidates: frequent target words
    ed_sims = np.array([[edit_similarity(src_words[s], tgt_words[c]) for c in cand] for s in te_s])
    co_s = cooccurrence_vectors(ids_src, tr_s, size_ratio=size_ratio)
    co_t = cooccurrence_vectors(ids_tgt, tr_t)
    co_sims = co_s[te_s] @ co_t[cand].T
    res = {}
    for name, S in (("Edit Distance", ed_sims), ("Word Co-occurrence", co_sims), ("Translation Matrix", tm_sims[:, cand]),
                    ("ED + TM", combined_scores(tm_sims[:, cand], ed_sims, 0.5))):
        top = cand[np.argsort(-S, axis=1)[:, :5]]
        res[name] = {"P@1": float(np.mean(top[:, 0] == te_t)), "P@5": float(np.mean([t in row for t, row in zip(te_t, top)]))}
    return res, W


def e1(a):
    en, es = load_corpus("en", a.max_words), load_corpus("es", a.max_words)
    Xen, en_w, en_i, en_ids = train_cbow(en, a, a.dim_src)
    Xes, es_w, es_i, es_ids = train_cbow(es, a, a.dim_tgt)
    np.save(HERE / "vec_en.npy", Xen); np.save(HERE / "vec_es.npy", Xes)
    (HERE / "vocab_en.txt").write_text("\n".join(en_w)); (HERE / "vocab_es.txt").write_text("\n".join(es_w))
    out = {}
    for direction, (Xs, Xt, sw, tw, si, ti, sid, tid, pair) in {
            "EN->ES": (Xen, Xes, en_w, es_w, en_i, es_i, en_ids, es_ids, "en-es"),
            "ES->EN": (Xes, Xen, es_w, en_w, es_i, en_i, es_ids, en_ids, "es-en")}.items():
        train = pairs_in_vocab(load_dictionary(pair, "0-5000"), si, ti)
        test = pairs_in_vocab(load_dictionary(pair, "5000-6500"), si, ti, limit=1000)
        res, _ = evaluate_all(Xs, Xt, sw, tw, train, test, sid, tid, len(sid) / len(tid))
        res["dictionary pairs (train/test)"] = [len(train), len(test)]
        out[direction] = res
        print(f"  {direction}: " + ", ".join(f"{k} P@1 {v['P@1']:.2f}" for k, v in res.items() if isinstance(v, dict)), flush=True)
    return out


def e2(a):
    en, es = load_corpus("en", a.max_words), load_corpus("es", a.max_words)
    out = {}
    cache = {}
    for ds, dt in ((100, 100), (200, 100), (400, 100), (200, 200), (400, 200)):
        if ("en", ds) not in cache:
            cache[("en", ds)] = train_cbow(en, a, ds)
        if ("es", dt) not in cache:
            cache[("es", dt)] = train_cbow(es, a, dt)
        Xen, en_w, en_i, _ = cache[("en", ds)]
        Xes, es_w, es_i, _ = cache[("es", dt)]
        train = pairs_in_vocab(load_dictionary("en-es", "0-5000"), en_i, es_i)
        test = pairs_in_vocab(load_dictionary("en-es", "5000-6500"), en_i, es_i, limit=1000)
        tr_s, tr_t = np.array([p[0] for p in train]), np.array([p[1] for p in train])
        W = fit_translation_matrix(unit(Xen[tr_s]), unit(Xes[tr_t]), lr=0.01, epochs=20)
        te_s, te_t = np.array([p[0] for p in test]), np.array([p[1] for p in test])
        out[f"{ds}-d EN -> {dt}-d ES"] = {k: precision_at_k(W, unit(Xen[te_s]), te_t, unit(Xes), k) for k in (1, 5)}
    return out


def e3(a):
    if not (HERE / "vec_en.npy").exists():
        return {"note": "run e1 first"}
    Xen, Xes = np.load(HERE / "vec_en.npy"), np.load(HERE / "vec_es.npy")
    en_i = {w: i for i, w in enumerate((HERE / "vocab_en.txt").read_text().split("\n"))}
    es_i = {w: i for i, w in enumerate((HERE / "vocab_es.txt").read_text().split("\n"))}
    train = pairs_in_vocab(load_dictionary("en-es", "0-5000"), en_i, es_i)
    test = pairs_in_vocab(load_dictionary("en-es", "5000-6500"), en_i, es_i, limit=1000)
    tr_s, tr_t = np.array([p[0] for p in train]), np.array([p[1] for p in train])
    te_s, te_t = np.array([p[0] for p in test]), np.array([p[1] for p in test])
    W = fit_translation_matrix(unit(Xen[tr_s]), unit(Xes[tr_t]), lr=0.01, epochs=20)
    out = {}
    for th in (0.0, 0.5, 0.6, 0.7):
        cov, p1 = coverage_and_precision(W, unit(Xen[te_s]), te_t, unit(Xes), th, 1)
        _, p5 = coverage_and_precision(W, unit(Xen[te_s]), te_t, unit(Xes), th, 5)
        out[str(th)] = {"coverage": cov, "P@1": p1, "P@5": p5}
    return out


def e4(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not (HERE / "vec_en.npy").exists():
        return {"note": "run e1 first"}
    groups = [(["one", "two", "three", "four", "five"], ["uno", "dos", "tres", "cuatro", "cinco"]),
              (["dog", "cat", "horse", "cow", "pig"], ["perro", "gato", "caballo", "vaca", "cerdo"])]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for row, (en_words, es_words) in enumerate(groups):
        for col, (lang, ws) in enumerate((("en", en_words), ("es", es_words))):
            V = np.load(HERE / f"vec_{lang}.npy")
            idx = {w: i for i, w in enumerate((HERE / f"vocab_{lang}.txt").read_text().split("\n"))}
            ws = [w for w in ws if w in idx]
            if len(ws) < 3:
                continue
            Xw = V[[idx[w] for w in ws]]
            Xw = Xw - Xw.mean(0)
            Y = Xw @ np.linalg.svd(Xw, full_matrices=False)[2][:2].T
            axes[row, col].scatter(*Y.T)
            for w, p in zip(ws, Y):
                axes[row, col].annotate(w, p)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "e4_fig1.png", dpi=120); plt.close(fig)
    return {"figure": "figures/e4_fig1.png"}


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


def report(R, a):
    L = ["# Results", "", f"Europarl, up to {a.max_words or 'all'} words per language, CBOW" + (" (QUICK run)" if a.quick else ""), ""]
    paper = {"Edit Distance": "13 / 24", "Word Co-occurrence": "19 / 30", "Translation Matrix": "33 / 51", "ED + TM": "43 / 60"}
    if R.get("e1"):
        for d, res in R["e1"].items():
            L += [f"## E1: Table 2, {d}", "", "| method | P@1 | P@5 | paper EN->ES (WMT11) |", "|---|---|---|---|"]
            L += [f"| {k} | {100 * v['P@1']:.1f}% | {100 * v['P@5']:.1f}% | {paper[k]} |" for k, v in res.items() if isinstance(v, dict)]
            L += ["", f"dictionary pairs used (train/test): {res['dictionary pairs (train/test)']}", ""]
    if R.get("e2"):
        L += ["## E2: vector sizes (paper: source 2-4x bigger than target works best)", ""]
        L += [f"- {k}: P@1 {100 * v['1']:.1f}%, P@5 {100 * v['5']:.1f}%" for k, v in R["e2"].items()] + [""]
    if R.get("e3") and "note" not in R["e3"]:
        L += ["## E3: Table 3 (paper: coverage 92.5/78.4/54.0/17.0 %, P@1 53/59/71/78 %)", "",
              "| threshold | coverage | P@1 | P@5 |", "|---|---|---|---|"]
        L += [f"| {k} | {100 * v['coverage']:.1f}% | {100 * v['P@1']:.1f}% | {100 * v['P@5']:.1f}% |" for k, v in R["e3"].items()] + [""]
    if R.get("e4") and "figure" in R["e4"]:
        L += ["## E4: Figure 1", "", f"![]({R['e4']['figure']})"]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--max-words", type=int, default=None)
    ap.add_argument("--dim-src", type=int, default=400)
    ap.add_argument("--dim-tgt", type=int, default=200)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.max_words, a.dim_src, a.dim_tgt, a.epochs = a.max_words or 3_000_000, 200, 100, 1
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in EXPS.items():
            if a.only in (None, name):
                print(name, flush=True)
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
