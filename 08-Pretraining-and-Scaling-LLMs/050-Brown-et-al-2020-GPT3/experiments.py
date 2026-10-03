"""Reproduce GPT-3's claims (Brown et al. 2020) in miniature.

A 175B model is far out of reach; instead we train a FAMILY of small GPT-2-style models (049's code) on a synthetic
'web' that mixes ordinary text with naturally occurring task demonstrations, and measure the paper's central plots:

  E1  Figure 3.1: validation loss vs training compute (6 N D) for 6 model sizes; fit L(C) = a C^-b.
  E2  Figures 1.2 / 3.10-style: zero-, one- and few-shot accuracy vs model size on synthetic tasks (2- and 3-digit
      addition and subtraction, word unscrambling CL / A1 / RW, reversed words) presented with Section 2.4's
      prompt format. Paper: the gap between zero- and few-shot grows with size.
  E3  In-context learning curves: loss vs number of in-context tokens for each size, against the Bayes-optimal
      learner (the demo's Dirichlet task).
  E4  Data pipeline (Appendix A): quality classifier (logistic regression on hashed words, trained to tell a
      'reference' corpus from a 'raw' one) + Pareto(alpha = 9) sampling + MinHash de-duplication on Gutenberg text
      vs randomly corrupted text; how much survives each stage.
  E5  Section 4 contamination: n-gram overlap (8-grams, since these examples are short; the paper also lowers N for
      short examples) between test prompts and the training mix; zero-shot accuracy on 'clean' vs 'dirty' examples.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import torch

from gpt3 import (PFS_DAY, bayes_optimal_nll, dedup, dirichlet_sequences, is_dirty, ngram_set, pareto_keep,
                  scramble, training_flops)

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
spec = importlib.util.spec_from_file_location("gpt2_049", HERE.parent / "049-Radford-et-al-2019-GPT-2" / "gpt2.py")
gpt2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpt2)
SIZES = [(2, 32), (2, 64), (4, 96), (4, 128), (6, 192), (8, 256)]                 # (layers, d_model)
WORDS = ("apple banana cherry garden window picture yellow purple orange silver market rocket planet ocean river "
         "mountain forest winter summer animal").split()


# ---------------------------------------------------------------------------------------------------- the synthetic web
def task_example(rng, task):
    if task in ("add2", "add3", "sub2", "sub3"):
        d = int(task[-1]); a, b = rng.randrange(10 ** (d - 1), 10 ** d), rng.randrange(10 ** (d - 1), 10 ** d)
        return (f"{a} plus {b} =", f" {a + b}") if task.startswith("add") else (f"{a} minus {b} =", f" {a - b}")
    w = rng.choice(WORDS)
    kind = {"cycle": "CL", "anagram": "A1", "reverse": "RW"}[task]
    return f"unscramble {scramble(w, kind, rng)} =", f" {w}"


TASKS = ["add2", "add3", "sub2", "sub3", "cycle", "anagram", "reverse"]


def web_document(rng):
    """Mostly filler prose; sometimes a run of 1-8 demonstrations of a random task (naturally occurring tasks)."""
    out = []
    for _ in range(rng.randint(2, 5)):
        if rng.random() < 0.4:
            task = rng.choice(TASKS)
            out += ["".join(task_example(rng, task)) + " ." for _ in range(rng.randint(1, 8))]
        else:
            out.append(" ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 12))) + " .")
    return " ".join(out)


class CharTok:
    def __init__(self):
        chars = sorted(set("abcdefghijklmnopqrstuvwxyz0123456789 .=-\n"))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars

    def __call__(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ---------------------------------------------------------------------------------------------------- training
def train_family(a):
    rng = random.Random(0)
    tok = CharTok()
    text = "\n".join(web_document(rng) for _ in range(a.n_docs))
    ids = torch.tensor(tok(text))
    cut = int(0.98 * len(ids))
    train, val = ids[:cut], ids[cut:]
    models, curves = {}, {}
    for L, d in SIZES[:a.n_sizes]:
        torch.manual_seed(0)
        m = gpt2.GPT2(vocab=len(tok.itos), n_ctx=a.n_ctx, d=d, layers=L, heads=max(1, d // 32)).to(DEV)
        n_params = sum(p.numel() for n_, p in m.named_parameters() if "wte" not in n_ and "wpe" not in n_)
        opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.1)
        pts = []
        for step in range(1, a.steps + 1):
            st = torch.randint(0, len(train) - a.n_ctx - 1, (a.batch,))
            x = torch.stack([train[s:s + a.n_ctx] for s in st]).to(DEV)
            loss = m.loss(x)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % max(1, a.steps // 10) == 0:
                pts.append({"compute (PF-days)": training_flops(n_params, step * a.batch * a.n_ctx) / PFS_DAY,
                            "val loss": val_loss(m, val, a)})
        models[(L, d)] = m.cpu().eval()
        curves[f"{L}x{d}"] = {"non-embedding params": n_params, "curve": pts}
        print(f"  trained {L}x{d}: {pts[-1]}", flush=True)
    return models, curves, tok, text


@torch.no_grad()
def val_loss(m, val, a, n=50):
    m.eval()
    g = torch.Generator().manual_seed(0)
    tot = 0.0
    for _ in range(n):
        s = torch.randint(0, len(val) - a.n_ctx - 1, (1,), generator=g)
        tot += m.loss(val[s:s + a.n_ctx][None].to(DEV)).item()
    m.train()
    return tot / n


@torch.no_grad()
def greedy(m, tok, prompt, n=6):
    m.to(DEV)
    out = m.generate(tok(prompt)[-m.n_ctx:], n, greedy=True)
    return tok.decode(out).split(" .")[0].split("\n")[0]


def k_shot_accuracy(m, tok, task, K, n, rng):
    ok = 0
    for _ in range(n):
        demos = [task_example(rng, task) for _ in range(K)]
        q, ans = task_example(rng, task)
        prompt = " ".join(f"{x}{y} ." for x, y in demos) + (" " if demos else "") + q
        ok += greedy(m, tok, prompt).strip() == ans.strip()
    return ok / n


# ---------------------------------------------------------------------------------------------------- experiments
def e1(a):
    _, curves, _, _ = train_family(a)
    pts = [(c["curve"][-1]["compute (PF-days)"], c["curve"][-1]["val loss"]) for c in curves.values()]
    X = torch.tensor([[1.0, math.log(c)] for c, _ in pts])
    y = torch.tensor([math.log(l) for _, l in pts])
    coef = torch.linalg.lstsq(X, y[:, None]).solution.flatten()
    return {"curves": curves, "power-law fit L = a C^-b": {"a": math.exp(coef[0]), "b": -coef[1].item()}}


def e2(a):
    models, _, tok, _ = train_family(a)
    out = {}
    for (L, d), m in models.items():
        rng = random.Random(1)
        out[f"{L}x{d}"] = {t: {f"K={K}": k_shot_accuracy(m, tok, t, K, a.n_eval, rng) for K in (0, 1, a.few_k)}
                           for t in TASKS}
        print("  E2", L, d, out[f"{L}x{d}"], flush=True)
    return out


def e3(a):
    V, T, alpha = 10, 64, 0.3
    xt = dirichlet_sequences(2000, T, V, alpha, generator=torch.Generator().manual_seed(1))
    out = {"Bayes-optimal": bayes_optimal_nll(xt, V, alpha)[:, 1:].mean(0).tolist()}
    for L, d in SIZES[:a.n_sizes]:
        torch.manual_seed(0)
        m = gpt2.GPT2(vocab=V, n_ctx=T, d=d, layers=L, heads=max(1, d // 32)).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        for _ in range(a.steps):
            loss = m.loss(dirichlet_sequences(64, T, V, alpha).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lg = m(xt.to(DEV)).cpu()
        nll = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, V), xt[:, 1:].reshape(-1), reduction="none")
        out[f"{L}x{d}"] = nll.view(2000, T - 1).mean(0).tolist()
    return out


def e4(a):
    import urllib.request
    d = HERE.parents[1] / "data" / "gutenberg"; d.mkdir(parents=True, exist_ok=True)
    f = d / "1342.txt"
    if not f.exists():
        urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", f)
    paras = [p.strip().lower() for p in f.read_text(encoding="utf-8", errors="ignore").split("\n\n") if len(p.split()) > 20]
    rng = random.Random(0)
    good = paras[: len(paras) // 2]
    junk = [" ".join(rng.sample(p.split(), len(p.split()))) for p in paras[len(paras) // 2:]]   # word salad
    dupes = [p + " " for p in rng.sample(good, len(good) // 10)]                                 # near-duplicates
    feat = lambda t: torch.bincount(torch.tensor([hash(w) % 4096 for w in t.split()]), minlength=4096).float()
    X = torch.stack([feat(t) for t in good + junk]); y = torch.tensor([1.0] * len(good) + [0.0] * len(junk))
    w = torch.zeros(4096, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], 0.05)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y)
        opt.zero_grad(); loss.backward(); opt.step()
    pool = good + junk + dupes
    scores = torch.sigmoid(torch.stack([feat(t) for t in pool]) @ w + b).tolist()
    kept = [t for t, s in zip(pool, scores) if pareto_keep(s, rng)]
    final = dedup(kept, threshold=0.7)
    n_good = lambda xs: sum(x.strip() in set(good) for x in xs)
    return {"pool": {"good": len(good), "junk": len(junk), "near-duplicates": len(dupes)},
            "after Pareto quality filter": {"total": len(kept), "good or duplicate": n_good(kept)},
            "after MinHash dedup": {"total": len(final), "good": n_good(final)}}


def e5(a):
    models, _, tok, text = train_family(a)
    train_grams = ngram_set(text, 8)                                               # short examples: smaller N
    rng = random.Random(2)
    m = models[SIZES[a.n_sizes - 1]]
    res = {"clean": [0, 0], "dirty": [0, 0]}
    for _ in range(a.n_eval * 4):
        task = rng.choice(TASKS)
        q, ans = task_example(rng, task)
        key = "dirty" if is_dirty(q + ans, train_grams, n=8) else "clean"
        res[key][1] += 1
        res[key][0] += greedy(m, tok, q).strip() == ans.strip()
    return {k: {"n": v[1], "zero-shot accuracy": v[0] / max(1, v[1])} for k, v in res.items()}


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        L += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.n_docs, a.n_ctx, a.batch, a.lr, a.steps, a.n_sizes, a.n_eval, a.few_k = 200000, 256, 64, 1e-3, 20000, 6, 200, 8
    if a.quick:
        a.n_docs, a.n_ctx, a.batch, a.steps, a.n_sizes, a.n_eval, a.few_k = 500, 64, 8, 30, 2, 3, 2
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
