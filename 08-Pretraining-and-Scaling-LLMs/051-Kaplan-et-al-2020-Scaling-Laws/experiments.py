"""Measure scaling laws for real (small) Transformer language models, following Kaplan et al. (2020).

Corpus: OpenWebText via Hugging Face `datasets` if installed, otherwise 20 Gutenberg books; byte-level BPE and the
GPT-2 model from paper 049. All 'N' are NON-embedding parameters (12 n_layer d^2), as the paper insists.

  E1  L(N) (Eq. 1.1, Figure 1 centre): ~8 model sizes, each trained to near convergence on plenty of data; fit
      a_N and N_c. Also the same with embedding parameters included, to see the cleaner fit the paper reports.
  E2  Shape independence (Figure 5): at fixed N, vary depth vs width and heads; loss change should be small.
  E3  L(D) and L(N, D) (Eqs. 1.2, 1.5, Figure 4 left): train sizes x dataset fractions with early stopping; fit
      the four parameters of Eq. 1.5 and the overfitting rule D ~ N^0.74.
  E4  L(N, S) learning curves (Eq. 1.6, Figure 4 right) and the compute-efficient frontier: for each compute
      budget, which N reached the lowest loss? Fit N_opt ~ C^p (paper: 0.73).
  E5  Critical batch size (Eq. 5.1, Figure 18): steps vs examples to reach a target loss at several batch sizes;
      fit S_min, E_min, B_crit = E_min / S_min at several target losses (Eq. 1.4).

!! HEAVY: hundreds of training runs. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import importlib.util
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from scaling import fit_L_ND, fit_power_law, non_embedding_params

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
spec = importlib.util.spec_from_file_location("gpt2_049", HERE.parent / "049-Radford-et-al-2019-GPT-2" / "gpt2.py")
gpt2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpt2)
BOOKS = [1342, 84, 11, 2701, 1661, 98, 1952, 345, 2600, 4300, 174, 1400, 16328, 76, 5200, 1080, 2554, 120, 768, 158]


def corpus(a):
    try:
        import datasets
        rows = []
        for r in datasets.load_dataset("Skylion007/openwebtext", split="train", streaming=True):
            rows.append(r["text"])
            if len(rows) >= a.n_docs:
                break
        return "\n\n".join(rows)
    except Exception:
        d = DATA / "gutenberg"; d.mkdir(parents=True, exist_ok=True)
        out = []
        for b in BOOKS[:a.n_books]:
            f = d / f"{b}.txt"
            if not f.exists():
                urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt", f)
            out.append(f.read_text(encoding="utf-8", errors="ignore").split("*** START")[-1].split("*** END")[0])
        return "\n\n".join(out)


def setup(a):
    text = corpus(a)
    bpe = gpt2.ByteBPE(text[:a.bpe_chars], a.merges)
    ids = torch.tensor(bpe.encode(text[:a.max_chars]))
    cut = int(0.98 * len(ids))
    return ids[:cut], ids[cut:], len(bpe)


def batches(ids, a, B=None):
    B = B or a.batch
    while True:
        st = torch.randint(0, len(ids) - a.n_ctx - 1, (B,))
        yield torch.stack([ids[s:s + a.n_ctx] for s in st]).to(DEV)


@torch.no_grad()
def evaluate(m, val, a, n=40):
    m.eval()
    g = torch.Generator().manual_seed(0)
    tot = 0.0
    for _ in range(n):
        s = torch.randint(0, len(val) - a.n_ctx - 1, (1,), generator=g)
        tot += m.loss(val[s:s + a.n_ctx][None].to(DEV)).item()
    m.train()
    return tot / n


def run(train, val, vocab, a, L, d, steps, B=None, heads=None, eval_every=None, target=None):
    """Train one model; returns its non-embedding N, a learning curve and the final validation loss.
    With `target`, stops when the validation loss first reaches it and returns the step count."""
    torch.manual_seed(0)
    m = gpt2.GPT2(vocab=vocab, n_ctx=a.n_ctx, d=d, layers=L, heads=heads or max(1, d // 64)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=0.01)
    warm = min(1000, steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    curve, it = [], batches(train, a, B)
    every = eval_every or max(1, steps // 20)
    for step in range(1, steps + 1):
        loss = m.loss(next(it))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % every == 0:
            v = evaluate(m, val, a)
            curve.append((step, v))
            if target is not None and v <= target:
                return {"N": non_embedding_params(L, d), "steps_to_target": step}
    out = {"N": non_embedding_params(L, d), "N_with_embeddings": sum(p.numel() for p in m.parameters()),
           "curve": curve, "final": min(v for _, v in curve)}
    if target is not None:
        out["steps_to_target"] = None
    return out


SIZES = [(2, 64), (2, 128), (4, 128), (4, 256), (6, 256), (6, 384), (8, 512), (12, 768)]


def e1(a):
    tr, va, V = setup(a)
    res = [run(tr, va, V, a, L, d, a.steps) for L, d in SIZES[:a.n_sizes]]
    N = np.array([r["N"] for r in res]); Ne = np.array([r["N_with_embeddings"] for r in res])
    Lv = np.array([r["final"] for r in res])
    aN, Nc = fit_power_law(N, Lv)
    aNe, _ = fit_power_law(Ne, Lv)
    resid = lambda x, a_, c_: float(np.std(np.log(Lv) - (a_ * np.log(c_) - a_ * np.log(x))))
    return {"runs": [{k: r[k] for k in ("N", "N_with_embeddings", "final")} for r in res],
            "fit non-embedding": {"a_N": aN, "N_c": Nc, "log-residual std": resid(N, aN, Nc)},
            "fit with embeddings": {"a_N": aNe, "log-residual std": resid(Ne, aNe, fit_power_law(Ne, Lv)[1])},
            "paper": {"a_N": 0.076, "N_c": 8.8e13}}


def e2(a):
    tr, va, V = setup(a)
    target_N = non_embedding_params(6, 256)
    out = {}
    for L in (2, 3, 6, 12, 24):
        d = int(round(math.sqrt(target_N / (12 * L)) / 16) * 16)
        for heads in (1, 4):
            if d % heads == 0:
                r = run(tr, va, V, a, L, d, a.steps, heads=heads)
                out[f"L={L}, d={d}, heads={heads}"] = {"N": r["N"], "loss": r["final"]}
                print("  E2", L, d, heads, r["final"], flush=True)
    return out


def e3(a):
    tr, va, V = setup(a)
    rows = []
    for L, d in SIZES[:a.n_sizes]:
        for frac in a.fracs:
            sub = tr[: max(a.n_ctx * 4, int(len(tr) * frac))]
            r = run(sub, va, V, a, L, d, a.steps)
            rows.append({"N": r["N"], "D": len(sub), "L": r["final"]})
            print("  E3", L, d, frac, r["final"], flush=True)
    fit = fit_L_ND(np.array([r["N"] for r in rows], float), np.array([r["D"] for r in rows], float),
                   np.array([r["L"] for r in rows]))
    return {"runs": rows, "Eq. 1.5 fit": fit, "implied D ~ N^(a_N/a_D)": fit["a_N"] / fit["a_D"],
            "paper": {"a_N": 0.076, "a_D": 0.103, "N_c": 6.4e13, "D_c": 1.8e13, "exponent": 0.74}}


def e4(a):
    tr, va, V = setup(a)
    curves = {}
    for L, d in SIZES[:a.n_sizes]:
        r = run(tr, va, V, a, L, d, a.steps)
        curves[r["N"]] = [(6 * r["N"] * s * a.batch * a.n_ctx / 8.64e19, v) for s, v in r["curve"]]   # (PF-days, L)
    budgets = np.logspace(np.log10(min(c[0][0] for c in curves.values())),
                          np.log10(max(c[-1][0] for c in curves.values())), 12)
    frontier = []
    for C in budgets:
        best = min(((N, min((v for c, v in pts if c <= C), default=float("inf"))) for N, pts in curves.items()),
                   key=lambda t: t[1])
        if math.isfinite(best[1]):
            frontier.append({"C (PF-days)": float(C), "N_opt": int(best[0]), "loss": best[1]})
    p = np.polyfit(np.log([f["C (PF-days)"] for f in frontier]), np.log([f["N_opt"] for f in frontier]), 1)[0]
    return {"frontier": frontier, "N_opt ~ C^p, p =": p, "paper": 0.73}


def e5(a):
    tr, va, V = setup(a)
    L, d = SIZES[min(3, a.n_sizes - 1)]
    out = {}
    for target in a.targets:
        pts = []
        for B in a.batch_sizes:
            r = run(tr, va, V, a, L, d, a.steps, B=B, eval_every=max(1, a.steps // 100), target=target)
            if r["steps_to_target"]:
                pts.append((r["steps_to_target"], r["steps_to_target"] * B * a.n_ctx))
        if len(pts) >= 3:                                                           # fit (S/Smin - 1)(E/Emin - 1) = 1
            S, E = np.array(pts, float).T
            best, best_err = None, float("inf")
            for Smin in np.linspace(0.3 * S.min(), 0.99 * S.min(), 60):
                for Emin in np.linspace(0.3 * E.min(), 0.99 * E.min(), 60):
                    err = float(np.mean(((S / Smin - 1) * (E / Emin - 1) - 1) ** 2))
                    if err < best_err:
                        best, best_err = (Smin, Emin), err
            out[f"target {target}"] = {"points (S, E)": pts, "S_min": best[0], "E_min": best[1],
                                       "B_crit (tokens)": best[1] / best[0]}
            print("  E5", target, out[f"target {target}"]["B_crit (tokens)"], flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        L += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.n_docs, a.n_books, a.bpe_chars, a.merges, a.max_chars = 300000, 20, 3_000_000, 8000, 1_000_000_000
    a.n_ctx, a.batch, a.lr, a.steps, a.n_sizes = 512, 64, 6e-4, 20000, 8
    a.fracs, a.targets, a.batch_sizes = (1 / 64, 1 / 16, 1 / 4, 1.0), (5.0, 4.5, 4.0), (8, 16, 32, 64, 128, 256)
    if a.quick:
        a.n_docs, a.n_books, a.bpe_chars, a.merges, a.max_chars = 200, 2, 50_000, 200, 300_000
        a.n_ctx, a.batch, a.steps, a.n_sizes = 32, 4, 20, 3
        a.fracs, a.targets, a.batch_sizes = (0.25, 1.0), (7.0,), (2, 4, 8)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
