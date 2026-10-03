"""Chinchilla's experiments (Hoffmann et al. 2022) at small scale, with real Transformer language models.

Corpus: OpenWebText via Hugging Face `datasets` if installed, otherwise Gutenberg books; byte-level BPE and the
GPT-2 model from paper 049; parameters COUNTED WITH embeddings (as this paper does) and FLOPs = 6 N D.

  E1  Approach 2 (IsoFLOP profiles, Figure 3): for 5 budgets, train ~7 model sizes each with tokens = C / 6N
      (cosine schedule matched to that length, decaying 10x); fit a parabola in log N per budget; N_opt ~ C^a.
  E2  Approach 1 (Figure 2): train each size for several cosine lengths; take the lower envelope of all curves
      against FLOPs; N_opt ~ C^a.
  E3  Approach 3: fit L = E + A/N^alpha + B/D^beta to every final loss from E1 + E2 (Huber 1e-3, L-BFGS, grid of
      starts); report alpha, beta, a = beta/(alpha+beta), and tokens per parameter on the frontier.
  E4  Appendix B / Figure A1: cosine cycle length 1x, 1.1x, 1.25x, 1.5x, 2x, 5x the number of training steps;
      overshooting by more than ~25% should hurt.
  E5  The headline: at one fixed budget, train (a) the Chinchilla-style allocation (~20 tokens per parameter) and
      (b) the Kaplan-style allocation (N = 1.3e9 C^0.73 in PF-days, rescaled to the budget); compare final losses.

!! HEAVY. Not run on the author's laptop.
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

from chinchilla import approach1_envelope, approach2_isoflop, fit_parametric

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
spec = importlib.util.spec_from_file_location("gpt2_049", HERE.parent / "049-Radford-et-al-2019-GPT-2" / "gpt2.py")
gpt2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpt2)
BOOKS = [1342, 84, 11, 2701, 1661, 98, 1952, 345, 2600, 4300, 174, 1400, 16328, 76, 5200, 1080, 2554, 120, 768, 158]
SHAPES = [(2, 64), (2, 96), (3, 128), (4, 160), (4, 192), (6, 256), (6, 320), (8, 384), (10, 512), (12, 640)]


def setup(a):
    try:
        import datasets
        rows = []
        for r in datasets.load_dataset("Skylion007/openwebtext", split="train", streaming=True):
            rows.append(r["text"])
            if len(rows) >= a.n_docs:
                break
        text = "\n\n".join(rows)
    except Exception:
        d = DATA / "gutenberg"; d.mkdir(parents=True, exist_ok=True)
        parts = []
        for b in BOOKS[:a.n_books]:
            f = d / f"{b}.txt"
            if not f.exists():
                urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt", f)
            parts.append(f.read_text(encoding="utf-8", errors="ignore").split("*** START")[-1].split("*** END")[0])
        text = "\n\n".join(parts)
    bpe = gpt2.ByteBPE(text[:a.bpe_chars], a.merges)
    ids = torch.tensor(bpe.encode(text[:a.max_chars]))
    cut = int(0.99 * len(ids))
    return ids[:cut], ids[cut:], len(bpe)


def make(L, d, V, a):
    torch.manual_seed(0)
    m = gpt2.GPT2(vocab=V, n_ctx=a.n_ctx, d=d, layers=L, heads=max(1, d // 64)).to(DEV)
    return m, sum(p.numel() for p in m.parameters())


@torch.no_grad()
def val_loss(m, val, a, n=40):
    m.eval()
    g = torch.Generator().manual_seed(0)
    tot = sum(m.loss(val[s:s + a.n_ctx][None].to(DEV)).item()
              for s in torch.randint(0, len(val) - a.n_ctx - 1, (n,), generator=g))
    m.train()
    return tot / n


def train(m, N, tr, va, a, tokens, cycle_mult=1.0, record=0):
    """Train for `tokens` tokens; cosine decays 10x over cycle_mult x the run length. Returns (final loss, curve)."""
    steps = max(1, int(tokens / (a.batch * a.n_ctx)))
    cycle = max(1, int(steps * cycle_mult))
    opt = torch.optim.AdamW(m.parameters(), a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, steps // 50)
    lr_at = lambda s: (s / warm if s < warm else 1.0) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, cycle) / cycle)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    curve = []
    for s in range(1, steps + 1):
        st = torch.randint(0, len(tr) - a.n_ctx - 1, (a.batch,))
        loss = m.loss(torch.stack([tr[i:i + a.n_ctx] for i in st]).to(DEV))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if record and s % max(1, steps // record) == 0:
            curve.append((6 * N * s * a.batch * a.n_ctx, val_loss(m, va, a, 10)))
    return val_loss(m, va, a), curve


def e1(a):
    tr, va, V = setup(a)
    iso, rows = {}, []
    for C in a.budgets:
        iso[C] = []
        for L, d in SHAPES[:a.n_shapes]:
            m, N = make(L, d, V, a)
            tokens = C / (6 * N)
            if tokens < a.min_tokens_per_param * N * 0.05 or tokens > len(tr) * a.max_epochs:
                continue
            loss, _ = train(m, N, tr, va, a, tokens)
            iso[C].append((N, loss)); rows.append((N, tokens, loss))
            print(f"  E1 C={C:.1e} N={N:.2e} D={tokens:.2e} loss={loss:.3f}", flush=True)
    iso = {C: r for C, r in iso.items() if len(r) >= 3}
    pts, a2 = approach2_isoflop(iso)
    json.dump(rows, open(HERE / "runs_e1.json", "w"))
    return {"IsoFLOP minima": pts, "a (N_opt ~ C^a)": a2, "paper": 0.49}


def e2(a):
    tr, va, V = setup(a)
    curves, rows = {}, []
    for L, d in SHAPES[:a.n_shapes]:
        for mult in a.lengths:
            m, N = make(L, d, V, a)
            loss, curve = train(m, N, tr, va, a, a.base_tokens * mult, record=20)
            curves.setdefault(N, []).extend(curve)
            rows.append((N, a.base_tokens * mult, loss))
            print(f"  E2 N={N:.2e} tokens={a.base_tokens * mult:.2e} loss={loss:.3f}", flush=True)
    curves = {N: sorted(c) for N, c in curves.items()}
    pts, a1 = approach1_envelope(curves)
    json.dump(rows, open(HERE / "runs_e2.json", "w"))
    return {"envelope": pts, "a (N_opt ~ C^a)": a1, "paper": 0.50}


def e3(a):
    rows = []
    for f in ("runs_e1.json", "runs_e2.json"):
        if (HERE / f).exists():
            rows += json.load(open(HERE / f))
    if len(rows) < 6:
        return {"note": "run e1 and/or e2 first"}
    N, D, L = (np.array(c, float) for c in zip(*rows))
    E_, A_, B_, al, be = fit_parametric(N, D, L)
    a_ = be / (al + be)
    G = (al * A_ / (be * B_)) ** (1 / (al + be))
    C = 6 * np.median(N) * np.median(D)
    N_opt, D_opt = G * (C / 6) ** a_, (C / 6) ** (1 - a_) / G
    return {"E": E_, "A": A_, "B": B_, "alpha": al, "beta": be, "a": a_,
            "tokens per parameter on the frontier at a typical budget": D_opt / N_opt,
            "paper": {"E": 1.69, "alpha": 0.34, "beta": 0.28, "a": 0.46}}


def e4(a):
    tr, va, V = setup(a)
    L, d = SHAPES[min(4, a.n_shapes - 1)]
    out = {}
    for mult in (1.0, 1.1, 1.25, 1.5, 2.0, 5.0):
        m, N = make(L, d, V, a)
        out[f"cycle = {mult}x steps"] = train(m, N, tr, va, a, a.base_tokens * 4, cycle_mult=mult)[0]
        print("  E4", mult, out[f"cycle = {mult}x steps"], flush=True)
    return out


def e5(a):
    tr, va, V = setup(a)
    C = a.budgets[-1]
    sizes = [(L, d, make(L, d, V, a)[1]) for L, d in SHAPES[:a.n_shapes]]
    pick = lambda target: min(sizes, key=lambda s: abs(math.log(s[2] / target)))
    chin = pick(math.sqrt(C / 120))                                                # 6 N (20 N) = C
    kap_target = 1.3e9 * (C / 8.64e19) ** 0.73                                    # Kaplan's Table 5 (tiny budgets)
    kap = pick(kap_target)
    out = {}
    for name, (L, d, N) in (("Chinchilla-style (~20 tokens/param)", chin), ("Kaplan-style (N ~ C^0.73)", kap)):
        m, N = make(L, d, V, a)
        tokens = C / (6 * N)
        out[name] = {"N": N, "tokens": tokens, "tokens/param": tokens / N, "loss": train(m, N, tr, va, a, tokens)[0]}
        print("  E5", name, out[name], flush=True)
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
    a.n_docs, a.n_books, a.bpe_chars, a.merges, a.max_chars = 500000, 20, 3_000_000, 8000, 2_000_000_000
    a.n_ctx, a.batch, a.lr, a.n_shapes, a.max_epochs, a.min_tokens_per_param = 512, 32, 6e-4, 10, 1.0, 20
    a.budgets, a.base_tokens, a.lengths = (1e15, 3e15, 1e16, 3e16, 1e17), 2e8, (0.25, 0.5, 1.0, 2.0)
    if a.quick:
        a.n_docs, a.n_books, a.bpe_chars, a.merges, a.max_chars = 200, 2, 50_000, 200, 300_000
        a.n_ctx, a.batch, a.n_shapes = 32, 4, 4
        a.budgets, a.base_tokens, a.lengths = (3e9, 1e10), 2e4, (0.5, 1.0)
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
