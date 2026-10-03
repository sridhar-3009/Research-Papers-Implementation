"""LLaMA's claims at small scale (Touvron et al. 2023).

Corpus: OpenWebText via Hugging Face `datasets` if installed, else Gutenberg books (downloaded), else this repository's
own markdown; byte-level BPE from paper 049.

  E1  Section 2.2 ablation: GPT-2-style baseline (LayerNorm, GELU 4d, learned positions), then swap in RMSNorm,
      SwiGLU (2/3 4d), RoPE one at a time, and the full LLaMA block; equal parameters (+-2%) and steps; 3 seeds.
  E2  Introduction / Figure 1: train small models FAR past ~20 tokens per parameter (up to 500) and record the
      validation loss along the way: does it keep improving? Fit L(D) = E + B / D^beta per size.
  E3  Tokenizer: digits split into single tokens vs merged by BPE, on a synthetic arithmetic corpus; exact-match
      accuracy on held-out additions of 3-5 digit numbers.
  E4  Section 2.4: wall-clock time and peak memory of standard vs memory-efficient (blockwise, future-skipping)
      causal attention as the context grows (CUDA memory stats if available).
  E5  The inference-budget trade-off with E2's runs: for target losses, the (N, D) minimising training FLOPs vs the
      one minimising training + inference FLOPs for 1e9 ... 1e13 served tokens.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import importlib.util
import json
import math
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import llama as L

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
spec = importlib.util.spec_from_file_location("gpt2_049", HERE.parent / "049-Radford-et-al-2019-GPT-2" / "gpt2.py")
gpt2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpt2)
BOOKS = [1342, 84, 11, 2701, 1661, 98, 1952, 345, 2600, 4300]


# ---------------------------------------------------------------------------------------------------- data
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
        pass
    try:
        d = DATA / "gutenberg"; d.mkdir(parents=True, exist_ok=True)
        parts = []
        for b in BOOKS[:a.n_books]:
            f = d / f"{b}.txt"
            if not f.exists():
                urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt", f)
            parts.append(f.read_text(encoding="utf-8", errors="ignore"))
        return "\n\n".join(parts)
    except Exception:
        return "\n".join(p.read_text(errors="ignore") for p in sorted(HERE.parents[1].rglob("*.md")))


_cache = {}


def setup(a):
    if "d" not in _cache:
        text = corpus(a)
        bpe = gpt2.ByteBPE(text[:a.bpe_chars], a.merges)
        ids = torch.tensor(bpe.encode(text[:a.max_chars]))
        cut = int(0.98 * len(ids))
        _cache["d"] = (ids[:cut], ids[cut:], len(bpe))
    return _cache["d"]


# ---------------------------------------------------------------------------------------------------- variants
class Variant(nn.Module):
    """A decoder whose norm / MLP / positions can each be GPT-2-style or LLaMA-style."""

    def __init__(self, V, d, layers, heads, ctx, rms=False, swiglu=False, rope=False):
        super().__init__()
        self.rope, self.h = rope, heads
        self.tok = nn.Embedding(V, d)
        self.pos = None if rope else nn.Parameter(torch.randn(ctx, d) * 0.01)
        norm = (lambda: L.RMSNorm(d)) if rms else (lambda: nn.LayerNorm(d))
        mlp = (lambda: L.SwiGLU(d, int(2 * 4 * d / 3))) if swiglu else \
            (lambda: nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)))
        self.n1 = nn.ModuleList(norm() for _ in range(layers)); self.n2 = nn.ModuleList(norm() for _ in range(layers))
        self.qkv = nn.ModuleList(nn.Linear(d, 3 * d) for _ in range(layers))
        self.o = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.mlp = nn.ModuleList(mlp() for _ in range(layers))
        self.nf, self.out = norm(), nn.Linear(d, V, bias=False)
        cos, sin = L.rope_frequencies(d // heads, ctx)
        self.register_buffer("cos", cos, persistent=False); self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok(ids) + (0 if self.pos is None else self.pos[:T])
        for n1, qkv, o, n2, mlp in zip(self.n1, self.qkv, self.o, self.n2, self.mlp):
            q, k, v = qkv(n1(x)).view(B, T, 3, self.h, -1).permute(2, 0, 3, 1, 4)
            if self.rope:
                q, k = L.apply_rope(q, self.cos, self.sin), L.apply_rope(k, self.cos, self.sin)
            x = x + o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, -1))
            x = x + mlp(n2(x))
        return self.out(self.nf(x))

    def loss(self, ids):
        lg = self(ids)[:, :-1]
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1))


def train(m, tr, va, a, steps, record=0):
    opt = torch.optim.AdamW(m.parameters(), a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: L.lr_schedule(s, 1.0, warmup=max(1, steps // 50), total=steps))
    curve = []
    for s in range(1, steps + 1):
        st = torch.randint(0, len(tr) - a.ctx - 1, (a.batch,))
        loss = m.loss(torch.stack([tr[i:i + a.ctx] for i in st]).to(DEV))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if record and s % max(1, steps // record) == 0:
            curve.append((s * a.batch * a.ctx, val(m, va, a, 20)))
    return val(m, va, a), curve


@torch.no_grad()
def val(m, va, a, n=60):
    m.eval()
    g = torch.Generator().manual_seed(0)
    r = sum(m.loss(va[s:s + a.ctx][None].to(DEV)).item() for s in torch.randint(0, len(va) - a.ctx - 1, (n,), generator=g)) / n
    m.train()
    return r


# ---------------------------------------------------------------------------------------------------- experiments
def e1(a):
    tr, va, V = setup(a)
    configs = {"GPT-2 baseline": {}, "+ RMSNorm": {"rms": True}, "+ SwiGLU": {"swiglu": True}, "+ RoPE": {"rope": True},
               "LLaMA (all three)": {"rms": True, "swiglu": True, "rope": True}}
    out = {}
    for name, kw in configs.items():
        losses = []
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            m = Variant(V, a.d, a.layers, a.heads, a.ctx, **kw).to(DEV)
            losses.append(train(m, tr, va, a, a.steps)[0])
        out[name] = {"params": sum(p.numel() for p in m.parameters()), "val loss mean": float(np.mean(losses)),
                     "std": float(np.std(losses)), "seeds": losses}
        print("  E1", name, out[name], flush=True)
    return out


def e2(a):
    tr, va, V = setup(a)
    out = {}
    for d, layers in a.sizes:
        torch.manual_seed(0)
        m = L.LLaMA(vocab=V, d=d, layers=layers, heads=max(1, d // 64), ctx=a.ctx).to(DEV)
        N = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(("tok", "out")))
        steps = int(a.max_tokens_per_param * N / (a.batch * a.ctx))
        _, curve = train(m, tr, va, a, steps, record=30)
        D = np.array([c[0] for c in curve], float); Ls = np.array([c[1] for c in curve])
        best = None
        for E in np.linspace(0.5 * Ls.min(), 0.99 * Ls.min(), 40):                 # grid over E, linear fit for B, beta
            y = np.log(np.maximum(Ls - E, 1e-6))
            beta, logB = np.polyfit(np.log(D), y, 1)
            err = np.mean((np.exp(logB) * D ** beta + E - Ls) ** 2)
            best = min(best or (err, E, logB, beta), (err, E, logB, beta))
        out[f"N={N:.2e}"] = {"curve (tokens, loss)": curve, "tokens per param at end": D[-1] / N,
                             "loss at 20 tok/param": float(np.interp(20 * N, D, Ls)), "loss at end": float(Ls[-1]),
                             "fit E, B, beta": [best[1], math.exp(best[2]), -best[3]]}
        print("  E2", N, out[f"N={N:.2e}"]["loss at 20 tok/param"], out[f"N={N:.2e}"]["loss at end"], flush=True)
    json.dump(out, open(HERE / "runs_e2.json", "w"), default=float)
    return out


def arithmetic_corpus(n, rng):
    lines = []
    for _ in range(n):
        x, y = rng.randint(0, 99999), rng.randint(0, 99999)
        lines.append(f"{x} + {y} = {x + y}")
    return "\n".join(lines) + "\n"


def e3(a):
    rng = random.Random(0)
    text = arithmetic_corpus(a.n_arith, rng)
    tests = [(rng.randint(100, 99999), rng.randint(100, 99999)) for _ in range(a.n_arith_test)]
    out = {}
    for split in (False, True):
        bpe = gpt2.ByteBPE(text[:200_000], a.arith_merges)
        enc = (lambda s: sum((bpe.encode(p) for p in L.split_digits(s)), [])) if split else bpe.encode
        ids = torch.tensor(enc(text))
        torch.manual_seed(0)
        m = L.LLaMA(vocab=len(bpe), d=a.d, layers=a.layers, heads=a.heads, ctx=a.ctx).to(DEV)
        train(m, ids, ids[-5000:], a, a.steps)
        m.eval(); correct = 0
        with torch.no_grad():
            for x, y in tests:
                p = enc(f"{x} + {y} = ")
                for _ in range(12):
                    nxt = int(m(torch.tensor([p[-a.ctx:]], device=DEV))[0, -1].argmax())
                    p.append(nxt)
                    if bpe.decode([nxt]).endswith("\n"):
                        break
                ans = bpe.decode(p).split("= ")[-1].strip()
                correct += ans == str(x + y)
        out["digits split" if split else "BPE merges digits"] = {"exact match": correct / len(tests),
                                                                 "tokens in corpus": len(ids)}
        print("  E3", split, out, flush=True)
    return out


def e4(a):
    out = {}
    for T in a.lengths:
        q, k, v = (torch.randn(1, 8, T, 64, device=DEV) for _ in range(3))
        row = {}
        for name, fn in (("naive (materialised T x T)", lambda: torch.softmax(
                (q @ k.transpose(-1, -2) / 8).masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=DEV), 1), -1e9), -1) @ v),
                         ("blockwise + future skipping", lambda: L.memory_efficient_causal_attention(q, k, v, block=128))):
            if DEV == "cuda":
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t = time.time(); fn()
            if DEV == "cuda":
                torch.cuda.synchronize()
            row[name] = {"seconds": time.time() - t,
                         "peak MB": torch.cuda.max_memory_allocated() / 1e6 if DEV == "cuda" else None}
        out[T] = row
        print("  E4", T, row, flush=True)
    return out


def e5(a):
    f = HERE / "runs_e2.json"
    if not f.exists():
        return {"note": "run e2 first"}
    runs = json.load(open(f))
    pts = [(float(k.split("=")[1]), D, Lv) for k, r in runs.items() for D, Lv in r["curve (tokens, loss)"]]
    out = {}
    for target in np.quantile([p[2] for p in pts], [0.2, 0.4, 0.6]):
        ok = [(N, D) for N, D, Lv in pts if Lv <= target]
        if not ok:
            continue
        train_opt = min(ok, key=lambda nd: 6 * nd[0] * nd[1])
        row = {"cheapest to train (N, D)": train_opt}
        for served in (1e9, 1e11, 1e13):
            row[f"cheapest overall with {served:.0e} served tokens"] = min(ok, key=lambda nd: 6 * nd[0] * nd[1] + 2 * nd[0] * served)
        out[f"loss <= {target:.3f}"] = row
    return out


def report(R, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.n_docs, a.n_books, a.bpe_chars, a.merges, a.max_chars = 300000, 10, 3_000_000, 8000, 1_500_000_000
    a.d, a.layers, a.heads, a.ctx, a.batch, a.lr, a.steps, a.seeds = 512, 8, 8, 512, 32, 6e-4, 10000, 3
    a.sizes, a.max_tokens_per_param = [(128, 2), (256, 4), (384, 6), (512, 8)], 500
    a.n_arith, a.n_arith_test, a.arith_merges, a.lengths = 200000, 500, 400, (512, 1024, 2048, 4096, 8192)
    if a.quick:
        a.n_docs, a.n_books, a.bpe_chars, a.merges, a.max_chars = 100, 1, 20_000, 100, 200_000
        a.d, a.layers, a.heads, a.ctx, a.batch, a.steps, a.seeds = 32, 1, 2, 32, 4, 5, 1
        a.sizes, a.max_tokens_per_param = [(32, 1)], 2
        a.n_arith, a.n_arith_test, a.arith_merges, a.lengths = 300, 3, 30, (64, 128)
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
