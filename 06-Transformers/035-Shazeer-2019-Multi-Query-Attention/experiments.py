"""Reproduce 'Fast Transformer Decoding: One Write-Head is All You Need' (Shazeer 2019) at laptop/GPU scale.

The paper's language-model experiment (Table 3): a 6-layer transformer decoder, d_model 1024, d_ff 8192, h 8,
d_k = d_v = 128, on the Billion-Word benchmark; every variant has the SAME parameter count (the feed-forward width
absorbs the difference). Its speed experiment (Table 2): training cost per token and incremental-decoding cost per
token, with and without a local window of 32 positions.

  E1  Table 3 (and Table 1's rows) on Penn Treebank words (the paper's data is too big): multi-head, multi-query,
      and the 'simpler alternatives' with fewer heads / smaller keys, all parameter-matched. Dev perplexity.
      Paper (Billion-Word): MHA 29.9, MQA 30.2, h=1 31.2, h=2 31.1, h=4 31.0, d_k=16 30.9.
  E2  Table 2: training time per token and incremental decoding time per token (batch 128, prompt 1, 128 new
      tokens) for multi-head / multi-query, full and local (window 32) decoder self-attention, plus grouped-query.
      Paper (TPUv2): decoder 46 -> 3.8 us/token (multi-head -> multi-query); local 23 -> 3.3.
  E3  The scaling picture: decoding time per token vs batch size and sequence length, multi-head vs multi-query.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # small models, ~30-60 minutes on CPU
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from mqa import DecoderLM, matching_ffn_width

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{}.txt"


def ptb():
    (DATA / "ptb").mkdir(parents=True, exist_ok=True)
    words = {}
    for split in ("train", "valid"):
        p = DATA / "ptb" / f"ptb.{split}.txt"
        if not p.exists():
            urllib.request.urlretrieve(URL.format(split), p)
        words[split] = [w for line in p.read_text().splitlines() for w in line.split() + ["<eos>"]]
    vocab = {w: i for i, w in enumerate(sorted(set(words["train"])))}
    return {k: torch.tensor([vocab[w] for w in v]) for k, v in words.items()}, len(vocab)


def variants(a):
    """name -> (h, kv_heads, key size, d_ff), all with (about) the same parameter count as the baseline."""
    d, h, k, f, L = a.d, a.h, a.d // a.h, a.d_ff, a.layers
    base = (h, h, k, f)

    def widen(h2, g2, k2):
        from mqa import attention_params
        saved = L * (attention_params(d, h, k, k, h) - attention_params(d, h2, k2, k2, g2))
        return f + int(round(saved / (L * 2 * d)))
    return {"multi-head": base,
            "multi-query": (h, 1, k, widen(h, 1, k)),
            "grouped (2 kv heads)": (h, 2, k, widen(h, 2, k)),
            # the paper's 'simpler alternatives' all shrink the total key/value width to ONE head's worth:
            # (h, d_k) = (1, 128), (2, 64), (4, 32), (8, 16) at d_model 1024; here (1, k), (2, k/2), (4, k/4), (8, k/8)
            "multi-head h=1": (1, 1, k, widen(1, 1, k)),
            "multi-head h=2": (2, 2, k // 2, widen(2, 2, k // 2)),
            "multi-head h=4": (4, 4, k // 4, widen(4, 4, k // 4)),
            f"multi-head h={h} d_k={k // h}": (h, h, k // h, widen(h, h, k // h))}


def build(V, h, g, k, d_ff, a, window=None):
    m = DecoderLM(V, d=a.d, h=h, kv_heads=g, d_ff=d_ff, layers=a.layers, max_len=a.ctx + 1, window=window)
    if k != a.d // h:                                           # smaller (or larger) keys: resize the projections
        import torch.nn as nn
        for blk in m.blocks:
            att = blk.attn
            att.k = k
            att.W_q, att.W_k = nn.Linear(a.d, h * k, bias=False), nn.Linear(a.d, g * k, bias=False)
            att.W_v, att.W_o = nn.Linear(a.d, g * k, bias=False), nn.Linear(h * k, a.d, bias=False)
    return m.to(DEV)


def e1(a):
    data, V = ptb()
    out = {}
    for name, (h, g, k, f) in variants(a).items():
        torch.manual_seed(0)
        m = build(V, h, g, k, f, a)
        opt = torch.optim.Adam(m.parameters(), lr=a.lr)
        tr = data["train"]
        for step in range(a.steps):
            idx = torch.randint(0, len(tr) - a.ctx - 1, (a.batch,))
            x = torch.stack([tr[i:i + a.ctx + 1] for i in idx]).to(DEV)
            loss = F.cross_entropy(m(x[:, :-1])[0].reshape(-1, V), x[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); tot = cnt = 0.0
        va = data["valid"]
        with torch.no_grad():
            for s in range(0, len(va) - a.ctx - 1, a.ctx):
                x = va[s:s + a.ctx + 1][None].to(DEV)
                tot += F.cross_entropy(m(x[:, :-1])[0][0], x[0, 1:], reduction="sum").item(); cnt += a.ctx
        out[name] = {"dev perplexity": math.exp(tot / cnt), "params": sum(p.numel() for p in m.parameters()),
                     "h": h, "kv heads": g, "d_k": k, "d_ff": f}
        print("  E1", name, {k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in out[name].items()}, flush=True)
    return out


@torch.no_grad()
def decode_time(m, b, steps):
    p = torch.zeros(b, 1, dtype=torch.long, device=DEV)
    m.generate(p, 4)                                            # warm-up
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    m.generate(p, steps)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / (steps * b)


def e2(a):
    out = {}
    V = 10000
    for name, g, window in (("multi-head", a.h, None), ("multi-query", 1, None), ("grouped (2 kv heads)", 2, None),
                            ("multi-head local", a.h, 32), ("multi-query local", 1, 32)):
        torch.manual_seed(0)
        m = build(V, a.h, g, a.d // a.h, a.d_ff, a, window).eval()
        x = torch.randint(0, V, (a.batch, a.ctx + 1), device=DEV)
        m.train()
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        t0 = time.time()
        for _ in range(5):
            loss = F.cross_entropy(m(x[:, :-1])[0].reshape(-1, V), x[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        train_us = 1e6 * (time.time() - t0) / (5 * a.batch * a.ctx)
        m.eval()
        out[name] = {"training us/token": train_us, "decoding us/token": 1e6 * decode_time(m, 128, 128)}
        print("  E2", name, {k: round(v, 2) for k, v in out[name].items()}, flush=True)
    return out


def e3(a):
    out = {}
    V = 10000
    for g, name in ((a.h, "multi-head"), (1, "multi-query")):
        torch.manual_seed(0)
        m = build(V, a.h, g, a.d // a.h, a.d_ff, a).eval()
        for b in (1, 16, 128):
            for steps in (32, 128, 256):
                if steps > a.ctx:
                    continue
                out[f"{name} b={b} n={steps}"] = 1e6 * decode_time(m, b, steps)
        print("  E3", name, {k: round(v, 1) for k, v in out.items() if k.startswith(name)}, flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: parameter-matched variants, PTB dev perplexity"), ("e2", "E2: Table 2 speeds"),
                 ("e3", "E3: decoding time per token (us) vs batch and length")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["e1", "e2", "e3"])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.d, a.h, a.d_ff, a.layers, a.ctx, a.batch, a.steps, a.lr = 512, 8, 2048, 6, 256, 32, 20000, 3e-4
    if a.quick:
        a.d, a.h, a.d_ff, a.layers, a.ctx, a.batch, a.steps, a.lr = 128, 8, 512, 2, 128, 16, 1500, 1e-3
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
