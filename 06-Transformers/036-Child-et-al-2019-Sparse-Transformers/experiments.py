"""Reproduce 'Generating Long Sequences with Sparse Transformers' (Child et al. 2019) at single-GPU scale.

  E1  Table 2 (Enwik8 part): dense vs fixed vs strided attention on enwik8 bytes, bits per byte. Paper (context
      12,288, 30 layers, d 512, 8 heads, stride 128, c 32, merged heads): dense 1.00, fixed 0.99, strided 1.13.
      Here: context 1,024, stride 32, c 8 (or smaller with --quick).
  E2  Table 2 (CIFAR-10 part): the same three patterns on CIFAR-10 images as sequences of 3,072 bytes (rows of
      32 pixels x 3 channels; stride 96 = one image row). Paper (128 layers, d 256): dense 2.82, fixed 2.85,
      strided 2.80 bits per dim. Data embeddings use the (row, column, channel) of every byte (Section 5.3).
  E3  Table 3: bits per byte on enwik8 test as the minimum context available at evaluation grows (evaluate fewer
      tokens per window). Paper: 0.9952 at 6,144 tokens -> 0.9908 at 12,160.
  E4  Table 4's trade-off: peak memory of one training step vs sequence length for dense vs strided attention,
      using the reshape/transpose kernels for the sparse case (Section 5.5) and recomputation (Section 5.4).

Training (Section 6): Adam, 5,000-step linear warm-up then cosine decay, gradient clipping 1.0, weight decay 0.01.
Note: the model in sparse.py applies patterns as MASKS on a dense score matrix (simple and exact but not faster);
E4 measures the efficient blocked kernels separately.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # tiny contexts, ~1 hour on CPU
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import math
import pickle
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

from sparse import SparseTransformerLM, block_local_attention, masked_attention, causal, strided_column_attention

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def enwik8():
    d = DATA / "enwik8"; d.mkdir(parents=True, exist_ok=True)
    f = d / "enwik8"
    if not f.exists():
        z = d / "enwik8.zip"
        urllib.request.urlretrieve("http://mattmahoney.net/dc/enwik8.zip", z)
        zipfile.ZipFile(z).extractall(d)
    raw = torch.tensor(list(f.read_bytes()), dtype=torch.long)
    return raw[:90_000_000], raw[90_000_000:95_000_000], raw[95_000_000:]


def cifar_bytes():
    d = DATA / "cifar10"; d.mkdir(parents=True, exist_ok=True)
    t = d / "cifar-10-python.tar.gz"
    if not t.exists():
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", t)
        tarfile.open(t).extractall(d)
    def load(names):
        xs = []
        for nm in names:
            with open(d / "cifar-10-batches-py" / nm, "rb") as f:
                xs.append(torch.tensor(pickle.load(f, encoding="bytes")[b"data"]))
        # (N, 3072) stored channel-major -> raster order: row, column, channel
        x = torch.cat(xs).view(-1, 3, 32, 32).permute(0, 2, 3, 1).reshape(-1, 3072)
        return x.long()
    train = load([f"data_batch_{i}" for i in range(1, 6)])
    return train[:48000], train[48000:], load(["test_batch"])


def schedule(step, warmup, total):
    if step < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train_lm(make_batch, V, n_ctx, pattern, a, evaluate):
    torch.manual_seed(0)
    m = SparseTransformerLM(V, n_ctx, d=a.d, layers=a.layers, heads=a.heads, pattern=pattern, stride=a.stride,
                            c=a.c, mode=a.mode, dropout=a.dropout, recompute=True).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: schedule(s, a.warmup, a.steps))
    t0 = time.time()
    for step in range(a.steps):
        x = make_batch().to(DEV)
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % 1000 == 0:
            print(f"    {pattern} step {step}: {loss.item() / math.log(2):.3f} bits", flush=True)
    t_iter = (time.time() - t0) / a.steps
    m.eval()
    return m, {"bits": evaluate(m), "seconds/iter": t_iter}


def e1(a):
    tr, va, te = enwik8()
    if a.quick:
        tr, te = tr[:5_000_000], te[:300_000]
    n = a.ctx

    def batch():
        idx = torch.randint(0, len(tr) - n - 1, (a.batch,))
        return torch.stack([tr[i:i + n + 1] for i in idx])

    def evaluate(m):
        tot = cnt = 0.0
        with torch.no_grad():
            for s in range(0, len(te) - n - 1, n):
                x = te[s:s + n + 1][None].to(DEV)
                tot += F.cross_entropy(m(x[:, :-1])[0], x[0, 1:], reduction="sum").item(); cnt += n
        return tot / cnt / math.log(2)

    out = {}
    for pat in ("dense", "fixed", "strided"):
        m, r = train_lm(batch, 256, n + 1, pat, a, evaluate)
        out[pat] = r
        if pat == "fixed":
            torch.save(m.state_dict(), HERE / "enwik8_fixed.pt")
        print("  E1", pat, r, flush=True)
    return out


def e2(a):
    tr, va, te = cifar_bytes()
    if a.quick:
        tr, te = tr[:5000], te[:200]
    n = 3072 if not a.quick else 3 * 16 * 16                       # --quick: the top-left 16 x 16 crop
    if a.quick:
        crop = lambda x: x.view(-1, 32, 32, 3)[:, :16, :16].reshape(-1, n)
        tr, te = crop(tr), crop(te)
    stride = 96 if not a.quick else 48                              # one image row of bytes

    def batch():
        x = tr[torch.randint(0, len(tr), (a.batch,))]
        return torch.cat([torch.zeros(len(x), 1, dtype=torch.long), x], 1)   # a start byte

    def evaluate(m):
        tot = 0.0
        with torch.no_grad():
            for s in range(0, len(te), 16):
                x = torch.cat([torch.zeros(len(te[s:s + 16]), 1, dtype=torch.long), te[s:s + 16]], 1).to(DEV)
                tot += F.cross_entropy(m(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1), reduction="sum").item()
        return tot / (len(te) * n) / math.log(2)

    out = {}
    saved = a.stride
    a.stride = stride
    for pat in ("dense", "fixed", "strided"):
        _, r = train_lm(batch, 256, n + 1, pat, a, evaluate)
        out[pat] = r
        print("  E2", pat, r, flush=True)
    a.stride = saved
    return out


def e3(a):
    tr, va, te = enwik8()
    if not (HERE / "enwik8_fixed.pt").exists():
        return {"note": "run e1 first"}
    n = a.ctx
    m = SparseTransformerLM(256, n + 1, d=a.d, layers=a.layers, heads=a.heads, pattern="fixed", stride=a.stride,
                            c=a.c, mode=a.mode).to(DEV)
    m.load_state_dict(torch.load(HERE / "enwik8_fixed.pt", map_location=DEV)); m.eval()
    te = te[:a.e3_bytes]
    out = {}
    for keep in (n // 2, 3 * n // 4, 7 * n // 8, n - 16):
        # score only the LAST (n - keep) tokens of each window, so every scored token has >= keep tokens of context
        step = n - keep
        tot = cnt = 0.0
        with torch.no_grad():
            for s in range(0, len(te) - n - 1, step):
                x = te[s:s + n + 1][None].to(DEV)
                lp = F.cross_entropy(m(x[:, :-1])[0], x[0, 1:], reduction="none")[keep:]
                tot += lp.sum().item(); cnt += len(lp)
        out[f"min context {keep}"] = tot / cnt / math.log(2)
        print("  E3", keep, out[f"min context {keep}"], flush=True)
    return out


def e4(a):
    out = {}
    for n in a.e4_lengths:
        l = int(math.sqrt(n))
        n = l * l
        q, k, v = (torch.randn(1, 4, n, 64, device=DEV, requires_grad=True) for _ in range(3))
        res = {}
        for name in ("dense", "strided (blocked kernels)"):
            if DEV == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if name == "dense":
                y = masked_attention(q, k, v, causal(n).to(DEV))
            else:
                y = block_local_attention(q, k, v, l) + strided_column_attention(q, k, v, l)
            y.sum().backward()
            res[name] = {"seconds": time.time() - t0,
                         "peak MB": torch.cuda.max_memory_allocated() / 2 ** 20 if DEV == "cuda" else None}
        out[n] = res
        print("  E4", n, res, flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: enwik8 bits/byte (Table 2)"), ("e2", "E2: CIFAR-10 bits/dim (Table 2)"),
                 ("e3", "E3: context at evaluation (Table 3)"), ("e4", "E4: time / memory vs length")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["e1", "e2", "e3", "e4"])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.ctx, a.d, a.layers, a.heads, a.stride, a.c, a.mode = 1024, 256, 12, 8, 32, 8, "merged"
    a.dropout, a.lr, a.batch, a.steps, a.warmup, a.e3_bytes = 0.1, 3.5e-4, 16, 50_000, 5000, 500_000
    a.e4_lengths = (1024, 4096, 16384)
    if a.quick:
        a.ctx, a.d, a.layers, a.heads, a.stride, a.c = 256, 64, 2, 2, 16, 4
        a.batch, a.steps, a.warmup, a.e3_bytes, a.e4_lengths = 8, 800, 100, 50_000, (256, 1024)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R, default=str))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
