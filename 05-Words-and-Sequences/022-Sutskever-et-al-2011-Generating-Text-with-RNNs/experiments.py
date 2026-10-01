"""Reproduce Sutskever, Martens & Hinton (2011) at small scale.

The paper trained a 1500-unit, 1500-factor MRNN (4.9M parameters) with Hessian-free optimization
for 5 days on 8 GPUs, on ~100 MB of Wikipedia / NYT / NIPS+JMLR text. We use text8 (the first
100 MB of Wikipedia as lowercase letters and spaces, http://mattmahoney.net/dc/text8.zip) and
Anna Karenina from Project Gutenberg for debagging.

  E1  Section 3.2: RNN (500 units) vs MRNN (350 units, 350 factors), about the same number of
      parameters. Paper (ML dataset): RNN 1.65, MRNN 1.56 bits per character.
  E2  Hessian-free vs Adam on a small MRNN (sparse init): bits per character vs time.
  E3  Section 6: samples from the trained MRNN ('the meaning of life is'), and the list completion
      'england spain france germany'.
  E4  Section 5.4: debagging. 11 consecutive words of Anna Karenina: 2 + 2 words of context, the
      middle 7 shuffled; the model scores all 5040 orders. Paper (Wikipedia-trained MRNN): 34% correct,
      sequence memoizer 27%.

Training follows Section 5.2: sequences of 250 characters, the loss on the last 200 only.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick        # 5M characters, small models, ~30-60 minutes
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import itertools
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

from mrnn import MRNN, CharRNN, HessianFree, bits_per_char, count_params, nll, one_hot, sample

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
TEXT8 = "http://mattmahoney.net/dc/text8.zip"
ANNA = "https://www.gutenberg.org/cache/epub/1399/pg1399.txt"
ALPHABET = " abcdefghijklmnopqrstuvwxyz"


def load_text8(n_chars):
    z = DATA / "text8.zip"
    if not z.exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(TEXT8, z)
    with zipfile.ZipFile(z) as f:
        text = f.read("text8").decode()[:n_chars]
    return torch.tensor([ALPHABET.index(c) for c in text])


def batches(ids, n, T=250, generator=None):
    """n random windows of T characters, shape (T, n)."""
    starts = torch.randint(0, len(ids) - T - 1, (n,), generator=generator)
    return torch.stack([ids[s:s + T] for s in starts], 1)


def train_adam(model, train_ids, steps, a, lr=2e-3):
    g = torch.Generator().manual_seed(0)
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr)
    for step in range(steps):
        b = batches(train_ids, a.batch, generator=g).to(DEV)
        total, n = nll(model, b, skip=50)                       # predict only the last 200 of 250
        opt.zero_grad(); (total / n).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"    step {step}: {(total / n / 0.6931).item():.3f} bpc (train batch)", flush=True)
    return model


def evaluate(model, test_ids, a):
    g = torch.Generator().manual_seed(1)
    b = batches(test_ids, 200, generator=g).to(DEV)
    return bits_per_char(model, b, skip=50)


def e1(data, a):
    train_ids, test_ids = data
    out = {}
    for name, model in (("RNN 500", CharRNN(27, 500)), ("MRNN 350x350", MRNN(27, 350, 350))):
        model = train_adam(model, train_ids, a.steps, a)
        out[name] = {"params": count_params(model), "test bpc": evaluate(model, test_ids, a)}
        if name.startswith("MRNN"):
            torch.save(model.state_dict(), HERE / "mrnn.pt")
        print(f"  E1 {name}: {out[name]}", flush=True)
    return out


def e2(data, a):
    train_ids, test_ids = data
    out = {}
    g = torch.Generator().manual_seed(2)
    m = MRNN(27, 100, 100).sparse_init(k=15, scale=0.3)
    hf = HessianFree(m, lam=10.0, cg_iters=a.cg_iters)
    t0, log = time.time(), []
    for it in range(a.hf_steps):
        grad_b, curv_b = batches(train_ids, a.hf_grad_seqs, 100, g), batches(train_ids, a.hf_grad_seqs // 10, 100, g)
        hf.step(grad_b, curv_b)
        log.append((time.time() - t0, evaluate(m, test_ids, a)))
        print(f"  HF step {it}: {log[-1][1]:.3f} bpc, lambda {hf.lam:.3g}", flush=True)
    out["Hessian-free"] = log
    m2 = MRNN(27, 100, 100).sparse_init(k=15, scale=0.3)
    t0, log2 = time.time(), []
    opt = torch.optim.Adam(m2.parameters(), 2e-3)
    while time.time() - t0 < log[-1][0]:                          # same wall-clock budget
        b = batches(train_ids, 32, 100, g)
        total, n = nll(m2, b)
        opt.zero_grad(); (total / n).backward(); opt.step()
        if len(log2) == 0 or time.time() - t0 - log2[-1][0] > log[-1][0] / 20:
            log2.append((time.time() - t0, evaluate(m2, test_ids, a)))
    out["Adam (same time)"] = log2
    return out


def e3(data, a):
    if not (HERE / "mrnn.pt").exists():
        return {"note": "run e1 first"}
    m = MRNN(27, 350, 350)
    m.load_state_dict(torch.load(HERE / "mrnn.pt", map_location="cpu"))
    enc = lambda s: [ALPHABET.index(c) for c in s]
    dec = lambda ids: "".join(ALPHABET[i] for i in ids)
    g = torch.Generator().manual_seed(0)
    return {"the meaning of life is": [dec(sample(m, enc("the meaning of life is"), 300, generator=g)) for _ in range(2)],
            "england spain france germany": [dec(sample(m, enc("england spain france germany"), 60, generator=g)) for _ in range(4)]}


def e4(data, a):
    if not (HERE / "mrnn.pt").exists():
        return {"note": "run e1 first"}
    m = MRNN(27, 350, 350).to(DEV)
    m.load_state_dict(torch.load(HERE / "mrnn.pt", map_location=DEV))
    path = DATA / "anna_karenina.txt"
    if not path.exists():
        urllib.request.urlretrieve(ANNA, path)
    words = re.sub(r"[^a-z ]", " ", path.read_text(encoding="utf-8").lower()).split()
    g = torch.Generator().manual_seed(0)
    correct = 0
    for k in range(a.bags):
        s = int(torch.randint(1000, len(words) - 20, (1,), generator=g))
        ctx_before, bag, ctx_after = words[s:s + 2], words[s + 2:s + 9], words[s + 9:s + 11]
        texts = [" ".join(ctx_before + list(p) + ctx_after) for p in itertools.permutations(bag)]
        L = max(len(t) for t in texts)
        ids = torch.tensor([[ALPHABET.index(c) for c in t.ljust(L)] for t in texts]).T.to(DEV)   # (L, 5040)
        with torch.no_grad():
            logits, _ = m(one_hot(ids[:-1], 27))
            lp = F.log_softmax(logits, -1).gather(-1, ids[1:, :, None])[..., 0]
            lens = torch.tensor([len(t) for t in texts], device=DEV)
            mask = torch.arange(L - 1, device=DEV)[:, None] < (lens - 1)[None, :]
            scores = (lp * mask).sum(0)
        correct += int(scores.argmax().item() == 0)                # permutation 0 is the original order
        if k % 50 == 0:
            print(f"  E4 bag {k}: {correct}/{k + 1} correct", flush=True)
    return {"correct": correct / a.bags, "bags": a.bags}


EXPS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4}


def report(R, a):
    L = ["# Results", "", f"text8, {a.chars} characters" + (" (QUICK run)" if a.quick else ""), ""]
    if R.get("e1"):
        L += ["## E1: RNN vs MRNN at equal size (paper, ML data: 1.65 vs 1.56 bpc)", ""]
        L += [f"- {k}: {v['params']:,} parameters, {v['test bpc']:.3f} bits/char" for k, v in R["e1"].items()] + [""]
    if R.get("e2"):
        L += ["## E2: Hessian-free vs Adam (small MRNN)", ""]
        for k, v in R["e2"].items():
            L.append(f"- {k}: final {v[-1][1]:.3f} bpc after {v[-1][0]:.0f}s")
        L += [""]
    if R.get("e3") and "note" not in R["e3"]:
        L += ["## E3: samples", ""]
        for k, v in R["e3"].items():
            L += [f"**{k}**", ""] + [f"> {k}{s}" for s in v] + [""]
    if R.get("e4") and "note" not in R["e4"]:
        L += [f"## E4: debagging (paper: MRNN 34%, memoizer 27%)", "", f"{100 * R['e4']['correct']:.1f}% of {R['e4']['bags']} bags"]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--chars", type=int, default=100_000_000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hf-steps", type=int, default=50)
    ap.add_argument("--hf-grad-seqs", type=int, default=2000)
    ap.add_argument("--cg-iters", type=int, default=50)
    ap.add_argument("--bags", type=int, default=500)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.chars, a.steps, a.hf_steps, a.hf_grad_seqs, a.cg_iters, a.bags = 5_000_000, 2000, 10, 200, 20, 50
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        ids = load_text8(a.chars)
        split = len(ids) - min(10_000_000, len(ids) // 10)          # 'the last 10 million characters ... test set'
        data = (ids[:split], ids[split:])
        t0 = time.time()
        for name, fn in EXPS.items():
            if a.only in (None, name):
                print(name, flush=True)
                R[name] = fn(data, a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
