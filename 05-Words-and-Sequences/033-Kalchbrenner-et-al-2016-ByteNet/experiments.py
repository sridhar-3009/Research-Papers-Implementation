"""Reproduce ByteNet (Kalchbrenner et al. 2016) at a scale a single machine can manage.

  E1  Table 3, character-level language modelling on enwik8 (Hutter Prize Wikipedia): 90M / 5M / 5M bytes, the
      ByteNet decoder with 30 residual multiplicative blocks (6 sets of dilations 1-16), d = 512, kernel 3, Adam
      lr 3e-4, weight decay 1e-4, dropout 0.1 before the softmax, sequences of 500 bytes predicting the last 400.
      Paper: 1.31 bits/byte (test).
  E2  Tables 2 and 4, character-to-character translation English -> German. The paper uses WMT (NewsTest 2014/15,
      BLEU 23.75 / 26.26, 0.521 / 0.532 bits/char); we use Multi30k (29k sentence pairs). ReLU residual blocks,
      a = 1.2, b = 0 (Eq. 2), sentences padded to the next multiple of 50 plus 20% for the source, beam 12 over
      total log-likelihood without length normalization, only candidates that end in EOS.
  E3  Figure 5: correlation of source and target lengths in characters (paper: rho = 0.968 on NewsTest 2013), and
      the least-squares slope that motivates a = 1.2.
  E4  Figure 6: gradient saliency of a few translations (outputs vs source and target characters).
  E5  'Linear time': wall-clock of one training forward+backward pass vs length, ByteNet vs the attention
      RNNsearch of Paper 028 (imported).

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # small models, ~1-2 hours on CPU
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import gzip
import importlib.util
import json
import math
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

from bytenet import BOS, EOS, PAD, ByteNet, ByteNetLM, saliency

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------- E1 enwik8
def enwik8():
    d = DATA / "enwik8"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "enwik8"
    if not f.exists():
        z = d / "enwik8.zip"
        urllib.request.urlretrieve("http://mattmahoney.net/dc/enwik8.zip", z)
        zipfile.ZipFile(z).extractall(d)
    raw = torch.tensor(list(f.read_bytes()), dtype=torch.long)
    return raw[:90_000_000], raw[90_000_000:95_000_000], raw[95_000_000:]


def e1(a):
    train, valid, test = enwik8()
    if a.quick:
        train, valid, test = train[:5_000_000], valid[:500_000], test[:500_000]
    torch.manual_seed(0)
    m = ByteNetLM(256, d=a.lm_d, n_blocks=a.lm_blocks, k=3, block="mu", dropout=0.1).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-4)
    L, ctx = 500, 100                                          # 500 bytes, first 100 are context only

    def bits(split, n_seq):
        m.eval(); tot = cnt = 0.0
        with torch.no_grad():
            for s in range(0, min(len(split) - L - 1, n_seq * 400), 400):
                x = split[s:s + L + 1][None].to(DEV)
                lp = F.cross_entropy(m(x[:, :-1])[:, ctx:].reshape(-1, 256), x[:, ctx + 1:].reshape(-1), reduction="sum")
                tot += lp.item(); cnt += L - ctx
        m.train()
        return tot / cnt / math.log(2)

    for step in range(a.lm_steps):
        idx = torch.randint(0, len(train) - L - 1, (a.lm_batch,))
        x = torch.stack([train[i:i + L + 1] for i in idx]).to(DEV)
        loss = F.cross_entropy(m(x[:, :-1])[:, ctx:].reshape(-1, 256), x[:, ctx + 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()            # the paper does not decay the learning rate
        if step % 2000 == 0:
            print(f"    step {step}: train {loss.item() / math.log(2):.3f} bits/byte", flush=True)
    return {"valid bits/byte": bits(valid, 2000), "test bits/byte": bits(test, 2000),
            "receptive field": m.receptive_field, "parameters": sum(p.numel() for p in m.parameters())}


# ---------------------------------------------------------------------------------------------------- E2 translation
def multi30k():
    d = DATA / "multi30k"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, name in (("train", "train"), ("valid", "val"), ("test", "test_2016_flickr")):
        pair = []
        for lang in ("en", "de"):
            p = d / f"{name}.{lang}.gz"
            if not p.exists():
                urllib.request.urlretrieve(f"https://github.com/multi30k/dataset/raw/master/data/task1/raw/{name}.{lang}.gz", p)
            with gzip.open(p, "rt", encoding="utf-8") as f:
                pair.append([line.strip() for line in f])
        out[split] = list(zip(*pair))
    return out


class Chars:
    def __init__(self, texts):
        self.itos = ["<pad>", "<s>", "</s>"] + sorted({c for t in texts for c in t})
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, t):
        return [self.stoi.get(c, PAD) for c in t]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i > EOS)


def pad_to(seqs, multiple=50, extra=0.0, eos=False):
    seqs = [s + [EOS] if eos else s for s in seqs]
    L = max(map(len, seqs))
    L = int(math.ceil(L / multiple) * multiple * (1 + extra))   # 'padded to the nearest greater multiple of 50'
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])


@torch.no_grad()
def beam_search(m, src, beam=12, max_len=400):
    """Total log-likelihood, no length normalization, only candidates that end in EOS (Section 6)."""
    rep = m.encode(src)
    hyps, done = [(0.0, [BOS])], []
    for _ in range(max_len):
        x = torch.tensor([h[1] for h in hyps], device=src.device)
        lp = F.log_softmax(m.decode(rep.expand(len(hyps), -1, -1), x)[:, -1], -1)
        cand = []
        for (score, seq), row in zip(hyps, lp):
            top = row.topk(beam)
            cand += [(score + s, seq + [k]) for s, k in zip(top.values.tolist(), top.indices.tolist())]
        cand.sort(key=lambda h: -h[0])
        hyps = []
        for h in cand:
            (done if h[1][-1] == EOS else hyps).append(h)
            if len(hyps) == beam:
                break
        if not hyps or (done and max(d[0] for d in done) >= hyps[0][0]):
            break
    best = max(done, key=lambda h: h[0]) if done else hyps[0]
    return best[1][1:]


def bleu(hyps, refs, n=4):
    from collections import Counter
    match, total, hl, rl = [0] * n, [0] * n, 0, 0
    for h, r in zip(hyps, refs):
        hl, rl = hl + len(h), rl + len(r)
        for k in range(1, n + 1):
            hc = Counter(tuple(h[i:i + k]) for i in range(len(h) - k + 1))
            rc = Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1))
            match[k - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            total[k - 1] += max(len(h) - k + 1, 0)
    if min(match) == 0:
        return 0.0
    return 100 * min(1.0, math.exp(1 - rl / max(hl, 1))) * math.exp(sum(math.log(m / t) for m, t in zip(match, total)) / n)


def train_translator(D, src_v, tgt_v, a):
    torch.manual_seed(0)
    m = ByteNet(len(src_v.itos), len(tgt_v.itos), d=a.mt_d, n_blocks=a.mt_blocks, k=3, a=1.2, b=0).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    pairs = sorted(D["train"], key=lambda p: len(p[0]))         # bucketing by length
    for ep in range(a.mt_epochs):
        order = torch.randperm(len(pairs) // a.mt_batch)
        for b in order.tolist():
            chunk = pairs[b * a.mt_batch:(b + 1) * a.mt_batch]
            src = pad_to([src_v.encode(s) for s, _ in chunk], extra=0.2).to(DEV)
            tgt = pad_to([tgt_v.encode(t) for _, t in chunk], eos=True).to(DEV)
            loss = -m.log_prob(src, tgt).sum() / (tgt != PAD).sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        print(f"    epoch {ep + 1}: train {loss.item() / math.log(2):.3f} bits/char", flush=True)
    return m.eval()


def e2(a):
    D = multi30k()
    if a.quick:
        D["train"] = D["train"][:5000]
    src_v, tgt_v = Chars([s for s, _ in D["train"]]), Chars([t for _, t in D["train"]])
    m = train_translator(D, src_v, tgt_v, a)
    torch.save({"model": m.state_dict(), "src": src_v.itos, "tgt": tgt_v.itos}, HERE / "bytenet_mt.pt")
    test = D["test"][:a.n_test]
    tot = cnt = 0.0
    hyps, refs = [], []
    for s, t in test:
        src = torch.tensor([src_v.encode(s)], device=DEV)
        tgt = torch.tensor([tgt_v.encode(t) + [EOS]], device=DEV)
        with torch.no_grad():
            tot -= m.log_prob(src, tgt).item(); cnt += tgt.shape[1]
        hyps.append(tgt_v.decode(beam_search(m, src, beam=a.beam)).split())
        refs.append(t.split())
    return {"test bits/char": tot / cnt / math.log(2), "test BLEU": bleu(hyps, refs),
            "examples": [" ".join(h) for h in hyps[:5]]}


# ---------------------------------------------------------------------------------------------------- E3-E5
def e3(a):
    D = multi30k()
    xs = torch.tensor([float(len(s)) for s, _ in D["valid"]])
    ys = torch.tensor([float(len(t)) for _, t in D["valid"]])
    rho = torch.corrcoef(torch.stack([xs, ys]))[0, 1].item()
    slope = (xs * ys).sum().item() / (xs * xs).sum().item()     # least squares through the origin
    return {"pearson rho": rho, "slope de/en": slope, "share of targets longer than 1.2 x source":
            (ys > 1.2 * xs).float().mean().item()}


def e4(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ck = HERE / "bytenet_mt.pt"
    if not ck.exists():
        return {"note": "run e2 first"}
    st = torch.load(ck, map_location="cpu")
    src_v, tgt_v = Chars([]), Chars([])
    src_v.itos, tgt_v.itos = st["src"], st["tgt"]
    src_v.stoi, tgt_v.stoi = {c: i for i, c in enumerate(src_v.itos)}, {c: i for i, c in enumerate(tgt_v.itos)}
    m = ByteNet(len(src_v.itos), len(tgt_v.itos), d=a.mt_d, n_blocks=a.mt_blocks, k=3)
    m.load_state_dict(st["model"]); m.eval()
    (HERE / "figures").mkdir(exist_ok=True)
    files = []
    for k, (s, t) in enumerate(multi30k()["test"][:3]):
        S, T = saliency(m, torch.tensor([src_v.encode(s)]), torch.tensor([tgt_v.encode(t) + [EOS]]))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].imshow((S / S.max(1, keepdim=True).values).detach(), aspect="auto", cmap="gray_r")
        ax[0].set_title("outputs vs source characters"); ax[1].set_title("outputs vs previous target characters")
        ax[1].imshow((T / T.max(1, keepdim=True).values).detach(), aspect="auto", cmap="gray_r")
        f = HERE / "figures" / f"e4_saliency_{k}.png"
        fig.savefig(f, dpi=100); plt.close(fig)
        files.append(str(f.relative_to(HERE)))
    return {"figures": files}


def e5(a):
    spec = importlib.util.spec_from_file_location("attention028", HERE.parent / "028-Bahdanau-et-al-2015-Attention-NMT" / "attention.py")
    att = importlib.util.module_from_spec(spec); spec.loader.exec_module(att)
    out = {}
    for n in a.lengths:
        src = torch.randint(3, 50, (4, n))
        tgt = torch.randint(3, 50, (4, n))
        bn = ByteNet(50, 50, d=128, n_blocks=10)
        rs = att.RNNsearch(50, 50, m=128, n=128, l=128, n_align=128)
        t0 = time.time(); (-bn.log_prob(src, tgt).sum()).backward(); tb = time.time() - t0
        t0 = time.time(); (-rs.log_prob(src.T, tgt.T).sum()).backward(); tr = time.time() - t0
        out[n] = {"ByteNet s": tb, "RNNsearch s": tr}
        print(f"  E5 length {n}: ByteNet {tb:.2f}s, RNNsearch {tr:.2f}s", flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: enwik8 (paper 1.31 bits/byte)"), ("e2", "E2: char-level translation, Multi30k En->De"),
                 ("e3", "E3: length correlation (Figure 5; paper rho 0.968)"), ("e4", "E4: saliency (Figure 6)"),
                 ("e5", "E5: time vs length")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["e1", "e2", "e3", "e4", "e5"])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.lm_d, a.lm_blocks, a.lm_steps, a.lm_batch = 512, 30, 200_000, 16
    a.mt_d, a.mt_blocks, a.mt_epochs, a.mt_batch, a.beam, a.n_test = 400, 30, 20, 32, 12, 1000
    a.lengths = (50, 100, 200, 400, 800)
    if a.quick:
        a.lm_d, a.lm_blocks, a.lm_steps, a.lm_batch = 128, 10, 3000, 8
        a.mt_d, a.mt_blocks, a.mt_epochs, a.mt_batch, a.beam, a.n_test = 128, 10, 3, 32, 4, 100
        a.lengths = (50, 100, 200)
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
