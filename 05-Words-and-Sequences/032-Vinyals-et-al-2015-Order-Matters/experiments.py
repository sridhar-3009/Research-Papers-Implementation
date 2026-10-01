"""Reproduce 'Order Matters: Sequence to Sequence for Sets' (Vinyals, Bengio & Kudlur, ICLR 2016).

  E1  Table 1, sorting: Ptr-Net (LSTM encoder) vs Read-Process-Write with P = 0, 1, 5, 10 process steps, each with
      0 or 1 glimpse, for N = 5, 10, 15 numbers, 10,000 updates. Paper (N = 5, no glimpse / glimpse): Ptr-Net
      81/90%, P=0 65/84%, P=1 84/92%, P=5 88/94%, P=10 90/94%.
  E2  Section 5.1.1, language modelling order: PTB with the 'medium' regularized LSTM of Zaremba et al. (Paper 024)
      on natural, reversed and 3-word-reversed text. Paper: 86 / 86 / 96 validation perplexity.
  E3  Section 5.1.2, parse-tree order: depth-first vs breadth-first linearizations with an attention seq2seq parser
      (Paper 030's LSTM+A, imported) on NLTK's WSJ sample. Paper: 89.5 vs 81.5 F1.
  E4  Section 5.1.3, the equivalence class of outputs: sorting where the target lists (index, rank) pairs in
      increasing order vs in a random one of the n! orders. Paper: random orders never reach the same accuracy.
  E5  Section 5.1.4, star graphical models: head-first vs head-last LSTM, over the number of variables (10-50),
      the training-set size (200-20,000) and the peakiness of the distributions; 10,000 updates each.
  E6  Table 2, finding orders with Eq. 9 on PTB 5-grams turned into sets of (word, position): fixed (1,2,3,4,5),
      fixed (5,1,3,4,2), 'easy' (the data mixes those two orders; Eq. 9 picks) and 'hard' (all 120 orders).
      Paper: 225, 280, 225, 225 perplexity.

Choices the paper leaves open (all in orderset.py): dot-product attention in the process block; a glimpse ADDS its
readout to the decoder state (replacing it stalled our set model); pointer outputs are not masked.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # fewer updates and smaller grids, ~1-2 hours on CPU
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import collections
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from orderset import (ReadProcessWrite, SetLM, best_order, bfs_delinearize, bfs_linearize, dfs_linearize, eq9_step,
                      order_log_prob, reverse_words, sort_batch, star_log_prob, star_model, star_sample, star_tokens,
                      three_word_reversal)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PTB_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{}.txt"


# ---------------------------------------------------------------------------------------------------- E1 sorting
def train_sorter(n, a, iters, **kw):
    torch.manual_seed(0)
    m = ReadProcessWrite(d=a.d, **kw).to(DEV)
    opt = torch.optim.Adam(m.parameters(), a.lr)
    for _ in range(iters):
        X, o = sort_batch(a.batch, n)
        loss = -m.log_prob(X.to(DEV), o.to(DEV)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2); opt.step()
    X, o = sort_batch(2000, n, torch.Generator().manual_seed(123))
    with torch.no_grad():
        return 100 * (m.greedy(X.to(DEV)).cpu() == o).all(1).float().mean().item()


def e1(a):
    out = {}
    for n in a.lengths:
        for gl in (0, 1):
            out[f"N={n} Ptr-Net g{gl}"] = train_sorter(n, a, a.iters, encoder="lstm", steps=0, glimpses=gl)
            for P in (0, 1, 5, 10):
                out[f"N={n} P={P} g{gl}"] = train_sorter(n, a, a.iters, steps=P, glimpses=gl)
            print(f"  E1 N={n} glimpses={gl}:", {k: round(v, 1) for k, v in out.items() if k.startswith(f'N={n}') and k.endswith(f'g{gl}')},
                  flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- E2 LM order
def ptb():
    (DATA / "ptb").mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "valid", "test"):
        p = DATA / "ptb" / f"ptb.{split}.txt"
        if not p.exists():
            urllib.request.urlretrieve(PTB_URL.format(split), p)
        out[split] = [line.split() + ["<eos>"] for line in p.read_text().splitlines() if line.strip()]
    return out


class MediumLSTM(nn.Module):
    """Zaremba et al.'s regularized LSTM LM: 2 layers, dropout on the non-recurrent connections only."""

    def __init__(self, V, n=650, p=0.5):
        super().__init__()
        self.emb, self.drop = nn.Embedding(V, n), nn.Dropout(p)
        self.lstm = nn.LSTM(n, n, 2, dropout=p)
        self.out = nn.Linear(n, V)
        for w in self.parameters():
            nn.init.uniform_(w, -0.05, 0.05)

    def forward(self, x, state=None):
        h, state = self.lstm(self.drop(self.emb(x)), state)
        return self.out(self.drop(h)), state


def lm_perplexity(order_fn, a):
    data = ptb()
    vocab = {w: i for i, w in enumerate(sorted({w for s in data["train"] for w in s} | {"<pad>"}))}
    stream = lambda split: torch.tensor([vocab[w] for s in data[split] for w in order_fn(s)])
    train, valid = stream("train"), stream("valid")
    if a.quick:
        train = train[:200_000]
    B, T = 20, 35

    def batches(s):
        s = s[: len(s) // B * B].view(B, -1).t()
        for i in range(0, s.shape[0] - 1, T):
            yield s[i:i + T], s[i + 1:i + 1 + T]

    torch.manual_seed(0)
    m = MediumLSTM(len(vocab), n=a.lm_size).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)

    def evaluate():
        m.eval(); state, tot, cnt = None, 0.0, 0
        with torch.no_grad():
            for x, y in batches(valid):
                out, state = m(x.to(DEV), state)
                tot += F.cross_entropy(out[: len(y)].reshape(-1, out.shape[-1]), y.reshape(-1).to(DEV), reduction="sum").item()
                cnt += y.numel()
        return math.exp(tot / cnt)

    best = float("inf")
    for ep in range(a.lm_epochs):
        if ep >= 6:
            for g in opt.param_groups:
                g["lr"] /= 1.2
        m.train(); state = None
        for x, y in batches(train):
            if len(y) < len(x):
                x = x[: len(y)]
            out, state = m(x.to(DEV), state)
            state = tuple(s.detach() for s in state)
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1).to(DEV))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
        best = min(best, evaluate())
        print(f"    epoch {ep + 1}: valid perplexity {best:.1f}", flush=True)
    return best


def e2(a):
    return {name: lm_perplexity(fn, a) for name, fn in (("natural", lambda s: s), ("reverse", reverse_words),
                                                          ("3-word reversal", three_word_reversal))}


# ---------------------------------------------------------------------------------------------------- E3 parse order
def e3(a):
    import importlib.util
    gfl_dir = HERE.parent / "030-Vinyals-et-al-2015-Grammar-as-Foreign-Language"
    sys.path.insert(0, str(gfl_dir))                           # Paper 030's modules, loaded by explicit path

    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, gfl_dir / file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    P = load("parser", "parser.py")
    gfl = load("gfl_experiments", "experiments.py")
    PAD, LSTMA, Vocab, batch_pairs, beam_search, evalb, words = (P.PAD, P.LSTMA, P.Vocab, P.batch_pairs,
                                                                 P.beam_search, P.evalb, P.words)

    def strip_words(t):
        return (t[0], []) if len(t[1]) == 1 and isinstance(t[1][0], str) else (t[0], [strip_words(k) for k in t[1]])

    def attach(t, ws, k=None):
        """Put the words back on the leaves (pre-terminals), left to right."""
        k = k if k is not None else [0]
        if not t[1]:
            w = ws[k[0]] if k[0] < len(ws) else "<w>"
            k[0] += 1
            return (t[0], [w])
        return (t[0], [attach(c, ws, k) for c in t[1]])

    def dfs_tree(seq):
        stack, root = [("ROOT", [])], None
        for s in seq:
            if s.startswith("!"):
                if len(stack) > 1:
                    stack.pop()
            else:
                n = (s, []); stack[-1][1].append(n); stack.append(n)
        return stack[0][1][0] if stack[0][1] else ("S", [])

    D = gfl.load()
    if a.quick:
        D = {k: v[:500] if k == "train" else v[:100] for k, v in D.items()}
    out = {}
    for name, lin, delin in (("depth first", dfs_linearize, dfs_tree), ("breadth first", bfs_linearize, bfs_delinearize)):
        wv = Vocab([w.lower() for t in D["train"] for w in words(t)])
        lv = Vocab([s for t in D["train"] for s in lin(strip_words(t))])
        enc = lambda t: (wv.encode([w.lower() for w in words(t)]), lv.encode(lin(strip_words(t))))
        pairs = [enc(t) for t in D["train"]]
        torch.manual_seed(0)
        m = LSTMA(len(wv), len(lv), d=a.parse_d, emb=a.parse_d, layers=2, dropout=0.3)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        for ep in range(a.parse_epochs):
            random.Random(ep).shuffle(pairs)
            for b in range(0, len(pairs), 32):
                src, tgt = batch_pairs(pairs[b:b + 32])
                loss = -m.log_prob(src, tgt).sum() / (tgt != PAD).sum()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
        m.eval()
        preds, valid = [], 0
        for t in D["test"]:
            seq = [lv.itos[s] for s in beam_search(m, batch_pairs([enc(t)])[0], beam=1)[1]]
            try:
                tree = attach(delin(seq), words(t)); valid += 1
            except Exception:
                tree = ("S", [("XX", [w]) for w in words(t)])
            preds.append(tree)
        out[name] = {"F1": evalb(D["test"], preds)["F1"], "valid trees %": 100 * valid / len(D["test"])}
        print("  E3", name, out[name], flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- E4 output sets
class RankedPointer(ReadProcessWrite):
    """Sorting with outputs as a SET of (index, rank) pairs: each step points at an element AND names its rank."""

    def __init__(self, n, **kw):
        super().__init__(**kw)
        self.rank = nn.Linear(self.d, n)

    def pair_log_prob(self, X, idx_seq, rank_seq):
        M, h = self.encode(X)
        c = torch.zeros_like(h)
        x = self.go.expand(X.shape[0], -1)
        r = torch.arange(X.shape[0])
        total = X.new_zeros(X.shape[0])
        for t in range(idx_seq.shape[1]):
            h, c = self.dec(x, (h, c))
            total = total + F.log_softmax(self.pointer_logits(M, h), -1)[r, idx_seq[:, t]]
            total = total + F.log_softmax(self.rank(h), -1)[r, rank_seq[:, t]]
            x = M[r, idx_seq[:, t]]
        return total

    @torch.no_grad()
    def predict_pairs(self, X):
        M, h = self.encode(X)
        c = torch.zeros_like(h)
        x = self.go.expand(X.shape[0], -1)
        r = torch.arange(X.shape[0])
        pairs = []
        for _ in range(X.shape[1]):
            h, c = self.dec(x, (h, c))
            k = self.pointer_logits(M, h).argmax(-1)
            pairs.append(torch.stack([k, self.rank(h).argmax(-1)], 1))
            x = M[r, k]
        return torch.stack(pairs, 1)


def e4(a):
    out = {}
    for n in (5, 10):
        for name in ("increasing order", "random order"):
            torch.manual_seed(0)
            m = RankedPointer(n, d=a.d, steps=1, glimpses=1)
            opt = torch.optim.Adam(m.parameters(), a.lr)
            for _ in range(a.iters):
                X, o = sort_batch(a.batch, n)                           # o[:, r] = index of the rank-r element
                ranks = torch.arange(n).expand(a.batch, -1)
                if name == "random order":
                    perm = torch.stack([torch.randperm(n) for _ in range(a.batch)])
                    o, ranks = o.gather(1, perm), ranks.gather(1, perm)
                loss = -m.pair_log_prob(X, o, ranks).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 2); opt.step()
            X, o = sort_batch(1000, n, torch.Generator().manual_seed(5))
            pred = m.predict_pairs(X)
            truth = [{(o[i, r].item(), r) for r in range(n)} for i in range(len(X))]
            acc = sum({tuple(p) for p in pred[i].tolist()} == truth[i] for i in range(len(X))) / len(X)
            out[f"n={n} {name}"] = 100 * acc
            print(f"  E4 n={n} {name}: {100 * acc:.1f}%", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- E5 graphical models
def e5(a):
    out = {}
    for n_vars in a.star_vars:
        for n_train in a.star_sizes:
            for peaky in a.star_peaky:
                g = torch.Generator().manual_seed(0)
                model = star_model(n_vars, peaky=peaky, generator=g)
                tr, te = star_sample(model, n_train, g), star_sample(model, 5000, g)
                res = {"true model NLL": -star_log_prob(model, te).mean().item()}
                for head_first in (True, False):
                    torch.manual_seed(0)
                    lm = SetLM(n_vars * 10, d=128).to(DEV)
                    opt = torch.optim.Adam(lm.parameters(), 3e-3)
                    T, TE = star_tokens(tr, head_first).to(DEV), star_tokens(te, head_first).to(DEV)
                    best = float("inf")
                    for it in range(a.star_iters):
                        loss = -lm.seq_log_prob(T[torch.randint(0, n_train, (64,))]).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                        if it % 200 == 199:
                            with torch.no_grad():
                                best = min(best, -lm.seq_log_prob(TE).mean().item())
                    res["head first" if head_first else "head last"] = best
                out[f"vars={n_vars} train={n_train} peaky={peaky}"] = res
                print("  E5", n_vars, n_train, peaky, {k: round(v, 2) for k, v in res.items()}, flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- E6 Eq. 9
def e6(a):
    data = ptb()
    vocab = {w: i for i, w in enumerate(sorted({w for s in data["train"] for w in s}))}

    def fivegrams(split, limit):
        out = []
        for s in data[split]:
            ids = [vocab[w] for w in s]
            out += [ids[i:i + 5] for i in range(len(ids) - 4)]
        random.Random(0).shuffle(out)
        return torch.tensor(out[:limit])

    V = len(vocab)
    tr, va = fivegrams("train", a.ngram_train), fivegrams("valid", 5000)
    tok = lambda w: w * 5 + torch.arange(5)                     # (word, position) token: the 5-gram as a SET
    tr_set, va_set = tok(tr), tok(va)
    nat = torch.arange(5)
    scr = torch.tensor([4, 0, 2, 3, 1])                         # the paper's (5, 1, 3, 4, 2)
    out = {}
    for name in ("(1,2,3,4,5)", "(5,1,3,4,2)", "easy", "hard"):
        torch.manual_seed(0)
        lm = SetLM(V * 5, d=a.ngram_d).to(DEV)
        opt = torch.optim.Adam(lm.parameters(), 1e-3)
        g = torch.Generator().manual_seed(0)
        for it in range(a.ngram_iters):
            b = tr_set[torch.randint(0, len(tr_set), (128,))].to(DEV)
            if name == "(1,2,3,4,5)":
                eq9_step(lm, opt, b, "given")
            elif name == "(5,1,3,4,2)":
                eq9_step(lm, opt, b[:, scr], "given")
            elif name == "easy":                               # the data shows one of the two orders at random
                b = b.gather(1, (b % 5).argsort(1))             # a set: put tokens back in position order first
                cands = torch.stack([nat, scr]).to(b.device)    # the two orders Eq. 9 may choose between
                if it < a.pretrain:                             # uniform prior over the two orders
                    pick = torch.randint(0, 2, (len(b),), generator=g).to(b.device)
                    loss = -order_log_prob(lm, b, cands[pick]).mean()
                else:                                           # Eq. 9: max over the two orders
                    lps = torch.stack([order_log_prob(lm, b, c.expand(len(b), -1)) for c in cands], 1)
                    loss = -lps.max(1).values.mean()
                opt.zero_grad(); loss.backward(); opt.step()
            else:                                               # hard: any of the 120 orders
                b = torch.stack([row[torch.randperm(5, generator=g)] for row in b])
                eq9_step(lm, opt, b, "uniform" if it < a.pretrain else "sample", g)
        with torch.no_grad():
            if name in ("(1,2,3,4,5)", "(5,1,3,4,2)"):
                o = (nat if name.startswith("(1") else scr).expand(len(va_set), -1)
                nll = -order_log_prob(lm, va_set.to(DEV), o.to(DEV)).mean().item()
                orders = {}
            else:
                o, _ = best_order(lm, va_set[:2000].to(DEV))
                nll = -order_log_prob(lm, va_set[:2000].to(DEV), o).mean().item()
                orders = dict(collections.Counter(str(tuple((x + 1).tolist())) for x in o).most_common(3))
        out[name] = {"perplexity": math.exp(nll / 5), "favourite orders": orders}
        print("  E6", name, out[name], flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- driver
def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    titles = {"e1": "E1: Table 1, sorting accuracy (%)", "e2": "E2: PTB validation perplexity by word order",
              "e3": "E3: parsing, depth-first vs breadth-first", "e4": "E4: sorting as an output set",
              "e5": "E5: star graphical models (test NLL)", "e6": "E6: Table 2, Eq. 9 on PTB 5-grams"}
    for k, t in titles.items():
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 7)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.d, a.lr, a.batch, a.iters, a.lengths = 128, 1e-3, 128, 10000, (5, 10, 15)
    a.lm_size, a.lm_epochs = 650, 39
    a.parse_d, a.parse_epochs = 256, 20
    a.star_vars, a.star_sizes, a.star_peaky, a.star_iters = (10, 30, 50), (200, 2000, 20000), (0.5, 2.0, 8.0), 10000
    a.ngram_train, a.ngram_iters, a.ngram_d, a.pretrain = 500_000, 20000, 256, 1000
    if a.quick:
        a.d, a.lr, a.batch, a.iters, a.lengths = 64, 1e-2, 64, 1500, (5, 10)
        a.lm_size, a.lm_epochs = 200, 4
        a.parse_d, a.parse_epochs = 64, 3
        a.star_vars, a.star_sizes, a.star_peaky, a.star_iters = (10, 30), (200, 2000), (2.0,), 1000
        a.ngram_train, a.ngram_iters, a.ngram_d, a.pretrain = 50_000, 2000, 64, 300
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
