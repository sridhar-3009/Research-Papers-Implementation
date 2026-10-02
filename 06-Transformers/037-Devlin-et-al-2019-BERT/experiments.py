"""Reproduce BERT (Devlin et al. 2019) as a 'mini-BERT' that one GPU can pre-train.

The paper pre-trains on BooksCorpus + English Wikipedia (3.3B words, 1M steps of 256 x 512 tokens, 4 days on 16
TPU chips). Here: a document-level corpus of public-domain Project Gutenberg books (a stand-in for BooksCorpus), a
WordPiece vocabulary built by our own `build_wordpiece_vocab`, and fine-tuning on two GLUE tasks: SST-2 (single
sentence) and RTE (sentence pairs, small).

  E1  The core claim (Table 1, Section 4.1): pre-train mini-BERT with MLM + NSP, then fine-tune on SST-2 and RTE;
      compare with the same network trained from scratch. Fine-tuning: batch 32, 3 epochs, lr chosen among
      {5e-5, 4e-5, 3e-5, 2e-5} on dev (Section 4.1).
  E2  Table 5, the pre-training tasks: BERT (MLM + NSP), No NSP, LTR & No NSP (left-to-right LM, left-only at
      fine-tuning too).
  E3  Table 6, model size: (L, H, A) from (2, 128, 2) to (6, 512, 8); MLM perplexity on held-out text and dev
      accuracy.
  E4  Appendix C.2, the masking mix: (MASK, SAME, RND) = (80,10,10), (100,0,0), (80,0,20), (80,20,0), (0,20,80),
      (0,0,100), each fine-tuned and feature-based (frozen features, linear classifier).
  E5  Section 5.3 / Table 7, feature-based vs fine-tuning: frozen features (last layer, sum or concatenation of the
      top four layers) + a linear classifier on SST-2.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # tiny corpus and models, ~1-2 hours on CPU
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import collections
import json
import math
import random
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

from bert import (IGNORE, PAD, Bert, Classifier, build_wordpiece_vocab, collate, feature_based, mask_tokens,
                  nsp_pairs, optimizer, pack_pair, warmup_linear, wordpiece_tokenize)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOOKS = [1342, 84, 11, 2701, 1661, 98, 1952, 345, 2600, 4300, 174, 1400, 16328, 76, 5200, 1080, 2554, 120, 768, 158]
GLUE = "https://dl.fbaipublicfiles.com/glue/data/{}.zip"


# ---------------------------------------------------------------------------------------------------- data
def words_of(text):
    return re.findall(r"[a-z]+|[0-9]+|[^\sa-z0-9]", text.lower())


def corpus(a):
    d = DATA / "gutenberg"; d.mkdir(parents=True, exist_ok=True)
    docs = []
    for b in BOOKS[: a.n_books]:
        f = d / f"{b}.txt"
        if not f.exists():
            urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt", f)
        text = f.read_text(encoding="utf-8", errors="ignore")
        body = text.split("*** START")[-1].split("*** END")[0]
        paras = [p for p in re.split(r"\n\s*\n", body) if len(p.split()) > 8]
        # a 'document' = 40 consecutive paragraphs; a 'sentence' = a paragraph (BERT's sentences are text spans)
        for k in range(0, len(paras), 40):
            docs.append([words_of(p) for p in paras[k:k + 40]])
    return docs


def glue(task):
    d = DATA / "glue"; d.mkdir(parents=True, exist_ok=True)
    if not (d / task).exists():
        z = d / f"{task}.zip"
        urllib.request.urlretrieve(GLUE.format(task), z)
        zipfile.ZipFile(z).extractall(d)
    out = {}
    for split in ("train", "dev"):
        rows = (d / task / f"{split}.tsv").read_text(encoding="utf-8").splitlines()[1:]
        if task == "SST-2":
            out[split] = [(words_of(r.split("\t")[0]), None, int(r.split("\t")[1])) for r in rows]
        else:                                                   # RTE: index, sentence1, sentence2, label
            parts = [r.split("\t") for r in rows]
            out[split] = [(words_of(p[1]), words_of(p[2]), int(p[3] == "not_entailment")) for p in parts if len(p) == 4]
    return out


class Tok:
    def __init__(self, docs, size):
        counts = collections.Counter(w for doc in docs for s in doc for w in s)
        self.itos = build_wordpiece_vocab(dict(counts.most_common(20000)), size)
        self.stoi = {p: i for i, p in enumerate(self.itos)}
        self.cache = {}

    def __call__(self, words):
        out = []
        for w in words:
            if w not in self.cache:
                self.cache[w] = [self.stoi.get(p, 1) for p in wordpiece_tokenize(w, self.stoi)]
            out += self.cache[w]
        return out


# ---------------------------------------------------------------------------------------------------- pre-training
def pretrain(docs, tok, a, L, H, A, nsp=True, causal=False, probs=(0.8, 0.1, 0.1), tag=""):
    torch.manual_seed(0)
    rng = random.Random(0)
    V = len(tok.itos)
    m = Bert(V, H=H, L=L, A=A, max_len=a.seq, causal=causal).to(DEV)
    opt = optimizer(m, a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_linear(s, a.warmup, a.steps))
    enc_docs = [[tok(s) for s in doc] for doc in docs]
    half = a.seq // 2
    for step in range(a.steps):
        batch = nsp_pairs(enc_docs, a.batch, rng)
        packed = [pack_pair(x[:half], y[:half], max_len=a.seq) for x, y, _ in batch]
        ids, seg, mask = (t.to(DEV) for t in collate(packed))
        nsp_lab = torch.tensor([l for *_, l in batch], device=DEV)
        if causal:                                             # 'LTR & No NSP': a left-to-right LM objective
            T, _ = m(ids, seg, mask)
            tgt = ids[:, 1:].masked_fill(ids[:, 1:] == PAD, IGNORE)
            loss = F.cross_entropy(m.mlm_logits(T)[:, :-1].reshape(-1, V), tgt.reshape(-1), ignore_index=IGNORE)
        else:
            inp, lab = mask_tokens(ids.cpu(), V, probs=probs)
            loss, mlm, nl = m.pretrain_loss(inp.to(DEV), seg, mask, lab.to(DEV), nsp_lab)
            if not nsp:
                loss = mlm
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % 2000 == 0:
            print(f"    {tag} step {step}: loss {loss.item():.3f}", flush=True)
    return m


def mlm_perplexity(m, docs, tok, a):
    V = len(tok.itos)
    g = torch.Generator().manual_seed(0)
    tot = cnt = 0.0
    m.eval()
    with torch.no_grad():
        for doc in docs[:50]:
            for s in doc[:10]:
                ids, seg, mask = collate([pack_pair(tok(s)[: a.seq - 2])])
                inp, lab = mask_tokens(ids, V, generator=g)
                if (lab != IGNORE).sum() == 0:
                    continue
                T, _ = m(inp.to(DEV), seg.to(DEV), mask.to(DEV))
                tot += F.cross_entropy(m.mlm_logits(T).reshape(-1, V), lab.reshape(-1).to(DEV), ignore_index=IGNORE,
                                       reduction="sum").item()
                cnt += (lab != IGNORE).sum().item()
    return math.exp(tot / max(cnt, 1))


# ---------------------------------------------------------------------------------------------------- fine-tuning
def finetune(bert_state, make_model, task, tok, a):
    best = 0.0
    for lr in a.ft_lrs:
        torch.manual_seed(0)
        clf = Classifier(make_model(), 2).to(DEV)
        if bert_state is not None:
            clf.bert.load_state_dict(bert_state)
        opt = optimizer(clf, lr)
        train = task["train"][: a.ft_train]
        steps = a.ft_epochs * math.ceil(len(train) / 32)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_linear(s, int(0.1 * steps), steps))
        for ep in range(a.ft_epochs):
            random.Random(ep).shuffle(train)
            for b in range(0, len(train), 32):
                chunk = train[b:b + 32]
                ids, seg, mask = (t.to(DEV) for t in collate([pack_pair(tok(x), tok(y) if y else None, a.seq)
                                                               for x, y, _ in chunk]))
                loss = F.cross_entropy(clf(ids, seg, mask), torch.tensor([l for *_, l in chunk], device=DEV))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        best = max(best, accuracy(clf, task["dev"], tok, a))
    return best


def accuracy(clf, data, tok, a):
    clf.eval(); right = 0
    with torch.no_grad():
        for b in range(0, len(data), 64):
            chunk = data[b:b + 64]
            ids, seg, mask = (t.to(DEV) for t in collate([pack_pair(tok(x), tok(y) if y else None, a.seq)
                                                           for x, y, _ in chunk]))
            right += (clf(ids, seg, mask).argmax(-1).cpu() == torch.tensor([l for *_, l in chunk])).sum().item()
    clf.train()
    return 100 * right / len(data)


# ---------------------------------------------------------------------------------------------------- experiments
def setup(a):
    docs = corpus(a)
    random.Random(0).shuffle(docs)
    tok = Tok(docs, a.vocab)
    tasks = {"SST-2": glue("SST-2"), "RTE": glue("RTE")}
    return docs[5:], docs[:5], tok, tasks


def e1(a):
    docs, held, tok, tasks = setup(a)
    L, H, A = a.L, a.H, a.A
    m = pretrain(docs, tok, a, L, H, A, tag="BERT")
    torch.save(m.state_dict(), HERE / "mini_bert.pt")
    make = lambda: Bert(len(tok.itos), H=H, L=L, A=A, max_len=a.seq)
    out = {}
    for name, task in tasks.items():
        out[name] = {"pre-trained": finetune(m.state_dict(), make, task, tok, a),
                     "from scratch": finetune(None, make, task, tok, a)}
        print("  E1", name, out[name], flush=True)
    out["MLM perplexity (held-out)"] = mlm_perplexity(m, held, tok, a)
    return out


def e2(a):
    docs, held, tok, tasks = setup(a)
    out = {}
    for name, kw in (("BERT (MLM + NSP)", {}), ("No NSP", dict(nsp=False)), ("LTR & No NSP", dict(nsp=False, causal=True))):
        m = pretrain(docs, tok, a, a.L, a.H, a.A, tag=name, **kw)
        causal = kw.get("causal", False)
        make = lambda: Bert(len(tok.itos), H=a.H, L=a.L, A=a.A, max_len=a.seq, causal=causal)
        out[name] = {t: finetune(m.state_dict(), make, task, tok, a) for t, task in tasks.items()}
        print("  E2", name, out[name], flush=True)
    return out


def e3(a):
    docs, held, tok, tasks = setup(a)
    out = {}
    for L, H, A in a.sizes:
        m = pretrain(docs, tok, a, L, H, A, tag=f"L{L} H{H}")
        make = lambda: Bert(len(tok.itos), H=H, L=L, A=A, max_len=a.seq)
        out[f"L={L} H={H} A={A}"] = {"MLM ppl": mlm_perplexity(m, held, tok, a),
                                     **{t: finetune(m.state_dict(), make, task, tok, a) for t, task in tasks.items()}}
        print("  E3", L, H, out[f"L={L} H={H} A={A}"], flush=True)
    return out


def linear_probe(m, task, tok, a, how):
    """Frozen features at the [CLS] position + logistic regression (feature-based approach)."""
    def feats(data):
        X, y = [], []
        for b in range(0, len(data), 64):
            chunk = data[b:b + 64]
            ids, seg, mask = (t.to(DEV) for t in collate([pack_pair(tok(x), tok(z) if z else None, a.seq)
                                                           for x, z, _ in chunk]))
            X.append(feature_based(m, ids, seg, mask, how)[:, 0].cpu())
            y += [l for *_, l in chunk]
        return torch.cat(X), torch.tensor(y)
    Xtr, ytr = feats(task["train"][: a.ft_train])
    Xdv, ydv = feats(task["dev"])
    W = torch.zeros(Xtr.shape[1], 2, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([W, b], 1e-2)
    for _ in range(300):
        loss = F.cross_entropy(Xtr @ W + b, ytr) + 1e-4 * (W ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return 100 * ((Xdv @ W + b).argmax(-1) == ydv).float().mean().item()


def e4(a):
    docs, held, tok, tasks = setup(a)
    out = {}
    for probs in ((0.8, 0.1, 0.1), (1.0, 0.0, 0.0), (0.8, 0.0, 0.2), (0.8, 0.2, 0.0), (0.0, 0.2, 0.8), (0.0, 0.0, 1.0)):
        # probs = (MASK, random, keep); the paper's table lists (MASK, SAME, RND)
        m = pretrain(docs, tok, a, a.L, a.H, a.A, probs=probs, tag=str(probs))
        make = lambda: Bert(len(tok.itos), H=a.H, L=a.L, A=a.A, max_len=a.seq)
        key = f"MASK {probs[0]:.0%} SAME {probs[2]:.0%} RND {probs[1]:.0%}"
        out[key] = {"SST-2 fine-tune": finetune(m.state_dict(), make, tasks["SST-2"], tok, a),
                    "SST-2 feature-based": linear_probe(m.eval(), tasks["SST-2"], tok, a, "concat_last_four")}
        print("  E4", key, out[key], flush=True)
    return out


def e5(a):
    docs, held, tok, tasks = setup(a)
    if not (HERE / "mini_bert.pt").exists():
        return {"note": "run e1 first"}
    m = Bert(len(tok.itos), H=a.H, L=a.L, A=a.A, max_len=a.seq).to(DEV)
    m.load_state_dict(torch.load(HERE / "mini_bert.pt", map_location=DEV)); m.eval()
    out = {how: linear_probe(m, tasks["SST-2"], tok, a, how) for how in ("last", "sum_last_four", "concat_last_four")}
    make = lambda: Bert(len(tok.itos), H=a.H, L=a.L, A=a.A, max_len=a.seq)
    out["fine-tuning"] = finetune(m.state_dict(), make, tasks["SST-2"], tok, a)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: pre-trained vs from scratch (GLUE dev accuracy)"), ("e2", "E2: Table 5 ablation"),
                 ("e3", "E3: Table 6 model size"), ("e4", "E4: Appendix C.2 masking mix"),
                 ("e5", "E5: feature-based vs fine-tuning (SST-2 dev)")):
        if R.get(k):
            L += [f"## {t}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["e1", "e2", "e3", "e4", "e5"])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.n_books, a.vocab, a.seq, a.L, a.H, a.A = 20, 8000, 128, 4, 256, 4
    a.steps, a.warmup, a.batch, a.lr = 50_000, 2500, 64, 1e-4
    a.ft_lrs, a.ft_epochs, a.ft_train = (5e-5, 4e-5, 3e-5, 2e-5), 3, 100_000
    a.sizes = ((2, 128, 2), (4, 256, 4), (6, 384, 6), (6, 512, 8))
    if a.quick:
        a.n_books, a.vocab, a.seq, a.L, a.H, a.A = 4, 2000, 64, 2, 128, 2
        a.steps, a.warmup, a.batch, a.lr = 1500, 150, 32, 5e-4
        a.ft_lrs, a.ft_epochs, a.ft_train = (5e-5,), 2, 3000
        a.sizes = ((1, 64, 2), (2, 128, 2))
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
