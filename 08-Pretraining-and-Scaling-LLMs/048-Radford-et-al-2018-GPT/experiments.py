"""Reproduce GPT (Radford et al. 2018) at single-GPU scale.

Pre-training corpus: 20 public-domain Gutenberg books (long contiguous text, the property the paper wanted from
BooksCorpus, which is no longer distributed). BPE learned on it (034's learn_bpe). Fine-tuning on GLUE / SuperGLUE
tasks that cover the four input transformations of Figure 1:
  SST-2 (classification), RTE (entailment), MRPC (similarity: both orders summed), COPA (multiple choice).

  E1  Pre-train a mini-GPT (default 6 layers, d = 384) with the paper's schedule (warm-up + cosine, AdamW 0.01);
      report held-out perplexity.
  E2  Table 5 in miniature: fine-tune (lambda = 0.5) vs without auxiliary LM (lambda = 0) vs no pre-training.
  E3  Figure 2 (left): number of transferred layers 0..L on RTE and SST-2.
  E4  Figure 2 (right): zero-shot heuristics over pre-training checkpoints: SST-2 ('very' + positive/negative),
      CoLA (average token log-prob, threshold chosen on train), COPA (average log-prob of each alternative).
  E5  Table 5's LSTM row: a 1-layer LSTM language model in the same pre-train / fine-tune framework.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import collections
import copy
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt import (GPT, FineTuner, Specials, apply_bpe, avg_logprob, classification_input, entailment_input,
                 learn_bpe, multiple_choice_inputs, next_token_logprobs, optimizer, pad_batch, similarity_inputs,
                 transfer_layers, warmup_cosine, warmup_linear)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOOKS = [1342, 84, 11, 2701, 1661, 98, 1952, 345, 2600, 4300, 174, 1400, 16328, 76, 5200, 1080, 2554, 120, 768, 158]
GLUE = "https://dl.fbaipublicfiles.com/glue/data/{}.zip"
COPA = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip"


# ---------------------------------------------------------------------------------------------------- data
def words_of(text):
    return re.findall(r"[a-z]+|[0-9]+|[^\sa-z0-9]", text.lower())


def books(n):
    d = DATA / "gutenberg"; d.mkdir(parents=True, exist_ok=True)
    texts = []
    for b in BOOKS[:n]:
        f = d / f"{b}.txt"
        if not f.exists():
            urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt", f)
        texts.append(words_of(f.read_text(encoding="utf-8", errors="ignore").split("*** START")[-1].split("*** END")[0]))
    return texts


def fetch_zip(url, name):
    d = DATA / "glue"; d.mkdir(parents=True, exist_ok=True)
    if not (d / name).exists():
        z = d / f"{name}.zip"
        urllib.request.urlretrieve(url, z)
        zipfile.ZipFile(z).extractall(d)
    return d / name


def glue(task):
    d = fetch_zip(GLUE.format(task), task)
    out = {}
    for split in ("train", "dev"):
        f = d / f"{split}.tsv"
        if task == "MRPC" and not f.exists():
            f = d / ("msr_paraphrase_train.txt" if split == "train" else "msr_paraphrase_test.txt")
        rows = [r.split("\t") for r in f.read_text(encoding="utf-8").splitlines()[1:]]
        if task == "SST-2":
            out[split] = [(words_of(r[0]), None, int(r[1])) for r in rows]
        elif task == "CoLA":
            rows = [r.split("\t") for r in (d / f"{split}.tsv").read_text(encoding="utf-8").splitlines()]
            out[split] = [(words_of(r[3]), None, int(r[1])) for r in rows if len(r) == 4]
        elif task == "RTE":
            out[split] = [(words_of(r[1]), words_of(r[2]), int(r[3] == "entailment")) for r in rows if len(r) == 4]
        else:                                                                      # MRPC: label, id, id, s1, s2
            out[split] = [(words_of(r[3]), words_of(r[4]), int(r[0])) for r in rows if len(r) == 5]
    return out


def copa():
    d = fetch_zip(COPA, "COPA")
    out = {}
    for split, fn in (("train", "train.jsonl"), ("dev", "val.jsonl")):
        out[split] = []
        for line in (d / fn).read_text().splitlines():
            ex = json.loads(line)
            joiner = "because" if ex["question"] == "cause" else "so"
            out[split].append((words_of(ex["premise"]) + [joiner], [words_of(ex["choice1"]), words_of(ex["choice2"])],
                               ex["label"]))
    return out


class BPETok:
    def __init__(self, texts, merges, base=("<pad>",)):
        counts = collections.Counter(w for t in texts for w in t)
        self.merges = learn_bpe(dict(counts.most_common(30000)), merges)
        syms = set()
        self.cache = {}
        for w in counts:
            syms.update(self.pieces(w))
        self.itos = list(base) + sorted(syms) + ["<unk>"]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def pieces(self, w):
        if w not in self.cache:
            self.cache[w] = apply_bpe(w, self.merges)
        return self.cache[w]

    def __call__(self, words):
        unk = self.stoi["<unk>"]
        return [self.stoi.get(p, unk) for w in words for p in self.pieces(w)]


# ---------------------------------------------------------------------------------------------------- pre-training
def make_gpt(a, vocab, lstm=False):
    if lstm:
        return LSTMLM(vocab, a.d, a.n_ctx)
    return GPT(vocab, n_ctx=a.n_ctx, d=a.d, layers=a.layers, heads=a.heads, ff=4 * a.d, dropout=0.1)


class LSTMLM(nn.Module):
    """Table 5's comparison: one LSTM layer in place of the transformer, same embeddings / tied output / API."""

    def __init__(self, vocab, d, n_ctx):
        super().__init__()
        self.n_ctx = n_ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(n_ctx, d)                                          # unused; keeps transfer_layers' API
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.blocks = nn.ModuleList()

    def hidden(self, ids, n_layers=None):
        return self.lstm(self.tok(ids))[0]

    def logits(self, h):
        return h @ self.tok.weight.T

    def lm_loss(self, ids, mask=None):
        return GPT.lm_loss(self, ids, mask)

    def extend_vocab(self, n_new):
        return GPT.extend_vocab(self, n_new)


def pretrain(stream, vocab, a, lstm=False, checkpoints=()):
    torch.manual_seed(0)
    m = make_gpt(a, vocab, lstm).to(DEV)
    opt = optimizer(m, a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, a.warmup, a.steps))
    saved = {}
    for step in range(1, a.steps + 1):
        starts = torch.randint(0, len(stream) - a.n_ctx - 1, (a.batch,))
        ids = torch.stack([stream[s:s + a.n_ctx] for s in starts]).to(DEV)     # contiguous windows
        loss = m.lm_loss(ids)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step in checkpoints:
            saved[step] = copy.deepcopy(m).cpu()
    return m.cpu(), saved


@torch.no_grad()
def perplexity(m, stream, a, n=200):
    m.eval().to(DEV)
    tot = 0.0
    g = torch.Generator().manual_seed(0)
    for _ in range(n):
        s = torch.randint(0, len(stream) - a.n_ctx - 1, (1,), generator=g)
        tot += m.lm_loss(stream[s:s + a.n_ctx][None].to(DEV)).item()
    m.cpu()
    return float(torch.exp(torch.tensor(tot / n)))


# ---------------------------------------------------------------------------------------------------- fine-tuning
def encode(task, ex, tok, sp, n_ctx):
    x1, x2, y = ex
    if task in ("SST-2", "CoLA"):
        return [classification_input(tok(x1)[:n_ctx - 2], sp)], y
    if task == "RTE":
        a, b = tok(x1)[:(n_ctx - 3) // 2], tok(x2)[:(n_ctx - 3) // 2]
        return [entailment_input(a, b, sp)], y
    if task == "MRPC":
        a, b = tok(x1)[:(n_ctx - 3) // 2], tok(x2)[:(n_ctx - 3) // 2]
        return similarity_inputs(a, b, sp), y
    return multiple_choice_inputs(tok(x1)[:n_ctx // 2], [tok(c)[:n_ctx // 2 - 3] for c in x2], sp), y   # COPA


def finetune(base, task, data, tok, a, lam=0.5, layers=None, scratch_vocab=None):
    torch.manual_seed(0)
    if scratch_vocab:
        g = make_gpt(a, scratch_vocab)
    elif layers is not None:
        g = transfer_layers(base, make_gpt(a, base.tok.weight.shape[0]), layers)
    else:
        g = copy.deepcopy(base)
    sp = Specials(*g.extend_vocab(3))
    f = FineTuner(g, 2 if task != "COPA" else 1).to(DEV)
    train = [encode(task, ex, tok, sp, a.n_ctx) for ex in data["train"]]
    opt = optimizer(f, a.lr_ft)
    total = a.ft_epochs * (len(train) // a.ft_batch + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_linear(s, max(1, int(0.002 * total)), total))
    for _ in range(a.ft_epochs):
        f.train()
        for k in torch.randperm(len(train)).split(a.ft_batch):
            loss = batch_loss(f, task, [train[i] for i in k], lam)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return evaluate(f, task, [encode(task, ex, tok, sp, a.n_ctx) for ex in data["dev"]])


def batch_loss(f, task, items, lam):
    seqs = [s for seqs, _ in items for s in seqs]
    ids, mask = pad_batch(seqs)
    ids, mask = ids.to(DEV), mask.to(DEV)
    y = torch.tensor([lab for _, lab in items], device=DEV)
    feats, h = f.features(ids, mask)
    if task == "MRPC":                                                            # add the two orders' states
        logits = f.Wy(f.drop(feats.view(len(items), 2, -1).sum(1)))
    elif task == "COPA":                                                          # one score per choice
        logits = f.Wy(f.drop(feats)).view(len(items), 2)
    else:
        logits = f.Wy(f.drop(feats))
    l2 = F.cross_entropy(logits, y)
    if lam == 0:
        return l2
    lg = f.gpt.logits(h)[:, :-1]
    m = mask[:, 1:].float()
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none").view_as(m)
    return l2 + lam * (ce * m).sum() / m.sum()


@torch.no_grad()
def evaluate(f, task, items):
    f.eval()
    preds, ys = [], []
    for k in range(0, len(items), 32):
        chunk = items[k:k + 32]
        ids, mask = pad_batch([s for seqs, _ in chunk for s in seqs])
        feats, _ = f.features(ids.to(DEV), mask.to(DEV))
        if task == "MRPC":
            logits = f.Wy(feats.view(len(chunk), 2, -1).sum(1))
        elif task == "COPA":
            logits = f.Wy(feats).view(len(chunk), 2)
        else:
            logits = f.Wy(feats)
        preds += logits.argmax(1).cpu().tolist(); ys += [lab for _, lab in chunk]
    acc = sum(p == y for p, y in zip(preds, ys)) / len(ys)
    if task == "MRPC":
        tp = sum(p == y == 1 for p, y in zip(preds, ys))
        prec, rec = tp / max(1, sum(preds)), tp / max(1, sum(ys))
        return {"accuracy": acc, "F1": 2 * prec * rec / max(1e-9, prec + rec)}
    return {"accuracy": acc}


# ---------------------------------------------------------------------------------------------------- experiments
def setup(a):
    texts = books(a.n_books)
    tok = BPETok(texts, a.merges)
    stream = torch.tensor([i for t in texts for i in tok(t)])
    cut = int(0.98 * len(stream))
    return tok, stream[:cut], stream[cut:]


def tasks(a):
    out = {t: glue(t) for t in ("SST-2", "RTE", "MRPC")}
    out["COPA"] = copa()
    for t in out:
        out[t]["train"] = out[t]["train"][:a.max_train]
    return out


def e1(a):
    tok, tr, va = setup(a)
    m, _ = pretrain(tr, len(tok.itos), a)
    torch.save(m.state_dict(), HERE / "gpt_mini.pt")
    return {"vocab": len(tok.itos), "held-out perplexity": perplexity(m, va, a), "paper (BooksCorpus)": 18.4}


def load_or_pretrain(a, tok, tr):
    m = make_gpt(a, len(tok.itos))
    if (HERE / "gpt_mini.pt").exists():
        m.load_state_dict(torch.load(HERE / "gpt_mini.pt", map_location="cpu"))
        return m
    return pretrain(tr, len(tok.itos), a)[0]


def e2(a):
    tok, tr, _ = setup(a)
    base = load_or_pretrain(a, tok, tr)
    T = tasks(a)
    out = {}
    for name, kw in (("full (aux LM, lambda 0.5)", dict(lam=0.5)), ("no aux LM", dict(lam=0.0)),
                     ("no pre-training", dict(lam=0.5, scratch_vocab=len(tok.itos)))):
        out[name] = {t: finetune(base, t, T[t], tok, a, **kw) for t in T}
        print("  E2", name, out[name], flush=True)
    return out


def e3(a):
    tok, tr, _ = setup(a)
    base = load_or_pretrain(a, tok, tr)
    T = tasks(a)
    return {t: {n: finetune(base, t, T[t], tok, a, layers=n)["accuracy"] for n in range(a.layers + 1)}
            for t in ("RTE", "SST-2")}


def e4(a):
    tok, tr, _ = setup(a)
    ckpts = sorted({int(a.steps * f) for f in (0.1, 0.25, 0.5, 0.75, 1.0)})
    _, saved = pretrain(tr, len(tok.itos), a, checkpoints=ckpts)
    sst, cola, cp = glue("SST-2"), glue("CoLA"), copa()
    very, pos, neg = tok(["very"])[0], tok(["positive"])[0], tok(["negative"])[0]
    out = {}
    for step, m in saved.items():
        acc_sst = sum((next_token_logprobs(m, tok(x)[:a.n_ctx - 2] + [very])[pos] >
                       next_token_logprobs(m, tok(x)[:a.n_ctx - 2] + [very])[neg]) == bool(y)
                      for x, _, y in sst["dev"][:a.n_zero]) / min(a.n_zero, len(sst["dev"]))
        score = lambda x: avg_logprob(m, tok(x)[:1], tok(x)[1:a.n_ctx]) if len(tok(x)) > 1 else 0.0
        tr_s = [(score(x), y) for x, _, y in cola["train"][:a.n_zero]]
        thr = max((t for t, _ in tr_s), key=lambda t: sum((s > t) == bool(y) for s, y in tr_s))
        acc_cola = sum((score(x) > thr) == bool(y) for x, _, y in cola["dev"][:a.n_zero]) / min(a.n_zero, len(cola["dev"]))
        acc_copa = sum(int(avg_logprob(m, tok(p), tok(c[1])) > avg_logprob(m, tok(p), tok(c[0]))) == y
                       for p, c, y in cp["dev"]) / len(cp["dev"])
        out[f"step {step}"] = {"SST-2 zero-shot": acc_sst, "CoLA zero-shot": acc_cola, "COPA zero-shot": acc_copa}
        print("  E4", step, out[f"step {step}"], flush=True)
    return out


def e5(a):
    tok, tr, _ = setup(a)
    lstm, _ = pretrain(tr, len(tok.itos), a, lstm=True)
    T = tasks(a)
    return {t: finetune(lstm, t, T[t], tok, a) for t in T}


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        L += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.n_books, a.merges, a.n_ctx, a.d, a.layers, a.heads = 20, 8000, 256, 384, 6, 6
    a.lr, a.warmup, a.steps, a.batch = 2.5e-4, 2000, 50000, 32
    a.lr_ft, a.ft_epochs, a.ft_batch, a.max_train, a.n_zero = 6.25e-5, 3, 32, 100000, 1000
    if a.quick:
        a.n_books, a.merges, a.n_ctx, a.d, a.layers, a.heads = 2, 500, 64, 64, 2, 2
        a.warmup, a.steps, a.batch, a.ft_epochs, a.max_train, a.n_zero = 20, 200, 8, 1, 200, 50
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
