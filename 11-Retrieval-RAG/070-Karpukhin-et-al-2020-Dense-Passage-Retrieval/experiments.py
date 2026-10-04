"""Dense Passage Retrieval experiments (Karpukhin et al. 2020): toy sweeps of the paper's ablations, and a real dual
encoder fine-tuned from BERT on a real QA dataset.

  E1  Table 3 on the toy, several seeds and an EQUAL number of steps: negatives random / BM25 / in-batch gold for
      batch 8, 32, 128, and in-batch + 1 or 2 BM25 negatives.
  E2  Figure 1 on the toy: training-set size 50 ... 1120 (with equal steps), DPR vs BM25 top-k.
  E3  Similarity and loss (Section 5.2): dot product vs cosine vs (negative) L2 distance.
  E4  Shared initialisation of the two encoders (like starting both from one BERT) vs unrelated random weights.
  E5  Real DPR: two `bert-base-uncased` encoders ([CLS] vectors), in-batch + 1 BM25 negative, Adam lr 1e-5, trained on
      Natural Questions pairs from Hugging Face `datasets` ('sentence-transformers/natural-questions'); evaluated as
      top-k accuracy over a sub-corpus of the answers' passages + distractors, against BM25.

!! HEAVY for E5 (GPU + BERT and data download); E1-E4 run in minutes on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dpr as D

HERE = Path(__file__).parent


def setup(a, seed=0):
    W = D.make_world(n_people=a.people, n_filler=a.filler, seed=seed)
    return W, D.make_questions(W, seed=seed), D.BM25(W["passages"])


def acc(model, W, qs, ks=(1, 5, 20)):
    return D.top_k_accuracy(D.DenseIndex(model, W["passages"]).scores([q for q, *_ in qs]), qs, W["passages"], ks)


def e1(a):
    out = {}
    schemes = {"7 random": dict(negatives="random", n_extra=7, batch=128),
               "7 BM25": dict(negatives="bm25", n_extra=7, batch=128),
               "in-batch 8": dict(negatives="gold", batch=8), "in-batch 32": dict(negatives="gold", batch=32),
               "in-batch 128": dict(negatives="gold", batch=128),
               "in-batch 128 + 1 BM25": dict(negatives="gold+bm25", batch=128, n_extra=1),
               "in-batch 128 + 2 BM25": dict(negatives="gold+bm25", batch=128, n_extra=2)}
    for seed in range(a.seeds):
        W, Q, bm = setup(a, seed)
        seen = Q["test seen people, overlap"] + Q["test seen people, paraphrase"]
        for name, kw in schemes.items():
            epochs = max(1, round(a.steps * kw["batch"] / len(Q["train"])))           # equal number of steps
            out.setdefault(name, []).append(acc(D.train_dpr(W, Q["train"], bm25=bm, epochs=epochs, seed=seed, **kw), W, seen))
    return out


def e2(a):
    W, Q, bm = setup(a)
    qs = Q["test seen people, paraphrase"]
    out = {"BM25": D.top_k_accuracy(D.bm25_matrix(bm, qs), qs, W["passages"], (1, 5, 20))}
    for n in a.sizes:
        b = min(128, n)
        epochs = max(1, round(a.steps * b / n))
        out[f"DPR {n}"] = acc(D.train_dpr(W, Q["train"], n_train=n, batch=b, epochs=epochs, bm25=bm), W, qs)
    return out


def e3(a):
    """Train with each similarity function; re-implements the training loop with a pluggable score."""
    W, Q, bm = setup(a)
    seen = Q["test seen people, overlap"] + Q["test seen people, paraphrase"]
    sims = {"dot": lambda q, p: q @ p.T,
            "cosine": lambda q, p: F.normalize(q, dim=-1) @ F.normalize(p, dim=-1).T,
            "neg L2": lambda q, p: -torch.cdist(q, p)}
    out = {}
    for name, sim in sims.items():
        rng = random.Random(0)
        torch.manual_seed(0)
        vocab = D.Vocab(W["passages"] + [q for q, *_ in Q["train"]])
        m = D.DPR(vocab, 128)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        qs = list(Q["train"])
        for _ in range(a.epochs):
            rng.shuffle(qs)
            for s in range(0, len(qs), 128):
                b = qs[s:s + 128]
                S = sim(m.q([q for q, *_ in b]), m.p([W["passages"][W["gold"][(e, r)]] for _, _, e, r in b]))
                loss = F.cross_entropy(S, torch.arange(len(b)))
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            S = sim(m.q([q for q, *_ in seen]), m.p(W["passages"])).numpy()
        out[name] = D.top_k_accuracy(S, seen, W["passages"], (1, 5, 20))
    return out


def e4(a):
    W, Q, bm = setup(a)
    out = {}
    for same in (True, False):
        m = D.train_dpr(W, Q["train"], bm25=bm, same_init=same, epochs=a.epochs)
        out[f"same init {same}"] = {k: acc(m, W, qs) for k, qs in Q.items() if k != "train"}
    return out


def e5(a):
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.bert)
    eq, ep = AutoModel.from_pretrained(a.bert).to(dev), AutoModel.from_pretrained(a.bert).to(dev)
    ds = load_dataset("sentence-transformers/natural-questions", split="train").shuffle(seed=0)
    train = ds.select(range(a.nq_train))
    test = ds.select(range(a.nq_train, a.nq_train + a.nq_test))
    corpus = list(dict.fromkeys(test["answer"] + ds.select(range(a.nq_train + a.nq_test, a.nq_train + a.nq_test + a.distract))["answer"]))
    bm = D.BM25([c.lower() for c in corpus])

    def enc(model, texts):
        x = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        return model(**x).last_hidden_state[:, 0]                                   # the [CLS] vector

    opt = torch.optim.Adam(list(eq.parameters()) + list(ep.parameters()), lr=1e-5)
    bm_train = D.BM25([c.lower() for c in train["answer"]])
    for _ in range(a.nq_epochs):
        idx = np.random.permutation(len(train))
        for s in range(0, len(idx), a.nq_batch):
            b = train.select(idx[s:s + a.nq_batch].tolist())
            hard = []
            for q, ans in zip(b["query"], b["answer"]):                             # top BM25 passage that is not gold
                o = np.argsort(-bm_train.scores(q.lower()))[:5]
                hard.append(next((train[int(i)]["answer"] for i in o if train[int(i)]["answer"] != ans), ans))
            loss = D.in_batch_loss(enc(eq, b["query"]), enc(ep, b["answer"]), enc(ep, hard))
            opt.zero_grad()
            loss.backward()
            opt.step()
    eq.eval(), ep.eval()
    with torch.no_grad():
        P = torch.cat([enc(ep, corpus[i:i + 64]) for i in range(0, len(corpus), 64)])
        Qv = torch.cat([enc(eq, test["query"][i:i + 64]) for i in range(0, len(test), 64)])
    gold = [corpus.index(x) for x in test["answer"]]
    out = {}
    for name, S in (("DPR", (Qv @ P.T).cpu().numpy()), ("BM25", np.stack([bm.scores(q.lower()) for q in test["query"]]))):
        order = np.argsort(-S, 1)
        out[name] = {k: float(np.mean([g in order[i, :k] for i, g in enumerate(gold)])) for k in (1, 5, 20, 100)}
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
    a.people, a.filler, a.seeds, a.steps, a.epochs, a.sizes = 500, 1500, 3, 400, 40, (50, 100, 200, 500, 1120)
    a.bert, a.nq_train, a.nq_test, a.distract, a.nq_epochs, a.nq_batch = "bert-base-uncased", 20000, 1000, 9000, 2, 32
    if a.quick:
        a.people, a.filler, a.seeds, a.steps, a.epochs, a.sizes = 60, 100, 1, 4, 1, (50,)
        a.bert, a.nq_train, a.nq_test, a.distract, a.nq_epochs, a.nq_batch = "prajjwal1/bert-tiny", 8, 4, 8, 1, 4
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                try:
                    R[name] = fn(a)
                except ImportError as e:
                    R[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(R, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
