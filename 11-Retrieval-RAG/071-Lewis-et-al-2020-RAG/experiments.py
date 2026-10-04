"""RAG experiments (Lewis et al. 2020): toy sweeps of the paper's ablations, and a real RAG model from Hugging Face.

  E1  Table 6 on the toy, several seeds: closed-book / BM25 / frozen DPR / learned DPR, RAG-Sequence and RAG-Token
      training, both decoding schemes.
  E2  Figure 3: test-time k in {1, 2, 3, 5, 10, 20} for models trained with k = 5 and k = 10, both marginalisations.
  E3  Warm-up and retriever learning rate: how fragile is end-to-end retriever learning with a from-scratch generator?
  E4  The two-fact task with a TRAINED generator: a bigger world (2000 people) so the generator has enough examples
      to learn to read instead of memorising; generator first trained on single-fact QA (the 'pre-trained BART' role),
      then on 'describe' questions with RAG-Sequence vs RAG-Token.
  E5  Hot-swap with partial change (10% ... 100% of people change), matched vs mismatched index.
  E6  Real RAG: `facebook/rag-sequence-nq` and `facebook/rag-token-nq` with the dummy Wikipedia index from Hugging
      Face `transformers` / `datasets`, exact match on a slice of Natural Questions.

!! HEAVY for E6 (model + index download, ~GBs); E1-E5 take minutes on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import rag as R

HERE = Path(__file__).parent
dpr = R.dpr


def setup(a, seed=0, n_people=None):
    W = dpr.make_world(n_people=n_people or a.people, n_filler=a.filler, seed=seed)
    Q = dpr.make_questions(W, seed=seed)
    d = R.pretrain_dpr(W, Q["train"], epochs=a.dpr_epochs)
    V = R.make_vocab(W, Q["train"] + R.describe_questions(W, W["people"]))
    return W, Q, d, V


def e1(a):
    out = []
    for seed in range(a.seeds):
        W, Q, d, V = setup(a, seed)
        tests = {k: Q[k] for k in ("test seen people, paraphrase", "test seen people, overlap")}
        for name, mk, learn in (("closed-book", lambda: None, False),
                                ("BM25", lambda: R.BM25Retriever(W["passages"]), False),
                                ("frozen DPR", lambda: R.DenseRetriever(d, W["passages"]), False),
                                ("learned DPR", lambda: R.DenseRetriever(d, W["passages"]), True)):
            for mode in ("sequence", "token"):
                torch.manual_seed(seed)
                ret, g = R.train_rag(mk(), R.Generator(V, seed=seed), Q["train"], mode=mode, steps=a.steps,
                                     train_retriever=learn, seed=seed)
                row = {"seed": seed, "retriever": name, "train mode": mode}
                for tn, qs in tests.items():
                    for dm, th in (("token", True), ("sequence", True), ("sequence", False)):
                        row[f"{tn} | decode {dm}{'' if th else ' fast'}"] = R.exact_match(ret, g, qs, mode=dm, thorough=th)
                out.append(row)
    return out


def e2(a):
    W, Q, d, V = setup(a)
    qs = Q["test seen people, paraphrase"]
    out = {}
    for k_train in (5, 10):
        for mode in ("sequence", "token"):
            ret, g = R.train_rag(R.DenseRetriever(d, W["passages"]), R.Generator(V), Q["train"], k=k_train, mode=mode,
                                 steps=a.steps)
            out[f"train k={k_train}, {mode}"] = {k: R.exact_match(ret, g, qs, k=k, mode=mode) for k in a.test_ks}
    return out


def e3(a):
    W, Q, d, V = setup(a)
    qs = Q["test seen people, paraphrase"]
    out = {}
    for warm in (0, 100, 200, 400):
        for rlr in (3e-3, 1e-3, 3e-4):
            ret, g = R.train_rag(R.DenseRetriever(d, W["passages"]), R.Generator(V), Q["train"], steps=a.steps,
                                 warmup=warm, retriever_lr=rlr)
            out[f"warmup {warm}, retriever lr {rlr}"] = {"EM": R.exact_match(ret, g, qs),
                                                         "recall@5": R.retrieval_recall(ret, qs, W)}
    return out


def e4(a):
    W, Q, d, V = setup(a, n_people=a.big_people)
    P = W["people"]
    n_seen = int(len(P) * 0.8)
    split = int(n_seen * 0.75)
    test_people = set(P[split:n_seen])
    qa = [x for x in Q["train"] if x[2] not in test_people]                # generator never sees test people's facts
    dtr, dte = R.describe_questions(W, P[:split]), R.describe_questions(W, P[split:n_seen])
    ret0, g0 = R.train_rag(R.DenseRetriever(d, W["passages"]), R.Generator(V), qa, steps=a.steps, train_retriever=False)
    out = {"recall (both passages in top 5)": R.retrieval_recall(ret0, dte, W)}
    for mode in ("sequence", "token"):
        ret, g = R.train_rag(copy.deepcopy(ret0), copy.deepcopy(g0), dtr, mode=mode, steps=a.steps, warmup=0,
                             train_retriever=False)
        out[f"train {mode}"] = {f"decode {dm}{'' if th else ' fast'}": R.exact_match(ret, g, dte, mode=dm, length=2, thorough=th)
                                for dm, th in (("token", True), ("sequence", True), ("sequence", False))}
        out[f"train {mode}"]["train EM"] = R.exact_match(ret, g, dtr[:200], mode=mode, length=2)
    return out


def e5(a):
    W, Q, d, V = setup(a)
    ret, g = R.train_rag(R.DenseRetriever(d, W["passages"]), R.Generator(V), Q["train"], steps=a.steps)
    out = {}
    for frac in (0.1, 0.5, 1.0):
        W2 = R.changed_world(W, frac=frac)
        changed = [p for p in W["people"][:400] if W2["facts"][p]["job"] != W["facts"][p]["job"]]
        asked = [(dpr.OVERLAP_Q[r].format(e=p), p, r) for p in changed for r in ("job", "instrument", "team")]
        row = {}
        for iname, iw in (("old", W), ("new", W2)):
            ret.swap_index(d, iw["passages"])
            for aname, aw in (("old", W), ("new", W2)):
                row[f"{iname} index / {aname} facts"] = R.exact_match(ret, g, [(q, aw["facts"][p][r], p, r) for q, p, r in asked])
        out[f"{frac:.0%} changed"] = row
        ret.swap_index(d, W["passages"])
    return out


def e6(a):
    from datasets import load_dataset
    from transformers import RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer
    nq = load_dataset("nq_open", split=f"validation[:{a.nq}]")
    out = {}
    for name, cls in (("facebook/rag-sequence-nq", RagSequenceForGeneration), ("facebook/rag-token-nq", RagTokenForGeneration)):
        tok = RagTokenizer.from_pretrained(name)
        retr = RagRetriever.from_pretrained(name, index_name="exact", use_dummy_dataset=True)
        model = cls.from_pretrained(name, retriever=retr).eval()
        hits = []
        for ex in nq:
            x = tok(ex["question"], return_tensors="pt")
            with torch.no_grad():
                g = model.generate(input_ids=x["input_ids"], n_docs=a.n_docs)
            pred = tok.batch_decode(g, skip_special_tokens=True)[0].strip().lower()
            hits.append(any(pred == ans.lower() for ans in ex["answer"]))
        out[name] = {"EM (dummy 10k-passage index, so far below the paper)": float(np.mean(hits))}
    return out


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 7)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.people, a.filler, a.dpr_epochs, a.seeds, a.steps = 500, 1500, 40, 3, 600
    a.test_ks, a.big_people, a.nq, a.n_docs = (1, 2, 3, 5, 10, 20), 2000, 500, 5
    if a.quick:
        a.people, a.filler, a.dpr_epochs, a.seeds, a.steps = 60, 60, 2, 1, 3
        a.test_ks, a.big_people, a.nq, a.n_docs = (1, 2), 80, 2, 2
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6)):
            if a.only in (None, name):
                try:
                    Rs[name] = fn(a)
                except ImportError as e:
                    Rs[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(Rs, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(Rs, a)


if __name__ == "__main__":
    main()
