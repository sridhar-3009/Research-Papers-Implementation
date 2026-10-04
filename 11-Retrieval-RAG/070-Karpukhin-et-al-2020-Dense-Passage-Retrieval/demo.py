"""Dense Passage Retrieval in ~20 seconds: BM25 vs a trained dual encoder on a toy Wikipedia, with paraphrased vs
word-overlap questions, the training-scheme ablation (Table 3), sample efficiency (Figure 1), the BM25 + DPR hybrid,
end-to-end exact match with a simple reader, and search speed."""

import time

import numpy as np
import torch

from dpr import (BM25, REPORTED, DenseIndex, bm25_matrix, exact_match, hybrid, in_batch_loss, make_questions,
                 make_world, top_k_accuracy, train_dpr, tune_lambda)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


W = make_world()
Q = make_questions(W)
P = W["passages"]
bm = BM25(P)
tests = {k: v for k, v in Q.items() if k != "train"}

section("1. The toy Wikipedia")
print(f"  {len(P)} passages: 500 people x 4 facts, plus 1500 filler passages that mention people in passing, e.g.")
print(f"    '{P[0]}'   '{P[1]}'   '{P[-1]}'")
q0 = tests["test seen people, overlap"][0]
q1 = tests["test seen people, paraphrase"][0]
print(f"  overlap question:    '{q0[0]}'  (re-uses the passage's words, like SQuAD)")
print(f"  paraphrase question: '{q1[0]}'  (different words, like Natural Questions); answer: {q0[1]}")
print(f"  {len(Q['train'])} training questions about 400 'seen' people; tests: their held-out facts, and 100 people never")
print("  asked about in training")

section("2. In-batch negatives (Section 3.2)")
torch.manual_seed(0)
qv, pv, hv = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
print("  B = 4 questions, their 4 gold passages and 4 BM25 negatives: S = Q [P; P_bm25]^T is 4 x 8; row i's target is")
print(f"  column i, so each question gets 3 gold + 4 BM25 = 7 negatives for free; loss = {in_batch_loss(qv, pv, hv).item():.3f}"
      f" for random vectors")
print(f"  (if all 8 scores were equal it would be ln 8 = {np.log(8):.3f}; training pushes it toward 0)")

section("3. Retrieval accuracy: does a top-k passage contain the answer? (Table 2)")
model = train_dpr(W, Q["train"], bm25=bm)
idx = DenseIndex(model, P)
lam = tune_lambda(bm, idx, Q["train"][:300], P)
print(f"  hybrid weight lambda = {lam} (picked on 300 training questions, as the paper tunes it on dev sets)")
print("    test set                           BM25 top-1/top-5     DPR top-1/top-5     BM25+DPR top-1/top-5")
res = {}
for name, qs in tests.items():
    b, d = bm25_matrix(bm, qs), idx.scores([q for q, *_ in qs])
    res[name] = [top_k_accuracy(m, qs, P, (1, 5)) for m in (b, d, hybrid(b, d, lam))]
    print(f"    {name:34s} " + "     ".join(f"{r[1]:5.1%} / {r[5]:5.1%}" for r in res[name]))
sp, so = res["test seen people, paraphrase"], res["test seen people, overlap"]
up, uo = res["test unseen people, paraphrase"], res["test unseen people, overlap"]
print(f"  -> paraphrased questions: DPR {sp[1][1]:.0%} top-1 vs BM25 {sp[0][1]:.0%} (BM25 can only match the name, and that")
print("     name also appears in the person's other facts and in filler passages); dense vectors learned that 'hometown'")
print("     means 'born', 'profession' means 'works as'.")
print(f"  -> rare names: for people never seen in training, DPR drops to {uo[1][1]:.0%} on overlap questions where BM25 gets"
      f" {uo[0][1]:.0%}:")
print("     an exact rare word is BM25's strength (the paper's 'Thoros of Myr' example and its SQuAD result).")
print(f"  -> the BM25 + {lam} x DPR hybrid gets the best of both: {min(r[2][1] for r in res.values()):.0%} or more top-1 on every"
      " test set (in the paper the hybrid")
print("     helped on some datasets but not on NQ: 76.6 vs 78.4 top-20).")

section("4. Training schemes (Table 3): which negatives? top-5 on seen-people questions (both styles)")
seen = tests["test seen people, overlap"] + tests["test seen people, paraphrase"]
for label, kw in (("7 random negatives, no in-batch", dict(negatives="random", n_extra=7, batch=128)),
                  ("7 BM25 negatives, no in-batch", dict(negatives="bm25", n_extra=7, batch=128)),
                  ("in-batch golds, batch 8 (7 negatives)", dict(negatives="gold", batch=8)),
                  ("in-batch golds, batch 32 (31 negatives)", dict(negatives="gold", batch=32)),
                  ("in-batch golds, batch 128 (127 negatives)", dict(negatives="gold", batch=128)),
                  ("in-batch 128 + 1 BM25 each (the paper's)", dict(negatives="gold+bm25", batch=128))):
    m = train_dpr(W, Q["train"], bm25=bm, **kw)
    acc = top_k_accuracy(DenseIndex(m, P).scores([q for q, *_ in seen]), seen, P, (1, 5))
    print(f"    {label:44s} top-1 {acc[1]:5.1%}   top-5 {acc[5]:5.1%}")
print("  -> in-batch negatives work, and many beat few (batches of 32 and 128 beat batch 8). Honest differences from")
print("     the paper: (a) here batch 32 beats 128 -- every run gets 40 epochs, so small batches take 4x more steps;")
print("     (b) the extra BM25 negative does not help here, and BM25 negatives ALONE fail: in our toy the top BM25 passage")
print("     is usually another fact about the SAME person, so with only those negatives the name never has to be matched")
print("     against the rest of the corpus (in the paper BM25-only negatives were about as good as random ones).")

section("5. Sample efficiency (Figure 1): paraphrase questions about seen people, top-5")
bacc = top_k_accuracy(bm25_matrix(bm, tests["test seen people, paraphrase"]), tests["test seen people, paraphrase"], P, (5,))[5]
for n in (50, 200, 1120):
    m = train_dpr(W, Q["train"], n_train=n, bm25=bm, batch=min(128, n))
    qs = tests["test seen people, paraphrase"]
    a = top_k_accuracy(DenseIndex(m, P).scores([q for q, *_ in qs]), qs, P, (5,))[5]
    print(f"    DPR trained on {n:5d} questions: {a:5.1%}   (BM25 {bacc:.1%})")
print("  -> more questions help a lot; here DPR needs the full 1120 to pass BM25 (the paper: 1,000 NQ questions were already")
print("     enough -- its encoders start from pre-trained BERT, ours from random embeddings that merely match each other).")

section("6. End-to-end QA: retrieve top-k, then a reader extracts the answer (exact match)")
for name in ("test seen people, paraphrase", "test unseen people, overlap"):
    qs = tests[name]
    b, d = bm25_matrix(bm, qs), idx.scores([q for q, *_ in qs])
    print(f"    {name:32s} " + "   ".join(f"{lab} k={k}: {exact_match(m, qs, P, k):5.1%}"
                                          for lab, m in (("BM25", b), ("DPR", d), ("hybrid", hybrid(b, d, lam))) for k in (1, 5)))
print("  (the reader is rule-based, a stand-in for the paper's BERT reader: it can only answer from what was retrieved)")

section("7. Search speed")
qs = [q for q, *_ in seen]
t = time.time()
_ = bm25_matrix(bm, seen)
tb = time.time() - t
t = time.time()
_ = idx.scores(qs)
td = time.time() - t
print(f"  {len(qs)} questions over {len(P)} passages: BM25 (Python inverted index) {len(qs) / tb:,.0f} q/s, dense (one matrix"
      f" product over pre-computed passage vectors) {len(qs) / td:,.0f} q/s")
print("  (the paper: 995 q/s with FAISS vs 23.7 for Lucene per thread -- but encoding + indexing 21M passages took hours)")

section("8. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
