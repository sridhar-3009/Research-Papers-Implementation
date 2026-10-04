"""RAG in ~20 seconds: the two marginalisations on numbers, open-domain QA on the toy Wikipedia of paper 070
(closed-book vs RAG with a frozen / learned / BM25 retriever), test-time k, index hot-swapping, and a two-fact task
where RAG-Token can combine documents and RAG-Sequence cannot."""

import copy
import time

import numpy as np
import torch

from rag import (REPORTED, BM25Retriever, DenseRetriever, FixedReader, Generator, changed_world, describe_questions,
                 dpr, exact_match, make_vocab, marginal_logp, pretrain_dpr, retrieval_recall, token_posteriors,
                 train_rag)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The two marginalisations on one example")
pz = np.array([0.7, 0.3])                                                   # p(z|x) for 2 retrieved documents
py = np.array([[0.9, 0.1], [0.2, 0.8]])                                     # p(y_i | x, z): rows = docs, cols = tokens
seq = (pz * py.prod(1)).sum()
tok = (pz[:, None] * py).sum(0).prod()
print("  p(z|x) = [0.7, 0.3]; doc 1 supports token 1 (0.9) but not token 2 (0.1); doc 2 the reverse (0.2, 0.8)")
print(f"  RAG-Sequence: sum_z p(z) prod_i p(y_i|z) = 0.7*0.9*0.1 + 0.3*0.2*0.8 = {seq:.3f}")
print(f"  RAG-Token:    prod_i sum_z p(z) p(y_i|z) = (0.7*0.9+0.3*0.2) * (0.7*0.1+0.3*0.8) = {tok:.3f}")
print("  -> RAG-Token can take each token from a different document; RAG-Sequence needs ONE document that supports all.")

section("2. Open-domain QA on the toy Wikipedia (Tables 1 and 6)")
W = dpr.make_world()
Q = dpr.make_questions(W)
para, over = Q["test seen people, paraphrase"], Q["test seen people, overlap"]
d = pretrain_dpr(W, Q["train"])
V = make_vocab(W, Q["train"] + describe_questions(W, W["people"]))
print("  retriever: DPR pre-trained on OVERLAP-style questions only (the paper starts from DPR trained on NQ + TriviaQA);")
print("  generator: a tiny MLP over (document words, question words, previous token), trained from scratch (BART's role);")
print("  RAG fine-tuning: 1120 question/answer pairs, k = 5, loss -log p(answer | question) -- no document labels.")
print(f"  before fine-tuning, the retriever finds the gold passage in its top 5 for {retrieval_recall(DenseRetriever(d, W['passages']), para, W):.0%}"
      f" of paraphrased and {retrieval_recall(DenseRetriever(d, W['passages']), over, W):.0%} of overlap questions")
print("    model                                    EM paraphrase   EM overlap   gold passage in top-5 (para / overlap)")
rows = {}
for name, ret, learn in (("closed-book generator (no retrieval)", None, False),
                         ("RAG, BM25 retriever", BM25Retriever(W["passages"]), False),
                         ("RAG, frozen DPR retriever", DenseRetriever(d, W["passages"]), False),
                         ("RAG, retriever fine-tuned end to end", DenseRetriever(d, W["passages"]), True)):
    ret, g = train_rag(ret, Generator(V), Q["train"], train_retriever=learn)
    rows[name] = (ret, g, exact_match(ret, g, para), exact_match(ret, g, over))
    rec = f"{retrieval_recall(ret, para, W):5.1%} / {retrieval_recall(ret, over, W):5.1%}" if ret is not None else "   -"
    print(f"    {name:40s}   {rows[name][2]:6.1%}        {rows[name][3]:6.1%}      {rec}")
ret_l, g_l = rows["RAG, retriever fine-tuned end to end"][:2]
print("  -> closed-book can only memorise the TRAINING facts; held-out facts are guesses. Retrieval reads them from the")
print("     index. Fine-tuning the question encoder through the generator's loss alone (no document labels) teaches")
print("     the retriever the paraphrases: the paper's 'learned retrieval improves results for all tasks' (except BM25 on")
print("     FEVER). Note: our from-scratch generator needs 200 warm-up steps before the retriever may learn -- without")
print("     them its noisy gradient wrecked the retriever.")
print(f"  closed-book on 400 of its own training questions: {exact_match(None, rows['closed-book generator (no retrieval)'][1], Q['train'][:400]):.0%} "
      "(parametric memory: what it memorised)")

section("3. Number of documents at test time (Figure 3, left)")
print("    k            " + "   ".join(f"{k:5d}" for k in (1, 2, 5, 10)))
print("    EM para.     " + "   ".join(f"{exact_match(ret_l, g_l, para, k=k):5.1%}" for k in (1, 2, 5, 10)))
print("  -> honest note: here (RAG-Token, trained with k = 5) one document is best and extra documents only add noise for")
print("     our tiny generator; the paper found RAG-Sequence improves monotonically with k and RAG-Token peaks at k = 10.")

section("4. Index hot-swapping: update the world's knowledge without retraining")
W2 = changed_world(W)
asked = [(dpr.OVERLAP_Q[r].format(e=p), p, r) for p in W["people"][:400] for r in ("job", "instrument", "team")]
ret_h, g_h = copy.deepcopy(ret_l), g_l
res = {}
for idx_name, idx_world in (("old", W), ("new", W2)):
    ret_h.swap_index(d, idx_world["passages"])
    for ans_name, ans_world in (("old", W), ("new", W2)):
        res[(idx_name, ans_name)] = exact_match(ret_h, g_h, [(q, ans_world["facts"][p][r], p, r) for q, p, r in asked])
print("  every person changes job, instrument and team; we rebuild the index from the new passages with the SAME frozen")
print("  document encoder and do not touch the model")
print(f"    old index, old facts: {res[('old', 'old')]:.1%}     new index, new facts: {res[('new', 'new')]:.1%}")
print(f"    old index, new facts: {res[('old', 'new')]:.1%}      new index, old facts: {res[('new', 'old')]:.1%}")
print("  -> swapping the index updates the answers (the paper: 70% / 68% matched vs 12% / 4% mismatched).")

section("5. Two facts from two documents: RAG-Token vs RAG-Sequence ('where was X born and what does X work as?')")
dte = describe_questions(W, W["people"][300:400])
ret_f, reader = DenseRetriever(d, W["passages"]), FixedReader(V)
print("  to isolate the marginalisation we use a FIXED reader: if a document states the needed fact it gives that value")
print("  probability 0.9, otherwise it guesses uniformly. Both gold passages are in the top 5 for "
      f"{retrieval_recall(ret_f, dte, W):.0%} of the 100 questions.")
for label, kw in (("RAG-Sequence, thorough decoding", dict(mode="sequence", thorough=True)),
                  ("RAG-Sequence, fast decoding", dict(mode="sequence", thorough=False)),
                  ("RAG-Token", dict(mode="token"))):
    print(f"    {label:32s} exact match {exact_match(ret_f, reader, dte, length=2, **kw):5.1%}")
qs_, as_ = [q for q, *_ in dte], [a for _, a, *_ in dte]
print(f"  mean p(correct answer): RAG-Sequence {marginal_logp(ret_f, reader, qs_, as_, 5, 'sequence').exp().mean():.3f}, "
      f"RAG-Token {marginal_logp(ret_f, reader, qs_, as_, 5, 'token').exp().mean():.3f}")
docs, post = token_posteriors(ret_f, reader, dte[0][0], dte[0][1])
with torch.no_grad():
    prior = torch.softmax(ret_f.topk([dte[0][0]], 5)[1], -1)[0].numpy()
print(f"  Figure 2 analogue -- '{dte[0][0]}' -> '{dte[0][1]}':")
print("    document                                          prior p(z|x)   posterior for token 1   for token 2")
for j, doc in enumerate(docs):
    print(f"    {doc[:48]:48s}   {prior[j]:8.3f}        {post[0, j]:8.3f}            {post[1, j]:8.3f}")
print("  -> each answer token can lean on the document that supports it (the posterior on the job passage rises")
print("     well above its prior for token 2); RAG-Sequence needs a single document with both facts, and none exists.")
print("  (with a TRAINED tiny generator this task failed for both models: with 300 examples it memorised the training")
print("   answers instead of learning to read -- see experiments.py E4)")

section("6. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
