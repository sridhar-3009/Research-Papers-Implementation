"""Atlas in ~30 seconds: an unsupervised retriever, a Fusion-in-Decoder reader, joint masked-LM pre-training, 64-shot
question answering against a closed-book model and a model without joint pre-training, the reader teaching a weak
retriever (PDist / EMDR2) and the retriever targets on one example, index swapping, and product-quantised index
compression."""

import copy
import random
import time

import numpy as np

from atlas import (REPORTED, FiDReader, Index, Retriever, Vocab, answer_logp, contriever_pretrain, dpr, evaluate,
                   make_world, mlm_examples, product_quantize, qa_examples, recall, retriever_target, train_atlas)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


W = make_world()
P = W["people"]
V = Vocab(W["passages"] + [q for s in ("overlap", "paraphrase") for q, *_ in qa_examples(W, P, s)])
mlm = mlm_examples(W)
held = mlm[::8]
rng = random.Random(0)
few = qa_examples(W, P[:100], "overlap") + qa_examples(W, P[:100], "paraphrase")
rng.shuffle(few)
few = few[:64]
test_o, test_p = qa_examples(W, P[200:], "overlap"), qa_examples(W, P[200:], "paraphrase")

section("1. The corpus and the unsupervised retriever")
print(f"  {len(W['passages'])} passages: 300 people x 4 facts, each fact written TWICE (two templates), + 900 filler")
print(f"  masked-LM pre-training example: '{mlm[0][0]}' -> '{mlm[0][1]}'")
print("    (the passage itself is excluded from retrieval; the answer must come from ANOTHER passage)")
r0 = Retriever(V).init_idf(W["passages"])
ix0 = Index(r0, W["passages"])
print(f"  unsupervised retriever (random embeddings pooled with idf weights): masked-LM recall@5 {recall(r0, ix0, held, W):.0%}, "
      f"QA overlap {recall(r0, ix0, test_o, W):.0%}, paraphrase {recall(r0, ix0, test_p, W):.0%}")
weak = Retriever(V).init_idf(W["passages"])
contriever_pretrain(weak, W["passages"])
print(f"  (honest note: our Contriever-style training on random crops made it WORSE -- masked-LM recall@5 "
      f"{recall(weak, Index(weak, W['passages']), held, W):.0%} -- so we start from the idf retriever; section 3 re-uses the")
print("   weakened one to show the reader teaching a retriever)")

section("2. Joint pre-training, then 64-shot question answering (Tables 1 and 8)")
print("  pre-training: 800 steps of masked LM, k = 5; fine-tuning: 64 questions (half paraphrased), 300 steps with")
print("  query-side fine-tuning and PDist (for every row); test: 800 questions about 100 people never used in fine-tuning")
print("    model                                          masked-LM acc   recall@5 | 64-shot EM overlap   paraphrase")
results = {}
for name, pre_loss, closed, pretrain in (("closed-book (same pre-training, no retrieval)", None, True, True),
                                         ("retrieval, NO joint pre-training", None, False, False),
                                         ("joint pre-training, fixed retriever", None, False, True),
                                         ("joint pre-training, PDist retriever", "pdist", False, True)):
    r, rd = copy.deepcopy(r0), FiDReader(V)
    ix = Index(r, W["passages"])
    if pretrain:
        train_atlas(r, rd, ix, mlm, steps=800, loss=pre_loss, train_q=pre_loss is not None, closed_book=closed)
    pre_acc = f"{evaluate(r, rd, ix, held, closed_book=closed):5.2f}" if pretrain else "  -  "
    pre_rec = recall(r, ix, held, W)
    train_atlas(r, rd, ix, few, steps=300, loss=None if closed else "pdist", train_q=True, closed_book=closed)
    eo, ep = evaluate(r, rd, ix, test_o, closed_book=closed), evaluate(r, rd, ix, test_p, closed_book=closed)
    results[name] = (r, rd, ix, eo, ep)
    print(f"    {name:46s}   {pre_acc}        {'  -  ' if closed else f'{pre_rec:4.2f}'}   |     {eo:5.1%}         {ep:5.1%}")
cb, nj = results["closed-book (same pre-training, no retrieval)"], results["retrieval, NO joint pre-training"]
fx, pd = results["joint pre-training, fixed retriever"], results["joint pre-training, PDist retriever"]
print(f"  -> joint pre-training is what makes few-shot work: {fx[3]:.0%} / {pd[3]:.0%} vs {nj[3]:.0%} without it and {cb[3]:.0%}"
      " closed-book (overlap")
print("     questions); the closed-book model can only use the facts it memorised during pre-training.")
print(f"  -> PDist vs fixed retriever: {pd[3]:.1%} vs {fx[3]:.1%} (overlap) -- a small difference from one seed; this retriever")
print("     already finds the right passage for most masked-LM queries, so there is little left for it to learn.")

section("3. The reader teaching a WEAK retriever (no document labels anywhere)")
print(f"    start (crop-trained retriever):  masked-LM recall@5 {recall(weak, Index(weak, W['passages']), held, W):.0%}")
weak_rec = {}
for loss in ("pdist", "emdr"):
    r, rd = copy.deepcopy(weak), FiDReader(V)
    ix = Index(r, W["passages"])
    train_atlas(r, rd, ix, mlm, steps=600, loss=loss)
    weak_rec[loss] = recall(r, ix, held, W)
    print(f"    after 600 joint steps with {loss.upper():5s}: masked-LM recall@5 {weak_rec[loss]:.0%}, accuracy {evaluate(r, rd, ix, held):.2f}")
print(f"  -> with EMDR2 the reader pulls the useful passages up the ranking ({weak_rec['emdr']:.0%} recall, from 48%), the paper's")
print(f"     'strong improvements' on the pre-training metric. Honest note: PDist did NOT help here ({weak_rec['pdist']:.0%}): from a weak")
print("     start the untrained reader's per-document perplexities are nearly flat, so its target carries little signal.")
r, rd, ix = fx[:3]
q, a, ex = mlm[0]
docs = [[W["passages"][i] for i in ix.topk(r, [q], 5, exclude=[ex])[0].tolist()]]
_, alpha, vnorm = answer_logp(rd, [q], docs, [a])
print(f"  the four targets for '{q}' -> '{a}', using the trained reader (docs in retrieval order):")
for i, dtxt in enumerate(docs[0]):
    print(f"      doc {i + 1}: {dtxt}")
for name in ("adist", "pdist", "loop"):
    print(f"    {name:6s} {np.round(retriever_target(name, rd, [q], docs, [a], alpha, vnorm)[0].numpy(), 3)}")
print("    (EMDR2 is not a target distribution: it maximises log sum_k p_LM(a|d_k) p_retr(d_k) directly)")
print("  -> PDist favours the two passages that mention hanoi; LOOP is nearly flat (dropping one of five documents barely")
print("     changes the answer) apart from down-weighting the delhi filler; our one-head attention (ADist) points at that")
print("     delhi filler -- in our toy ADist was the unreliable target (experiments.py E1 compares all four).")

section("4. Updating knowledge by swapping the index (TempLAMA, Table 11)")
r, rd, ix = pd[:3]
rng2 = random.Random(5)
W2 = copy.deepcopy(W)
for p in P:
    for rel in ("job", "instrument", "team"):
        W2["facts"][p][rel] = rng2.choice([v for v in dpr.VALUES[rel] if v != W["facts"][p][rel]])
        for i, t in zip(W2["gold"][(p, rel)], dpr.PASSAGE_TEMPLATES[rel]):
            W2["passages"][i] = t.format(e=p, v=W2["facts"][p][rel])
asked = [(dpr.OVERLAP_Q[rel].format(e=p), p, rel) for p in P[200:] for rel in ("job", "instrument", "team")]
out = {}
for iname, iw in (("old", W), ("new", W2)):
    ix_s = Index(r, iw["passages"])
    for aname, aw in (("old", W), ("new", W2)):
        out[(iname, aname)] = evaluate(r, rd, ix_s, [(q, aw["facts"][p][rel], -1) for q, p, rel in asked])
print("  everyone changes job, instrument and team; same model, index rebuilt from the new passages:")
print(f"    old index: old answers {out[('old', 'old')]:.1%}, new answers {out[('old', 'new')]:.1%}   |   "
      f"new index: old answers {out[('new', 'old')]:.1%}, new answers {out[('new', 'new')]:.1%}")
print("  -> the new index moves the answers toward the new facts, but only partly: with the new index the model still")
print(f"     gives the OLD answer {out[('new', 'old')]:.0%} of the time -- our reader memorised facts during pre-training and")
print("     sometimes trusts that memory over the passage. (The paper: Atlas 57.7% on 2017 questions with a 2017 index,")
print("     53.1% on 2020 answers after swapping to a 2020 index; closed-book T5 12.1% and no way to update.)")

section("5. Compressing the index with product quantisation (Figure 4)")
D = ix.D.clone()
base = recall(r, ix, held, W)
print(f"    float32, 256 bytes/vector: recall@5 {base:.0%}")
for m, bits in ((16, 8), (8, 8), (4, 8), (4, 4)):
    recon, nbytes = product_quantize(D, m, bits)
    ix.D = recon
    print(f"    PQ {m} sub-vectors x {bits} bits = {nbytes:4.1f} bytes/vector ({256 / nbytes:4.0f}x smaller): "
          f"recall@5 {recall(r, ix, held, W):.0%}")
ix.D = D
print("  -> 16x compression costs almost nothing; squeezing harder loses recall fast (the paper: 49 GB -> 4 GB, about")
print("     12x, with a negligible drop).")

section("6. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
