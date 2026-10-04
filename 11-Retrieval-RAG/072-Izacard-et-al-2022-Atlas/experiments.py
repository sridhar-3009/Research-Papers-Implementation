"""Atlas experiments (Izacard et al. 2022): toy versions of the paper's ablations, and a real few-shot run.

  E1  Table 1: retriever objective during pre-training -- closed-book / no joint pre-training / fixed / ADist / EMDR2 /
      PDist / LOOP -- from the idf retriever AND from the weak crop-trained retriever; masked-LM accuracy, recall and
      64-shot EM; 3 seeds.
  E2  Table 4: how to treat the retriever while fine-tuning on 64 examples -- fixed, query-side, full (both towers +
      re-indexing every 50 steps), and top-L re-ranking (retrieve L = 20 with the stale index, re-score with the
      current encoders, keep 5).
  E3  Few-shot curve: 16 / 64 / 256 / 1024 fine-tuning examples, Atlas vs closed-book.
  E4  Number of retrieved documents k in {1, 3, 5, 10, 20} for pre-training and fine-tuning (FiD cost is linear in k).
  E5  Index swap with partial change (10% / 50% / 100% of people), and the share of stale (old) answers.
  E6  PQ compression sweep: sub-vectors {2, 4, 8, 16, 32} x bits {2, 4, 6, 8}, recall@5 and 64-shot EM.
  E7  Real retrieval-augmented few-shot QA: Contriever (`facebook/contriever`) over a Wikipedia slice
      (`wikipedia` 20220301.simple) + a FiD-style reader built from `google/flan-t5-base` (encode each passage with the
      question, concatenate encoder states), 64 NQ-open training examples, PDist on the query encoder.

!! HEAVY for E7 (GPU, model + Wikipedia downloads); E1-E6 take minutes to an hour on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import atlas as A

HERE = Path(__file__).parent


def setup(a, seed=0):
    W = A.make_world(n_people=a.people, n_filler=a.filler, seed=seed)
    P = W["people"]
    V = A.Vocab(W["passages"] + [q for s in ("overlap", "paraphrase") for q, *_ in A.qa_examples(W, P, s)])
    rng = random.Random(seed)
    n_ft = len(P) // 3
    pool = A.qa_examples(W, P[:n_ft], "overlap") + A.qa_examples(W, P[:n_ft], "paraphrase")
    rng.shuffle(pool)
    test = A.qa_examples(W, P[2 * len(P) // 3:], "overlap") + A.qa_examples(W, P[2 * len(P) // 3:], "paraphrase")
    return W, V, A.mlm_examples(W), pool, test


def pretrain(a, W, V, mlm, start, loss, closed=False, k=5):
    r, rd = copy.deepcopy(start), A.FiDReader(V)
    ix = A.Index(r, W["passages"])
    train_atlas_kw = dict(steps=a.pre_steps, loss=loss, train_q=loss is not None, train_d=loss is not None,
                          reindex_every=a.reindex if loss else 0, closed_book=closed, k=k)
    A.train_atlas(r, rd, ix, mlm, **train_atlas_kw)
    return r, rd, ix


def e1(a):
    out = []
    for seed in range(a.seeds):
        W, V, mlm, pool, test = setup(a, seed)
        held = mlm[::8]
        idf = A.Retriever(V, seed=seed).init_idf(W["passages"])
        weak = A.contriever_pretrain(A.Retriever(V, seed=seed).init_idf(W["passages"]), W["passages"], steps=a.crop_steps)
        for start_name, start in (("idf", idf), ("crop-trained", weak)):
            for name, loss, closed, pre in (("closed-book", None, True, True), ("no joint pre-training", None, False, False),
                                            ("fixed", None, False, True), ("adist", "adist", False, True),
                                            ("emdr", "emdr", False, True), ("pdist", "pdist", False, True),
                                            ("loop", "loop", False, True)):
                if pre:
                    r, rd, ix = pretrain(a, W, V, mlm, start, loss, closed)
                else:
                    r, rd = copy.deepcopy(start), A.FiDReader(V)
                    ix = A.Index(r, W["passages"])
                row = {"seed": seed, "start": start_name, "model": name,
                       "mlm acc": A.evaluate(r, rd, ix, held, closed_book=closed) if pre else None,
                       "mlm recall@5": A.recall(r, ix, held, W)}
                A.train_atlas(r, rd, ix, pool[:64], steps=a.ft_steps, loss=None if closed else "pdist", closed_book=closed)
                row["64-shot EM"] = A.evaluate(r, rd, ix, test, closed_book=closed)
                out.append(row)
    return out


def rerank_topk(ret, index, queries, k, L):
    """Re-ranking (Section 2.4): retrieve L with the stale index, re-embed those L documents with the current
    document encoder, keep the best k."""
    stale = index.topk(ret, queries, L)
    out = []
    with torch.no_grad():
        for q, row in zip(queries, stale.tolist()):
            s = ret.q([q]) @ ret.d([index.passages[i] for i in row]).T
            out.append([row[j] for j in s[0].topk(k).indices.tolist()])
    return torch.tensor(out)


def e2(a):
    W, V, mlm, pool, test = setup(a)
    start = A.Retriever(V).init_idf(W["passages"])
    r0, rd0, _ = pretrain(a, W, V, mlm, start, "pdist")
    out = {}
    for name, kw in (("fixed retriever", dict(loss=None)), ("query-side", dict(train_q=True)),
                     ("full + re-index every 50", dict(train_q=True, train_d=True, reindex_every=50)),
                     ("full, stale index", dict(train_q=True, train_d=True))):
        r, rd = copy.deepcopy(r0), copy.deepcopy(rd0)
        ix = A.Index(r, W["passages"])
        A.train_atlas(r, rd, ix, pool[:64], steps=a.ft_steps, **kw)
        out[name] = A.evaluate(r, rd, ix, test)
        if name == "full, stale index":                                  # same model, re-rank top-20 instead
            idx = rerank_topk(r, ix, [x[0] for x in test], 5, 20)
            docs = [[ix.passages[i] for i in row] for row in idx.tolist()]
            with torch.no_grad():
                pred = rd([x[0] for x in test], docs)[0].argmax(-1)
            inv = {i: w for w, i in V.ix.items()}
            out["full, top-20 re-ranking"] = float(np.mean([inv[int(p)] == x[1] for p, x in zip(pred, test)]))
    return out


def e3(a):
    W, V, mlm, pool, test = setup(a)
    start = A.Retriever(V).init_idf(W["passages"])
    out = {}
    for closed in (False, True):
        r0, rd0, _ = pretrain(a, W, V, mlm, start, None if closed else "pdist", closed)
        for n in a.shots:
            r, rd = copy.deepcopy(r0), copy.deepcopy(rd0)
            ix = A.Index(r, W["passages"])
            A.train_atlas(r, rd, ix, pool[:n], steps=a.ft_steps, loss=None if closed else "pdist", closed_book=closed)
            out[f"{'closed-book' if closed else 'atlas'} {n}-shot"] = A.evaluate(r, rd, ix, test, closed_book=closed)
    return out


def e4(a):
    W, V, mlm, pool, test = setup(a)
    start = A.Retriever(V).init_idf(W["passages"])
    out = {}
    for k in a.ks:
        t = time.time()
        r, rd, ix = pretrain(a, W, V, mlm, start, "pdist", k=k)
        A.train_atlas(r, rd, ix, pool[:64], steps=a.ft_steps, k=k)
        out[f"k={k}"] = {"64-shot EM": A.evaluate(r, rd, ix, test, k=k), "seconds": time.time() - t}
    return out


def e5(a):
    W, V, mlm, pool, test = setup(a)
    r, rd, ix = pretrain(a, W, V, mlm, A.Retriever(V).init_idf(W["passages"]), "pdist")
    A.train_atlas(r, rd, ix, pool[:64], steps=a.ft_steps)
    out = {}
    for frac in (0.1, 0.5, 1.0):
        rng = random.Random(1)
        W2 = copy.deepcopy(W)
        changed = []
        for p in W["people"]:
            if rng.random() < frac:
                changed.append(p)
                for rel in ("job", "instrument", "team"):
                    W2["facts"][p][rel] = rng.choice([v for v in A.dpr.VALUES[rel] if v != W["facts"][p][rel]])
                    for i, t in zip(W2["gold"][(p, rel)], A.dpr.PASSAGE_TEMPLATES[rel]):
                        W2["passages"][i] = t.format(e=p, v=W2["facts"][p][rel])
        asked = [(A.dpr.OVERLAP_Q[rel].format(e=p), p, rel) for p in changed for rel in ("job", "instrument", "team")]
        row = {}
        for iname, iw in (("old", W), ("new", W2)):
            ixs = A.Index(r, iw["passages"])
            for aname, aw in (("old", W), ("new", W2)):
                row[f"{iname} index / {aname} answers"] = A.evaluate(r, rd, ixs, [(q, aw["facts"][p][rel], -1) for q, p, rel in asked])
        out[f"{frac:.0%} changed"] = row
    return out


def e6(a):
    W, V, mlm, pool, test = setup(a)
    r, rd, ix = pretrain(a, W, V, mlm, A.Retriever(V).init_idf(W["passages"]), "pdist")
    A.train_atlas(r, rd, ix, pool[:64], steps=a.ft_steps)
    D = ix.D.clone()
    out = {"float32": {"recall": A.recall(r, ix, mlm[::8], W), "EM": A.evaluate(r, rd, ix, test)}}
    for m in a.subvectors:
        for bits in (2, 4, 6, 8):
            ix.D, nbytes = A.product_quantize(D, m, bits)
            out[f"m={m}, {bits} bits ({nbytes:g} B)"] = {"recall": A.recall(r, ix, mlm[::8], W), "EM": A.evaluate(r, rd, ix, test)}
    ix.D = D
    return out


def e7(a):
    from datasets import load_dataset
    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ctok, cq = AutoTokenizer.from_pretrained("facebook/contriever"), AutoModel.from_pretrained("facebook/contriever").to(dev)
    cd = AutoModel.from_pretrained("facebook/contriever").to(dev).eval()
    rtok, reader = AutoTokenizer.from_pretrained(a.reader), AutoModelForSeq2SeqLM.from_pretrained(a.reader).to(dev)
    wiki = load_dataset("wikipedia", "20220301.simple", split=f"train[:{a.wiki_articles}]")
    passages = []
    for art in wiki:
        w = art["text"].split()
        passages += [" ".join(w[i:i + 100]) for i in range(0, min(len(w), 1000), 100)]
    nq_train = load_dataset("nq_open", split="train").shuffle(seed=0).select(range(64))
    nq_test = load_dataset("nq_open", split=f"validation[:{a.nq_test}]")

    def emb(model, texts):
        x = ctok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        h = model(**x).last_hidden_state
        m = x["attention_mask"][..., None]
        return (h * m).sum(1) / m.sum(1)                                             # Contriever: mean pooling

    with torch.no_grad():
        D = torch.cat([emb(cd, passages[i:i + 64]) for i in range(0, len(passages), 64)])

    def fid_logp(q, docs, answer):
        """FiD: encode 'question: q context: d' per passage, concatenate encoder states, decode the answer."""
        x = rtok([f"question: {q} context: {d}" for d in docs], padding=True, truncation=True, max_length=256,
                 return_tensors="pt").to(dev)
        enc = reader.get_encoder()(**x).last_hidden_state.reshape(1, -1, reader.config.d_model)
        mask = x["attention_mask"].reshape(1, -1)
        y = rtok([answer], return_tensors="pt").input_ids.to(dev)
        out = reader(encoder_outputs=(enc,), attention_mask=mask, labels=y)
        return -out.loss * y.shape[1]

    opt = torch.optim.Adam(list(reader.parameters()) + list(cq.parameters()), lr=a.lr)
    for _ in range(a.epochs):
        for ex in nq_train:
            qv = emb(cq, [ex["question"]])
            top = (qv @ D.T)[0].topk(a.k)
            docs = [passages[i] for i in top.indices.tolist()]
            lm = fid_logp(ex["question"], docs, ex["answer"][0])
            with torch.no_grad():                                                     # PDist target
                per = torch.stack([fid_logp(ex["question"], [d], ex["answer"][0]) for d in docs])
            target = torch.softmax(per, 0)
            log_pr = torch.log_softmax(top.values, 0)
            loss = -lm + (target * (target.clamp(min=1e-9).log() - log_pr)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    hits = []
    with torch.no_grad():
        for ex in nq_test:
            top = (emb(cq, [ex["question"]]) @ D.T)[0].topk(a.k).indices.tolist()
            x = rtok([f"question: {ex['question']} context: {passages[i]}" for i in top], padding=True, truncation=True,
                     max_length=256, return_tensors="pt").to(dev)
            enc = reader.get_encoder()(**x).last_hidden_state.reshape(1, -1, reader.config.d_model)
            g = reader.generate(encoder_outputs=type("O", (), {"last_hidden_state": enc})(),
                                attention_mask=x["attention_mask"].reshape(1, -1), max_new_tokens=16)
            pred = rtok.decode(g[0], skip_special_tokens=True).strip().lower()
            hits.append(any(pred == ans.lower() for ans in ex["answer"]))
    return {"64-shot NQ EM (small Wikipedia slice; far below the paper)": float(np.mean(hits))}


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 8)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.people, a.filler, a.seeds, a.pre_steps, a.ft_steps, a.reindex, a.crop_steps = 300, 900, 3, 800, 300, 200, 300
    a.shots, a.ks, a.subvectors = (16, 64, 256, 1024), (1, 3, 5, 10, 20), (2, 4, 8, 16, 32)
    a.reader, a.wiki_articles, a.nq_test, a.k, a.lr, a.epochs = "google/flan-t5-base", 20000, 500, 10, 1e-5, 2
    if a.quick:
        a.people, a.filler, a.seeds, a.pre_steps, a.ft_steps, a.reindex, a.crop_steps = 24, 24, 1, 3, 3, 2, 2
        a.shots, a.ks, a.subvectors = (16,), (1,), (4,)
        a.reader, a.wiki_articles, a.nq_test, a.k, a.epochs = "google/flan-t5-small", 5, 2, 2, 1
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6), ("e7", e7)):
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
