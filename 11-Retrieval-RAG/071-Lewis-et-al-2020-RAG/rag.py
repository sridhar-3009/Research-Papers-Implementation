"""Retrieval-Augmented Generation (Lewis et al. 2020): a seq2seq generator that conditions on documents fetched by a
dense retriever, treating the document as a LATENT variable that is marginalised out, trained end to end.

  retriever    p_eta(z|x) proportional to exp(d(z) . q(x))            (DPR, paper 070; d frozen, q fine-tuned)
  RAG-Sequence p(y|x) ~= sum_{z in top-k} p(z|x) prod_i p(y_i | x, z, y_<i)       (one document for the whole answer)
  RAG-Token    p(y|x) ~= prod_i sum_{z in top-k} p(z|x) p(y_i | x, z, y_<i)       (a document per output token)
  training     minimise -log p(y|x): NO label says which document is right; the retriever learns from the generator.

This file re-uses the toy Wikipedia, BM25 and dual encoder of paper 070 and adds:
  * a tiny conditional generator p(y_i | x, z, y_<i) (bag of document words + bag of question words + previous token
    -> MLP -> softmax), a stand-in for BART;
  * both marginalisations, exact top-k retrieval with gradients into the question encoder, a BM25 retriever whose
    scores act as logits (as in the paper's ablation), and a closed-book generator with no retrieval;
  * decoding: RAG-Token greedy on the mixed next-token distribution; RAG-Sequence 'thorough' decoding (candidates from
    each document, re-scored under every document) and 'fast' decoding (no re-scoring);
  * index hot-swapping: rebuild the index from a changed world without retraining;
  * a two-fact 'describe' task whose answer needs two different passages (the case the paper says suits RAG-Token).
"""

import copy
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location(
    "dpr070", Path(__file__).resolve().parent.parent / "070-Karpukhin-et-al-2020-Dense-Passage-Retrieval" / "dpr.py")
dpr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpr)

NODOC = "<nodoc>"
BOS = "<bos>"

# ----------------------------------------------------------------------------------------------- data

def describe_questions(world, people):
    """Two-fact task: 'where was X born and what does X work as ?' -> '<city> <job>' (needs two passages)."""
    return [(f"where was {p} born and what does {p} work as ?",
             f"{world['facts'][p]['born']} {world['facts'][p]['job']}", p, "describe") for p in people]


def changed_world(world, frac=1.0, seed=1):
    """The same people a few years later: a fraction of them have a new job, instrument and team and a new passage
    for each (birthplaces never change). Returns a new world dict."""
    rng = random.Random(seed)
    w = copy.deepcopy(world)
    for p in world["people"]:
        if rng.random() < frac:
            for r in ("job", "instrument", "team"):
                old = w["facts"][p][r]
                w["facts"][p][r] = rng.choice([v for v in dpr.VALUES[r] if v != old])
                w["passages"][w["gold"][(p, r)]] = rng.choice(dpr.PASSAGE_TEMPLATES[r]).format(e=p, v=w["facts"][p][r])
    return w

# ----------------------------------------------------------------------------------------------- the generator

class Generator(nn.Module):
    """p(y_i | x, z, y_<i) = softmax(MLP([mean emb(z words); mean emb(x words); emb(y_{i-1}); emb(position i)]))."""

    def __init__(self, vocab, d=96, hidden=256, max_len=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab = vocab
        n = len(vocab.ix)
        self.doc, self.q = nn.EmbeddingBag(n, d, mode="mean"), nn.EmbeddingBag(n, d, mode="mean")
        self.prev, self.pos = nn.Embedding(n, d), nn.Embedding(max_len, d)
        self.mlp = nn.Sequential(nn.Linear(4 * d, hidden), nn.ReLU(), nn.Linear(hidden, n))

    def step_logits(self, docs, questions, prev_tokens, position):
        """docs, questions: lists of strings (same length N); prev_tokens: LongTensor (N,)."""
        h = torch.cat([self.doc(*self.vocab.encode(docs)), self.q(*self.vocab.encode(questions)),
                       self.prev(prev_tokens), self.pos(torch.full_like(prev_tokens, position))], -1)
        return self.mlp(h)

    def token_logps(self, docs, questions, answers):
        """log p(y_i | x, z, y_<i) for every answer token (teacher forcing): (N, L)."""
        ys = torch.tensor([[self.vocab.ix.get(w, 0) for w in a.split()] for a in answers])
        prev = torch.full((len(docs),), self.vocab.ix[BOS])
        out = []
        for i in range(ys.shape[1]):
            out.append(torch.log_softmax(self.step_logits(docs, questions, prev, i), -1).gather(1, ys[:, i:i + 1]))
            prev = ys[:, i]
        return torch.cat(out, 1)


class FixedReader(Generator):
    """A hand-written reader with no parameters, used to isolate the two marginalisations on the 'describe' task:
    token i of the answer is the value of DESCRIBE_RELATIONS[i]; if document z states that relation, p(y_i|x,z) puts
    `confidence` on its value (the rest spread evenly over the vocabulary); otherwise it is uniform over that relation's
    values (the reader has to guess)."""

    def __init__(self, vocab, confidence=0.9):
        super().__init__(vocab)
        self.confidence = confidence

    def step_logits(self, docs, questions, prev_tokens, position):
        n = len(self.vocab.ix)
        rel = DESCRIBE_RELATIONS[position]
        vals = [self.vocab.ix[v] for v in dpr.VALUES[rel]]
        out = torch.full((len(docs), n), -30.0)
        for j, d in enumerate(docs):
            t = d.split()
            stated = [w for w in t if w in dpr.VALUES[rel]] if dpr.RELATION_WORDS[rel] & set(t) else []
            if stated:
                out[j] = np.log((1 - self.confidence) / n)
                out[j, self.vocab.ix[stated[0]]] = np.log(self.confidence)
            else:
                out[j, vals] = 0.0                                          # uniform guess over the relation's values
        return out


DESCRIBE_RELATIONS = ("born", "job")


@torch.no_grad()
def token_posteriors(retriever, gen, question, answer, k=5):
    """Figure 2: the document posterior p(z | x, y_i, y_<i) for each answer token under RAG-Token."""
    idx, scores = retriever.topk([question], k)
    log_pz = torch.log_softmax(scores, -1)[0]
    docs = [retriever.passages[i] for i in idx[0].tolist()]
    tok = gen.token_logps(docs, [question] * k, [answer] * k)               # (k, L)
    return docs, torch.softmax(log_pz[:, None] + tok, 0).T.numpy()          # (L, k)


def make_vocab(world, questions):
    v = dpr.Vocab(world["passages"] + [q for q, *_ in questions] + [a for _, a, *_ in questions])
    for w in [NODOC, BOS] + [x for vs in dpr.VALUES.values() for x in vs]:  # every possible answer word
        v.ix.setdefault(w, len(v.ix))
    return v

# ----------------------------------------------------------------------------------------------- retrievers

class DenseRetriever(nn.Module):
    """DPR with a FROZEN document index and a trainable question encoder (the paper fine-tunes only BERT_q)."""

    def __init__(self, dpr_model, passages):
        super().__init__()
        self.qenc = copy.deepcopy(dpr_model.eq)
        self.vocab = dpr_model.vocab
        with torch.no_grad():
            self.D = dpr_model.p(passages)                                  # the non-parametric memory
        self.passages = passages

    def swap_index(self, dpr_model, passages):
        """Index hot-swapping: re-encode a NEW corpus with the same frozen document encoder."""
        with torch.no_grad():
            self.D = dpr_model.p(passages)
        self.passages = passages

    def topk(self, questions, k):
        """Top-k document indices (B, k) and their scores WITH gradients into the question encoder."""
        s = self.qenc(self.vocab.encode(questions)) @ self.D.T
        idx = s.detach().topk(k, -1).indices
        return idx, s.gather(1, idx)


class BM25Retriever:
    """The paper's ablation: a fixed BM25 retriever whose scores are used as the logits of p(z|x)."""

    def __init__(self, passages):
        self.bm = dpr.BM25(passages)
        self.passages = passages
        self.cache = {}                                                     # BM25 never changes: score each q once

    def topk(self, questions, k):
        rows = []
        for q in questions:
            if (q, k) not in self.cache:
                s = torch.tensor(self.bm.scores(q), dtype=torch.float32)
                i = s.topk(k).indices
                self.cache[(q, k)] = (i, s[i])
            rows.append(self.cache[(q, k)])
        return torch.stack([r[0] for r in rows]), torch.stack([r[1] for r in rows])

# ----------------------------------------------------------------------------------------------- RAG

def marginal_logp(retriever, gen, questions, answers, k, mode):
    """log p(y|x) under RAG-Sequence ('sequence') or RAG-Token ('token'); retriever=None is closed-book."""
    B = len(questions)
    if retriever is None:
        return gen.token_logps([NODOC] * B, questions, answers).sum(-1)
    idx, scores = retriever.topk(questions, k)
    log_pz = torch.log_softmax(scores, -1)                                   # (B, k)
    docs = [retriever.passages[i] for i in idx.flatten().tolist()]
    tok = gen.token_logps(docs, [q for q in questions for _ in range(k)],
                          [a for a in answers for _ in range(k)]).view(B, k, -1)   # (B, k, L)
    if mode == "sequence":
        return torch.logsumexp(log_pz + tok.sum(-1), -1)
    return torch.logsumexp(log_pz[..., None] + tok, 1).sum(-1)


def train_rag(retriever, gen, data, k=5, mode="token", steps=600, batch=64, lr=3e-3, train_retriever=True, seed=0,
              retriever_lr=1e-3, warmup=200):
    """Minimise -log p(y|x). The retriever's question encoder is updated only if train_retriever (else 'frozen').
    Our generator starts from scratch (the paper's BART is pre-trained and can already read), so for the first
    `warmup` steps only the generator learns; until it reads documents, the retriever's gradient is just noise."""
    rng = random.Random(seed)
    groups = [{"params": list(gen.parameters()), "lr": lr}]
    if train_retriever and isinstance(retriever, DenseRetriever):
        groups.append({"params": list(retriever.qenc.parameters()), "lr": retriever_lr})
    opt = torch.optim.Adam(groups)
    for step in range(steps):
        b = rng.sample(data, min(batch, len(data)))
        loss = -marginal_logp(retriever, gen, [q for q, *_ in b], [a for _, a, *_ in b], k, mode).mean()
        opt.zero_grad()
        loss.backward()
        if step < warmup and len(groups) > 1:
            for p in groups[1]["params"]:
                p.grad = None
        opt.step()
    return retriever, gen


@torch.no_grad()
def decode(retriever, gen, questions, k=5, mode="token", length=1, thorough=True):
    """Greedy decoding. token: argmax of sum_z p(z|x) p(y_i|x,z,y_<i) at each step. sequence: decode greedily under each
    document separately, then score every candidate by sum_z p(z|x) p(y|x,z) (thorough) or only under the documents
    that produced it (fast)."""
    B = len(questions)
    bos = gen.vocab.ix[BOS]
    inv = {i: w for w, i in gen.vocab.ix.items()}
    if retriever is None:
        docs, log_pz, kk = [NODOC] * B, torch.zeros(B, 1), 1
    else:
        idx, scores = retriever.topk(questions, k)
        log_pz, kk = torch.log_softmax(scores, -1), k
        docs = [retriever.passages[i] for i in idx.flatten().tolist()]
    qs = [q for q in questions for _ in range(kk)]
    if mode == "token" or retriever is None:
        prev, out = torch.full((B,), bos), []
        for i in range(length):
            lp = torch.log_softmax(gen.step_logits(docs, qs, prev.repeat_interleave(kk), i), -1).view(B, kk, -1)
            tok = torch.logsumexp(log_pz[..., None] + lp, 1).argmax(-1)
            out.append(tok)
            prev = tok
        return [" ".join(inv[int(t[b])] for t in out) for b in range(B)]
    prev, toks = torch.full((B * kk,), bos), []                               # one greedy answer per document
    for i in range(length):
        t = gen.step_logits(docs, qs, prev, i).argmax(-1)
        toks.append(t)
        prev = t
    cands = [" ".join(inv[int(t[j])] for t in toks) for j in range(B * kk)]
    answers = []
    for b in range(B):
        cs = list(dict.fromkeys(cands[b * kk:(b + 1) * kk]))
        best, best_lp = None, -1e9
        for c in cs:
            if thorough:                                                      # extra forward passes for every doc
                lp = gen.token_logps(docs[b * kk:(b + 1) * kk], qs[b * kk:(b + 1) * kk], [c] * kk).sum(-1)
            else:                                                             # p(y|x,z) ~= 0 where y wasn't generated
                own = torch.tensor([cands[b * kk + j] == c for j in range(kk)])
                lp = gen.token_logps(docs[b * kk:(b + 1) * kk], qs[b * kk:(b + 1) * kk], [c] * kk).sum(-1)
                lp = torch.where(own, lp, torch.full_like(lp, -1e9))
            score = torch.logsumexp(log_pz[b] + lp, 0).item()
            if score > best_lp:
                best, best_lp = c, score
        answers.append(best)
    return answers


def exact_match(retriever, gen, data, **kw):
    preds = decode(retriever, gen, [q for q, *_ in data], **kw)
    return float(np.mean([p == a for p, (_, a, *_) in zip(preds, data)]))


@torch.no_grad()
def retrieval_recall(retriever, data, world, k=5):
    """How often the gold passage of the asked fact is among the top-k (for describe: both gold passages)."""
    idx, _ = retriever.topk([q for q, *_ in data], k)
    hits = []
    for row, (_, _, p, r) in zip(idx.tolist(), data):
        need = [world["gold"][(p, "born")], world["gold"][(p, "job")]] if r == "describe" else [world["gold"][(p, r)]]
        hits.append(all(n in row for n in need))
    return float(np.mean(hits))


def pretrain_dpr(world, questions, epochs=40):
    """The retriever we start from: DPR trained on OVERLAP-style questions only (the paper starts from a DPR trained
    on NQ + TriviaQA); paraphrases are new to it."""
    overlap = [(dpr.OVERLAP_Q[r].format(e=p), a, p, r) for _, a, p, r in questions if r in dpr.OVERLAP_Q]
    return dpr.train_dpr(world, overlap, epochs=epochs)


REPORTED = {
    "open-domain QA, exact match (NQ / TQA / WQ / CT)": "RAG-Seq 44.5 / 56.8 / 45.2 / 52.2, RAG-Token 44.1 / 55.2 / 45.5 / "
                                                       "50.0; DPR (extractive) 41.5 NQ; T5-11B closed-book 34.5 NQ",
    "answers not in any retrieved document (NQ)": "RAG still gets 11.8% right (an extractive reader: 0%)",
    "Table 6 ablation, NQ dev EM": "RAG-Seq: BM25 31.8, frozen retriever 41.2, learned 44.0 (RAG-Token 29.7 / 37.8 / 43.5)",
    "FEVER": "BM25 retrieval works best there (entity-centric claims); 3-way within 4.3% of pipeline SotA",
    "Jeopardy human eval (452 pairs)": "RAG more factual in 42.7% vs BART in 7.1%; RAG-Token > RAG-Seq on Q-BLEU-1",
    "index hot-swap (82 world leaders)": "2016 index/2016 leaders 70%, 2018/2018 68%; mismatched 12% and 4%",
    "setup": "DPR retriever (BERT-base, doc encoder + index frozen), BART-large generator (400M), 21M 100-word "
             "Wikipedia passages, k in {5, 10} for training",
}
