"""Atlas (Izacard et al. 2022): a retrieval-augmented language model that is jointly PRE-TRAINED with its retriever, so
that it becomes a strong FEW-SHOT learner (42.4% on NaturalQuestions from 64 examples with 11B parameters).

Components (all tiny here):
  * retriever: a Contriever-style dual encoder (word embeddings pooled with learned per-word weights, dot product),
    started WITHOUT labels -- weights from corpus idf, optionally trained on random crops of passages (two crops of the
    same passage are positives, other passages in the batch negatives);
  * reader: Fusion-in-Decoder (FiD): each (query, document) pair is encoded SEPARATELY, the decoder attends over the
    concatenation of all encoded documents (cost linear in the number of documents, not quadratic);
  * four ways to train the retriever from the reader, with no document labels (Section 2.2), each a target
    distribution over the top-K documents that the retriever's p_retr(d|q) = softmax(s(d, q)/theta) is pulled toward:
      ADist  attention distillation: alpha_n * ||v_n|| summed over each document's tokens
      EMDR2  -log sum_k p_LM(a|q, d_k) p_retr(d_k)            (documents as latent variables, LM frozen in this loss)
      PDist  perplexity distillation: softmax_k log p_LM(a | q, d_k)
      LOOP   leave-one-out: softmax_k -log p_LM(a | q, all docs except d_k)
  * self-supervised joint pre-training with masked language modelling: mask a fact's value in a passage and let the
    model retrieve OTHER passages to fill it in; the index is refreshed every few hundred steps;
  * few-shot fine-tuning on 64 question/answer pairs with a fixed retriever, query-side fine-tuning, or full
    fine-tuning (with re-indexing);
  * product quantisation (PQ) of the document index.

The toy corpus re-uses paper 070's people and templates, but writes every fact in BOTH of its templates, so that a
masked passage can be completed from another passage (as on real Wikipedia, where facts recur).
"""

import importlib.util
import math
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

MASK, PAD = "<mask>", "<pad>"

# ----------------------------------------------------------------------------------------------- corpus

def make_world(n_people=300, n_filler=900, seed=0):
    """Every (person, relation) fact is written twice, once in each template; plus filler passages."""
    rng = random.Random(seed)
    names = set()
    while len(names) < n_people:
        names.add("".join(rng.choice(dpr.SYLLABLES) for _ in range(3)))
    people = sorted(names)
    rng.shuffle(people)
    facts = {p: {r: rng.choice(dpr.VALUES[r]) for r in dpr.RELATIONS} for p in people}
    passages, gold, fact_of = [], {}, {}
    for p in people:
        for r in dpr.RELATIONS:
            gold[(p, r)] = []
            for t in dpr.PASSAGE_TEMPLATES[r]:
                gold[(p, r)].append(len(passages))
                fact_of[len(passages)] = (p, r)
                passages.append(t.format(e=p, v=facts[p][r]))
    for _ in range(n_filler):
        e, e2 = rng.sample(people, 2)
        passages.append(rng.choice(dpr.FILLER).format(e=e, e2=e2, v=rng.choice(dpr.CITIES)))
    return {"people": people, "facts": facts, "passages": passages, "gold": gold, "fact_of": fact_of}


def mlm_examples(world):
    """Pre-training data: (masked passage, value, index of the passage itself -- excluded from retrieval)."""
    out = []
    for i, (p, r) in world["fact_of"].items():
        v = world["facts"][p][r]
        out.append((" ".join(MASK if w == v else w for w in world["passages"][i].split()), v, i))
    return out


def qa_examples(world, people, style):
    T = dpr.OVERLAP_Q if style == "overlap" else dpr.PARAPHRASE_Q
    return [(T[r].format(e=p), world["facts"][p][r], -1) for p in people for r in dpr.RELATIONS]


class Vocab:
    def __init__(self, texts):
        self.ix = {PAD: 0, MASK: 1}
        for t in texts:
            for w in t.split():
                self.ix.setdefault(w, len(self.ix))
        for vs in dpr.VALUES.values():
            for v in vs:
                self.ix.setdefault(v, len(self.ix))

    def pad(self, texts, length=None):
        ids = [[self.ix.get(w, 0) for w in t.split()] for t in texts]
        L = length or max(len(x) for x in ids)
        return torch.tensor([x[:L] + [0] * (L - len(x[:L])) for x in ids])

# ----------------------------------------------------------------------------------------------- retriever

class Retriever(nn.Module):
    """Contriever-style dual encoder: word embeddings pooled with a learned per-word importance weight
    (softmax-weighted average; a transformer learns such weighting itself) -> linear. The towers start identical."""

    def __init__(self, vocab, d=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        n = len(vocab.ix)
        self.vocab = vocab
        self.q_emb, self.d_emb = nn.Embedding(n, d, padding_idx=0), nn.Embedding(n, d, padding_idx=0)
        self.q_w, self.d_w = nn.Embedding(n, 1), nn.Embedding(n, 1)
        nn.init.zeros_(self.q_w.weight)
        nn.init.zeros_(self.d_w.weight)
        self.q_lin, self.d_lin = nn.Linear(d, d), nn.Linear(d, d)
        self.d_emb.load_state_dict(self.q_emb.state_dict())
        self.d_lin.load_state_dict(self.q_lin.state_dict())

    @staticmethod
    def _pool(emb, w, lin, ids):
        a = w(ids).squeeze(-1).masked_fill(ids == 0, -1e9)
        return lin(torch.einsum("bt,btd->bd", torch.softmax(a, -1), emb(ids)))

    def init_idf(self, passages, scale=0.8):
        """Unsupervised start: importance weight = scale * idf(word) from corpus counts (no labels involved)."""
        df = {}
        for t in passages:
            for w in set(t.split()):
                df[w] = df.get(w, 0) + 1
        with torch.no_grad():
            for w, i in self.vocab.ix.items():                          # words absent from the corpus (the mask,
                idf = math.log(len(passages) / (1 + df[w])) if w in df else 0.0   # question-only words): neutral
                self.q_w.weight[i, 0] = self.d_w.weight[i, 0] = scale * idf
        return self

    def q_params(self):
        return list(self.q_emb.parameters()) + list(self.q_w.parameters()) + list(self.q_lin.parameters())

    def d_params(self):
        return list(self.d_emb.parameters()) + list(self.d_w.parameters()) + list(self.d_lin.parameters())

    def q(self, texts):
        return self._pool(self.q_emb, self.q_w, self.q_lin, self.vocab.pad(texts))

    def d(self, texts):
        return self._pool(self.d_emb, self.d_w, self.d_lin, self.vocab.pad(texts))


def contriever_pretrain(ret, passages, steps=300, batch=128, lr=3e-3, crop=0.6, temp=0.1, seed=0):
    """Unsupervised contrastive training on independent random crops (Contriever / MoCo, simplified to in-batch)."""
    rng = random.Random(seed)
    opt = torch.optim.Adam(ret.parameters(), lr=lr)

    def crop_of(t):
        w = t.split()
        keep = [x for x in w if rng.random() < crop] or [rng.choice(w)]
        return " ".join(keep)

    for _ in range(steps):
        b = rng.sample(passages, batch)
        qv, dv = F.normalize(ret.q([crop_of(t) for t in b]), dim=-1), F.normalize(ret.d([crop_of(t) for t in b]), dim=-1)
        loss = F.cross_entropy(qv @ dv.T / temp, torch.arange(batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ret


class Index:
    """Pre-computed document embeddings (the expensive part Atlas refreshes every few hundred steps)."""

    def __init__(self, ret, passages):
        self.passages = passages
        self.refresh(ret)

    def refresh(self, ret):
        with torch.no_grad():
            self.D = ret.d(self.passages)

    def topk(self, ret, queries, k, exclude=None):
        with torch.no_grad():
            s = ret.q(queries) @ self.D.T
            if exclude is not None:
                for i, e in enumerate(exclude):
                    if e >= 0:
                        s[i, e] = -1e9
        return s.topk(k, -1).indices

# ----------------------------------------------------------------------------------------------- the FiD reader

class FiDReader(nn.Module):
    """Fusion-in-Decoder in miniature: every document token is encoded together with the query (per document,
    independently), and one decoder query attends over ALL tokens of ALL documents to predict the answer word."""

    def __init__(self, vocab, d=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        n = len(vocab.ix)
        self.vocab, self.d = vocab, d
        self.E = nn.Embedding(n, d, padding_idx=0)
        self.pos = nn.Embedding(24, d)
        self.enc = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.uq = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(2 * d, n)

    def query_vec(self, q_ids):
        m = (q_ids > 0).float()[..., None]
        return (self.E(q_ids) * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, queries, docs, doc_keep=None):
        """queries: B strings; docs: B lists of K strings (K may be 0: closed-book); doc_keep: optional (B, K) bool
        mask of documents to use. Returns logits (B, V), attention alpha (B, K, T), value norms (B, K, T)."""
        qv = self.query_vec(self.vocab.pad(queries))                                      # (B, d)
        u = self.uq(qv)
        B, K = len(queries), len(docs[0]) if docs else 0
        if K == 0:
            return self.out(torch.cat([torch.zeros_like(u), u], -1)), None, None
        ids = self.vocab.pad([t for row in docs for t in row], 20).view(B, K, -1)          # (B, K, T)
        T = ids.shape[-1]
        x = self.E(ids) + self.pos(torch.arange(T))
        h = self.enc(torch.cat([x, qv[:, None, None, :].expand(-1, K, T, -1)], -1))       # per-document encoding
        mask = ids > 0
        if doc_keep is not None:
            mask = mask & doc_keep[..., None]
        att = torch.einsum("bktd,bd->bkt", self.Wk(h), u) / math.sqrt(self.d)
        att = att.masked_fill(~mask, -1e9)
        alpha = torch.softmax(att.view(B, -1), -1).view(B, K, T)                         # over ALL docs' tokens
        v = self.Wv(h)
        c = torch.einsum("bkt,bktd->bd", alpha, v)
        return self.out(torch.cat([c, u], -1)), alpha, v.norm(dim=-1)


def answer_logp(reader, queries, docs, answers, doc_keep=None):
    logits, alpha, vnorm = reader(queries, docs, doc_keep)
    a = torch.tensor([reader.vocab.ix[x] for x in answers])
    return torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1), alpha, vnorm

# ----------------------------------------------------------------------------------------------- retriever losses

def retriever_target(loss, reader, queries, docs, answers, alpha, vnorm):
    """The target distribution over the K retrieved documents (no gradient), for ADist / PDist / LOOP."""
    B, K = len(queries), len(docs[0])
    with torch.no_grad():
        if loss == "adist":                                     # alpha_n ||v_n||, summed per document, normalised
            s = (alpha * vnorm).sum(-1)
            return s / s.sum(-1, keepdim=True)
        rep_q = [q for q in queries for _ in range(K)]
        rep_a = [a for a in answers for _ in range(K)]
        rep_d = [row for row in docs for _ in range(K)]
        eye = torch.eye(K, dtype=torch.bool).repeat(B, 1)                                  # (B*K, K)
        if loss == "pdist":                                     # each document alone
            lp = answer_logp(reader, rep_q, rep_d, rep_a, doc_keep=eye)[0].view(B, K)
            return torch.softmax(lp, -1)
        if loss == "loop":                                      # all documents except one
            lp = answer_logp(reader, rep_q, rep_d, rep_a, doc_keep=~eye)[0].view(B, K)
            return torch.softmax(-lp, -1)
    raise ValueError(loss)


def retriever_loss(loss, ret, reader, queries, docs, answers, alpha, vnorm, theta=1.0, train_docs=False):
    """KL(target || p_retr) for ADist / PDist / LOOP; the EMDR2 objective for 'emdr'."""
    B, K = len(queries), len(docs[0])
    qv = ret.q(queries)
    if train_docs:
        dv = ret.d([t for row in docs for t in row]).view(B, K, -1)
    else:
        with torch.no_grad():
            dv = ret.d([t for row in docs for t in row]).view(B, K, -1)
    log_pr = torch.log_softmax(torch.einsum("bd,bkd->bk", qv, dv) / theta, -1)
    if loss == "emdr":
        B_, K_ = B, K
        with torch.no_grad():
            rep = [x for x in range(B_) for _ in range(K_)]
            lp = answer_logp(reader, [queries[i] for i in rep], [docs[i] for i in rep], [answers[i] for i in rep],
                             doc_keep=torch.eye(K_, dtype=torch.bool).repeat(B_, 1))[0].view(B_, K_)
        return -torch.logsumexp(lp + log_pr, -1).mean()
    tgt = retriever_target(loss, reader, queries, docs, answers, alpha, vnorm)
    return (tgt * (torch.log(tgt.clamp(min=1e-9)) - log_pr)).sum(-1).mean()

# ----------------------------------------------------------------------------------------------- training

def train_atlas(ret, reader, index, data, k=5, steps=600, batch=32, lr=3e-3, ret_lr=3e-4, loss="pdist",
                train_q=True, train_d=False, reindex_every=0, closed_book=False, seed=0):
    """Joint training: the reader learns -log p(answer | query, top-k docs); the retriever is pulled toward the
    reader's document preferences via `loss` (None = fixed retriever). data: (query, answer, exclude_index)."""
    rng = random.Random(seed)
    groups = [{"params": list(reader.parameters()), "lr": lr}]
    rparams = (ret.q_params() if train_q else []) + (ret.d_params() if train_d else [])
    use_ret = loss is not None and rparams and not closed_book
    if use_ret:
        groups.append({"params": rparams, "lr": ret_lr})
    opt = torch.optim.Adam(groups)
    for step in range(steps):
        b = rng.sample(data, min(batch, len(data)))
        qs, ans, ex = [x[0] for x in b], [x[1] for x in b], [x[2] for x in b]
        if closed_book:
            docs = [[] for _ in b]
        else:
            idx = index.topk(ret, qs, k, exclude=ex)
            docs = [[index.passages[i] for i in row] for row in idx.tolist()]
        lp, alpha, vnorm = answer_logp(reader, qs, docs, ans)
        total = -lp.mean()
        if use_ret:
            total = total + retriever_loss(loss, ret, reader, qs, docs, ans, alpha, vnorm, train_docs=train_d)
        opt.zero_grad()
        total.backward()
        opt.step()
        if reindex_every and (step + 1) % reindex_every == 0:
            index.refresh(ret)
    return ret, reader


@torch.no_grad()
def evaluate(ret, reader, index, data, k=5, closed_book=False):
    qs = [x[0] for x in data]
    if closed_book:
        docs = [[] for _ in data]
    else:
        idx = index.topk(ret, qs, k, exclude=[x[2] for x in data])          # never the masked passage itself
        docs = [[index.passages[i] for i in row] for row in idx.tolist()]
    logits = reader(qs, docs)[0]
    inv = {i: w for w, i in reader.vocab.ix.items()}
    return float(np.mean([inv[int(p)] == a for p, (_, a, _) in zip(logits.argmax(-1), data)]))


@torch.no_grad()
def recall(ret, index, data, world, k=5):
    """Fraction of queries with a passage stating the asked fact (other than the query passage) in the top k."""
    hits = []
    for (q, a, ex), row in zip(data, index.topk(ret, [x[0] for x in data], k, exclude=[x[2] for x in data]).tolist()):
        hits.append(any(world["fact_of"].get(i) is not None and i != ex and
                        world["facts"][world["fact_of"][i][0]][world["fact_of"][i][1]] == a and
                        world["fact_of"][i][0] in q.split() for i in row))
    return float(np.mean(hits))

# ----------------------------------------------------------------------------------------------- product quantisation

def kmeans(x, k, iters=15, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c, torch.cdist(x, c).argmin(1)


def product_quantize(D, m, bits):
    """Split each vector into m sub-vectors and replace each by the nearest of 2^bits centroids (one byte or less per
    sub-vector). Returns the reconstructed matrix and the bytes per vector."""
    n, d = D.shape
    sub = d // m
    recon = torch.empty_like(D)
    for j in range(m):
        c, a = kmeans(D[:, j * sub:(j + 1) * sub], min(2 ** bits, n), seed=j)
        recon[:, j * sub:(j + 1) * sub] = c[a]
    return recon, m * bits / 8


REPORTED = {
    "NaturalQuestions, 64 examples": "Atlas-11B 42.4% (45.1% with a Wikipedia-only index) vs PaLM-540B 39.6%",
    "NaturalQuestions, full data": "60.4% (64.0% with the Dec-2018 Wikipedia index, temporally matched to NQ)",
    "MMLU 5-shot": "Atlas-11B beats GPT-3 by 4 points with 15x fewer parameters; zero-shot de-biased 47.1% > GPT-3 "
                   "5-shot 43.9%",
    "retriever losses (Table 1)": "ADist, EMDR2, PDist and LOOP perform similarly; PDist chosen (stable, cheap); "
                                  "the closed-book baseline is poor and joint pre-training is key for few-shot",
    "pretext tasks (Table 2, 64-shot avg)": "prefix LM 40.1, masked LM 42.4, title-to-section 40.8",
    "fine-tuning the retriever (Table 4)": "fixed retriever clearly worse; top-100 re-ranking ~ full re-indexing; "
                                           "query-side fine-tuning best at 64 shots",
    "temporal (TempLAMA)": "2017 questions: Atlas 57.7% vs closed-book T5 12.1%; swap to a 2020 index -> 53.1% on "
                           "2020 answers without retraining",
    "index compression": "PQ shrinks the Wikipedia index from 49 GB to 4 GB with negligible accuracy loss",
    "pre-training cost": "index refreshed every 1,000 steps: ~30% overhead vs a fixed retriever; 20 documents retrieved",
}
