"""Dense Passage Retrieval (Karpukhin et al. 2020): two encoders map questions and passages to vectors, relevance is
their dot product, and training uses in-batch negatives (+ one BM25 hard negative per question).

  sim(q, p) = E_Q(q) . E_P(p)                                                                      (Eq. 1)
  loss      = -log  exp(sim(q_i, p_i+)) / ( exp(sim(q_i, p_i+)) + sum_j exp(sim(q_i, p_j-)) )      (Eq. 2)
  in-batch: S = Q P^T (B x B); row i's positive is column i, the other B-1 golds are negatives;
            the extra BM25 negatives are shared by every question in the batch

This file has:
  * a toy 'Wikipedia': people with four facts each (birthplace, job, instrument, team), written as short passages,
    plus filler passages that mention people in passing;
  * questions in two styles: OVERLAP (re-uses the passage's words, like SQuAD) and PARAPHRASE (different words for the
    same relation, like Natural Questions), about seen people (held-out facts) and about people never asked about in
    training (rare 'salient phrases', like the paper's 'Thoros of Myr' example);
  * Okapi BM25 (k1 = 0.9, b = 0.4, the paper's tuned values);
  * the dual encoder (two independent bag-of-words encoders; the paper uses two BERT-base [CLS] vectors), in-batch +
    BM25-negative training, exact maximum-inner-product search (FAISS's job), BM25 + lambda * DPR hybrid;
  * top-k retrieval accuracy (a hit = a retrieved passage CONTAINS the answer string, as in the paper);
  * a simple rule-based reader for end-to-end exact match.
"""

import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------- the toy corpus

CITIES = ["paris", "lima", "oslo", "cairo", "delhi", "quito", "rome", "seoul", "tunis", "hanoi", "dakar", "minsk",
          "accra", "kyoto", "bern", "riga", "sofia", "lagos", "perth", "porto"]
JOBS = ["baker", "pilot", "nurse", "farmer", "lawyer", "painter", "plumber", "teacher", "chemist", "sailor", "dentist",
        "tailor", "jeweller", "surveyor", "geologist", "librarian"]
INSTRUMENTS = ["violin", "cello", "flute", "drums", "banjo", "harp", "oboe", "sitar", "tuba", "piano", "lute", "accordion"]
TEAMS = ["rovers", "united", "wanderers", "athletic", "rangers", "albion", "city", "hotspur", "olympic", "dynamo"]
VALUES = {"born": CITIES, "job": JOBS, "instrument": INSTRUMENTS, "team": TEAMS}
RELATIONS = list(VALUES)

PASSAGE_TEMPLATES = {
    "born": ["{e} was born in {v} and grew up there .", "{e} was born in the city of {v} ."],
    "job": ["{e} works as a {v} in a small town .", "by trade {e} works as a {v} ."],
    "instrument": ["{e} plays the {v} in a local band .", "in the evenings {e} plays the {v} ."],
    "team": ["{e} is a lifelong fan of {v} .", "{e} has always been a fan of {v} ."],
}
OVERLAP_Q = {"born": "where was {e} born ?", "job": "what does {e} work as ?",
             "instrument": "what does {e} play ?", "team": "{e} is a fan of which club ?"}
PARAPHRASE_Q = {"born": "what is the hometown of {e} ?", "job": "what is the profession of {e} ?",
                "instrument": "which musical instrument is {e} known for ?", "team": "which club does {e} support ?"}
FILLER = ["{e} met {e2} at a party in {v} .", "{e} and {e2} once travelled to {v} together .",
          "{e} wrote a letter to {e2} about {v} .", "a neighbour of {e} named {e2} visited {v} ."]
SYLLABLES = ["ka", "zor", "mi", "vel", "thu", "ran", "dos", "pek", "lio", "mar", "sun", "gri", "fae", "bol", "nyx", "tor",
             "qua", "ves", "lun", "dra"]


def make_world(n_people=500, n_filler=1500, seed=0):
    """People with four facts each; one passage per fact plus filler passages that mention people in passing."""
    rng = random.Random(seed)
    names = set()
    while len(names) < n_people:
        names.add("".join(rng.choice(SYLLABLES) for _ in range(3)))
    people = sorted(names)
    rng.shuffle(people)
    facts = {p: {r: rng.choice(VALUES[r]) for r in RELATIONS} for p in people}
    passages, gold = [], {}
    for p in people:
        for r in RELATIONS:
            gold[(p, r)] = len(passages)
            passages.append(rng.choice(PASSAGE_TEMPLATES[r]).format(e=p, v=facts[p][r]))
    for _ in range(n_filler):
        e, e2 = rng.sample(people, 2)
        passages.append(rng.choice(FILLER).format(e=e, e2=e2, v=rng.choice(CITIES)))
    return {"people": people, "facts": facts, "passages": passages, "gold": gold}


def make_questions(world, seen_frac=0.8, train_frac=0.7, seed=0):
    """Train on some (person, relation) facts of 'seen' people; test on their held-out facts and on unseen people.
    Every question gets an OVERLAP and a PARAPHRASE version. Returns dicts of (question, answer, person, relation)."""
    rng = random.Random(seed)
    people = world["people"]
    n_seen = int(len(people) * seen_frac)
    seen_pairs = [(p, r) for p in people[:n_seen] for r in RELATIONS]
    rng.shuffle(seen_pairs)
    k = int(len(seen_pairs) * train_frac)

    def q(pairs, style):
        T = OVERLAP_Q if style == "overlap" else PARAPHRASE_Q
        return [(T[r].format(e=p), world["facts"][p][r], p, r) for p, r in pairs]

    train = [x for p, r in seen_pairs[:k] for x in q([(p, r)], rng.choice(["overlap", "paraphrase"]))]
    unseen_pairs = [(p, r) for p in people[n_seen:] for r in RELATIONS]
    return {"train": train,
            "test seen people, overlap": q(seen_pairs[k:], "overlap"),
            "test seen people, paraphrase": q(seen_pairs[k:], "paraphrase"),
            "test unseen people, overlap": q(unseen_pairs, "overlap"),
            "test unseen people, paraphrase": q(unseen_pairs, "paraphrase")}

# ----------------------------------------------------------------------------------------------- BM25

class BM25:
    """Okapi BM25: sum over query terms t of idf(t) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))."""

    def __init__(self, docs, k1=0.9, b=0.4):
        self.toks = [d.split() for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = np.mean([len(t) for t in self.toks])
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.index = {}                                                    # inverted index: word -> docs
        for i, t in enumerate(self.tf):
            for w in t:
                self.index.setdefault(w, []).append(i)

    def scores(self, query):
        s = np.zeros(len(self.toks))
        for w in set(query.split()):
            for i in self.index.get(w, []):
                tf, dl = self.tf[i][w], len(self.toks[i])
                s[i] += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

# ----------------------------------------------------------------------------------------------- the dual encoder

class Vocab:
    def __init__(self, texts):
        self.ix = {"<unk>": 0}
        for t in texts:
            for w in t.split():
                self.ix.setdefault(w, len(self.ix))

    def encode(self, texts):
        ids = [[self.ix.get(w, 0) for w in t.split()] for t in texts]
        flat = torch.tensor([i for x in ids for i in x])
        offsets = torch.tensor([0] + list(np.cumsum([len(x) for x in ids])[:-1]))
        return flat, offsets


class Encoder(nn.Module):
    """Bag of words -> mean embedding -> linear: a tiny stand-in for BERT's [CLS] vector."""

    def __init__(self, n_vocab, d=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.bag = nn.EmbeddingBag(n_vocab, d, mode="mean")
        self.proj = nn.Linear(d, d)

    def forward(self, enc):
        return self.proj(self.bag(*enc))


class DPR(nn.Module):
    """Two INDEPENDENT encoders, E_Q for questions and E_P for passages (as in the paper). Like the paper's two BERTs,
    which both start from the same pre-trained checkpoint, they start from the SAME weights (same_init=True) and then
    train separately; same_init=False starts them from unrelated random weights."""

    def __init__(self, vocab, d=64, same_init=True):
        super().__init__()
        self.vocab = vocab
        self.eq, self.ep = Encoder(len(vocab.ix), d, seed=1), Encoder(len(vocab.ix), d, seed=1 if same_init else 2)

    def q(self, texts):
        return self.eq(self.vocab.encode(texts))

    def p(self, texts):
        return self.ep(self.vocab.encode(texts))


def in_batch_loss(qv, pv_pos, pv_extra=None):
    """Eq. 2 with in-batch negatives: S = Q [P_pos; P_extra]^T, the target for row i is column i."""
    P = pv_pos if pv_extra is None else torch.cat([pv_pos, pv_extra])
    S = qv @ P.T
    return F.cross_entropy(S, torch.arange(len(qv)))


def bm25_negative(bm25, world, question, answer, rng, top=10):
    """The highest-ranked BM25 passage that does NOT contain the answer (the paper's hard negative)."""
    s = bm25.scores(question)
    for i in np.argsort(-s)[:top]:
        if answer not in world["passages"][i].split():
            return world["passages"][i]
    return rng.choice(world["passages"])


def train_dpr(world, train_qs, negatives="gold+bm25", batch=128, epochs=40, lr=1e-2, d=128, n_train=None, seed=0,
              n_extra=1, bm25=None, same_init=True):
    """negatives: 'gold' (in-batch only), 'gold+bm25' (in-batch + n_extra BM25 negatives per question, shared by the
    batch), 'random' / 'bm25' (no in-batch: each question gets n_extra negatives of that kind and only those)."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    vocab = Vocab(world["passages"] + [q for q, *_ in train_qs])
    model = DPR(vocab, d, same_init)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    qs = train_qs[:n_train] if n_train else list(train_qs)
    bm25 = bm25 or BM25(world["passages"])
    hard = {q: bm25_negative(bm25, world, q, a, rng) for q, a, *_ in qs} if "bm25" in negatives else {}
    for _ in range(epochs):
        rng.shuffle(qs)
        for s in range(0, len(qs), batch):
            b = qs[s:s + batch]
            if len(b) < 2:
                continue
            qv = model.q([q for q, *_ in b])
            pos = model.p([world["passages"][world["gold"][(e, r)]] for _, _, e, r in b])
            if negatives in ("gold", "gold+bm25"):
                extra = model.p([hard[q] for q, *_ in b for _ in range(n_extra)]) if negatives == "gold+bm25" else None
                loss = in_batch_loss(qv, pos, extra)
            else:                                                          # per-question negatives, no in-batch
                pick = (lambda q: rng.choice(world["passages"])) if negatives == "random" else (lambda q: hard[q])
                neg = model.p([pick(q) for q, *_ in b for _ in range(n_extra)]).view(len(b), n_extra, -1)
                S = torch.cat([(qv * pos).sum(-1, keepdim=True), torch.einsum("bd,bkd->bk", qv, neg)], 1)
                loss = F.cross_entropy(S, torch.zeros(len(b), dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


class DenseIndex:
    """Pre-compute every passage vector once; search = one matrix product (exact MIPS; FAISS does this at scale)."""

    def __init__(self, model, passages):
        self.model = model
        with torch.no_grad():
            self.P = model.p(passages)

    def scores(self, questions):
        with torch.no_grad():
            return (self.model.q(questions) @ self.P.T).numpy()

# ----------------------------------------------------------------------------------------------- evaluation

def top_k_accuracy(score_matrix, questions, passages, ks=(1, 5, 20)):
    """Fraction of questions where a top-k passage CONTAINS the answer string (the paper's metric)."""
    toks = [set(p.split()) for p in passages]
    order = np.argsort(-score_matrix, 1)
    out = {}
    for k in ks:
        out[k] = float(np.mean([any(a in toks[i] for i in order[n, :k]) for n, (_, a, *_) in enumerate(questions)]))
    return out


def bm25_matrix(bm25, questions):
    return np.stack([bm25.scores(q) for q, *_ in questions])


def hybrid(bm25_s, dense_s, lam):
    """The paper's BM25(q, p) + lambda * sim(q, p) (they re-rank the union of both top-2000 lists)."""
    return bm25_s + lam * dense_s


RELATION_WORDS = {"born": {"born", "hometown"}, "job": {"work", "works", "profession"},
                  "instrument": {"play", "plays", "instrument"}, "team": {"fan", "club", "support"}}


TEMPLATE_WORDS = {w for T in (OVERLAP_Q, PARAPHRASE_Q, PASSAGE_TEMPLATES) for v in T.values()
                  for t in (v if isinstance(v, list) else [v]) for w in t.split() if "{" not in w} | \
                 {w for t in FILLER for w in t.split() if "{" not in w}


def subject(tokens):
    """The first word that is not template vocabulary or a value: the name of the person a sentence is about."""
    vals = {v for vs in VALUES.values() for v in vs}
    return next((w for w in tokens if w not in TEMPLATE_WORDS and w not in vals), None)


def rule_reader(question, retrieved):
    """A stand-in for the paper's BERT reader: find the question's person and relation, then read the value from the
    first retrieved passage that is ABOUT that person and states that relation. Returns the answer or None."""
    words = question.split()
    rel = next(r for r, ws in RELATION_WORDS.items() if set(words) & ws)
    person = subject(words)
    for p in retrieved:
        t = p.split()
        if subject(t) == person and RELATION_WORDS[rel] & set(t):
            vals = [w for w in t if w in VALUES[rel]]
            if vals:
                return vals[0]
    return None


def tune_lambda(bm25, index, dev_qs, passages, grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0)):
    """Pick the hybrid weight on development questions (top-1 accuracy), as the paper tunes lambda on dev sets."""
    b, d = bm25_matrix(bm25, dev_qs), index.scores([q for q, *_ in dev_qs])
    return max(grid, key=lambda lam: top_k_accuracy(hybrid(b, d, lam), dev_qs, passages, (1,))[1])


def exact_match(score_matrix, questions, passages, k):
    order = np.argsort(-score_matrix, 1)[:, :k]
    return float(np.mean([rule_reader(q, [passages[i] for i in order[n]]) == a
                          for n, (q, a, *_) in enumerate(questions)]))


REPORTED = {
    "top-20 accuracy, NQ test": "BM25 59.1, DPR 78.4, BM25+DPR 76.6 (single-dataset training)",
    "top-100 accuracy, NQ test": "BM25 73.7, DPR 85.4",
    "SQuAD top-20": "BM25 68.8 > DPR 63.2 (questions written while looking at the passage: high word overlap)",
    "training": "in-batch negatives, batch 128 + 1 BM25 negative per question, Adam lr 1e-5, up to 40 epochs",
    "corpus": "Wikipedia split into 100-word passages: 21,015,324 passages; BERT-base [CLS], d = 768",
    "sample efficiency": "DPR trained on only 1,000 NQ questions already beats BM25",
    "Table 3 (NQ dev top-20)": "7 random/BM25/gold negatives 64.3/63.3/63.1; in-batch 7/31/127 golds 69.1/70.8/73.0; "
                               "in-batch 127 + 1 BM25 each 78.0",
    "end-to-end QA, NQ exact match": "DPR 41.5 vs ORQA 33.3",
    "speed": "995 questions/s with FAISS vs 23.7 for BM25/Lucene per CPU thread; but the dense index takes hours to build",
}
