import math

import numpy as np
import torch

from rag import (DenseRetriever, FixedReader, Generator, changed_world, describe_questions, dpr, exact_match,
                 make_vocab, marginal_logp)


class _FixedRetriever:
    """Two documents with p(z|x) = [0.7, 0.3]."""

    def __init__(self, passages):
        self.passages = passages

    def topk(self, questions, k):
        s = torch.log(torch.tensor([[0.7, 0.3]] * len(questions)))
        return torch.tensor([[0, 1]] * len(questions)), s


def test_marginalisations_match_formulas():
    W = dpr.make_world(n_people=20, n_filler=10)
    V = make_vocab(W, describe_questions(W, W["people"]))
    p = W["people"][0]
    docs = [W["passages"][W["gold"][(p, "born")]], W["passages"][W["gold"][(p, "job")]]]
    reader = FixedReader(V, confidence=0.9)
    ret = _FixedRetriever(docs)
    q, a = describe_questions(W, [p])[0][:2]
    tok = reader.token_logps(docs, [q, q], [a, a]).exp().numpy()             # (2 docs, 2 tokens)
    pz = np.array([0.7, 0.3])
    seq = (pz * tok.prod(1)).sum()
    tokm = (pz[:, None] * tok).sum(0).prod()
    assert abs(marginal_logp(ret, reader, [q], [a], 2, "sequence").exp().item() - seq) < 1e-5
    assert abs(marginal_logp(ret, reader, [q], [a], 2, "token").exp().item() - tokm) < 1e-5
    assert tokm > seq                                                         # each doc supports one token


def test_retriever_gradients_flow_and_index_swap():
    W = dpr.make_world(n_people=40, n_filler=40)
    Q = dpr.make_questions(W)
    d = dpr.train_dpr(W, Q["train"], epochs=2)
    ret = DenseRetriever(d, W["passages"])
    g = Generator(make_vocab(W, Q["train"]))
    lp = marginal_logp(ret, g, [q for q, *_ in Q["train"][:4]], [a for _, a, *_ in Q["train"][:4]], 3, "token")
    lp.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ret.qenc.parameters())
    W2 = changed_world(W)
    ret.swap_index(d, W2["passages"])
    assert ret.passages is W2["passages"] and W2["facts"] != W["facts"]
    assert all(W2["facts"][p]["born"] == W["facts"][p]["born"] for p in W["people"])


def test_fixed_reader_rag_token_beats_rag_sequence_on_two_facts():
    W = dpr.make_world(n_people=20, n_filler=10)
    V = make_vocab(W, describe_questions(W, W["people"]))
    data = describe_questions(W, W["people"])
    ret = _FixedRetriever(None)
    reader = FixedReader(V)
    hits = {"token": 0, "sequence": 0}
    for q, a, p, _ in data:
        ret.passages = [W["passages"][W["gold"][(p, "born")]], W["passages"][W["gold"][(p, "job")]]]
        for m in hits:
            hits[m] += exact_match(ret, reader, [(q, a, p, "describe")], k=2, mode=m, length=2)
    assert hits["token"] == len(data) and hits["sequence"] < len(data) / 2
