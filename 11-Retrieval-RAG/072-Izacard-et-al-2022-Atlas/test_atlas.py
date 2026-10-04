import math

import numpy as np
import torch

from atlas import (FiDReader, Index, Retriever, Vocab, answer_logp, make_world, mlm_examples, product_quantize,
                   retriever_loss, retriever_target, train_atlas, evaluate)


def _setup():
    W = make_world(n_people=30, n_filler=30)
    V = Vocab(W["passages"])
    return W, V


def test_world_mlm_and_fid_shapes():
    W, V = _setup()
    assert all(len(v) == 2 for v in W["gold"].values())                          # every fact written twice
    q, a, ex = mlm_examples(W)[0]
    assert "<mask>" in q and a not in q.split() and a in W["passages"][ex].split()
    rd = FiDReader(V)
    docs = [W["passages"][:3], W["passages"][3:6]]
    logits, alpha, vnorm = rd(["x was born in <mask>"] * 2, docs)
    assert logits.shape == (2, len(V.ix)) and alpha.shape[:2] == (2, 3)
    assert torch.allclose(alpha.sum((1, 2)), torch.ones(2))                      # one softmax over all docs' tokens
    keep = torch.tensor([[True, False, False]] * 2)
    _, alpha1, _ = rd(["x was born in <mask>"] * 2, docs, keep)
    assert alpha1[:, 1:].sum() < 1e-6                                            # masked documents get no attention


def test_retriever_targets_are_distributions_and_emdr_formula():
    W, V = _setup()
    ret, rd = Retriever(V).init_idf(W["passages"]), FiDReader(V)
    q, a, _ = mlm_examples(W)[0]
    docs = [W["passages"][:4]]
    lp, alpha, vnorm = answer_logp(rd, [q], docs, [a])
    for name in ("adist", "pdist", "loop"):
        t = retriever_target(name, rd, [q], docs, [a], alpha, vnorm)
        assert t.shape == (1, 4) and abs(t.sum().item() - 1) < 1e-5 and (t >= 0).all()
    single = torch.stack([answer_logp(rd, [q], docs, [a], torch.eye(4, dtype=torch.bool)[k:k + 1])[0][0] for k in range(4)])
    assert torch.allclose(retriever_target("pdist", rd, [q], docs, [a], alpha, vnorm)[0], torch.softmax(single, 0), atol=1e-5)
    with torch.no_grad():
        log_pr = torch.log_softmax(ret.q([q]) @ ret.d(docs[0]).T, -1)[0]
    emdr = retriever_loss("emdr", ret, rd, [q], docs, [a], alpha, vnorm)
    assert abs(emdr.item() + torch.logsumexp(single.detach() + log_pr, 0).item()) < 1e-4


def test_pq_and_pretraining_learns():
    torch.manual_seed(0)
    D = torch.randn(300, 16)
    r8, b = product_quantize(D, 4, 8)
    r2, _ = product_quantize(D, 4, 2)
    assert b == 4 and (r8 - D).pow(2).mean() < (r2 - D).pow(2).mean()          # more bits, less error
    W, V = _setup()
    ret, rd = Retriever(V).init_idf(W["passages"]), FiDReader(V)
    ix = Index(ret, W["passages"])
    data = mlm_examples(W)
    before = evaluate(ret, rd, ix, data)
    train_atlas(ret, rd, ix, data, steps=150, loss=None)
    assert evaluate(ret, rd, ix, data) > before + 0.3
