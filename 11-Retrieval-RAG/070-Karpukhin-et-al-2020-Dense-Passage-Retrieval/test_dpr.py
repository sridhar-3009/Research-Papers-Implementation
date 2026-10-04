import math

import numpy as np
import torch

from dpr import (BM25, DenseIndex, in_batch_loss, make_questions, make_world, rule_reader, top_k_accuracy, train_dpr)


def test_bm25_formula():
    docs = ["a b c", "a a d", "e f g h"]
    bm = BM25(docs, k1=0.9, b=0.4)
    n, df_a = 3, 2
    idf = math.log(1 + (n - df_a + 0.5) / (df_a + 0.5))
    avgdl = (3 + 3 + 4) / 3
    tf, dl = 2, 3
    expect = idf * tf * 1.9 / (tf + 0.9 * (1 - 0.4 + 0.4 * dl / avgdl))
    s = bm.scores("a")
    assert abs(s[1] - expect) < 1e-12 and s[1] > s[0] > 0 and s[2] == 0


def test_in_batch_loss_is_cross_entropy_over_shared_negatives():
    torch.manual_seed(0)
    q, p, h = torch.randn(3, 4), torch.randn(3, 4), torch.randn(2, 4)
    S = q @ torch.cat([p, h]).T                                                     # 3 x 5
    manual = -(S.diag() - S.logsumexp(1)).mean()
    assert abs(in_batch_loss(q, p, h).item() - manual.item()) < 1e-6
    assert in_batch_loss(10 * p, p).item() < 0.01                                   # aligned vectors: loss ~ 0


def test_world_reader_and_training_beats_bm25_on_paraphrase():
    W = make_world(n_people=120, n_filler=300)
    Q = make_questions(W)
    assert rule_reader("which musical instrument is kazormi known for ?", ["in the evenings kazormi plays the harp ."]) == "harp"
    bm = BM25(W["passages"])
    m = train_dpr(W, Q["train"], bm25=bm, epochs=40, batch=32)
    qs = Q["test seen people, paraphrase"]
    dense = top_k_accuracy(DenseIndex(m, W["passages"]).scores([q for q, *_ in qs]), qs, W["passages"], (1,))[1]
    sparse = top_k_accuracy(np.stack([bm.scores(q) for q, *_ in qs]), qs, W["passages"], (1,))[1]
    assert dense > sparse + 0.2
