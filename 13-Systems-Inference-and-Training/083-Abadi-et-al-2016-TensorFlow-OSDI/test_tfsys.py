import numpy as np

from tfsys import (T, backup_worker_curve, full_softmax_loss, run_with_failures, sampled_softmax_grad,
                   sharded_embedding, sparse_update, sync_step_time, train_replicated)


def test_sharded_embedding_equals_dense_lookup_and_sparse_update():
    rng = np.random.default_rng(0)
    G = T.Graph()
    ids = G.placeholder("ids")
    emb, shards, table = sharded_embedding(G, ids, 50, 4, 3, rng)
    I = np.array([0, 49, 7, 7, 13])
    assert np.allclose(T.Session(G).run(emb, {ids: I.astype(float)}), table[I])
    before = G.variables["emb_shard1"].copy()
    n = sparse_update(G, "emb_shard1", np.array([2, 2, 5]), np.ones((3, 4)), 0.5)
    changed = np.where((G.variables["emb_shard1"] != before).any(1))[0]
    assert n == 2 and changed.tolist() == [2, 5]


def test_sampled_softmax_touches_few_rows_and_is_finite():
    rng = np.random.default_rng(1)
    W, h, y = rng.standard_normal((1000, 8)), rng.standard_normal((16, 8)), rng.integers(0, 1000, 16)
    loss, classes, gW, gh = sampled_softmax_grad(h, W, y, 64, rng)
    assert len(classes) <= 80 and set(y) <= set(classes) and np.isfinite(loss)
    assert gW.shape == (len(classes), 8) and gh.shape == h.shape
    assert np.isfinite(full_softmax_loss(h, W, y))


def test_backups_checkpoints_and_replication():
    rng = np.random.default_rng(2)
    assert sync_step_time(rng, 10, 0, ps_cost=0.0) >= 0.0
    c = backup_worker_curve(n=50, max_b=4, steps=500)
    assert c[4]["median step"] < c[0]["median step"]
    assert run_with_failures(fail_at=())["wasted steps"] == 0
    assert run_with_failures(ckpt_every=100, fail_at=(150,))["wasted steps"] == 50
    a = train_replicated("async", n_workers=5, wall=5)
    s = train_replicated("sync", n_workers=5, wall=5)
    assert a["updates"] > s["updates"] and a["mean staleness"] > 0 and s["mean staleness"] == 0
    assert a["final loss"] < a["curve"][0][1] and s["final loss"] < s["curve"][0][1]
