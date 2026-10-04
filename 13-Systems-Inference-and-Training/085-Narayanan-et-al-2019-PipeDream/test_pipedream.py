import numpy as np

from pipedream import (backward, config_string, forward, init_stages, noam, partition_dp, simulate_1f1b,
                       simulate_gpipe, train_pipelined)


def test_partitioner_finds_hybrid_layout():
    compute = [4, 4, 6, 6, 6, 5, 5, 5, 3, 3, 1, 1]
    weights = [0.01, 0.04, 0.1, 0.15, 0.3, 0.6, 0.6, 0.6, 1, 1, 40, 8]
    acts = [3, 3, 2, 2, 1.5, 1, 1, 0.8, 0.5, 0.5, 0.05, 0.01]
    t, st = partition_dp(compute, weights, acts, 8, bandwidth=0.5)
    assert config_string(st) == "7-1" and noam(st) == 2
    assert sum(m for _, _, m in st) == 8 and st[0][0] == 0 and st[-1][1] == 11
    t_fast, st_fast = partition_dp(compute, weights, acts, 8, bandwidth=4.0)
    assert config_string(st_fast) == "8"


def test_1f1b_schedule_and_memory():
    t1, peak = simulate_1f1b([1.0] * 4, 16)
    tg, _ = simulate_gpipe([1.0] * 4, 4)
    assert peak == [4, 3, 2, 1] and t1 < 4 * tg
    assert simulate_1f1b([1.0] * 4, 4)[0] == simulate_gpipe([1.0] * 4, 4)[0]      # same window, no flush difference


def test_manual_backprop_and_weight_semantics():
    rng = np.random.default_rng(0)
    Ws = init_stages(3, d_in=4, width=5)
    x, y = rng.standard_normal((6, 4)), rng.standard_normal((6, 1))
    g = backward(Ws, forward(Ws, x), y)
    eps, W0 = 1e-6, Ws[1].copy()
    Ws[1][2, 3] += eps
    lp = float(((forward(Ws, x)[-1] - y) ** 2).mean())
    Ws[1][2, 3] -= 2 * eps
    lm = float(((forward(Ws, x)[-1] - y) ** 2).mean())
    Ws[1] = W0
    assert abs((lp - lm) / (2 * eps) - g[1][2, 3]) < 1e-6
    c = {m: train_pipelined(m, 3, steps=200, lr=0.1) for m in ("sgd", "naive", "stash")}
    assert all(v[-1] < v[0] for v in c.values())
    assert c["sgd"][-1] <= min(c["naive"][-1], c["stash"][-1]) + 1e-3
