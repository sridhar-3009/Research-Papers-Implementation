import numpy as np

from zero import Cluster, comm_volume, init_params, loss_and_grad, model_state_bytes, train, unflatten


def test_memory_formulas_reproduce_figure1_and_table1():
    gb = [round(model_state_bytes(7.5e9, 64, s) / 1e9, 1) for s in range(4)]
    assert gb == [120.0, 31.4, 16.6, 1.9]
    assert model_state_bytes(1e12, 4, 1) / 1e9 == 7000 and model_state_bytes(1e12, 1024, 3) / 1e9 == 15.625
    assert comm_volume(3) / comm_volume(1) == 1.5


def test_collectives_and_gradient():
    cl = Cluster(4)
    vecs = [np.arange(8, dtype=np.float32) * (r + 1) for r in range(4)]
    full = cl.all_reduce(vecs)
    assert np.allclose(full, np.arange(8) * 10) and np.allclose(cl.sent, 2 * 3 / 4 * 8)
    flat, shapes = init_params(d_in=3, hidden=4)
    X, y = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32), np.ones((5, 1), np.float32)
    _, g = loss_and_grad(flat, shapes, X, y)
    i, eps = 2, 1e-3
    fp, fm = flat.copy(), flat.copy()
    fp[i] += eps
    fm[i] -= eps
    num = (loss_and_grad(fp, shapes, X, y)[0] - loss_and_grad(fm, shapes, X, y)[0]) / (2 * eps)
    assert abs(num - g[i]) < 1e-3 and [a.shape for a in unflatten(flat, shapes)] == [(3, 4), (4,), (4, 1), (1,)]


def test_all_stages_identical_weights_and_less_memory():
    res = {s: train(s, nd=4, steps=10) for s in range(4)}
    assert all(np.array_equal(res[0]["weights"], res[s]["weights"]) for s in (1, 2, 3))
    mem = [res[s]["persistent bytes"] for s in range(4)]
    assert mem[0] > mem[1] > mem[2] > mem[3]
    assert abs(res[3]["sent per step (x psi)"] / res[0]["sent per step (x psi)"] - 1.5) < 1e-9
    assert res[0]["losses"][-1] < res[0]["losses"][0]
