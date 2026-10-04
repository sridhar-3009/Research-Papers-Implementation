import torch

from gpipe import (bubble_fraction, gpipe_step, make_model, normalised_throughput, partition, reference_grads, simulate,
                   split_stages)


def test_schedule_bubble_matches_formula():
    for K, M in ((2, 1), (4, 4), (8, 32)):
        total, busy, _ = simulate([1.0] * K, M, remat=False)
        assert abs((1 - busy.mean() / total) - bubble_fraction(K, M)) < 1e-9


def test_partition_and_throughput():
    assert partition([1] * 8, 4) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    st = partition([5, 1, 1, 1, 1, 1, 1, 5], 3)
    assert sum(len(s) for s in st) == 8 and all(st)
    u = [1.0] * 48
    assert normalised_throughput(u, 8, 32) > normalised_throughput(u, 4, 32) > normalised_throughput(u, 2, 32)
    assert normalised_throughput(u, 8, 1) == 1.0                                 # no overlap with one micro-batch
    heavy = [1.0] * 47 + [47.0]                                                  # one layer = half the work
    assert normalised_throughput(heavy, 8, 32) < normalised_throughput(u, 8, 32) / 2


def test_pipeline_gradients_equal_full_batch_and_remat_saves_memory():
    m = make_model(L=6, d=16)
    x, y = torch.randn(16, 16), torch.randn(16, 1)
    _, g_ref = reference_grads(m, x, y)
    kept = {}
    for remat in (False, True):
        m.zero_grad()
        _, kept[remat] = gpipe_step(split_stages(m, 3), x, y, M=4, remat=remat)
        assert all(torch.allclose(p.grad, g, atol=1e-6) for p, g in zip(m.parameters(), g_ref))
    assert kept[True] * 3 < kept[False]
