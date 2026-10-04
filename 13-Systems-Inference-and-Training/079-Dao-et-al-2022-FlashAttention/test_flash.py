import math

import torch

from flash import block_sizes, flash_attention, flash_backward, online_softmax, standard_attention


def test_online_softmax_matches():
    x = torch.randn(37)
    p, m, l = online_softmax(x, 5)
    assert torch.allclose(p, torch.softmax(x, 0), atol=1e-6) and abs(m - x.max().item()) < 1e-7


def test_flash_forward_backward_exact():
    torch.manual_seed(0)
    N, d, M = 96, 16, 1024
    Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
    O_std, _ = standard_attention(Q, K, V)
    O, m, l, _ = flash_attention(Q, K, V, M)
    assert torch.allclose(O, O_std, atol=1e-5)
    Qr, Kr, Vr = (t.clone().requires_grad_() for t in (Q, K, V))
    (torch.softmax(Qr @ Kr.T / math.sqrt(d), -1) @ Vr).backward(dO := torch.randn(N, d))
    dQ, dK, dV, _ = flash_backward(Q, K, V, O, dO, m, l, M)
    for a, b in ((dQ, Qr.grad), (dK, Kr.grad), (dV, Vr.grad)):
        assert torch.allclose(a, b, atol=1e-4)


def test_io_scaling_and_block_sparsity():
    d = 16
    assert block_sizes(4096, d) == (64, 16)
    counts = {}
    for N in (128, 256):
        for M in (2048, 8192):
            Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
            counts[(N, M)] = flash_attention(Q, K, V, M)[3].total
    assert 3.0 < counts[(256, 8192)] / counts[(128, 8192)] < 4.5           # ~N^2
    assert counts[(256, 2048)] / counts[(256, 8192)] > 2.5                   # ~1/M
    Q, K, V = torch.randn(256, d), torch.randn(256, d), torch.randn(256, d)
    bc, br = block_sizes(4096, d)
    mask = torch.zeros(math.ceil(256 / br), math.ceil(256 / bc), dtype=torch.bool)
    mask[:, 0] = True
    sparse = flash_attention(Q, K, V, 4096, block_mask=mask)
    dense = flash_attention(Q, K, V, 4096)
    assert sparse[3].total < dense[3].total / 2
    # with only the first key block, each row attends to keys 0..63 exactly
    ref = torch.softmax(Q @ K[:bc].T / 4, -1) @ V[:bc]
    assert torch.allclose(sparse[0], ref, atol=1e-5)
