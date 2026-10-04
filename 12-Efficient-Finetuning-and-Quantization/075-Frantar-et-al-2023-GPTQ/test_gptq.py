import torch

from gptq import grid, gptq, gptq_naive, hessian, layer_error, obq, quant, rtn


def _layer(seed=0, d_row=8, d_col=12, n=200):
    torch.manual_seed(seed)
    X = ((torch.randn(d_col, d_col) @ torch.randn(d_col, n)) / 3).double()
    return torch.randn(d_row, d_col).double(), X


def test_grid_and_rtn():
    w = torch.tensor([[0.9, -0.4, 0.1, 0.35]])
    scale, zero = grid(w, 2)
    assert abs(scale.item() - 1.3 / 3) < 1e-6 and zero.item() == 1
    q = quant(w, scale, zero, 2)
    levels = {round(scale.item() * (k - 1), 5) for k in range(4)}
    assert all(round(v, 5) in levels for v in q.flatten().tolist())
    assert (rtn(w, 2) - w).abs().max() <= scale.item() / 2 + 1e-6


def test_gptq_equals_literal_update_and_beats_rtn():
    W, X = _layer()
    H = hessian(X)
    for bits in (4, 3):
        a, b, c = gptq_naive(W, H, bits), gptq(W, H, bits, blocksize=1), gptq(W, H, bits, blocksize=5)
        assert torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)    # blocking changes nothing
        assert layer_error(W, a.double(), X) < 0.8 * layer_error(W, rtn(W, bits).double(), X)


def test_obq_two_weight_example_and_order():
    X = torch.tensor([[1.0, 0.9, 1.1, 0.95], [1.0, 1.1, 0.9, 1.05]], dtype=torch.float64)
    W = torch.tensor([[0.3, 0.3]], dtype=torch.float64)
    H = 2 * X @ X.T
    Hinv = torch.linalg.inv(H)
    w2 = W[0, 1] - (W[0, 0] - 0.5) / Hinv[0, 0] * Hinv[0, 1]
    assert w2 < 0.25                                                                 # compensation lowers w2
    W2, X2 = _layer(1)
    H2 = hessian(X2)
    e_obq, e_gptq = layer_error(W2, obq(W2, H2, 3).double(), X2), layer_error(W2, gptq(W2, H2, 3).double(), X2)
    assert e_gptq < 1.5 * e_obq                                                       # fixed order ~ greedy order
