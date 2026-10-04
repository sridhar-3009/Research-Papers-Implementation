import torch

from awq import awq_scale, clip_search, wq


def test_scaling_salient_weight_reduces_its_error():
    w = torch.tensor([[0.07, -0.9, 0.42, 0.55, -0.31, 0.88, 0.12, -0.6]])
    x = torch.tensor([50.0, 1, 1, 1, 1, 1, 1, 1])
    errs = []
    for s in (1.0, 4.0):
        sc = torch.ones(8)
        sc[0] = s
        errs.append(((wq(w * sc, 3, None) / sc - w) @ x).abs().item())
    assert errs[1] < errs[0] / 3


def test_awq_search_prefers_activation_aware_scaling():
    torch.manual_seed(0)
    X = torch.randn(256, 32)
    X[:, :2] *= 40                                                       # two salient input channels
    W = torch.randn(16, 32)
    W[:, :2] *= 0.05                                                     # with small weights
    s, alpha, err = awq_scale([W], X, 3, 8)
    plain = float((X @ wq(W, 3, 8).T - X @ W.T).pow(2).sum())
    assert alpha > 0 and err < plain and s[:2].min() > s[2:].max()
    # the transformation is exact before quantization
    assert torch.allclose((X / s) @ (W * s).T, X @ W.T, atol=1e-3)


def test_clipping_never_increases_error():
    torch.manual_seed(1)
    X, W = torch.randn(128, 16), torch.randn(8, 16)
    W[0, 0] = 6.0                                                        # one extreme weight
    e0 = float((X @ wq(W, 3, 8).T - X @ W.T).pow(2).sum())
    e1 = float((X @ wq(clip_search(W, X, 3, 8), 3, 8).T - X @ W.T).pow(2).sum())
    assert e1 <= e0 + 1e-4
