import numpy as np
import torch

from smoothquant import (L, QuantLinear, effective_levels, inject_outliers, quantize, quantized_copy,
                         record_input_absmax, smooth, task_accuracy)


def test_quantizer_and_levels():
    x = torch.tensor([0.3, -1.2, 0.05, 2.0])
    q = quantize(x)
    assert (q - x).abs().max() <= 2.0 / 127 / 2 + 1e-7                           # error <= Delta / 2
    assert torch.equal(quantize(x, bits=8, scale=torch.tensor(1.0)), x.round())
    X = torch.tensor([[1.0, 100.0], [-0.5, 90.0]])
    assert torch.allclose(effective_levels(X), torch.tensor([2.56, 256.0]))
    rows = quantize(torch.tensor([[1.0, 0.01], [100.0, 1.0]]), dim=-1)
    assert abs(rows[0, 1].item() - 0.0079) < 1e-3                                 # per-row scale keeps small rows


def test_smoothing_is_exact_and_flattens_activations():
    torch.manual_seed(0)
    m = L.new_model()
    with torch.no_grad():                                                         # function-changing outliers, no retraining
        m = inject_outliers(m, np.random.default_rng(0), steps=0)
    calib = L.make_batch("copy", 32, np.random.default_rng(1))
    sm = smooth(m, calib, 0.5)
    assert torch.allclose(sm(calib), m(calib), atol=1e-3)
    before = record_input_absmax(m, calib)[(0, "attn", "wq")]
    after = record_input_absmax(sm, calib)[(0, "attn", "wq")]
    assert after.max() / after.min() < before.max() / before.min() / 10


def test_quant_linear_modes():
    lin = torch.nn.Linear(4, 3, bias=False)
    x = torch.randn(2, 5, 4)
    for kw in (dict(a_gran="tensor"), dict(a_gran="token"), dict(a_gran="channel"), dict(outlier=0.5)):
        y = QuantLinear(lin, **kw)(x)
        assert y.shape == (2, 5, 3) and torch.allclose(y, lin(x), atol=0.1)
    exact = QuantLinear(lin, w_bits=None, a_bits=None)(x)
    assert torch.allclose(exact, lin(x), atol=1e-6)
