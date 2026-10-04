import copy

import numpy as np
import torch

from qlora import (DATA_TYPES, L, add_lora_everywhere, bits_per_parameter, double_quantize, fake_quant, nf4_values,
                   quantize_blockwise, quantize_model)

PAPER_NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
             -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
             0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


def test_nf4_matches_appendix_and_beats_int4_on_gaussians():
    assert torch.allclose(nf4_values(), torch.tensor(PAPER_NF4), atol=1e-6)
    assert len(nf4_values()) == 16 and 0.0 in nf4_values().tolist()
    torch.manual_seed(0)
    W = torch.randn(256, 64)
    err = {k: float((fake_quant(W, v) - W).pow(2).mean()) for k, v in DATA_TYPES.items()}
    assert err["NF4"] < err["Int4"] and err["NF4"] < err["FP4 (E2M1)"]


def test_blockwise_and_double_quantization():
    torch.manual_seed(1)
    W = torch.randn(4, 128)
    codes, c = quantize_blockwise(W, nf4_values())
    assert codes.shape == (8, 64) and torch.allclose(c, W.reshape(8, 64).abs().amax(1))
    assert abs(bits_per_parameter() - 4.5) < 1e-9 and abs(bits_per_parameter(dq=True) - (4 + 8 / 64 + 32 / (64 * 256))) < 1e-9
    cc = torch.rand(1000) + 0.5
    assert (double_quantize(cc) - cc).abs().max() < 0.01


def test_qlora_base_is_frozen_and_quantized():
    base = L.new_model()
    q = add_lora_everywhere(quantize_model(base, nf4_values()), r=2)
    trainable = [n for n, p in q.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".A", ".B")) for n in trainable)
    assert len(trainable) == 2 * 7 * 2                                          # 2 blocks x 7 linears x (A, B)
    w = q.blocks[0].attn.wq.base.weight
    assert len(torch.unique(w[0, :64] / w[0, :64].abs().max())) <= 16           # each block lives on 16 NF4 levels
