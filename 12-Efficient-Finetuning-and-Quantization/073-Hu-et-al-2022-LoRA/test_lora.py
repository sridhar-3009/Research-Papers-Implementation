import copy

import numpy as np
import torch

from lora import (LoRALinear, accuracy, add_lora, amplification, finetune, lora_params_gpt3, new_model, set_adapters,
                  subspace_similarity, trainable)


def test_lora_layer_zero_start_merge_and_rank():
    base = torch.nn.Linear(16, 12, bias=False)
    lin = LoRALinear(copy.deepcopy(base), r=3, alpha=6)
    x = torch.randn(5, 16)
    assert torch.equal(lin(x), base(x))                                          # B = 0
    with torch.no_grad():
        lin.B.normal_()
    assert abs(lin.scale - 2.0) < 1e-12 and torch.linalg.matrix_rank(lin.delta()) == 3
    y = lin(x)
    lin.merge()
    assert torch.allclose(lin(x), y, atol=1e-5)
    lin.unmerge()
    assert torch.allclose(lin.base.weight, base.weight, atol=1e-6)


def test_injection_freezes_and_counts():
    m = add_lora(new_model(), ("wq", "wv"), r=4)
    assert trainable(m) == 2 * 2 * 4 * (64 + 64)                                  # 2 layers x 2 matrices x r(d+k)
    assert lora_params_gpt3(4, 2) == 96 * 2 * 4 * 2 * 12288                       # ~18.9M
    rng = np.random.default_rng(0)
    before = copy.deepcopy(m.state_dict())
    finetune(m, "desc", rng, steps=3, lr=1e-2)
    after = m.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed and all(k.endswith((".A", ".B")) for k in changed)          # only LoRA weights move
    set_adapters(m, False)
    fresh = new_model()
    x = torch.randint(0, 16, (2, 14))
    assert torch.allclose(m(x), fresh(x), atol=1e-6)                             # adapters off = base model


def test_analyses():
    torch.manual_seed(0)
    A = torch.randn(4, 32)
    assert abs(subspace_similarity(A, A, 4, 4) - 1) < 1e-5
    big = torch.cat([A, torch.randn(12, 32)])
    assert subspace_similarity(A, big, 4, 16) > 0.99                              # A's span lies inside big's
    W, dW = torch.randn(20, 20), torch.randn(20, 2) @ torch.randn(2, 20)
    out = amplification(W, dW, 2)
    assert out["on W's own top directions"] >= out["||U^T W V^T|| on dW's directions"] - 1e-6
