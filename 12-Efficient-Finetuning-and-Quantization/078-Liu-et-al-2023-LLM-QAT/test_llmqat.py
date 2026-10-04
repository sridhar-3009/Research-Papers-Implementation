import numpy as np
import torch

from llmqat import (BOS, LEN, QLinear, TASK, diversity, fake_quant, generate, make_seq, new_model, quantize, train_qat)


def test_fake_quant_values_and_ste():
    x = torch.tensor([[0.3, -1.2, 0.05, 2.0]], requires_grad=True)
    q = fake_quant(x, 4, -1)
    alpha = 2.0 / 7
    assert torch.allclose(q, torch.round(x.detach() / alpha) * alpha)
    q.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))                                  # straight-through
    w = torch.tensor([[1.0, 0.1], [100.0, 10.0]])
    qw = fake_quant(w, 4, 1)                                                         # per output channel
    assert abs(qw[0, 1].item() - 1 / 7) < 1e-6 and abs(qw[1, 1].item() - 100 / 7) < 1e-4


def test_quantized_model_and_generation_shapes():
    torch.manual_seed(0)
    m = new_model()
    q = quantize(m, 4, 8, 4)
    assert isinstance(q.blocks[0].attn.wk, QLinear) and q.blocks[0].attn.wk.kv_bits == 4
    assert q.blocks[0].attn.wq.kv_bits is None
    seq = make_seq("sort", 5, np.random.default_rng(0))
    assert seq.shape == (5, LEN) and (seq[:, 0] == BOS).all() and (seq[:, 1] == TASK["sort"]).all()
    g = generate(m, 7, "sample")
    assert g.shape == (7, LEN) and (g[:, 0] == BOS).all()
    assert generate(m, 4, "top1").unique(dim=0).shape[0] == 1                       # top-1 is deterministic
    d, v = diversity(make_seq("copy", 10, np.random.default_rng(1)))
    assert v == 1.0


def test_qat_updates_through_quantized_forward():
    torch.manual_seed(0)
    teacher = new_model()
    student = quantize(teacher, 3, 8, 8)
    before = student.blocks[0].attn.wq.weight.detach().clone()
    data = make_seq("copy", 32, np.random.default_rng(2))
    train_qat(student, teacher, data, steps=3, batch=8)
    assert not torch.equal(before, student.blocks[0].attn.wq.weight.detach())     # STE lets weights learn
    assert torch.equal(teacher.blocks[0].attn.wq.weight.detach(), before)          # teacher untouched
