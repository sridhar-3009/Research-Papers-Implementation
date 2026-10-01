"""Tests for Sparse Transformers (Child et al. 2019). A few seconds.

Run with:  python3 -m pytest -q
"""

import math

import torch
import torch.nn.functional as F

from sparse import (SparseTransformerLM, block_local_attention, causal, entries, fixed_patterns, gelu_approx,
                    masked_attention, reachable_in_two, strided_column_attention, strided_patterns)

torch.manual_seed(0)


def test_patterns_match_the_definitions():
    A1, A2 = strided_patterns(12, 4)
    assert A1[9].nonzero()[:, 0].tolist() == [6, 7, 8, 9]                          # the previous l positions
    assert A2[9].nonzero()[:, 0].tolist() == [1, 5, 9]                              # every l-th position back
    F1, F2 = fixed_patterns(12, 4, 2)
    assert F1[9].nonzero()[:, 0].tolist() == [8, 9]                                 # same block
    assert F2[9].nonzero()[:, 0].tolist() == [2, 3, 6, 7, 9]                        # summary cells (+ itself)
    for M in (A1, A2, F1, F2):
        assert not (M & ~causal(12)).any()                                          # never the future


def test_every_earlier_position_is_reachable_in_two_steps():
    n = 64
    A1, A2 = strided_patterns(n, 8)
    assert torch.equal(reachable_in_two(A1, A2), causal(n)) and torch.equal(reachable_in_two(A2, A1), causal(n))
    F1, F2 = fixed_patterns(n, 8, 2)
    assert torch.equal(reachable_in_two(F1, F2), causal(n))                         # block first, then summary
    assert not torch.equal(reachable_in_two(F2, F1), causal(n))                     # the other order is not enough


def test_cost_grows_like_n_sqrt_n():
    for n in (256, 1024, 4096):
        l = int(math.sqrt(n))
        A1, A2 = strided_patterns(n, l)
        assert entries(A1 | A2) < 2 * n * l                                         # ~ n (l + n/l) = 2 n sqrt(n)
        assert entries(causal(n)) == n * (n + 1) // 2


def test_efficient_blocked_and_strided_kernels_equal_the_masked_ones():
    q, k, v = torch.randn(3, 2, 64, 16).unbind(0)
    F1, _ = fixed_patterns(64, 8, 2)
    _, A2 = strided_patterns(64, 8)
    assert torch.allclose(block_local_attention(q, k, v, 8), masked_attention(q, k, v, F1), atol=1e-5)
    assert torch.allclose(strided_column_attention(q, k, v, 8), masked_attention(q, k, v, A2), atol=1e-5)


def test_head_modes():
    m = SparseTransformerLM(10, 16, d=16, layers=2, heads=2, pattern="strided", stride=4, mode="interleave")
    A1, A2 = strided_patterns(16, 4)
    assert torch.equal(m.blocks[0].attn.head_masks(16)[0], A1) and torch.equal(m.blocks[1].attn.head_masks(16)[0], A2)
    m.blocks[0].attn.mode = "merged"
    assert torch.equal(m.blocks[0].attn.head_masks(16)[1], A1 | A2)
    m.blocks[0].attn.mode = "multihead"
    assert [torch.equal(x, y) for x, y in zip(m.blocks[0].attn.head_masks(16), (A1, A2))] == [True, True]


def test_initialisation_and_residual_block():
    m = SparseTransformerLM(10, 16, d=32, layers=8, heads=2, stride=4)
    x = torch.randint(0, 10, (2, 16))
    assert torch.allclose(F.softmax(m(x), -1), torch.full((2, 16, 10), 0.1))      # zero logits: uniform at start
    std = m.blocks[0].W2.weight.std().item()
    assert abs(std / (0.125 / math.sqrt(128) / math.sqrt(16)) - 1) < 0.2            # scaled by 1/sqrt(2N)
    blk, H = m.blocks[0], torch.randn(2, 16, 32)
    a = blk.attn(blk.n1(H))
    b = blk.W2(gelu_approx(blk.W1(blk.n2(H + a))))
    assert torch.allclose(blk(H), a + b, atol=1e-6)                                 # Eqs. 12-14


def test_gelu_approximation():
    x = torch.linspace(-5, 5, 1001)
    assert (gelu_approx(x) - F.gelu(x)).abs().max() < 0.03


def test_recomputation_gives_the_same_gradients():
    torch.manual_seed(0)
    a = SparseTransformerLM(10, 16, d=16, layers=2, heads=2, stride=4)
    b = SparseTransformerLM(10, 16, d=16, layers=2, heads=2, stride=4, recompute=True)
    b.load_state_dict(a.state_dict())
    nn_out = torch.randn(10, 16)
    with torch.no_grad():
        a.out.weight.copy_(nn_out); b.out.weight.copy_(nn_out)
    x = torch.randint(0, 10, (3, 16))
    for m in (a, b):
        F.cross_entropy(m(x[:, :-1]).reshape(-1, 10), x[:, 1:].reshape(-1)).backward()
    assert torch.allclose(a.emb.weight.grad, b.emb.weight.grad, atol=1e-6)


def test_strided_model_learns_to_copy_the_row_above():
    torch.manual_seed(0)
    V, n, l = 8, 64, 8
    m = SparseTransformerLM(V, n, d=64, layers=2, heads=2, pattern="strided", stride=l)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(150):
        x = torch.randint(0, V, (32, l)).repeat(1, n // l)
        loss = F.cross_entropy(m(x[:, :-1])[:, l - 1:].reshape(-1, V), x[:, l:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1
