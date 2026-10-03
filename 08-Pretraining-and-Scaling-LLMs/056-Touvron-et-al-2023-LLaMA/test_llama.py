import math

import torch
import torch.nn.functional as F

from llama import (COMMON_SENSE, LLaMA, RMSNorm, SwiGLU, TABLE_1, TABLE_2, TABLE_15_GPU_HOURS, apply_rope, byte_fallback,
                   carbon, ffn_hidden, lr_schedule, memory_efficient_causal_attention, param_count, rope_frequencies,
                   split_digits, training_days)


def test_rmsnorm():
    x = torch.tensor([[3.0, -4.0, 0.0, 0.0]])                                    # mean square = 25/4 -> rms 2.5
    assert torch.allclose(RMSNorm(4, eps=0)(x), x / 2.5)
    y = torch.randn(5, 16) * 7 + 3
    out = RMSNorm(16)(y)
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(5), atol=1e-4)        # unit RMS, but the mean is NOT removed


def test_swiglu_hidden_size_and_parameter_parity():
    assert [ffn_hidden(d) for d in (4096, 5120, 6656, 8192)] == [11008, 13824, 17920, 22016]
    d = 768
    swiglu = sum(p.numel() for p in SwiGLU(d, hidden=int(2 * 4 * d / 3)).parameters())
    assert abs(swiglu / (2 * d * 4 * d) - 1) < 1e-3                              # same size as a 4d two-matrix MLP
    s = SwiGLU(4, hidden=3)
    x = torch.randn(2, 4)
    assert torch.allclose(s(x), s.w2(F.silu(s.w1(x)) * s.w3(x)))


def test_rope_is_a_rotation_and_depends_only_on_relative_position():
    cos, sin = rope_frequencies(8, 64)
    q, k = torch.randn(8), torch.randn(8)

    def at(v, m):
        return apply_rope(v.expand(64, 8), cos, sin)[m]
    assert abs(at(q, 17).norm() - q.norm()) < 1e-5                                # norms preserved
    s1 = at(q, 10) @ at(k, 3)
    s2 = at(q, 40) @ at(k, 33)                                                   # same offset 7
    assert abs(s1 - s2) < 1e-4 and abs(at(q, 0) @ at(k, 0) - q @ k) < 1e-5


def test_memory_efficient_attention_matches_standard():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 150, 16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(memory_efficient_causal_attention(q, k, v, block=32), ref, atol=1e-5)
    torch.manual_seed(0)
    m1 = LLaMA(vocab=50, d=32, layers=2, heads=4, ctx=40)
    torch.manual_seed(0)
    m2 = LLaMA(vocab=50, d=32, layers=2, heads=4, ctx=40, efficient=True)
    ids = torch.randint(0, 50, (2, 40))
    assert torch.allclose(m1(ids), m2(ids), atol=1e-4)


def test_model_is_causal_and_count_formula_matches():
    torch.manual_seed(0)
    m = LLaMA(vocab=100, d=64, layers=2, heads=4, ctx=32)
    assert sum(p.numel() for p in m.parameters()) == param_count(64, 2, vocab=100)
    ids = torch.randint(0, 100, (1, 12))
    ids2 = ids.clone(); ids2[0, -1] = (ids[0, -1] + 1) % 100
    assert torch.allclose(m(ids)[0, :-1], m(ids2)[0, :-1], atol=1e-5)


def test_table_2_parameter_counts():
    for name, (reported, d, heads, layers, lr, tokens) in TABLE_2.items():
        assert abs(param_count(d, layers) / reported - 1) < 0.006
        assert d % heads == 0 and d // heads == 128                             # every LLaMA uses 128-dim heads


def test_training_time_and_carbon():
    assert abs(training_days(1.4e12) - 21) < 0.5                                 # 'approximately 21 days'
    expected_t = {"7B": 14, "13B": 23, "33B": 90, "65B": 173, "OPT-175B": 137, "BLOOM-175B": 183}
    for k, t in expected_t.items():
        assert round(carbon(TABLE_15_GPU_HOURS[k])[1]) == t
    assert abs(sum(p for p, _, _ in TABLE_1.values()) - 1) < 1e-9


def test_tokenizer_rules_and_schedule():
    assert split_digits("in 2023 we") == ["in ", "2", "0", "2", "3", " we"]
    assert byte_fallback("a€", {"a"}) == ["a", "<0xE2>", "<0x82>", "<0xAC>"]
    assert lr_schedule(0, 3e-4) == 3e-4 / 2000 and abs(lr_schedule(1999, 3e-4) - 3e-4) < 1e-12
    assert abs(lr_schedule(250_000, 3e-4) - 3e-5) < 1e-12                        # ends at 10% of the peak


def test_13b_beats_gpt3_on_most_common_sense_tasks():
    wins = sum(a > b for a, b in zip(COMMON_SENSE["LLaMA-13B"], COMMON_SENSE["GPT-3 175B"]))
    assert wins == 5 and len(COMMON_SENSE["GPT-3 175B"]) == 7
