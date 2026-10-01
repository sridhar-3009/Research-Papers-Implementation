"""Tests for multi-query attention (Shazeer 2019). A few seconds.

Run with:  python3 -m pytest -q
"""

import torch
import torch.nn.functional as F

from mqa import (DecoderLM, GroupedAttention, attention_params, incremental_ratio, matching_ffn_width,
                 multihead_attention_batched, multihead_self_attention_incremental, multiquery_attention_batched,
                 multiquery_self_attention_incremental)

torch.manual_seed(0)
b, n, d, h, k = 2, 5, 12, 3, 4


def causal(n, heads=h):
    return torch.triu(torch.full((n, n), float("-inf")), 1).expand(b, heads, n, n)


def test_multi_head_einsum_equals_a_loop_over_heads():
    X = torch.randn(b, n, d)
    P_q, P_k, P_v, P_o = (torch.randn(h, d, k) for _ in range(4))
    out = multihead_attention_batched(X, X, causal(n), P_q, P_k, P_v, P_o)
    loop = 0
    for i in range(h):
        w = torch.softmax(X @ P_q[i] @ (X @ P_k[i]).transpose(1, 2) + causal(n)[:, 0], -1)
        loop = loop + w @ (X @ P_v[i]) @ P_o[i].T
    assert torch.allclose(out, loop, atol=1e-4)


def test_multi_query_is_multi_head_with_all_key_and_value_heads_tied():
    X = torch.randn(b, n, d)
    P_q, P_o = torch.randn(h, d, k), torch.randn(h, d, k)
    P_k, P_v = torch.randn(d, k), torch.randn(d, k)
    mqa = multiquery_attention_batched(X, X, causal(n), P_q, P_k, P_v, P_o)
    mha = multihead_attention_batched(X, X, causal(n), P_q, P_k.expand(h, d, k), P_v.expand(h, d, k), P_o)
    assert torch.allclose(mqa, mha, atol=1e-5)


def test_incremental_steps_equal_the_batched_causal_computation():
    X = torch.randn(b, n, d)
    P_q, P_k, P_v, P_o = (torch.randn(h, d, k) for _ in range(4))
    full = multihead_attention_batched(X, X, causal(n), P_q, P_k, P_v, P_o)
    K, V = torch.zeros(b, h, 0, k), torch.zeros(b, h, 0, k)
    for t in range(n):
        y, K, V = multihead_self_attention_incremental(X[:, t], K, V, P_q, P_k, P_v, P_o)
        assert torch.allclose(y, full[:, t], atol=1e-4)
    Pk1, Pv1 = torch.randn(d, k), torch.randn(d, k)
    fullq = multiquery_attention_batched(X, X, causal(n), P_q, Pk1, Pv1, P_o)
    K, V = torch.zeros(b, 0, k), torch.zeros(b, 0, k)
    for t in range(n):
        y, K, V = multiquery_self_attention_incremental(X[:, t], K, V, P_q, Pk1, Pv1, P_o)
        assert torch.allclose(y, fullq[:, t], atol=1e-4)
    assert K.shape == (b, n, k)                                                     # one cached head, not h


def test_widened_feed_forward_matches_the_paper():
    assert matching_ffn_width(1024, 8, 128, 4096, n_attention_layers=18, n_ffn_layers=12) == 5440   # Table 1
    assert matching_ffn_width(1024, 8, 128, 8192, n_attention_layers=6, n_ffn_layers=6) == 9088     # Table 3
    assert attention_params(1024, 8, 128, 128, 8) - attention_params(1024, 8, 128, 128, 1) == 2 * 1024 * 128 * 7


def test_grouped_attention_covers_mha_and_mqa():
    x = torch.randn(b, n, 16)
    for g in (4, 2, 1):
        att = GroupedAttention(16, 4, g)
        out, (K, V) = att(x)
        assert K.shape == (b, g, n, 4)                                              # g key/value heads cached
        Wq = att.W_q.weight.T.reshape(16, 4, 4).permute(1, 0, 2) / 2              # fold 1/sqrt(k) into P_q
        Wk = att.W_k.weight.T.reshape(16, g, 4).permute(1, 0, 2).repeat_interleave(4 // g, 0)
        Wv = att.W_v.weight.T.reshape(16, g, 4).permute(1, 0, 2).repeat_interleave(4 // g, 0)
        Wo = att.W_o.weight.reshape(16, 4, 4).permute(1, 0, 2)
        ref = multihead_attention_batched(x, x, causal(n, 4), Wq, Wk, Wv, Wo)
        assert torch.allclose(out, ref, atol=1e-5)


def test_kv_cache_generation_equals_recomputation_and_is_h_times_smaller_for_mqa():
    for g in (4, 1):
        m = DecoderLM(30, d=32, h=4, kv_heads=g, d_ff=64, layers=2).eval()
        p = torch.randint(0, 30, (3, 4))
        assert torch.equal(m.generate(p, 6, True), m.generate(p, 6, False))
    assert DecoderLM(30, d=32, h=4, kv_heads=4).cache_numbers(8, 100) == 4 * DecoderLM(30, d=32, h=4, kv_heads=1).cache_numbers(8, 100)


def test_local_attention_window():
    att = GroupedAttention(8, 2, 1, window=3)
    x = torch.randn(1, 10, 8, requires_grad=True)
    att(x)[0][0, 7].sum().backward()
    seen = (x.grad[0].abs().sum(-1) > 0).nonzero()[:, 0].tolist()
    assert seen == [5, 6, 7]                                                         # the current and previous 2


def test_memory_to_compute_ratio():
    assert abs(incremental_ratio(1024, 1024, 8, 128, "mha") - (1 + 1 / 128)) < 1e-12
    assert incremental_ratio(1024, 1024, 8, 128, "mqa") < incremental_ratio(1024, 1024, 8, 128, "mha") / 5


def test_multi_query_model_still_learns():
    torch.manual_seed(0)
    m = DecoderLM(12, d=32, h=4, kv_heads=1, d_ff=64, layers=2)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(200):
        x = torch.randint(3, 12, (32, 6))
        seq = torch.cat([x, x], 1)                                                  # second half repeats the first
        loss = F.cross_entropy(m(seq[:, :-1])[0][:, 5:].reshape(-1, 12), seq[:, 6:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.2
