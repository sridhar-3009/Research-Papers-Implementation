import math

import numpy as np

from inference import (Mesh, PALM, ffn_1d_ws, ffn_2d_ws, ffn_reference, ffn_weight_gathered, kv_bytes_per_token,
                       max_context, step_time)


def test_table1_max_context():
    assert abs(max_context("multihead", 128) - 1320) / 1320 < 0.02
    assert abs(max_context("baseline multiquery", 512) - 165) / 165 < 0.02
    assert abs(max_context("optimized multiquery", 128) - 43000) / 43000 < 0.02
    n, L, E, F, H, dh = PALM["540B"]
    assert abs(kv_bytes_per_token(L, H, 128, H) * 512 * 2048 / 1e12 - 3.0) < 0.1      # the paper's '3TB'


def test_partitioned_ffn_exact_and_comm_scaling():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 256))
    Wi, Wo = rng.standard_normal((256, 1024)) / 16, rng.standard_normal((1024, 256)) / 32
    ref = ffn_reference(x, Wi, Wo)
    sent = {}
    for n, X in ((16, 2), (64, 4)):
        m1, m2, m3 = Mesh(n), Mesh(n), Mesh(n)
        assert np.allclose(np.concatenate(ffn_1d_ws(np.array_split(x, n, axis=-1), Wi, Wo, m1), axis=-1), ref)
        assert np.allclose(ffn_2d_ws(x, Wi, Wo, m2, X, n // X), ref)
        assert np.allclose(ffn_weight_gathered(x, Wi, Wo, m3), ref)
        sent[n] = (m1.sent.mean() / x.nbytes, m2.sent.mean() / x.nbytes)
    assert abs(sent[64][0] - 2 * 63 / 64) < 1e-9                                      # 1D: ~2 BLE, constant
    assert sent[64][1] < sent[16][1] * 0.7                                            # 2D: shrinks with chips


def test_roofline_regimes():
    pre = step_time("540B", 64, 512, 2048, "prefill", "WG", 2)
    dec = step_time("540B", 64, 1, 2048, "decode", "2D WS", 2)
    assert pre["compute"] > pre["memory"] and dec["memory"] > dec["compute"]          # prefill compute-bound
    assert pre["MFU"] > dec["MFU"]
    assert step_time("540B", 64, 64, 2048, "decode", "2D WS", 1)["total"] < step_time("540B", 64, 64, 2048, "decode", "2D WS", 2)["total"]
