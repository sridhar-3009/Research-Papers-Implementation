import math

import numpy as np

from chinchilla import (ALPHA, BETA, GOPHER_FLOPS, TABLE_3, A, B, E, approach1_envelope, approach2_isoflop,
                        fit_parametric, flops_appendix_f, frontier, frontier_numeric, kaplan_N_opt, loss,
                        params_count, synthetic_curves, synthetic_isoflop)


def test_closed_form_frontier_matches_numeric_minimisation():
    for C in (1e19, 1e21, GOPHER_FLOPS, 1e25):
        N, D = frontier(C)
        Nn, Dn, _ = frontier_numeric(C)
        assert abs(math.log(N / Nn)) < 0.01 and abs(6 * N * D / C - 1) < 1e-9
    a, b = BETA / (ALPHA + BETA), ALPHA / (ALPHA + BETA)
    assert abs(a - 0.4516) < 1e-3 and abs(a + b - 1) < 1e-12                        # Table 2, Approach 3: 0.46, 0.54
    N1, _ = frontier(1e21); N2, _ = frontier(1e23)
    assert abs(math.log(N2 / N1) / math.log(100) - a) < 1e-9


def test_frontier_balances_the_two_terms():
    """At the optimum, d/dN of [A/N^alpha + B/(C/6N)^beta] = 0  <=>  alpha A/N^alpha = beta B/D^beta."""
    N, D = frontier(1e22)
    assert abs((ALPHA * A / N ** ALPHA) / (BETA * B / D ** BETA) - 1) < 1e-9


def test_table_3_is_consistent_with_6ND_and_about_20_tokens_per_parameter():
    for N, (flops, tokens) in TABLE_3.items():
        assert abs(6 * N * tokens / flops - 1) < 0.06
        assert 19 < tokens / N < 23


def test_chinchilla_vs_gopher_under_the_fitted_law():
    assert loss(70e9, 1.4e12) < loss(280e9, 300e9)                                   # same budget, lower loss
    assert abs(6 * 70e9 * 1.4e12 / GOPHER_FLOPS - 1) < 0.03
    assert kaplan_N_opt(GOPHER_FLOPS) > 10 * frontier(GOPHER_FLOPS)[0]              # Kaplan: far bigger model


def test_rounding_sensitivity_of_the_published_exponents():
    N_rounded, _ = frontier(GOPHER_FLOPS)
    N_precise, _ = frontier(GOPHER_FLOPS, (E, A, B, 0.3392, 0.2849))
    assert 30e9 < N_rounded < 35e9 and 38e9 < N_precise < 42e9                       # the paper quotes ~40B


def test_appendix_f_flops_close_to_6ND_for_large_models():
    cfg = dict(vocab=32000, d_model=8192, n_layers=80, ffw=32768, n_heads=64, kq_size=128)
    N = params_count(**cfg)
    ratio = flops_appendix_f(seq_len=2048, **cfg) / (6 * N * 2048)
    assert 0.95 < ratio < 1.15
    small = dict(vocab=32000, d_model=640, n_layers=10, ffw=2560, n_heads=10, kq_size=64)
    assert flops_appendix_f(2048, **small) / (6 * params_count(**small) * 2048) > ratio   # attention matters more


def test_approaches_1_2_3_recover_the_planted_law():
    curves = synthetic_curves(list(np.logspace(7.5, 11, 40)), 3e12, n_points=80)
    _, a1 = approach1_envelope(curves)
    assert abs(a1 - 0.452) < 0.03
    iso = synthetic_isoflop(np.logspace(18, 22, 9))
    _, a2 = approach2_isoflop(iso)
    assert abs(a2 - 0.452) < 0.02
    Ns, Ds, Ls = zip(*[(N, C / (6 * N), l) for C, rows in iso.items() for N, l in rows])
    fit = fit_parametric(Ns, Ds, Ls, grid=[(5, 5, 0, 0.5, 0.5), (6, 6, 0.5, 0.3, 0.3)])
    assert abs(fit[0] - E) < 0.02 and abs(fit[3] - ALPHA) < 0.02 and abs(fit[4] - BETA) < 0.02
