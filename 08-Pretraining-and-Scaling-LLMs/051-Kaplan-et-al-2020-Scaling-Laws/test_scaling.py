import math

import numpy as np

from scaling import (A_N, NS, B_crit, C_min, L_of_Cmin, L_of_D, L_of_N, L_of_ND, L_of_NS, S_min,
                     S_stop_lower_bound, allocation_exponents, compute_optimal_N, fit_L_ND, fit_power_law,
                     forward_flops_per_token, loss_at_compute, non_embedding_params, overfit_penalty,
                     spectrum_loss_vs_N, spectrum_toy, table5, tokens_to_avoid_overfitting, training_compute)


def test_doubling_parameters_multiplies_loss_by_0_95():
    assert abs(L_of_N(2e9) / L_of_N(1e9) - 2 ** -A_N) < 1e-12
    assert abs(2 ** -A_N - 0.949) < 1e-3                                           # '0.95' in Section 1.2


def test_L_ND_reduces_to_the_one_variable_laws():
    big = 1e30
    assert abs(L_of_ND(1e9, big) - (6.4e13 / 1e9) ** 0.076) < 1e-6                  # infinite data: L(N)
    assert abs(L_of_ND(big, 1e10) - (1.8e13 / 1e10) ** 0.103) < 1e-6                # infinite model: L(D)
    assert L_of_ND(1e9, 1e9) > L_of_ND(1e9, 1e11)                                   # less data, more loss
    assert overfit_penalty(1e9, 1e30) < 1e-9 and overfit_penalty(1e9, 1e8) > 0.02


def test_overfitting_rule_keeps_penalty_near_noise_level():
    for N in (1e6, 1e8, 1e9):
        p = overfit_penalty(N, tokens_to_avoid_overfitting(N))
        assert 0.005 < p < 0.05                                                     # ~0.02 seed noise
    assert S_stop_lower_bound(1e8, 1e9) > 0


def test_critical_batch_and_step_compute_tradeoff():
    L = 3.0
    Bc = B_crit(L)
    assert abs(Bc - 2e8 / 3 ** (1 / 0.21)) < 1
    assert abs(B_crit(L * 0.87) / Bc - 0.87 ** (-1 / 0.21)) < 1e-9                  # ~doubles per 13% lower loss
    S, B = 1e5, Bc                                                                   # training AT B_crit:
    assert abs(S_min(S, B, L) - S / 2) < 1e-6 and abs(C_min(1.0, B, L) - 0.5) < 1e-12   # 2 S_min steps, 2 C_min


def test_sizes_and_compute():
    assert non_embedding_params(12, 768) == 12 * 12 * 768 ** 2                     # ~85M for GPT-2 small
    assert forward_flops_per_token(1e8, 12, 1024, 768) == 2e8 + 2 * 12 * 1024 * 768
    assert training_compute(1e9, 1e10) == 6e19


def test_allocation_exponents_eq_1_8():
    e = allocation_exponents()
    assert abs(e["a_C_min"] - 1 / (1 / 0.76 + 1 / 0.21 + 1 / 0.077)) < 1e-12
    assert abs(e["a_C_min"] - 0.050) < 0.005                                         # vs the directly fitted 0.050
    assert 0.6 < e["N"] < 0.75 and 0.2 < e["B"] < 0.3 and e["S"] < 0.1             # N gets almost everything
    t = table5(1.0)
    assert t["N_opt"] == 1.3e9 and abs(t["D_opt"] - 2e10) < 1


def test_numerical_frontier_matches_the_paper():
    Cs = np.logspace(-4, 2, 7)
    Ns, Ls = zip(*[compute_optimal_N(C) for C in Cs])
    slope = np.polyfit(np.log(Cs), np.log(Ns), 1)[0]
    assert 0.6 < slope < 0.75                                                       # paper: 0.73
    for N, L in zip(Ns, Ls):
        assert abs(L / L_of_NS(N, 1e30) - (1 + NS["a_N"] / NS["a_S"])) < 0.01       # ~10% above converged
    assert abs(Ls[-1] / L_of_Cmin(Cs[-1]) - 1) < 0.02                               # agrees with Eq. 1.3
    assert loss_at_compute(1e8, 1.0) > loss_at_compute(1e8, 10.0)


def test_power_law_fits():
    x = np.logspace(5, 10, 12)
    a, xc = fit_power_law(x, (3e12 / x) ** 0.08)
    assert abs(a - 0.08) < 1e-9 and abs(xc / 3e12 - 1) < 1e-6
    N = np.array([1e6, 1e7, 1e8, 1e6, 1e7, 1e8]); D = np.array([1e8, 1e8, 1e8, 1e10, 1e10, 1e10])
    true = dict(a_N=0.08, a_D=0.1, N_c=5e13, D_c=2e13)
    fit = fit_L_ND(N, D, L_of_ND(N, D, true), steps=3000)
    assert abs(fit["a_N"] - 0.08) < 0.01 and abs(fit["a_D"] - 0.1) < 0.02


def test_spectrum_toy_gives_a_power_law():
    lam, w = spectrum_toy(alpha=1.5)
    Ns = np.array([16, 32, 64, 128, 256, 512])
    a, _ = fit_power_law(Ns, spectrum_loss_vs_N(lam, w, Ns))
    assert 0.3 < a < 0.8                                                            # theory: alpha - 1 = 0.5
