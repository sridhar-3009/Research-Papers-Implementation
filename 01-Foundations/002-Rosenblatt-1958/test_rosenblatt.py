"""Tests: each checks an equation or a conclusion of Rosenblatt (1958).

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest

from experiments import ideal_environment, prototype_classes
from perceptron import Photoperceptron
from theory import Pa, Pc, Pc_min, overlap_to_LG


# ---- Equations (1)-(3): the probability theory ----

@pytest.mark.parametrize("x, y, theta, R", [(10, 0, 3, 0.2), (5, 5, 1, 0.3), (10, 5, 2, 0.4)])
def test_Pa_and_Pc_match_simulation(x, y, theta, R):
    # Build a big random retina and many random A-units, and count directly.
    rng = np.random.default_rng(0)
    Ns, NA = 20000, 20000
    exc, inh = rng.integers(0, Ns, (NA, x)), rng.integers(0, Ns, (NA, y))
    fires = lambda s: (s[exc].sum(1) - s[inh].sum(1)) >= theta
    s1 = rng.random(Ns) < R
    # s2: same size as s1, sharing half of its lit points
    on, off = np.flatnonzero(s1), np.flatnonzero(~s1)
    keep = rng.choice(on, len(on) // 2, replace=False)
    new = rng.choice(off, len(on) - len(keep), replace=False)
    s2 = np.zeros(Ns, bool)
    s2[keep] = s2[new] = True
    a1, a2 = fires(s1), fires(s2)
    assert Pa(R, x, y, theta) == pytest.approx(a1.mean(), abs=0.01)
    C = len(keep) / len(on)
    assert Pc(R, *overlap_to_LG(R, C), x, y, theta) == pytest.approx(a2[a1].mean(), abs=0.02)


def test_Pa_falls_with_threshold_and_inhibition():
    # Figure 4: Pa goes down when theta rises or when more connections are inhibitory.
    assert all(Pa(0.3, 10, 0, t) > Pa(0.3, 10, 0, t + 1) for t in range(1, 10))
    assert Pa(0.3, 10, 0, 1) > Pa(0.3, 8, 2, 1) > Pa(0.3, 6, 4, 1)


def test_Pc_is_one_for_identical_stimuli_and_positive_for_disjoint():
    # Figures 5-6: Pc -> 1 as stimuli become identical; Pc > 0 even with no overlap.
    assert Pc(0.3, *overlap_to_LG(0.3, 1.0), 10, 0, 3) == pytest.approx(1.0)
    assert Pc(0.3, *overlap_to_LG(0.3, 0.0), 10, 0, 3) > 0


def test_Pc_min_is_reached_at_the_highest_threshold():
    # Eq. (3): with theta = x (all excitatory points must be lit), the unit keeps
    # firing only if none of its points go dark, so Pc = (1 - L)^x.
    L, G = overlap_to_LG(0.5, 0.6)
    assert Pc(0.5, L, G, 10, 0, 10) == pytest.approx(Pc_min(L, G, 10, 0))


# ---- The perceptron itself ----

def test_gamma_system_keeps_source_set_value_constant():
    # Table 1: in the gamma system a source-set's total value never changes.
    rng = np.random.default_rng(0)
    p = Photoperceptron(400, 500, x=5, y=5, theta=3, rng=0)
    p.train_forced(ideal_environment(50, rng), rng.integers(0, 2, 50), "gamma")
    for r in range(2):
        assert p.V[p.source == r].sum() == pytest.approx(0.0, abs=1e-9)


def ideal_run(n, system, mode, seed=0):
    rng = np.random.default_rng(seed)
    p = Photoperceptron(400, 2000, x=5, y=5, theta=3, rng=seed)
    S, lab = ideal_environment(2 * n, rng), np.repeat([0, 1], n)
    p.train_forced(S, lab, system)
    return p.p_correct(S, lab, mode), p.p_correct(ideal_environment(400, rng), rng.integers(0, 2, 400), mode)


def test_ideal_environment_recall_falls_and_no_generalization():
    # Conclusions 1-3: random stimuli can be learned, recall falls toward chance as
    # more are stored, and new random stimuli are at chance (nothing to generalize).
    pr = [ideal_run(n, "gamma", "mu")[0] for n in (10, 100, 1000)]
    assert pr[0] > pr[1] > pr[2] > 0.5
    pg = np.mean([ideal_run(200, "gamma", "mu", seed=s)[1] for s in range(3)])
    assert pg == pytest.approx(0.5, abs=0.05)


def test_mean_beats_sum_in_alpha_but_not_in_gamma():
    # "systems which respond to mean values have an advantage over systems which
    # respond to sums" - but in the gamma system it makes no difference.
    assert ideal_run(300, "alpha", "mu")[0] > ideal_run(300, "alpha", "sigma")[0] + 0.1
    assert ideal_run(300, "gamma", "mu")[0] == pytest.approx(ideal_run(300, "gamma", "sigma")[0], abs=0.02)


def test_gamma_beats_alpha_with_unequal_training():
    # Figure 10: the gamma system handles responses trained different amounts.
    rng = np.random.default_rng(1)
    lab = np.repeat(np.arange(10), rng.integers(25, 76, 10))
    S = ideal_environment(len(lab), rng)
    scores = {}
    for system in ["alpha", "gamma"]:
        p = Photoperceptron(400, 5000, n_resp=10, x=5, y=5, theta=3, rng=1)
        p.train_forced(S, lab, system)
        scores[system] = p.p_correct(S, lab, "sigma")
    assert scores["gamma"] > scores["alpha"] + 0.2


def test_differentiated_environment_generalizes():
    # Conclusions 4-5: with classes of similar stimuli, recall and generalization
    # approach the same asymptote, and more A-units raise it.
    pg = {}
    for NA in (100, 2000):
        rng = np.random.default_rng(3)
        p = Photoperceptron(400, NA, x=5, y=5, theta=3, rng=3)
        S, lab = prototype_classes(600, rng, flip=0.3)
        T, lt = prototype_classes(400, rng, flip=0.3)
        p.train_forced(S, lab, "gamma")
        pr, pg[NA] = p.p_correct(S, lab), p.p_correct(T, lt)
        assert abs(pr - pg[NA]) < 0.06
    assert pg[2000] > pg[100] + 0.1 and pg[2000] > 0.95


def test_memory_is_distributed():
    # Conclusion 9: removing part of the A-units causes a small, general loss.
    rng = np.random.default_rng(9)
    p = Photoperceptron(400, 2000, n_resp=5, x=5, y=5, theta=3, rng=9)
    S, lab = prototype_classes(500, rng, flip=0.3, n_classes=5)
    T, lt = prototype_classes(500, rng, flip=0.3, n_classes=5)
    p.train_forced(S, lab, "gamma")
    before = p.p_correct(T, lt)
    p.alive[rng.permutation(2000)[:400]] = False            # remove 20%
    after = p.p_correct(T, lt)
    assert before - 0.1 < after < before + 0.01
    assert all(p.p_correct(T[lt == k], lt[lt == k]) > 0.5 for k in range(5))  # no class wiped out


def test_bivalent_trial_and_error_learning():
    # Conclusion 7: with reward and punishment the mistakes go down.
    rng = np.random.default_rng(11)
    p = Photoperceptron(400, 1000, n_resp=4, x=5, y=5, theta=3, rng=11)
    S, lab = prototype_classes(400, rng, flip=0.35, n_classes=4)
    errors = p.train_bivalent(S, lab, epochs=5)
    assert errors[-1] < errors[0] / 2
