import warnings

import numpy as np
import pytest
from scipy import stats

import shiftdetect as S

warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation unsuccessful")


@pytest.fixture(scope="module")
def setup():
    (Xtr, ytr), (Xva, yva), (Xte, yte) = S.load_data()
    return S.Reducers(Xtr, ytr), Xva, yva, Xte, yte


def test_tests_are_correct():
    r = np.random.default_rng(0)
    A, B = r.normal(0, 1, (60, 3)), r.normal(0, 1, (60, 3))
    assert S.ks_bonferroni(A, B) > 0.05 and S.ks_bonferroni(A, B + 1.5) < 1e-3
    assert S.mmd_test(A, B, perms=50) > 0.05 and S.mmd_test(A, B + 1.0, perms=50) < 0.05
    K = np.ones((6, 6))
    assert abs(S.mmd2_unbiased(K, 3)) < 1e-12                         # identical points -> MMD^2 = 0
    a, b = np.repeat(np.arange(3), 30), np.repeat(np.arange(3), 30)
    assert S.chi2_test(a, b, 3) > 0.99
    assert stats.binomtest(50, 100, 0.5).pvalue == 1.0


def test_srp_matrix_follows_eq1(setup):
    red = setup[0]
    v, K = np.sqrt(64), red.K
    vals = set(np.round(np.unique(red.R_srp), 6))
    assert vals <= {0.0, round(np.sqrt(v / K), 6), round(-np.sqrt(v / K), 6)}
    assert 0.6 < (red.R_srp == 0).mean() < 0.95                        # density about 1 / sqrt(D) = 1/8


def test_classifier_and_shifts(setup):
    red, Xva, yva, Xte, yte = setup
    assert (red.clf.proba(Xva).argmax(1) == yva).mean() > 0.93
    Xs, ys = S.apply_shift(Xte, yte, "ko", 1.0)
    assert (ys == 0).sum() == 0 and len(ys) == (yte != 0).sum()
    Xo, yo = S.apply_shift(Xte, yte, "oz+m_img", 0.5)
    assert set(yo) == {0}
    Xa, ya = S.apply_shift(Xte, yte, "adv", 1.0, red)
    assert (red.clf.proba(Xa).argmax(1) == ya).mean() < 0.7          # FGSM hurts the classifier


def test_detects_large_shift_not_none(setup):
    red, Xva, yva, Xte, yte = setup
    Xs, ys = S.apply_shift(Xte, yte, "m_img", 1.0, seed=1)
    assert S.detection_rate(red, Xva, Xs, "BBSDs", "univ", 100, reps=5) >= 0.8
    assert S.detection_rate(red, Xva, Xte, "BBSDs", "univ", 100, reps=5) <= 0.2
    _, _, acc_top = S.most_anomalous(Xva[:200], Xs[:200], ys[:200], red)
    assert acc_top < 0.8
