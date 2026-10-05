import numpy as np

import dmls as D


def test_reservoir_is_uniform_and_hashing_formula():
    f = D.reservoir_uniformity(n=20, k=5, trials=4000)
    assert np.all(np.abs(f - 0.25) < 0.04)
    for B, v in D.hashing_collisions(n_values=3000, buckets=(2 ** 12,)).items():
        assert abs(v["measured"] - v["formula"]) < 0.03


def test_weak_supervision_and_imbalance():
    r = D.weak_supervision(n=4000)
    assert r["classifier on majority vote"] > r["classifier on 100 hand labels"] + 0.05
    c = D.class_imbalance(n=20000)
    assert c["accuracy of 'always negative'"] > 0.97 and c["model PR-AUC"] < c["model ROC-AUC"]
    assert c["recall @0.5 class-weighted"] > c["recall @0.5 unweighted"] + 0.3
    assert c["mean prediction class-weighted"] > 5 * c["mean prediction unweighted"]


def test_leakage_inflates_accuracy():
    r = D.leakage_feature_selection()
    assert r["selected on ALL data (leak)"] > 0.7 and r["selected on training data"] < 0.62
    d = D.leakage_duplicates(n=1500)
    assert d["random split (copies leak)"] > d["split by item (no leak)"] + 0.08


def test_calibration_and_behaviour():
    r = D.calibration_demo(n=8000)
    assert r["ECE after Platt"] < r["ECE raw"] / 5 and abs(r["AUC raw"] - r["AUC after Platt"]) < 1e-9
    p = np.array([0.9, 0.9, 0.1, 0.1]); y = np.array([1, 0, 0, 0])
    assert abs(D.expected_calibration_error(p, y)[0] - 0.5 * 0.4 - 0.5 * 0.1) < 1e-9
    b = D.baselines_and_behaviour(n=8000)
    assert b["invariance: decisions that flip"] > 0.05
    assert b["directional (model without it): share where +income lowers approval"] == 0.0


def test_production_testing():
    n = D.ab_sample_size(0.10, 0.01)
    assert 14000 < n < 15500
    r = D.bandit_vs_ab(users=5000, reps=4)
    assert r["Thompson sampling: regret"] < r["A/B/n then ship: regret (clicks lost)"]
    s = D.stateless_vs_stateful(days=12)
    assert s["stateful example-passes"] * 10 < s["stateless example-passes"]
