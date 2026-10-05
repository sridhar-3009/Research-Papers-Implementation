import numpy as np
import pytest

import hidden_debt as H


def test_direct_loop_greedy_gets_stuck_more_than_ucb():
    stuck = lambda pol: np.mean([H.direct_feedback_loop(pol, rounds=400, seed=s)["share of rounds on best item"] < 0.1
                                 for s in range(10)])
    assert stuck("greedy") > stuck("ucb") + 0.3


def test_hidden_loop_between_two_systems():
    lo, hi = H.two_systems(-1.0, n=20000), H.two_systems(1.0, n=20000)
    assert lo["B's learned review weight"] > hi["B's learned review weight"] + 0.3


def test_config_resolve_diff_validate_and_closure():
    configs = {"base": {"features": ["query"], "train_from": 1010, "memory_gb": 8},
               "x": {"parent": "base", "features+": ["feature_A", "feature_D"], "train_from": 901, "lr": 1}}
    cfg = H.resolve("x", configs)
    assert cfg["features"] == ["query", "feature_A", "feature_D"]
    assert "+ feature feature_A" in H.diff(H.resolve("base", configs), cfg)
    probs = " | ".join(H.validate(cfg))
    assert "logged incorrectly" in probs and "not available in serving" in probs and "'lr'" in probs
    assert H.validate(H.resolve("base", configs)) == []
    assert H.closure(["feature_D"]) == {"offline_join", "raw_logs", "geo_db"}
    assert H.consumers_of("legacy_db", configs) == []


def test_typed_values_and_monitoring():
    with pytest.raises(TypeError):
        H.LogOdds(1.0) > H.Probability(0.5)
    with pytest.raises(ValueError):
        H.Probability(1.5)
    X, y, c, names = H.world(n=8000)
    w = H.fit_logreg(X, y)
    Xb, yb, cb, _ = H.world(n=8000, broken=("IN", "units"), seed=1)
    sb = H.sliced_bias(w, Xb, yb, cb, names)
    assert sb["IN"] > 0.05 and abs(sb["US"]) < 0.02
    assert H.data_tests(H.make_schema(X), Xb) and not H.data_tests(H.make_schema(X), H.world(n=8000, seed=2)[0])


def test_reproducibility_and_pipeline():
    w0, w0b, w1 = H.sgd_runs(n=500, epochs=1)
    assert np.array_equal(w0, w0b) and not np.allclose(w0, w1)
    assert H.run_pipeline(n=500)["rows"] == 500
    ml, other = H.code_fraction()
    assert ml < other
