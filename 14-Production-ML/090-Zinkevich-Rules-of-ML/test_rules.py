import numpy as np

import rules as R


def test_all_43_rules_listed():
    nums = [k for rs in R.RULES.values() for k in rs]
    assert nums == list(range(1, 44))


def test_rule30_weighting_restores_calibration():
    r = R.rule30_importance_weighting(n=20000)
    true = r["all data"]["true rate"]
    assert abs(r["sampled, weight 10/3"]["mean prediction"] - true) < 0.01
    assert r["sampled, dropped (no weights)"]["mean prediction"] > 1.5 * true


def test_rule24_delta_bounds():
    s = np.arange(50.0)
    assert R.rule24_delta(s, s) == 0.0 and R.rule24_delta(s, -s) == 1.0
    s2 = s.copy()
    s2[49] = -1.0                                                    # the top item falls out of the top 10
    assert 0 < R.rule24_delta(s, s2) < 1
    assert R.rule24_delta(s, np.r_[s[:45], s[45:][::-1]]) == 0.0     # reordering inside the top 10 is not a delta


def test_rule34_and_rule36():
    r = R.rule34_filter_holdout(n=60000, holdout=0.03)
    true = r["oracle: all traffic labelled"]["true spam rate"]
    assert r["trained on shown (filtered) traffic"]["mean prediction"] < true - 0.1
    assert abs(r["trained on 1% held-out traffic"]["mean prediction"] - true) < 0.05
    p = R.rule36_position(impressions=60000)
    assert p["doc + position model (served at slot 0)"] > p["doc-only model"]


def test_rule37_jump_is_at_live_and_logging_removes_it():
    a, b = R.rule37_skew(), R.rule37_skew(log_at_serving=True)
    assert a["live"] - a["next day"] > 0.05 and abs(b["live"] - b["next day"]) < 0.02
    m = R.rule40_ensemble(n=5000)
    assert m["non-negative"] and m["share of examples whose ensemble score went DOWN when base A went up"] == 0.0
