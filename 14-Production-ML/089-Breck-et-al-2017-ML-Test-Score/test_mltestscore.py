import numpy as np

import mltestscore as M


def test_scoring_rule_uses_minimum_and_table_v():
    st = {"Data": ["automated"] * 7, "Model": ["manual"] * 6 + ["none"], "Infra": [1] * 4 + [0] * 3,
          "Monitor": ["automated"] * 2 + ["manual"] * 2 + ["none"] * 3}
    sums, final, text = M.ml_test_score(st)
    assert sums == {"Data": 7.0, "Model": 3.0, "Infra": 4.0, "Monitor": 3.0}
    assert final == 3.0 and text.startswith("Reasonably tested")
    assert M.ml_test_score({s: ["none"] * 7 for s in M.RUBRIC})[2].startswith("More of a research project")
    assert sum(len(v) for v in M.RUBRIC.values()) == 28


def test_clean_system_passes_infra_and_monitoring():
    R = M.run_suite()
    assert all(p for _, p, _ in R["Infra"]) and all(p for _, p, _ in R["Monitor"])
    sums, final, _ = M.score_suite(R)
    assert sums["Data"] == 6.5 and final == 6.5


def test_skew_vs_upstream_change_are_told_apart():
    cols = M.BASE_SPEC["features"]
    w = M.raw_world(n=500)
    assert M.training_serving_skew(w, cols) == {}
    assert M.training_serving_skew(w, cols, serving_bug=True) == {"spend": 500}
    schema = M.make_schema(M.raw_world(n=2000, seed=1))
    w["spend_raw"] = w["spend_raw"] * 100
    assert M.check_schema(schema, w) and M.training_serving_skew(w, cols) == {}


def test_registry_canary_and_unit_tests():
    tr, va = M.raw_world(n=3000), M.raw_world(n=3000, seed=1)
    good = M.train(M.BASE_SPEC, tr)
    reg = M.Registry()
    assert reg.push(good, va)[0] == "BLESSED"
    bad = M.train(M.BASE_SPEC, tr)
    bad.w = bad.w + 1.0
    assert reg.push(bad, va)[0] == "VETOED" and reg.serving == 0
    new = M.train(dict(M.BASE_SPEC, op_version=2), tr)
    assert not M.canary(new)[0] and M.canary(good)[0]
    assert all(M.spec_unit_tests(M.BASE_SPEC).values())


def test_monitors():
    alerts = []
    M.Model(dict(M.BASE_SPEC, lr=1e6)).fit(np.random.default_rng(0).standard_normal((50, 4)),
                                             np.ones(50) * (np.arange(50) % 2), monitor=M.numeric_monitor(alerts))
    assert alerts
    assert M.perf_regression(np.r_[np.full(20, 10.0), 30.0]) == [(20, "dramatic"), (20, "slow leak")]
    assert M.perf_regression(np.full(30, 10.0)) == []
    assert M.dead_relu_fraction(np.array([[1.0, -1.0], [2.0, -3.0]])) == 0.5
