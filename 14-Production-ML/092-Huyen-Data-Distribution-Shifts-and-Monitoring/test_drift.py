import numpy as np

import drift as D


def test_shift_types_and_monitors():
    t = D.shift_table(n=2000)
    assert t["source"]["inputs KS p"] > 0.01
    assert t["covariate"]["inputs KS p"] < 1e-3 and abs(t["covariate"]["mean prediction"] - t["covariate"]["true rate"]) < 0.03
    assert t["label"]["mean prediction"] > 1.5 * t["label"]["true rate"]
    assert t["concept"]["inputs KS p"] > 0.01 and t["concept"]["accuracy target"] < t["concept"]["accuracy source"] - 0.05


def test_adaptation_without_labels():
    r = D.importance_weighting(n=3000)
    assert r["weights from a domain classifier"]["target log-loss"] < r["unweighted"]["target log-loss"] - 0.05
    b = D.label_shift_prior(n=3000)
    assert abs(b["confusion-matrix estimate"] - b["true target positive rate"]) < \
        abs(b["naive estimate (mean prediction)"] - b["true target positive rate"])
    assert b["log-loss after prior correction"] < b["log-loss before"]


def test_windows_and_cumulative():
    s, _ = D.stream()
    h0 = 24 * 9
    seasonal = D.window_alerts(s, 6, seasonal=True)
    flat = D.window_alerts(s, 6, seasonal=False)
    assert any(h0 <= h < h0 + 12 for h in seasonal)
    assert len([h for h in flat if not h0 <= h < h0 + 12]) > len([h for h in seasonal if not h0 <= h < h0 + 12])
    cum, slide = D.cumulative_vs_sliding(s)
    assert abs(slide[h0 + 3] - slide[h0 - 1]) > 10 * abs(cum[h0 + 3] - cum[h0 - 1])


def test_retraining_and_feedback():
    r = D.retraining_strategies(eval_days=(25,))[25]
    assert r["fine-tune on last 2 days"] < r["scratch, all data"] < r["stale"]
    a, b = D.feedback_recommender(explore=0.0), D.feedback_recommender(explore=0.1)
    assert b["distinct items shown (last round)"] > 10 * a["distinct items shown (last round)"]
