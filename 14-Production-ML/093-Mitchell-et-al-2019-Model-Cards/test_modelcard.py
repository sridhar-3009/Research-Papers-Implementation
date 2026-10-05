import numpy as np

import modelcard as M


def test_confusion_rates_by_hand():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    r = M.confusion_rates(y, p)                                       # TP 2, FN 2, FP 1, TN 3
    assert (r["FPR"], r["FNR"], r["FDR"], r["FOR"]) == (0.25, 0.5, 1 / 3, 0.4)


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.5).astype(int)
    p = np.where(rng.random(2000) < 0.8, y, 1 - y)
    est, ci = M.confusion_rates(y, p), M.bootstrap_ci(y, p, reps=300)
    for k in ("FPR", "FNR", "FDR", "FOR"):
        assert ci[k][0] <= est[k] <= ci[k][1] and ci[k][1] - ci[k][0] < 0.08


def test_card_renders_all_sections_and_smiling_findings():
    card, rows, _ = M.smiling_card(reps=200)
    md = card.to_markdown()
    assert all(f"## {s}" in md for s in M.SECTIONS) and card.missing() == []
    assert M.ModelCard("x").missing() == M.SECTIONS
    r = {k: v[0] for k, v in rows.items()}
    inter = [k for k in r if " " in k]
    assert max(inter, key=lambda k: r[k]["FDR"]) == "male old"
    assert r["male"]["FNR"] > r["female"]["FNR"]


def test_toxicity_bias_and_mitigation():
    t = M.toxicity_versions(pinned_draws=5)
    v1, v2 = (t[k]["per term"] for k in t)
    for term in M.TARGETED:
        assert v1[term]["BPSN AUC"] < 0.8 < v2[term]["BPSN AUC"]
    assert v1["teacher"]["BPSN AUC"] == 1.0
    assert M.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
