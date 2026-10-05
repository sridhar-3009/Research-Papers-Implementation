import numpy as np

import tasks as T


def test_task_content_limits_and_monotonicity():
    G1, _ = T.task_content(0.4, 1.0, sigma=1.0)
    assert np.isclose(G1, 0.6)                                         # sigma = 1: Gamma = N - I
    G, _ = T.task_content(0.4, 1.0, sigma=0.8)
    assert T.task_content(0.5, 1.0, sigma=0.8)[0] < G < T.task_content(0.4, 1.1, sigma=0.8)[0]


def test_equilibrium_shares_add_up_and_eq2():
    e = T.equilibrium(0.4, 1.0, K=3.0)
    assert np.isclose(e["W"] * 1 + e["R"] * 3.0, e["Y"])               # factor payments exhaust output
    rel = (e["W"] / 1.0) / (e["R"] / 1.0)
    assert np.isclose(T.labour_share_eq2(e["Gamma"], rel, 0.8), e["labor share"])


def test_automation_lowers_labour_share_for_any_sigma_and_so_so_lowers_wages():
    for s in (0.5, 0.8, 1.0, 1.5):
        assert T.effects("automation", sigma=s)["change in labour share (d ln s_L)"] < 0
        assert T.effects("new tasks", sigma=s)["change in labour share (d ln s_L)"] > 0
    ss = T.so_so_automation()
    assert ss[0.5]["wage"] < 0 < ss[8.0]["wage"]
    assert all(v["capital cost"] < v["labour cost"] for v in ss.values())   # automation is cost-minimising


def test_decomposition_adds_up_and_bounds():
    d = T.synthetic_industries()
    dec = T.decompose(d)
    total = dec["productivity"] + dec["composition"] + dec["substitution"] + dec["task content"]
    assert np.abs(dec["wage bill"] - total).max() < 0.002
    assert abs(dec["task content"].mean() - T.planted_task_content(d).mean()) < 0.001
    pd, pr = T.planted_split(d)
    assert pd.mean() <= dec["displacement"].mean() <= 0 <= dec["reinstatement"].mean() <= pr.mean()
