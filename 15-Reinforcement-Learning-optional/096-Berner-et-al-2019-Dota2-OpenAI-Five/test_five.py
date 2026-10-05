import numpy as np

import five as F


def test_surgery_is_exact_for_widening_and_new_inputs():
    rng = np.random.default_rng(0)
    p = F.init_mlp(5, 8, 2, rng)
    X = rng.standard_normal((200, 5))
    assert F.max_policy_change(p, F.surgery_widen(p, 20, rng), X) < 1e-12
    assert F.max_policy_change(p, F.surgery_add_inputs(p, 4), X, pad=4) < 1e-12
    q = F.surgery_widen(F.surgery_add_inputs(p, 4), 20, rng)
    assert q["W1"].shape == (9, 20) and q["W2"].shape == (20, 2) and np.all(q["W2"][8:] == 0)
    assert F.max_policy_change(p, F.surgery_widen_recurrent(p, 20, rng, 0.5), X) > 0.05


def test_gae_matches_hand_computation():
    r = np.array([[1.0], [1.0], [1.0]])
    v = np.array([[0.5], [0.5], [0.5]])
    m = np.ones((3, 1), bool)
    adv, ret = F.gae(r, v, m, gamma=0.9, lam=0.5)
    d2 = 1 - 0.5                                                      # last step: no bootstrap
    d1 = 1 + 0.9 * 0.5 - 0.5
    d0 = d1
    assert np.isclose(adv[2, 0], d2) and np.isclose(adv[1, 0], d1 + 0.45 * d2)
    assert np.isclose(adv[0, 0], d0 + 0.45 * (d1 + 0.45 * d2))


def test_ppo_learns_and_staleness_hurts():
    c, hit, _ = F.train_ppo(updates=60, seed=0)
    assert hit is not None and hit < 40
    _, hit_stale, _ = F.train_ppo(updates=80, staleness=8, seed=0)
    assert hit_stale is None or hit_stale > hit * 1.5


def test_team_spirit_snr():
    _, s0 = F.team_spirit(0.0, steps=1)
    _, s1 = F.team_spirit(1.0, steps=1)
    assert s0 > 1.5 * s1
