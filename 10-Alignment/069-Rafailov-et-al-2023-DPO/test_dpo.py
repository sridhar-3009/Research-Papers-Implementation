import math

import numpy as np
import torch

from dpo import (GEN_LEN, LM, bandit_expected_dpo, dpo_grad_weight, dpo_loss, optimal_frontier, optimal_policy,
                 seq_logp, true_reward)


def test_dpo_loss_matches_formula_and_gradient():
    lw = torch.tensor([-10.0], requires_grad=True)
    ll = torch.tensor([-12.0], requires_grad=True)
    rw, rl = torch.tensor([-11.0]), torch.tensor([-11.0])
    beta = 0.5
    loss, r_w, r_l = dpo_loss(lw, ll, rw, rl, beta)
    assert abs(loss.item() - (-math.log(1 / (1 + math.exp(-1.0))))) < 1e-6         # -log sigmoid(0.5 - (-0.5))
    loss.backward()
    wgt = dpo_grad_weight(r_w, r_l).item()                                          # sigmoid(r_l - r_w)
    assert abs(lw.grad.item() - (-beta * wgt)) < 1e-6 and abs(ll.grad.item() - beta * wgt) < 1e-6


def test_bandit_dpo_recovers_closed_form_optimum():
    ref = np.array([0.4, 0.3, 0.2, 0.1])
    r = np.array([0.0, 1.0, -1.0, 2.0])
    for beta in (0.5, 2.0):
        assert np.abs(bandit_expected_dpo(ref, r, beta, steps=1500) - optimal_policy(ref, r, beta)).max() < 1e-4
    p = optimal_policy(ref, r, 1.0)                                                 # Eq. 5: rewards from the policy
    implied = np.log(p / ref)
    assert np.allclose(implied - implied[0], r - r[0])


def test_frontier_and_lm_shapes():
    k1, r1 = optimal_frontier(1.0)
    k2, r2 = optimal_frontier(2.0)
    assert k1 > k2 > 0 and r1 > r2 > 0                                               # smaller beta: more reward, more KL
    m = LM()
    x, y = torch.randint(1, 16, (5, 2)), torch.randint(1, 16, (5, GEN_LEN))
    lp = seq_logp(m, x, y)
    assert lp.shape == (5,) and (lp < 0).all()
    assert true_reward(np.array([[1, 2, 4, 7, 8, 9]])).tolist() == [1.0]
