import numpy as np

import alphago as A


def test_game_rules_and_solver():
    g = A.ConnectK(4, 5, 3)
    b = g.empty()
    b, won, _ = g.play(b, 0, 1)
    b, _, _ = g.play(b, 4, 2)
    b, _, _ = g.play(b, 1, 1)
    b, _, _ = g.play(b, 4, 2)
    assert g.move_values(b)[2] == 1                                   # 1 1 _ on the bottom row: column 2 wins
    _, won, _ = g.play(b, 2, 1)
    assert won
    assert g.solve(g.empty()) == 1                                    # first player wins with perfect play
    assert A.ConnectK(3, 3, 3).solve(A.ConnectK(3, 3, 3).empty()) in (-1, 0, 1)


def test_policy_learning_and_rollout_policy():
    g = A.ConnectK()
    rng = np.random.default_rng(0)
    X, M, Y, B = A.expert_games(g, 400, rng)
    sl = A.train_policy(A.Net(X.shape[1], g.C, seed=0), X, M, Y, epochs=60, lr=1e-2)
    _, _, _, Bt = A.expert_games(g, 100, np.random.default_rng(1))
    assert A.optimal_move_rate(sl, g, Bt) > A.optimal_move_rate(A.random_agent, g, Bt) + 0.2
    rp = A.RolloutPolicy().fit(g, B[:800], list(Y[:800]), epochs=50)
    assert rp.w[0] > 0                                                # 'wins now' gets a positive weight


def test_mcts_finds_the_winning_move_and_backup_signs():
    g = A.ConnectK()
    b = g.empty()
    for c, p in ((0, 1), (4, 2), (1, 1), (4, 2)):
        b, _, _ = g.play(b, c, p)
    uniform = A.Net(4 * 20, g.C, seed=0)
    value = A.Net(4 * 20, 1, seed=1, kind="value")
    rp = A.RolloutPolicy()
    move, visits = A.mcts(g, b, uniform, value, rp, sims=200, lam=0.0)
    assert move in (2,)                                               # the immediate win attracts the visits
    assert visits[2] == max(visits.values())


def test_value_training_reduces_error():
    g = A.ConnectK()
    rng = np.random.default_rng(0)
    X, M, Y, _ = A.expert_games(g, 300, rng)
    sl = A.train_policy(A.Net(X.shape[1], g.C, seed=0), X, M, Y, epochs=40, lr=1e-2)
    Xv, Zv, _ = A.self_play_positions(g, sl, 1500, rng, one_per_game=True)
    v = A.train_value(Xv, Zv, epochs=30)
    assert A.mse(v, Xv, Zv) < ((Zv - Zv.mean()) ** 2).mean()
