# The code, explained simply

How the code in this folder rebuilds the AlphaGo pipeline (Silver et al. 2016) on a game that can be solved exactly.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `alphago.py` | Connect-K with an exact solver, board features, a small MLP (policy or value), expert data, the SL policy, a linear rollout policy, RL self-play with an opponent pool, value-network data regimes, PUCT MCTS with λ-mixed leaf evaluation, match utilities |
| `experiments.py` | simulations × λ, priors and c_puct, value-data regimes, RL ablations, a harder board (E1–E5) |
| `demo.py` | the whole pipeline with every component scored against perfect play (~11 seconds) |
| `test_alphago.py` | 4 quick tests (~2 seconds) |

**Run it** (from `15-Reinforcement-Learning-optional/097-Silver-et-al-2016-AlphaGo`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `alphago.py`

### The game
| Name | What it does |
|---|---|
| `ConnectK(rows, cols, k)` | precomputes every K-cell line and the lines through each cell; boards are tuples (row 0 at the bottom) |
| `legal`, `drop_index`, `play`, `to_move` | rules: stones drop to the lowest empty cell; `play` returns (new board, won, draw) |
| `solve(b)` | memoised negamax value for the player to move (+1 / 0 / −1), stopping early on a win |
| `move_values(b)` | the exact value of each legal move: the ground truth for "perfect-move rate" |

### Networks and data
| Name | What it does |
|---|---|
| `board_features` | own / opponent / empty / playable-cell planes from the mover's point of view (4 × R·C values) |
| `Net(kind="policy" or "value")` | one tanh hidden layer; masked softmax over columns or a tanh value; `step` applies Adam given dLoss/d(output) |
| `expert_move`, `expert_games` | a perfect move (random among ties), or with probability 0.25 a random move |
| `train_policy`, `accuracy`, `optimal_move_rate` | cross-entropy training; match with the expert's moves; perfect-move rate on decisive positions (works for networks or agent functions) |
| `rollout_features`, `RolloutPolicy` | 5 features per move; a linear softmax fitted by maximum likelihood with features precomputed once |

### Self-play and value
| Name | What it does |
|---|---|
| `play_game`, `match` | play with optional random opening moves; `match` alternates colours and scores win = 1, draw = 0.5 |
| `policy_agent`, `perfect_agent`, `random_agent` | players |
| `rl_self_play(sl_net, iters, games_per_iter, pool_every)` | copies p_σ; each game is against a random pool member; one REINFORCE step per iteration on the current policy's moves weighted by its outcome z; a snapshot joins the pool every `pool_every` iterations |
| `self_play_positions(net, n_games, one_per_game)` | positions labelled with z from the mover's view: every position of each game, or one random position per game |
| `train_value`, `mse` | value regression with a tanh output |

### Search
| Name | What it does |
|---|---|
| `Node` | prior P, visit count N, total value W (from the view of the player who chose this move), children |
| `rollout_value` | plays to the end with the rollout policy; ±1 / 0 for the player to move |
| `mcts(b, prior_net, value_net, rp, sims, lam, c_puct)` | selection by Q + c_puct · P · √(ΣN + 1)/(1 + N); expansion with SL priors; leaf value (1 − λ)·v_θ + λ·rollout, negated to the chooser's view; terminal wins / draws scored directly; back up with a sign flip each ply; returns the most-visited move and the visit counts |
| `mcts_agent` | wraps the search as a player |

`REPORTED` holds the paper's numbers (57.0% / 24.2%, > 80% / 85%, 0.19 / 0.37 → 0.226 / 0.234, the search constants, 494 of 495, handicap results, hardware, Fan Hui 5–0), checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `build` | builds SL, rollout, RL and value components (shared by E1–E3 and E5) |
| `e1` | simulations 10–200 × λ 0–1: perfect-move rate and score vs the perfect player |
| `e2` | SL vs RL priors × c_puct |
| `e3` | value data: all positions vs one-per-game at 1×, 2×, 5×, 10× the size |
| `e4` | RL iterations 0–200, with and without the opponent pool |
| `e5` | Connect-4 on 4 × 5 (solve, then λ variants) |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_game_rules_and_solver` | a bottom-row three-in-a-row is detected; the move values find it; the first player wins 4 × 5 Connect-3 |
| `test_policy_learning_and_rollout_policy` | the SL policy beats random by > 20 points of perfect-move rate; "wins now" gets a positive rollout weight |
| `test_mcts_finds_the_winning_move_and_backup_signs` | with uniform priors and an untrained value net, search still concentrates its visits on the immediately winning move (checks the sign handling) |
| `test_value_training_reduces_error` | the value net beats predicting the mean |

---

## 5. Try it yourself

1. Implement AlphaGo Zero: one network with policy and value heads, trained from MCTS visit counts and outcomes, with no rollouts and no expert data.
2. Add virtual loss and run several simulations "in parallel" (interleaved) to see how it spreads the search.
3. Make the rollout policy weaker (drop the "wins now" and "blocks" features) and check whether λ = 0.5 then beats λ = 1.
4. Play the learned agent against the perfect player from the losing side, and measure how often the search finds swindles.
5. Train on Connect-4 4 × 5 (E5), where perfect play is a draw and strategy matters more than tactics.
