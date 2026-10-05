# The code, explained simply

How the code in this folder reproduces OpenAI Five's key ideas (Berner et al. 2019) at laptop scale.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `five.py` | surgery operations on a small MLP policy, PPO + GAE (manual gradients, Adam) on the vectorised CartPole from paper 095, staleness and sample-reuse controls, surgery vs restart, and a team-spirit toy |
| `experiments.py` | batch-size, staleness and reuse sweeps; surgery over more stages; the τ sweep (E1–E5) |
| `demo.py` | surgery checks, surgery vs restart, the three data-quality ablations, team spirit (~8 seconds) |
| `test_five.py` | 4 quick tests (~1.5 seconds) |

**Run it** (from `15-Reinforcement-Learning-optional/096-Berner-et-al-2019-Dota2-OpenAI-Five`; reuses `../095-…/es.py` for CartPole):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `five.py`

### Surgery
| Name | What it does |
|---|---|
| `init_mlp`, `mlp_forward`, `softmax` | a one-hidden-layer tanh MLP |
| `surgery_widen(p, new_hidden)` | Eq. 6: random new W1 columns and b1 entries, **zero** new W2 rows |
| `surgery_add_inputs(p, n_new)` | Eq. 9: **zero** rows in W1 for new observation features |
| `surgery_widen_recurrent(p, new_hidden, small)` | stands in for the LSTM case: new outgoing weights ~N(0, small) instead of zero |
| `max_policy_change(p, q, X, pad)` | the largest change in action probabilities between old and new models (new inputs padded with zeros) |

### PPO + GAE
| Name | What it does |
|---|---|
| `features(s)` | CartPole state, with the angle scaled by 5 |
| `rollout(policy, n_envs, rng, extra_obs)` | a batch of stochastic episodes; returns per-step observations, actions, log-probabilities, rewards and alive masks, plus episode returns |
| `gae(rews, values, masks, γ, λ)` | backward recursion Aₜ = δₜ + γλ Aₜ₊₁, with no bootstrapping past the end of an episode |
| `PPO` | separate policy and value MLPs; `update` runs `epochs` passes of 4 minibatches. The policy gradient is −A · r · (onehot − p) on samples where the clip isn't active; the value loss is squared error. Adam with lr 1e-3 (policy) and 1e-2 (value) |
| `train_ppo(n_envs, staleness, reuse, epochs, target)` | each update is one parameter version. **Staleness k:** rollouts use the policy from k versions ago (a queue). **Reuse k:** only n_envs/k fresh episodes per version, with the optimizer training on a buffer of the last k collections, so consumption / production ≈ k. Returns the training curve and the first version where the 3-version mean reaches `target` |

### Surgery vs restart and team spirit
| Name | What it does |
|---|---|
| `surgery_vs_restart(stages)` | at each stage, 2 extra observation features (copies of the angle and angular velocity) and 8 more hidden units. Surgery transforms the existing agent (and resets Adam); restart builds a new agent. Both train 25 versions |
| `team_spirit(τ)` | 5 Bernoulli "effort" agents; raw reward 0.4·effort + team bonus/5 + noise; REINFORCE on rᵢ = (1 − τ)ρᵢ + τ·mean(ρ); tracks the team's mean reward and the starting gradient signal-to-noise (measured on 20,000 samples) |

`REPORTED` holds the paper's facts (OG 2–0; 7,215 of 7,257 Arena games; 770 PFlops/s·days; the model; surgery; Rerun; batch, staleness and reuse findings; team spirit; reaction time), checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | episodes per version 4 to 128 × 10 seeds |
| `e2` | staleness 0 to 32 × 10 seeds |
| `e3` | reuse 1 to 16 × 10 seeds |
| `e4` | surgery vs restart over 5 stages × 10 seeds; recurrent-widening scale vs policy change |
| `e5` | τ ∈ {0, 0.3, 0.5, 0.75, 1} × 20 seeds |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_surgery_is_exact_for_widening_and_new_inputs` | exact function preservation; the right shapes and zero blocks; non-zero change for the recurrent variant |
| `test_gae_matches_hand_computation` | the 3-step example from EXPLAINED.md |
| `test_ppo_learns_and_staleness_hurts` | PPO reaches 200 within 40 versions; staleness 8 needs > 1.5× as many (or never gets there) |
| `test_team_spirit_snr` | τ = 0 has > 1.5× the gradient signal-to-noise of τ = 1 |

---

## 5. Try it yourself

1. Add an LSTM policy (for a partially observed CartPole that hides velocities) and implement true LSTM widening with small random weights.
2. Simulate asynchronous workers with random delays (staleness drawn per batch) instead of a fixed k.
3. Add self-play: a two-player game where 80% of games are against the current policy and 20% against past versions.
4. Measure the batch-size speedup at a higher target (e.g. 450) to see whether it grows later in training, as the paper speculates.
5. Anneal τ from 0.3 to 1 during training and compare with fixed τ.
