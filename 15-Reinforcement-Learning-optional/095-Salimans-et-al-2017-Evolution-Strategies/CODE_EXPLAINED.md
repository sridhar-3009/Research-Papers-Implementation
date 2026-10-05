# The code, explained simply

How the code in this folder implements evolution strategies (Salimans et al. 2017).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `es.py` | the ES estimator with mirrored sampling and rank shaping, Adam, simulated parallel ES with shared seeds (Algorithm 2), a batched numpy CartPole, REINFORCE, the variance-vs-horizon study and the duplicated-features study |
| `experiments.py` | CartPole data efficiency, frame-skip, an ablation with an MLP policy, time per update, and the variance table (E1–E5) |
| `demo.py` | the estimator, Algorithm 2, CartPole ES vs REINFORCE, variance vs T, intrinsic dimension (~2 seconds) |
| `test_es.py` | 4 quick tests (~0.5 seconds) |

**Run it** (from `15-Reinforcement-Learning-optional/095-Salimans-et-al-2017-Evolution-Strategies`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `es.py`

### ES core
| Name | What it does |
|---|---|
| `centered_ranks(x)` | ranks scaled to [−0.5, 0.5] (fitness shaping) |
| `es_gradient(F, theta, sigma, n_pairs, rng, shaping, antithetic)` | draws n_pairs noise vectors and their mirrors (or 2·n_pairs independent ones), evaluates the batch `F(theta + sigma eps)`, weights by ranks or by returns minus their mean, and returns `w @ eps / (n sigma)` |
| `Adam` | Adam for **ascent** |
| `es_optimize` | the paper's recipe: ES gradient, weight decay, Adam |

### Algorithm 2
`parallel_es(F_single, theta0, workers, iters)`:
- worker i's noise at step t comes from `default_rng([seed, t, i])`, so any worker can regenerate it;
- each worker evaluates its own perturbation;
- the returns (one scalar per worker) are turned into centred ranks;
- every worker regenerates all noise vectors and applies the same update.

It returns all workers' parameters (they are bit-identical), the scalars each worker sent, and the floats a gradient-sharing scheme would have sent.

### Environment and policies
| Name | What it does |
|---|---|
| `CartPole(n, rng, frame_skip)` | Gym's CartPole-v1 constants and Euler update, for n environments at once; finished episodes freeze; `step` repeats the action `frame_skip` times and returns the reward (sim steps survived) |
| `policy_logits(params, s, hidden)` | a batch of linear policies (5 parameters) or 4-h-1 tanh MLPs, one per row of `params` |
| `cartpole_returns(P, seed, frame_skip)` | runs every row of P as a deterministic policy (action = logit > 0) in one batch; returns the returns and the env steps used |
| `train_es_cartpole` | mirrored perturbations, rank shaping, Adam, weight decay 0.005; logs (env steps, mean return); final greedy evaluation |
| `train_reinforce_cartpole` | Bernoulli policy σ(w·s + b), reward-to-go, per-time-step mean baseline over the batch, gradient Σ advantage · (a − p) · features, Adam |

### The paper's two arguments
| Name | What it does |
|---|---|
| `estimator_variance_vs_T(Ts, noise)` | T Bernoulli actions; return = mean + Gaussian noise. REINFORCE uses (R − R̄) Σₜ(aₜ − p); ES uses mirrored differences with actions thresholded on shared uniforms. Returns each estimator's mean and variance / true² |
| `intrinsic_dimension()` | ES on regression with x vs (x, x) under four (σ, lr) settings; mean final loss over 20 seeds |

`REPORTED` holds the paper's results (Humanoid in 10 minutes on 1,440 cores, Atari 23 vs 28 games against A3C, Table 1, tricks, the variance argument), checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | CartPole: population and σ variants of ES vs REINFORCE over 10 seeds; steps to an average of 475 and final return |
| `e2` | frame-skip 1 / 2 / 4 for both methods |
| `e3` | {ranks, raw} × {mirrored, plain} with a 4-8-1 MLP policy |
| `e4` | time per ES update vs population size |
| `e5` | the variance table for T up to 10,000 and noise 0.1 / 0.3 / 1.0 |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_centered_ranks_and_unbiased_gradient` | rank shaping by hand; the averaged ES estimate on a quadratic ≈ the true gradient 2 |
| `test_parallel_workers_stay_identical_and_improve` | identical parameters across workers; the objective improves; scalars · d = gradient floats |
| `test_cartpole_physics_and_es_learns` | a zero policy (always push left) falls within 60 steps; ES reaches > 300 in 60 updates |
| `test_variance_grows_for_pg_not_es` | policy-gradient variance grows > 10× from T = 10 to 300; ES grows < 2× and stays unbiased |

---

## 5. Try it yourself

1. Use actual processes (`multiprocessing`) for Algorithm 2 and measure the speedup vs number of cores.
2. Add observation noise to CartPole (`obs_noise`) and compare ES and REINFORCE robustness.
3. Implement the paper's noise table: one big shared block of Gaussian noise, with workers sending only indices into it.
4. Compare ES with finite differences along coordinate axes (each worker perturbs one parameter).
5. Train a sparse-reward task (reward only at the end of a long episode) and compare ES and REINFORCE.
