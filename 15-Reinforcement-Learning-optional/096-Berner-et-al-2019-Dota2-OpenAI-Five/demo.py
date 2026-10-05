"""OpenAI Five at toy scale in ~20 seconds: surgery (changing the network and observations without changing the policy),
surgery vs restarting on an environment under development, PPO + GAE on CartPole with the paper's data-quality
ablations (batch size, staleness, sample reuse), and team spirit."""

import time

import numpy as np

import five as F

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Surgery: modify the model, keep the policy (Appendix B)")
rng = np.random.default_rng(0)
p = F.init_mlp(6, 16, 2, rng)
X = rng.standard_normal((1000, 6))
print(f"  widen hidden 16 -> 32 (random in, ZERO out, Eq. 6):        max change in action probs "
      f"{F.max_policy_change(p, F.surgery_widen(p, 32, rng), X):.1e}")
print(f"  add 3 observations (ZERO weights, Eq. 9):                   max change "
      f"{F.max_policy_change(p, F.surgery_add_inputs(p, 3), X, pad=3):.1e}")
for small in (0.01, 0.1, 0.5):
    print(f"  recurrent-style widening, new outgoing weights ~N(0, {small:<4}): max change "
          f"{F.max_policy_change(p, F.surgery_widen_recurrent(p, 32, rng, small), X):.3f}")
print("  -> exact for feed-forward growth and new inputs; for the LSTM (2048 -> 4096) OpenAI picked the largest random")
print("     scale that did not noticeably lower TrueSkill")

section("2. An environment under development: surgery vs restart (3 stages, 25 PPO versions each)")
r = F.surgery_vs_restart()
print("    stage   surgery: return at first version -> last     restart: return at first version -> last")
for i, (s, q) in enumerate(zip(r["surgery"], r["restart"])):
    print(f"      {i + 1}         {s['first update return']:6.0f} -> {s['final return']:5.0f}                   "
          f"{q['first update return']:6.0f} -> {q['final return']:5.0f}")
print("  -> each change adds observations and widens the network; surgery keeps the skill already learned and reaches")
print("     500, while every restart begins again at ~20 and is still below 500 after the third stage (the paper:")
print("     restarting after each of ~20 surgeries would have taken ~40 months instead of 10)")

section("3. Data quality (Figure 5): PPO versions needed to reach a mean return of 200")
def versions(**kw):
    return F.train_ppo(updates=80, seed=0, **kw)[1]
b = {n: versions(n_envs=n) for n in (8, 32)}
print(f"  batch size: 8 episodes -> {b[8]} versions, 32 episodes -> {b[32]} versions "
      f"(speedup {b[8] / b[32]:.2f}x for 4x the data: sublinear)")
s = {k: versions(staleness=k) for k in (0, 8)}
print(f"  staleness:  fresh data -> {s[0]} versions, data from parameters 8 versions old -> {s[8]} versions "
      f"({s[8] / s[0]:.1f}x slower)")
u = {k: versions(reuse=k) for k in (1, 8)}
print(f"  sample reuse: 1 -> {u[1]} versions, 8 (each sample consumed ~8 times, 1/8 the fresh data) -> {u[8]} "
      f"versions ({u[8] / u[1]:.1f}x slower)")
print("  -> the paper's ordering: stale data hurts most, reuse hurts, bigger batches help sublinearly. Honest note:")
print("     our reuse penalty is milder than the paper's (2x slowdown already at reuse 2-3) -- CartPole is easy")

section("4. Team spirit: r_i = (1 - tau) rho_i + tau mean(rho), five agents")
for tau in (0.0, 0.3, 1.0):
    c, snr = F.team_spirit(tau)
    print(f"    tau = {tau:.1f}: gradient signal-to-noise at start {snr:.3f}; team reward after 50 / 100 / 300 steps "
          f"{np.mean(c[45:55]):.2f} / {np.mean(c[95:105]):.2f} / {np.mean(c[-10:]):.2f}")
print("  -> with tau = 1 each agent's reward is diluted by four teammates' noise, halving the signal-to-noise and slowing")
print("     early learning; low tau gives clearer credit early, which is why OpenAI Five annealed it 0.3 -> 1.0")

section("What the paper reports (verified against the PDF)")
for k, v in F.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
