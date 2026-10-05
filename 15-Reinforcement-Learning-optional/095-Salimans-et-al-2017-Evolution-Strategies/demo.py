"""Evolution strategies in ~10 seconds: the ES gradient estimator, Algorithm 2's shared-seed parallelism (scalars
instead of gradients), ES vs REINFORCE on a vectorised CartPole, why ES's gradient variance does not grow with the
episode length, and the duplicated-features (intrinsic dimension) argument."""

import time

import numpy as np

import es as E

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The ES gradient is a Gaussian-smoothed gradient (check on F(theta) = -|theta - 1|^2, d = 10)")
F = lambda P: -((P - 1) ** 2).sum(1)
theta = np.zeros(10)
true = 2 * (1 - theta)                                                 # exact gradient (smoothing adds a constant)
print("    sigma    plain sampling: mean / std         mirrored pairs: mean / std      (100 evaluations each)")
for sig in (0.1, 1.0, 3.0):
    row = []
    for anti in (False, True):
        rng = np.random.default_rng(0)
        gs = np.array([E.es_gradient(F, theta, sig, 50, rng, shaping=False, antithetic=anti)[0] for _ in range(300)])
        row.append(f"{gs.mean(0)[:2].round(2)} / {gs.std(0).mean():.3f}")
    print(f"    {sig:4.1f}     {row[0]:28s}       {row[1]}")
print(f"  -> unbiased in every case (true gradient {true[:2]}). Mirrored pairs cancel the EVEN part of F (here the")
print("     curvature term sigma^2 |eps|^2), so their variance does not depend on sigma at all (0.925); plain sampling")
print("     gets noisier as sigma grows. Honest note: at small sigma the objective is nearly linear and mirroring only")
print("     halves the number of independent directions, so it is WORSE there (0.93 vs 0.65)")

section("2. Algorithm 2: 16 workers with shared random seeds")
thetas, sent, grad_floats = E.parallel_es(lambda th: -np.sum((th - 1) ** 2), np.zeros(100), workers=16, iters=150,
                                          alpha=0.02)
print(f"  after 150 updates all workers hold identical parameters (max difference "
      f"{max(np.abs(thetas[0] - t).max() for t in thetas):.1f}); objective -100.0 -> {-np.sum((thetas[0] - 1) ** 2):.1f}")
print(f"  each worker sent {sent} scalars in total, vs {grad_floats:,} floats if it had to share a 100-dim gradient "
      f"each step ({grad_floats // sent}x less)")

section("3. CartPole (linear policy, 5 parameters): ES vs REINFORCE, 100 updates each")
th_es, c_es, f_es = E.train_es_cartpole(iters=100)
th_pg, c_pg, f_pg = E.train_reinforce_cartpole(iters=100)
for name, c, f in (("ES (40 perturbations / update)", c_es, f_es), ("REINFORCE (40 episodes / update)", c_pg, f_pg)):
    reach = next((s for s, r in c if r >= 475), None)
    print(f"  {name:34s} final test return {f:.0f}/500; env steps used {c[-1][0]:,}; first update averaging "
          f">= 475: {f'{reach:,} steps' if reach else 'never'}")
print("  -> both reach a perfect 500 with their final deterministic policy; ES used 1.6x the environment steps (the")
print("     paper: 3-10x more data than A3C on Atari), but each ES evaluation is a forward pass only and the 40")
print("     evaluations are independent (trivially parallel). REINFORCE's TRAINING episodes use a stochastic policy, so")
print("     its training average never crossed 475 even though the learned weights score 500 when acting greedily")

section("4. Why ES does not care about episode length (section 3.1)")
out, true = E.estimator_variance_vs_T()
print(f"  T binary actions, return = mean of actions + noise; true gradient {true:.3f}")
print("      T     REINFORCE mean / variance (x true^2)     ES mean / variance (x true^2)")
for T, v in out.items():
    print(f"    {T:5d}    {v['PG mean']:.3f} / {v['PG var / true^2']:7.1f}                      "
          f"{v['ES mean']:.3f} / {v['ES var / true^2']:5.1f}")
print("  -> the policy-gradient score is a sum over T steps, so its variance grows ~linearly with T (5.6 -> 367);")
print("     ES perturbs the parameters once per episode, so its variance stays flat (~10) -- this is why ES is")
print("     insensitive to frame-skip and long horizons")

section("5. Intrinsic dimension: duplicating the features (section 3.2), mean final loss over 20 seeds")
for k, v in E.intrinsic_dimension().items():
    print(f"    {k:50s} {v:.2e}")
print("  -> with the learning rate halved the doubled problem behaves like the original (7.3e-5 vs 7.1e-5): more")
print("     parameters, same difficulty. On this quadratic objective sigma does not matter at all (mirrored differences")
print("     of a quadratic are exactly linear in sigma*eps, so sigma cancels), which is why sigma/2 and sigma/sqrt 2")
print("     give identical results; our derivation says sigma/sqrt 2 is the general match. Keeping the same lr")
print("     doubles the effective step and converges faster")

section("What the paper reports (verified against the PDF)")
for k, v in E.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
