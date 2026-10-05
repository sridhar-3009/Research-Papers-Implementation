"""Evolution Strategies as a Scalable Alternative to Reinforcement Learning (Salimans, Ho, Chen, Sidor, Sutskever;
OpenAI 2017).

  objective       maximise the Gaussian-blurred return  E_{eps ~ N(0, I)} F(theta + sigma eps)
  gradient        grad = E[ F(theta + sigma eps) eps ] / sigma     (score-function / zero-order estimator)
  Algorithm 1     sample eps_1..eps_n, evaluate F_i = F(theta + sigma eps_i), theta += alpha / (n sigma) sum F_i eps_i
  Algorithm 2     n workers share random SEEDS: each worker evaluates its own perturbation, broadcasts ONE SCALAR
                  (its return), and every worker rebuilds all perturbations from the seeds -> identical updates
                  everywhere with almost no bandwidth
  tricks          antithetic (mirrored) pairs eps, -eps; fitness shaping = rank transformation of the returns;
                  weight decay; fixed sigma
  claims          (3.1) the policy-gradient estimator's variance grows ~linearly with the episode length T, the ES
                  estimator's does not -> ES is invariant to frame-skip / action frequency; (3.2) what matters is the
                  intrinsic dimension: duplicating features does not make ES harder

This file implements ES and the parallel shared-seed version, a vectorised numpy CartPole (all perturbed policies are
simulated as one batch), REINFORCE as the policy-gradient comparison, and small experiments for each claim.
"""

import numpy as np

# ----------------------------------------------------------------------------------------------- ES core

def centered_ranks(x):
    """Fitness shaping: replace returns by their ranks, scaled to [-0.5, 0.5]."""
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def es_gradient(F, theta, sigma, n_pairs, rng, shaping=True, antithetic=True):
    """One ES gradient estimate. F maps a (P, d) matrix of parameter vectors to P returns."""
    d = len(theta)
    if antithetic:
        e = rng.standard_normal((n_pairs, d))
        eps = np.vstack([e, -e])
    else:
        eps = rng.standard_normal((2 * n_pairs, d))
    R = F(theta[None, :] + sigma * eps)
    w = centered_ranks(R) if shaping else R - R.mean()
    return (w @ eps) / (len(eps) * sigma), R


class Adam:
    def __init__(self, d, lr=0.03, b1=0.9, b2=0.999):
        self.m, self.v, self.t, self.lr, self.b1, self.b2 = np.zeros(d), np.zeros(d), 0, lr, b1, b2

    def step(self, g):                                                 # ASCENT direction g
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        return self.lr * (self.m / (1 - self.b1 ** self.t)) / (np.sqrt(self.v / (1 - self.b2 ** self.t)) + 1e-8)


def es_optimize(F, theta0, iters=50, sigma=0.1, n_pairs=20, lr=0.03, wd=0.005, seed=0, shaping=True,
                antithetic=True, callback=None):
    """ES with Adam, weight decay, antithetic sampling and rank shaping (the paper's recipe)."""
    rng = np.random.default_rng(seed)
    theta = theta0.astype(float).copy()
    opt = Adam(len(theta), lr)
    hist = []
    for it in range(iters):
        g, R = es_gradient(F, theta, sigma, n_pairs, rng, shaping, antithetic)
        theta = theta + opt.step(g - wd * theta)
        hist.append(float(R.mean()))
        if callback:
            callback(it, theta, R)
    return theta, hist

# ----------------------------------------------------------------------------------------------- Algorithm 2

def parallel_es(F_single, theta0, workers=8, iters=20, sigma=0.1, alpha=0.05, seed=0):
    """Simulated Algorithm 2: worker i draws eps_i from its own seeded stream, evaluates F_i, broadcasts the scalar;
    every worker then regenerates ALL eps_j from the known seeds and applies the same update. Returns each worker's
    final parameters, the floats each worker sent, and the floats a gradient-sharing method would have sent."""
    d = len(theta0)
    thetas = [theta0.astype(float).copy() for _ in range(workers)]
    sent = 0
    for t in range(iters):
        returns = []
        for i in range(workers):                                        # lines 4-7: local evaluation
            eps_i = np.random.default_rng([seed, t, i]).standard_normal(d)
            returns.append(F_single(thetas[i] + sigma * eps_i))
        sent += 1                                                       # line 8: one scalar per worker
        for i in range(workers):                                        # lines 9-12: identical update everywhere
            eps = np.stack([np.random.default_rng([seed, t, j]).standard_normal(d) for j in range(workers)])
            w = centered_ranks(np.array(returns))                       # fitness shaping, as in the paper
            thetas[i] = thetas[i] + alpha / (workers * sigma) * (w @ eps)
    return thetas, sent, iters * d

# ----------------------------------------------------------------------------------------------- CartPole (vectorised)

class CartPole:
    """Classic cart-pole (the constants of OpenAI Gym's CartPole-v1), simulated for a whole batch of policies at
    once. Reward 1 per step until the pole falls (|angle| > 12 degrees) or the cart leaves |x| > 2.4, max 500 steps.
    `frame_skip` repeats each chosen action that many simulator steps (max steps scale so the episode TIME is fixed)."""
    g, mc, mp, l, fmag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02

    def __init__(self, n, rng, frame_skip=1):
        self.s = rng.uniform(-0.05, 0.05, (n, 4))
        self.alive = np.ones(n, bool)
        self.frame_skip = frame_skip

    def step(self, action):
        reward = np.zeros(len(self.s))
        for _ in range(self.frame_skip):
            x, xd, th, thd = self.s.T
            f = np.where(action > 0, self.fmag, -self.fmag)
            ct, st = np.cos(th), np.sin(th)
            tmp = (f + self.mp * self.l * thd ** 2 * st) / (self.mc + self.mp)
            tha = (self.g * st - ct * tmp) / (self.l * (4 / 3 - self.mp * ct ** 2 / (self.mc + self.mp)))
            xa = tmp - self.mp * self.l * tha * ct / (self.mc + self.mp)
            new = np.column_stack([x + self.tau * xd, xd + self.tau * xa, th + self.tau * thd, thd + self.tau * tha])
            self.s = np.where(self.alive[:, None], new, self.s)
            reward += self.alive
            self.alive &= (np.abs(self.s[:, 0]) <= 2.4) & (np.abs(self.s[:, 2]) <= 12 * np.pi / 180)
        return reward


def policy_logits(params, s, hidden=0):
    """Linear policy (hidden = 0): params = (w[4], b). Or a tiny MLP 4-hidden-1 with tanh."""
    if hidden == 0:
        return np.einsum("pd,pd->p", params[:, :4], s) + params[:, 4]
    W1 = params[:, :4 * hidden].reshape(-1, 4, hidden)
    b1 = params[:, 4 * hidden:5 * hidden]
    W2 = params[:, 5 * hidden:6 * hidden]
    b2 = params[:, 6 * hidden]
    h = np.tanh(np.einsum("pd,pdh->ph", s, W1) + b1)
    return np.einsum("ph,ph->p", h, W2) + b2


def n_params(hidden=0):
    return 5 if hidden == 0 else 6 * hidden + 1


def cartpole_returns(P, seed=0, frame_skip=1, max_time_steps=500, hidden=0, obs_noise=0.0):
    """Deterministic policies (action = logit > 0) evaluated for every row of P in one batch. Returns the number of
    simulator steps survived (max 500) and the number of environment steps used."""
    rng = np.random.default_rng(seed)
    env = CartPole(len(P), rng, frame_skip)
    total = np.zeros(len(P))
    used = 0
    for _ in range(max_time_steps // frame_skip):
        if not env.alive.any():
            break
        obs = env.s + obs_noise * rng.standard_normal(env.s.shape)
        used += int(env.alive.sum())
        total += env.step(policy_logits(P, obs, hidden))
    return total, used


def train_es_cartpole(iters=40, n_pairs=20, sigma=0.1, lr=0.05, frame_skip=1, hidden=0, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros(n_params(hidden)) if hidden == 0 else 0.3 * rng.standard_normal(n_params(hidden))
    opt = Adam(len(theta), lr)
    steps, curve = 0, []
    for it in range(iters):
        e = rng.standard_normal((n_pairs, len(theta)))
        eps = np.vstack([e, -e])
        R, used = cartpole_returns(theta + sigma * eps, seed=seed * 10000 + it, frame_skip=frame_skip, hidden=hidden)
        steps += used
        g = (centered_ranks(R) @ eps) / (len(eps) * sigma)
        theta = theta + opt.step(g - 0.005 * theta)
        curve.append((steps, float(R.mean())))
    final, _ = cartpole_returns(np.repeat(theta[None], 20, 0), seed=seed + 999, frame_skip=frame_skip, hidden=hidden)
    return theta, curve, float(final.mean())

# ----------------------------------------------------------------------------------------------- REINFORCE

def train_reinforce_cartpole(iters=40, batch=40, lr=0.05, frame_skip=1, gamma=1.0, seed=0):
    """Policy gradient with a Bernoulli policy sigma(w.s + b), REINFORCE with reward-to-go and a mean baseline."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(5)
    opt = Adam(5, lr)
    steps, curve = 0, []
    for it in range(iters):
        env = CartPole(batch, rng, frame_skip)
        feats, acts, rews, alive_mask = [], [], [], []
        for _ in range(500 // frame_skip):
            if not env.alive.any():
                break
            s = env.s.copy()
            p = 1 / (1 + np.exp(-(s @ theta[:4] + theta[4])))
            a = (rng.random(batch) < p).astype(float)
            al = env.alive.copy()
            steps += int(al.sum())
            r = env.step(np.where(a > 0, 1.0, -1.0))
            feats.append(np.column_stack([s, np.ones(batch)]))
            acts.append(a - p)                                          # d log pi / d logit
            rews.append(r)
            alive_mask.append(al)
        T = len(rews)
        R = np.array(rews)
        G = np.zeros_like(R)
        run = np.zeros(batch)
        for t in range(T - 1, -1, -1):
            run = R[t] + gamma * run
            G[t] = run
        A = np.array(alive_mask)
        base = (G * A).sum(1, keepdims=True) / np.maximum(A.sum(1, keepdims=True), 1)
        adv = (G - base) * A
        grad = np.einsum("tb,tb,tbd->d", adv, np.array(acts), np.array(feats)) / batch
        theta = theta + opt.step(grad / max(T, 1))
        curve.append((steps, float(R.sum(0).mean())))
    final, _ = cartpole_returns(np.repeat(theta[None], 20, 0), seed=seed + 999, frame_skip=frame_skip)
    return theta, curve, float(final.mean())

# ----------------------------------------------------------------------------------------------- section 3.1: variance vs T

def estimator_variance_vs_T(Ts=(10, 30, 100, 300, 1000), n=4000, sigma=0.3, noise=0.3, seed=0):
    """A 1-parameter problem with T binary actions a_t ~ Bernoulli(sigmoid(theta)) and return
    R = mean(a_t) + environment noise N(0, noise^2) -- so, as the paper assumes for hard RL problems, each single action
    is only weakly correlated with the return, and Var[R] does not shrink with T. True gradient of E[R]: sigmoid'(theta).
    Single-episode estimators:
      REINFORCE   (R - mean R) * sum_t (a_t - p)                         (score is a sum of T terms)
      ES          (R(theta + sigma e) - R(theta - sigma e)) e / (2 sigma)  with actions a_t = 1[u_t < sigmoid(.)]
    Report each estimator's mean and its variance relative to the squared true gradient."""
    rng = np.random.default_rng(seed)
    theta = 0.0
    p = 1 / (1 + np.exp(-theta))
    true = p * (1 - p)
    out = {}
    for T in Ts:
        a = (rng.random((n, T)) < p).astype(float)
        R = a.mean(1) + noise * rng.standard_normal(n)
        pg = (R - R.mean()) * (a - p).sum(1)
        e = rng.standard_normal(n)
        u = rng.random((n, T))
        pp, pm = 1 / (1 + np.exp(-(theta + sigma * e))), 1 / (1 + np.exp(-(theta - sigma * e)))
        Rp = (u < pp[:, None]).mean(1) + noise * rng.standard_normal(n)
        Rm = (u < pm[:, None]).mean(1) + noise * rng.standard_normal(n)
        es = (Rp - Rm) * e / (2 * sigma)
        out[T] = {"PG mean": float(pg.mean()), "PG var / true^2": float(pg.var() / true ** 2),
                  "ES mean": float(es.mean()), "ES var / true^2": float(es.var() / true ** 2)}
    return out, true

# ----------------------------------------------------------------------------------------------- section 3.2: duplicated features

def intrinsic_dimension(iters=150, n_pairs=10, seed=0, reps=20, lr=0.02):
    """Regression y = x.w with 5 features vs the same with duplicated features x' = (x, x). The paper says ES does
    'exactly the same thing' on the doubled problem if sigma and the learning rate are both divided by two. Our
    derivation: the effective weight is w1 + w2 and its perturbation sigma'(e1 + e2) ~ N(0, 2 sigma'^2), so the exact
    match needs sigma / sqrt 2 and lr / 2. Mean final loss over `reps` seeds for each variant."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((200, 5))
    y = X @ rng.standard_normal(5)
    X2 = np.hstack([X, X])
    out = {}
    for name, Xf, sig, step in (("5 features (sigma, lr)", X, 0.1, lr),
                                ("10 features, sigma / sqrt 2, lr / 2", X2, 0.1 / np.sqrt(2), lr / 2),
                                ("10 features, sigma / 2, lr / 2 (paper's wording)", X2, 0.05, lr / 2),
                                ("10 features, unchanged sigma and lr", X2, 0.1, lr)):
        F = lambda P, Xf=Xf: -((P @ Xf.T - y) ** 2).mean(1)
        finals, curves = [], []
        for s_ in range(reps):
            r = np.random.default_rng(s_)
            theta = np.zeros(Xf.shape[1])
            for _ in range(iters):
                g, _ = es_gradient(F, theta, sig, n_pairs, r, shaping=False)
                theta = theta + step * g
            finals.append(-F(theta[None])[0])
        out[name] = float(np.mean(finals))
    return out


REPORTED = {
    "humanoid": "3D Humanoid (score 6000) solved in under 10 minutes on 1,440 CPU cores (80 machines); ~11 hours on one "
                "18-core machine; linear speedup in the number of cores",
    "Atari": "51 games, 1 hour per game on 720 CPUs; better than published A3C (1 day) on 23 games, worse on 28; ES used "
             "3x-10x as much data, partly offset by ~3x less computation (no backpropagation)",
    "MuJoCo (Table 1)": "ratio of ES to TRPO timesteps to reach 100% of TRPO's 5M-step progress: HalfCheetah 0.58, Hopper "
                        "6.94, InvertedDoublePendulum 1.23, InvertedPendulum 0.88, Swimmer 0.30, Walker2d 7.88 (within "
                        "10x on hard tasks, up to 3x better on simple ones)",
    "communication": "with shared seeds each worker only needs to communicate a scalar (its return)",
    "tricks": "antithetic (mirrored) sampling, rank-based fitness shaping, weight decay, fixed sigma, virtual batch "
              "normalisation for Atari, discretised actions for some MuJoCo tasks",
    "variance": "Var[PG] ~ Var[R] Var[sum_t grad log p(a_t)] grows ~linearly with T; Var[ES] is independent of T -> "
                "invariance to frame-skip (Pong with frame-skip 1-4 learns similarly)",
}
