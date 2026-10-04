"""Parallelized Stochastic Gradient Descent (Zinkevich, Weimer, Smola & Li, NeurIPS 2010): SimuParallelSGD.

  Algorithm 3   split the m examples randomly into k parts; on every machine, independently and WITHOUT any
                communication, run plain SGD with a fixed learning rate eta over its T = m/k examples (starting from
                w = 0); at the very end average the k parameter vectors:  v = (1/k) sum_i w_i
  loss          regularised risk  c_i(w) = (lambda/2) ||w||^2 + L(x_i, y_i, w . x_i)   (Huber, squared, logistic)
  theory        - with eta <= eta*, one SGD step is a contraction with constant (1 - eta lambda) (Lemma 3), so the
                  distribution of w converges exponentially fast (in the Wasserstein metric) to a stationary
                  distribution D*_eta (Theorem 8)
                - the MEAN of D*_eta is nearly optimal: c(E w) - min c <= 2 eta G^2  (Theorem 9; the 'bias')
                - its spread is small: E ||w - w*||^2 <= 4 eta G^2 / ((2 - eta lambda) lambda)  (Theorem 10)
                - averaging k independent runs divides the VARIANCE by k but leaves the bias (Theorem 12)
                So: few communication rounds (one), wall-clock speed-up from k machines, and the error can be made
                arbitrarily small by halving eta and doubling T.

This file simulates k machines at once with numpy (row i of a k x d matrix is machine i's parameter vector), on a
synthetic sparse, hashed-feature classification task like the paper's e-mail data, and measures the objective and
test error relative to one sequential pass over all the data, as the paper does.
"""

import numpy as np

# ----------------------------------------------------------------------------------------------- data

def make_data(m, d=512, active=12, noise=0.3, seed=0):
    """Sparse binary features (each example has `active` ones, hashed into d dimensions), normalised to unit length,
    labels y = sign(w_true . x + noise) in {-1, +1} -- a stand-in for the paper's hashed e-mail features.
    Use split() to cut one draw into train and test parts from the same distribution."""
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal(d)
    X = np.zeros((m, d))
    idx = rng.integers(0, d, size=(m, active))
    np.put_along_axis(X, idx, 1.0, axis=1)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.sign(X @ w_true + noise * rng.standard_normal(m) / np.sqrt(active))
    y[y == 0] = 1
    return X, y

def split(X, y, n_test):
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


def coupled_distance(X, y, eta, lam, kind="huber", steps=200, seed=0):
    """Contraction check: two SGD chains fed the SAME examples from different starting points. Lemma 3 says their
    distance shrinks by at least (1 - eta lambda) per step. Returns the distances."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    w1, w2 = np.zeros(d), rng.standard_normal(d)
    dist = [float(np.linalg.norm(w1 - w2))]
    for _ in range(steps):
        i = rng.integers(len(y))
        for w in (w1, w2):
            w -= eta * (dloss(kind, w @ X[i], y[i]) * X[i] + lam * w)
        dist.append(float(np.linalg.norm(w1 - w2)))
    return dist


# ----------------------------------------------------------------------------------------------- losses

def dloss(kind, margin_or_pred, y, delta=1.0):
    """Derivative of L with respect to the prediction p = w . x."""
    if kind == "squared":
        return margin_or_pred - y
    if kind == "huber":
        r = margin_or_pred - y
        return np.clip(r, -delta, delta)
    if kind == "logistic":
        return -y / (1 + np.exp(y * margin_or_pred))
    raise ValueError(kind)


def loss(kind, p, y, delta=1.0):
    if kind == "squared":
        return 0.5 * (p - y) ** 2
    if kind == "huber":
        r = np.abs(p - y)
        return np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))
    return np.log1p(np.exp(-y * p))


def objective(w, X, y, lam, kind):
    """c(w) = (1/m) sum_i c_i(w) for one w (d,) or several (k, d)."""
    W = np.atleast_2d(w)
    return (0.5 * lam * (W ** 2).sum(1) + loss(kind, X @ W.T, y[:, None]).mean(0)).squeeze()

# ----------------------------------------------------------------------------------------------- SimuParallelSGD

def simu_parallel_sgd(X, y, k, eta, lam, kind="huber", passes=1, seed=0, snapshots=None):
    """Algorithm 3: randomly partition into k machines with T = m // k examples each, shuffle locally, run SGD from
    w = 0 with a fixed learning rate, then average. Returns the averaged v, the k machine vectors, and (optionally)
    the average after each snapshot step count."""
    rng = np.random.default_rng(seed)
    m, d = X.shape
    T = m // k
    perm = rng.permutation(m)[:k * T].reshape(k, T)                      # machine i owns perm[i]
    W = np.zeros((k, d))
    snaps = {}
    for p in range(passes):
        order = np.stack([rng.permutation(T) for _ in range(k)])          # local shuffles
        for t in range(T):
            rows = perm[np.arange(k), order[:, t]]                        # every machine's t-th example
            xb, yb = X[rows], y[rows]
            g = dloss(kind, (W * xb).sum(1), yb)[:, None] * xb + lam * W
            W -= eta * g
            step = p * T + t + 1
            if snapshots and step in snapshots:
                snaps[step] = W.mean(0).copy()
    return W.mean(0), W, snaps


def full_batch_minimiser(X, y, lam, kind="huber", iters=3000, lr=1.0):
    """The risk minimiser w* by full-batch gradient descent (for measuring sub-optimality)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = X @ w
        w -= lr * (X.T @ dloss(kind, p, y) / len(y) + lam * w)
    return w


def eta_star(X, lam, c_star=1.0):
    """Lemma 3: SGD is a contraction if eta <= 1 / (max ||x|| c* + lambda) (c* bounds the loss-gradient Lipschitz)."""
    return 1.0 / (np.linalg.norm(X, axis=1).max() * c_star + lam)


def stationary_stats(X, y, eta, lam, kind="huber", runs=200, steps=4000, seed=0):
    """Run many independent fixed-eta SGD chains on the same data to sample the stationary distribution D*_eta:
    returns the mean of the chains' final w and their average squared distance to that mean."""
    rng = np.random.default_rng(seed)
    m, d = X.shape
    W = np.zeros((runs, d))
    for _ in range(steps):
        rows = rng.integers(0, m, size=runs)
        g = dloss(kind, (W * X[rows]).sum(1), y[rows])[:, None] * X[rows] + lam * W
        W -= eta * g
    mean = W.mean(0)
    return mean, float(((W - mean) ** 2).sum(1).mean()), W


REPORTED = {
    "algorithm": "SimuParallelSGD: k machines, T = m/k examples each, fixed learning rate, no communication until a "
                 "single final average -- one MapReduce pass",
    "theory": "contraction (1 - eta lambda) for eta <= eta* (Lemma 3, Theorem 8); bias of the stationary mean "
              "<= 2 eta G^2 (Theorem 9); E||w - w*||^2 <= 4 eta G^2 / ((2 - eta lambda) lambda) (Theorem 10); averaging "
              "reduces variance like 1/k but not bias (Theorem 12)",
    "data": "e-mail spam: 3,189,235 instances (2,508,220 train, 681,015 test), hashed to 2^18 features, ~313 active "
            "features each, unit-normalised",
    "setup": "Huber and squared loss, lambda in {1e-3, 1e-6}, eta = 1e-3, 1 / 10 / 100 machines, objective and test "
             "RMSE normalised so one sequential pass over all data = 1.0",
    "findings": "more machines reach a better objective for the same per-machine work (wall clock); 1 -> 10 machines "
                "helps much more than 10 -> 100; the gain is larger for the high-variance lambda = 1e-6 problem; "
                "parallel training needs slightly more total machine time",
}
