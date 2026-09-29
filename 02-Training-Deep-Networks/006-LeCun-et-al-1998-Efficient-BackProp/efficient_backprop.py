"""The tricks of "Efficient BackProp" (LeCun, Bottou, Orr & Mueller, 1998), in NumPy.

  Section 3    a multilayer net and backprop in matrix form, Eqs. (2)-(9)
  Section 4    the tricks: input normalization (4.3), the recommended sigmoid (4.4),
               target values (4.5), weight initialization (4.6), stochastic vs
               batch learning (4.1)
  Section 5    learning rates from the Hessian: eta_opt = 1/lambda_max, divergence
               above 2/lambda_max (Eqs. 38-39)
  Section 7    Hessian information: diagonal second derivatives by backprop
               (Eqs. 54-56), Hessian-vector products by finite differences (Eq. 59)
  Section 9    the stochastic diagonal Levenberg-Marquardt method (Eqs. 61-62) and
               the power method for the largest eigenvalue (Eqs. 60, 63-64)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Section 4.4: sigmoids
# ---------------------------------------------------------------------------

def lecun_tanh(x):
    """The recommended sigmoid, f(x) = 1.7159 tanh(2x/3).
    Chosen so that f(+-1) = +-1, the second derivative peaks at x = +-1, and the
    'gain' is about 1: unit-variance inputs give roughly unit-variance outputs."""
    return 1.7159 * np.tanh(2 * x / 3)


def lecun_tanh_prime(x):
    return 1.7159 * (2 / 3) * (1 - np.tanh(2 * x / 3) ** 2)


def logistic(x):
    return 0.5 * (1 + np.tanh(x / 2))           # 1 / (1 + e^-x), overflow-safe


def logistic_prime(x):
    s = logistic(x)
    return s * (1 - s)


SIGMOIDS = {"lecun_tanh": (lecun_tanh, lecun_tanh_prime), "logistic": (logistic, logistic_prime)}


# ---------------------------------------------------------------------------
# Section 4.3: transforming the inputs (Figure 3)
# ---------------------------------------------------------------------------

class InputTransform:
    """Fit on the TRAINING set, apply to any set. Three steps (Figure 3):
      1. mean cancellation   - subtract each input's average
      2. KL expansion (PCA)  - rotate so the inputs are uncorrelated   (optional)
      3. covariance equalization - scale each input to variance 1
    """

    def __init__(self, decorrelate=False, eps=1e-8):
        self.decorrelate, self.eps = decorrelate, eps

    def fit(self, X):
        self.mean = X.mean(0)
        Xc = X - self.mean
        if self.decorrelate:
            cov = Xc.T @ Xc / len(X)
            evals, evecs = np.linalg.eigh(cov)
            self.rot = evecs
            self.scale = 1 / np.sqrt(evals + self.eps)
        else:
            self.rot = None
            self.scale = 1 / (Xc.std(0) + self.eps)
        return self

    def __call__(self, X):
        Xc = X - self.mean
        if self.rot is not None:
            Xc = Xc @ self.rot
        return Xc * self.scale


# ---------------------------------------------------------------------------
# Section 3: the multilayer network, Eqs. (2)-(9)
# ---------------------------------------------------------------------------

class MLP:
    """Layers of weights and sigmoids: Y_n = W_n X_{n-1} (+ bias), X_n = F(Y_n).
    Cost: E = 1/2 ||D - output||^2 per pattern (Section 2), averaged over patterns.

    init="lecun": weights uniform with standard deviation m^(-1/2), m = fan-in
                  (Eq. 16). init=<number>: uniform in [-number, number].
    The last layer uses the same sigmoid as the hidden layers (as in the paper's
    classification examples, with targets +-1)."""

    def __init__(self, sizes, sigmoid="lecun_tanh", init="lecun", rng=None):
        rng = np.random.default_rng(rng)
        self.f, self.fprime = SIGMOIDS[sigmoid]
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for m, n in zip(sizes[:-1], sizes[1:]):
            r = np.sqrt(3 / m) if init == "lecun" else float(init)   # uniform(-r, r) has std r/sqrt(3)
            self.W.append(rng.uniform(-r, r, (m, n)))
            self.b.append(rng.uniform(-r, r, n) if init != "lecun" else np.zeros(n))

    # -- flat parameter vector (for Hessian tools) --
    def get(self):
        return np.concatenate([np.concatenate([W.ravel(), b]) for W, b in zip(self.W, self.b)])

    def set(self, w):
        i = 0
        for k in range(len(self.W)):
            self.W[k] = w[i:i + self.W[k].size].reshape(self.W[k].shape); i += self.W[k].size
            self.b[k] = w[i:i + self.b[k].size].copy(); i += self.b[k].size

    def forward(self, X):
        """Returns the weighted sums Y_n and the states X_n of every layer."""
        Ys, Xs = [], [X]
        for W, b in zip(self.W, self.b):
            Ys.append(Xs[-1] @ W + b)          # Eq. (2)
            Xs.append(self.f(Ys[-1]))          # Eq. (3)
        return Ys, Xs

    def predict(self, X):
        return self.forward(X)[1][-1]

    def cost(self, X, D):
        return float(0.5 * np.mean(np.sum((D - self.predict(X)) ** 2, axis=1)))

    def gradient(self, X, D):
        """dE/dW for every layer, averaged over the patterns in X. Eqs. (7)-(9)."""
        Ys, Xs = self.forward(X)
        dE_dX = -(D - Xs[-1]) / len(X)                     # derivative of 1/2 ||D - X||^2
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for n in reversed(range(len(self.W))):
            dE_dY = self.fprime(Ys[n]) * dE_dX             # Eq. (7)
            gW[n] = Xs[n].T @ dE_dY                        # Eq. (8)
            gb[n] = dE_dY.sum(0)
            dE_dX = dE_dY @ self.W[n].T                    # Eq. (9)
        return gW, gb

    def flat_gradient(self, X, D):
        gW, gb = self.gradient(X, D)
        return np.concatenate([np.concatenate([a.ravel(), b]) for a, b in zip(gW, gb)])

    # -----------------------------------------------------------------------
    # Section 7.4: the diagonal of the (Gauss-Newton) Hessian, by backprop
    # -----------------------------------------------------------------------
    def diag_hessian(self, X):
        """d2E/dw^2 for every weight, averaged over the patterns, Eqs. (54)-(56):
            d2E/dy_k^2   = d2E/do_k^2 * f'(y_k)^2         (54)
            d2E/dw_ki^2  = d2E/dy_k^2 * x_i^2             (55)
            d2E/dx_i^2   = sum_k d2E/dy_k^2 * w_ki^2      (56)
        Like backprop, but with SQUARED derivatives and squared weights. The
        f'' terms are dropped (Gauss-Newton), so every entry is >= 0.
        For E = 1/2 ||D - o||^2, d2E/do^2 = 1 at the output."""
        Ys, Xs = self.forward(X)
        d2E_dX = np.ones_like(Xs[-1]) / len(X)
        hW, hb = [None] * len(self.W), [None] * len(self.W)
        for n in reversed(range(len(self.W))):
            d2E_dY = d2E_dX * self.fprime(Ys[n]) ** 2      # Eq. (54)
            hW[n] = (Xs[n] ** 2).T @ d2E_dY                # Eq. (55)
            hb[n] = d2E_dY.sum(0)                          # bias: x = 1
            d2E_dX = d2E_dY @ (self.W[n] ** 2).T           # Eq. (56)
        return hW, hb


# ---------------------------------------------------------------------------
# Training: stochastic (Eq. 11) and batch (Eq. 10) gradient descent
# ---------------------------------------------------------------------------

def train_sgd(net, X, D, epochs=10, lr=0.01, rng=None, shuffle=True, per_weight_lr=None,
              X_eval=None, D_eval=None):
    """Stochastic (online) learning, Eq. (11): update after EVERY pattern.
    per_weight_lr: optional (list of arrays for W, list for b), one rate per
    weight, e.g. from stochastic_diag_lm_rates. Returns cost after each epoch."""
    rng = np.random.default_rng(rng)
    X_eval = X if X_eval is None else X_eval
    D_eval = D if D_eval is None else D_eval
    history = [net.cost(X_eval, D_eval)]
    for _ in range(epochs):
        order = rng.permutation(len(X)) if shuffle else np.arange(len(X))
        for p in order:
            gW, gb = net.gradient(X[p:p + 1], D[p:p + 1])
            for k in range(len(net.W)):
                rW = lr if per_weight_lr is None else per_weight_lr[0][k]
                rb = lr if per_weight_lr is None else per_weight_lr[1][k]
                net.W[k] -= rW * gW[k]
                net.b[k] -= rb * gb[k]
        history.append(net.cost(X_eval, D_eval))
    return history


def train_batch(net, X, D, epochs=10, lr=0.1):
    """Batch learning, Eq. (10): one update per pass, using the true gradient."""
    history = [net.cost(X, D)]
    for _ in range(epochs):
        gW, gb = net.gradient(X, D)
        for k in range(len(net.W)):
            net.W[k] -= lr * gW[k]
            net.b[k] -= lr * gb[k]
        history.append(net.cost(X, D))
    return history


# ---------------------------------------------------------------------------
# Section 9.1: stochastic diagonal Levenberg-Marquardt
# ---------------------------------------------------------------------------

def stochastic_diag_lm_rates(net, X_sample, eta=0.01, mu=0.1):
    """Per-weight learning rates, Eq. (61):  eta_ki = eta / (<d2E/dw_ki^2> + mu).
    <d2E/dw^2> is estimated on a sample of patterns (the paper: 'computed prior
    to training over e.g. a subset of the training set ... reestimated every few
    epochs'). Here the average over the sample plays the role of the running
    average of Eq. (62). mu stops the rate blowing up where the curvature is ~0.

    Note: diag_hessian averages over patterns, so per-pattern curvatures are
    rescaled by len(X_sample) to match the per-pattern updates of SGD."""
    hW, hb = net.diag_hessian(X_sample)
    n = len(X_sample)
    return ([eta / (h * n + mu) for h in hW], [eta / (h * n + mu) for h in hb])


# ---------------------------------------------------------------------------
# Sections 7.5 and 9.2: the largest eigenvalue of the Hessian, without the Hessian
# ---------------------------------------------------------------------------

def hessian_vector(net, X, D, v, alpha=1e-4):
    """H v ~ (grad(w + alpha v) - grad(w)) / alpha   (Eq. 59): two gradients,
    no Hessian needed."""
    w = net.get()
    g0 = net.flat_gradient(X, D)
    net.set(w + alpha * v)
    g1 = net.flat_gradient(X, D)
    net.set(w)
    return (g1 - g0) / alpha


def power_method(net, X, D, iters=50, rng=None):
    """Largest eigenvalue of the Hessian by the power method, Eq. (60):
    psi <- H psi / ||psi||. Returns (lambda_max, eigenvector)."""
    rng = np.random.default_rng(rng)
    psi = rng.normal(size=net.get().size)
    lam = 0.0
    for _ in range(iters):
        psi = psi / np.linalg.norm(psi)
        Hpsi = hessian_vector(net, X, D, psi)
        lam = float(np.linalg.norm(Hpsi))
        psi = Hpsi
    return lam, psi / np.linalg.norm(psi)


def online_eigenvalue(net, X, D, presentations=300, gamma=0.03, alpha=1e-4, rng=None):
    """The on-line estimate of Eq. (64): one pattern at a time,
        psi <- (1 - gamma) psi + gamma (1/alpha)(G_p(w + alpha psi/||psi||) - G_p(w))
    and ||psi|| -> lambda_max of the AVERAGE Hessian. Returns the history of ||psi||
    (Figure 24 plots exactly this)."""
    rng = np.random.default_rng(rng)
    w = net.get()
    psi = rng.normal(size=w.size)
    history = []
    for _ in range(presentations):
        p = rng.integers(len(X))
        g0 = net.flat_gradient(X[p:p + 1], D[p:p + 1])
        net.set(w + alpha * psi / np.linalg.norm(psi))
        g1 = net.flat_gradient(X[p:p + 1], D[p:p + 1])
        net.set(w)
        psi = (1 - gamma) * psi + gamma * (g1 - g0) / alpha
        history.append(float(np.linalg.norm(psi)))
    return history


# ---------------------------------------------------------------------------
# Section 5.2: the linear network (LMS) example
# ---------------------------------------------------------------------------

def two_gaussians(n=100, rng=None, centers=((-0.4, -0.8), (0.4, 0.8)), std=None):
    """Figure 10: 100 points from two Gaussian classes centred at (-0.4,-0.8) and
    (0.4,0.8), targets -1 / +1. The paper doesn't give the spread; std is chosen
    so the input covariance has eigenvalues close to the paper's 0.84 and 0.036."""
    rng = np.random.default_rng(rng)
    std = 0.19 if std is None else std
    labels = np.repeat([0, 1], n // 2)
    X = np.array(centers)[labels] + rng.normal(0, std, (n, 2))
    return X, np.where(labels == 1, 1.0, -1.0)[:, None]


def lms_hessian(X):
    """Eq. (29): for a linear unit y = w.x + b with E = 1/(2P) sum (d - y)^2, the
    Hessian is the (uncentred) covariance of the inputs, with the bias as an
    extra always-1 input."""
    Xb = np.c_[X, np.ones(len(X))]
    return Xb.T @ Xb / len(X)


def lms_batch(X, D, lr, epochs=10, w0=None):
    """Batch LMS on a linear unit (the network of Figure 9). Returns MSE per epoch."""
    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1]) if w0 is None else np.array(w0, float)
    hist = []
    for _ in range(epochs + 1):
        err = D[:, 0] - Xb @ w
        hist.append(float(0.5 * np.mean(err ** 2)))
        w = w + lr * Xb.T @ err / len(X)
    return hist, w
