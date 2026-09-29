"""Single-layer learning rules from Section 3 of Du et al. (2022).

  perceptron_train  Rosenblatt's perceptron learning algorithm, Eqs. (15)-(18)
  pocket_train      Gallant's pocket algorithm with ratchet (Section 3.2)
  lms_train         Widrow-Hoff LMS / Adaline, Eq. (19) with update (18) or the
                    normalized alpha-LMS update, Eq. (21)

All three learn a weight vector w and a threshold (bias) theta for one neuron:
    net = w . x + theta
"""

import numpy as np


def _step(net):
    """Hard limiter used by the perceptron, Eq. (16): 1 if net > 0, else 0."""
    return (net > 0).astype(int)


def perceptron_train(X, y, lr=0.5, max_epochs=100, rng=None):
    """Perceptron learning algorithm, Eqs. (15)-(18).

    X: (N, J) inputs. y: (N,) targets in {0, 1}.
    Returns (w, theta, converged, epochs, errors_per_epoch).

    Patterns are shown one at a time, over and over. Learning stops after the
    first epoch with zero errors. The perceptron convergence theorem says this
    always happens for linearly separable data; for inseparable data (e.g. XOR)
    it never happens, so we stop at max_epochs.
    """
    rng = np.random.default_rng(rng)
    N, J = X.shape
    # "w_ij are initialized at random" (Section 3.2).
    w = rng.uniform(-0.5, 0.5, J)
    theta = rng.uniform(-0.5, 0.5)
    errors = []
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for t in range(N):
            net = X[t] @ w + theta          # Eq. (15)
            o = int(net > 0)                # Eq. (16)
            e = y[t] - o                    # Eq. (17): -1, 0 or +1
            if e:
                w = w + lr * X[t] * e       # Eq. (18): only changes on a mistake
                theta = theta + lr * e
                mistakes += 1
        errors.append(mistakes)
        if mistakes == 0:
            return w, theta, True, epoch, errors
    return w, theta, False, max_epochs, errors


def accuracy(w, theta, X, y):
    """Fraction of patterns a hard-limiter neuron classifies correctly."""
    return float(np.mean(_step(X @ w + theta) == y))


def pocket_train(X, y, lr=0.5, steps=2000, rng=None):
    """Pocket algorithm with ratchet (Gallant, 1990).

    Runs the perceptron rule on randomly picked patterns, but keeps a copy "in
    its pocket" of the weights with the longest run of consecutive correct
    classifications. The ratchet only replaces the pocket weights if they also
    classify more of the whole training set correctly. Works on inseparable
    data, where the plain perceptron never settles.
    Returns (w, theta, accuracy).
    """
    rng = np.random.default_rng(rng)
    N, J = X.shape
    w, theta = np.zeros(J), 0.0
    best_w, best_theta, best_run = w.copy(), theta, 0
    best_acc = accuracy(w, theta, X, y)
    run = 0
    for _ in range(steps):
        t = rng.integers(N)
        e = y[t] - int(X[t] @ w + theta > 0)
        if e == 0:
            run += 1
            if run > best_run:
                acc = accuracy(w, theta, X, y)
                if acc > best_acc:              # ratchet: never accept a worse pocket
                    best_w, best_theta, best_run, best_acc = w.copy(), theta, run, acc
        else:
            w = w + lr * X[t] * e
            theta = theta + lr * e
            run = 0
    return best_w, best_theta, best_acc


def lms_train(X, y, lr=0.01, epochs=200, normalized=False, rng=None):
    """LMS (Widrow-Hoff) training of an Adaline, Section 3.3.

    X: (N, J) inputs. y: (N,) targets, usually in {-1, +1}.
    Unlike the perceptron, the error uses the LINEAR output (Eq. 19):
        e = y - net
    so the rule minimizes the mean squared error even when the classes can't
    be separated. After training, classify with sign(net).

    normalized=False: mu-LMS, update (18).
    normalized=True:  alpha-LMS, Eq. (21): the step is divided by
                      rho = ||[x, 1]||^2, and 0 < lr < 2 guarantees convergence.
    Returns (w, theta, mse_per_epoch).
    """
    rng = np.random.default_rng(rng)
    N, J = X.shape
    w, theta = np.zeros(J), 0.0
    history = []
    for _ in range(epochs):
        for t in rng.permutation(N):
            net = X[t] @ w + theta
            e = y[t] - net                              # Eq. (19)
            step = lr / (X[t] @ X[t] + 1) if normalized else lr
            w = w + step * X[t] * e                     # Eq. (18) / (21)
            theta = theta + step * e
        history.append(float(np.mean((y - (X @ w + theta)) ** 2)))
    return w, theta, history
