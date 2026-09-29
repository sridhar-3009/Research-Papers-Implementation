"""MLP training algorithms from Sections 4, 7 and 8 of Du et al. (2022).

Every method trains an MLP on (X, Y) and returns the training MSE after each
epoch. One "epoch" = one pass that updates the weights once (batch methods) or
once per sample (online BP).

  First order (Section 4, 7):
    "gd"    batch BP = gradient descent                         Eq. (39)
    "gdm"   BP with momentum                                    Eq. (32)
    "sgd"   online BP = stochastic gradient descent             Section 4.4
    "rprop" resilient propagation                               Section 7.5
  Second order (Section 8):
    "lm"    Levenberg-Marquardt                                 Eqs. (45)-(48)
    "bfgs"  BFGS quasi-Newton                                   Eqs. (49)-(53)
    "oss"   one-step secant (memoryless BFGS)                   Section 8.2.2
    "cgf"   conjugate gradient, Fletcher-Reeves                 Eqs. (55)-(56)
    "cgp"   conjugate gradient, Polak-Ribiere                   Eq. (57)
    "cgb"   conjugate gradient with Powell-Beale restarts       Section 8.3
    "scg"   scaled conjugate gradient (Moller 1993)             Section 8.3

MSE is reported as the mean of all squared output errors, the same measure the
paper uses in Table 1 (MATLAB's mse). The objective being minimized is E from
Eq. (26); the two differ only by a constant factor, 2/K.
"""

import numpy as np

METHODS = ["gd", "gdm", "sgd", "rprop", "lm", "bfgs", "oss", "cgf", "cgp", "cgb", "scg"]


def mse(net, X, Y):
    return float(np.mean((Y - net.predict(X)) ** 2))


def line_search(f, w, d, f0, slope, step, c=1e-4, max_iter=40):
    """Find a step a along direction d that decreases f enough (Eq. 49, inexactly).

    Backtracks (halves the step) until the Armijo condition
        f(w + a d) <= f0 + c * a * slope
    holds; if the first step already works, keeps doubling while f still falls.
    slope = g . d must be negative (d points downhill).
    Returns (a, f(w + a d)); a = 0 means no decrease was found.
    """
    a, fa = step, f(w + step * d)
    if fa <= f0 + c * a * slope:
        for _ in range(max_iter):                   # expand while it keeps improving
            f2 = f(w + 2 * a * d)
            if f2 >= fa:
                break
            a, fa = 2 * a, f2
        return a, fa
    for _ in range(max_iter):                       # shrink until it's good enough
        a /= 2
        fa = f(w + a * d)
        if fa <= f0 + c * a * slope:
            return a, fa
    return 0.0, f0


def train(net, X, Y, method="lm", epochs=1000, goal=1e-3, min_grad=1e-7, decay=0.0,
          lr=None, momentum=0.9, rng=None, val=None, patience=None, val_log=None,
          fault_rate=0.0):
    """Train `net` in place. Returns the list of training MSEs, one per epoch
    (index 0 = before training). Stops early when:
      - the MSE reaches `goal` (the paper's "performance goal", 0.001),
      - the gradient norm falls below `min_grad` (a minimum was reached),
      - the method can no longer make progress (line search fails, or the LM
        damping factor explodes).

    Generalization and fault tolerance options:
      decay       weight-decay strength, Eq. (34) (Section 5.3).
      val         (X_val, Y_val): early stopping (Section 5.2). The validation MSE
                  is checked every epoch; at the end the net gets the weights with
                  the LOWEST validation error, not the last ones. Validation MSEs
                  are appended to the list `val_log` if one is given.
      patience    with val: stop after this many epochs without improvement.
      fault_rate  online BP only: during training, each hidden node is broken
                  (output stuck at 0) with this probability, freshly for every
                  sample. This is random node-fault injection (Section 10.1).
    """
    rng = np.random.default_rng(rng)
    w = net.get_params()
    P = len(w)

    def f(v):                                   # objective E(w), Eq. (26) (+ decay)
        net.set_params(v)
        return net.loss(X, Y, decay)

    def grad(v):
        net.set_params(v)
        return net.gradient(X, Y, decay)

    history = [mse(net, X, Y)]
    best = {"val": np.inf, "w": w.copy(), "since": 0}

    def record(w_now, g_now):
        """Log this epoch; return True if training should stop."""
        net.set_params(w_now)
        history.append(mse(net, X, Y))
        if val is not None:
            v = mse(net, *val)
            if val_log is not None:
                val_log.append(v)
            if v < best["val"]:
                best.update(val=v, w=w_now.copy(), since=0)
            else:
                best["since"] += 1
                if patience is not None and best["since"] >= patience:
                    return True
        return history[-1] <= goal or np.linalg.norm(g_now) < min_grad

    def finish(w_final):
        """Leave the net with the final weights, or the best-on-validation ones."""
        net.set_params(best["w"] if val is not None else w_final)
        return history

    # State shared by several methods.
    g = grad(w)
    fw = f(w)
    velocity = np.zeros(P)                      # momentum
    step_sizes = np.full(P, 0.07)               # RProp: per-weight step, start 0.07
    g_prev = np.zeros(P)
    sigma = 0.01                                # LM damping, sigma(0) = 0.01
    H_inv = np.eye(P)                           # BFGS inverse Hessian, H^-1(0) = I
    d = -g                                      # CG / quasi-Newton search direction
    s = z = None                                # OSS: last step and gradient change
    a_prev = 0.01                               # previous line-search step
    scg = {"lam": 5e-7, "lam_bar": 0.0, "success": True, "k": 0}

    for epoch in range(1, epochs + 1):
        if method == "gd":
            # Plain batch BP: one step down the gradient, Eq. (39).
            w = w - (lr or 0.5) * g

        elif method == "gdm":
            # BP with momentum, Eq. (32): keep moving the way we were going,
            # which speeds up flat regions and damps zig-zagging.
            velocity = -(lr or 0.5) * g + momentum * velocity
            w = w + velocity

        elif method == "sgd":
            # Online BP: update after EVERY sample, in a new random order each
            # epoch (Section 4.4).
            for p in rng.permutation(len(X)):
                net.set_params(w)
                if fault_rate:                     # break random hidden nodes for this sample
                    net.masks = [(rng.random(len(m)) >= fault_rate) * 1.0 for m in net.masks]
                w = w - (lr or 0.05) * net.gradient(X[p:p + 1], Y[p:p + 1], decay)
            net.masks = [np.ones(len(m)) for m in net.masks]   # repair all nodes

        elif method == "rprop":
            # RProp: ignore the gradient's SIZE, use only its SIGN.
            # Each weight has its own step: grow it (x1.2) while the gradient
            # keeps its sign, shrink it (x0.5) when the sign flips (we jumped
            # over a minimum).
            same = g * g_prev
            step_sizes = np.where(same > 0, step_sizes * 1.2,
                                  np.where(same < 0, step_sizes * 0.5, step_sizes))
            step_sizes = np.clip(step_sizes, 1e-6, 50.0)
            w = w - step_sizes * np.sign(g)
            g_prev = g

        elif method == "lm":
            # Levenberg-Marquardt, Eqs. (45)-(48).
            # Gauss-Newton builds the Hessian from first derivatives only,
            # H ~ J^T J; LM adds sigma*I so it can always be inverted, Eq. (47).
            # Big sigma -> small gradient-descent step; small sigma -> fast
            # Gauss-Newton step. Eq. (48): shrink sigma after a success, grow it
            # (x10) after a failure and try again.
            net.set_params(w)
            e, J = net.jacobian(X, Y)
            N = len(X)
            H = J.T @ J / N + 2 * decay * np.eye(P)
            gv = J.T @ e / N + 2 * decay * w
            while True:
                step = np.linalg.solve(H + sigma * np.eye(P), -gv)
                f_new = f(w + step)
                if f_new < fw:
                    w = w + step
                    sigma = max(sigma / 10, 1e-20)
                    break
                sigma *= 10
                if sigma > 1e10:                     # can't improve any more
                    return finish(w)

        elif method == "bfgs":
            # BFGS, Eqs. (49)-(53): build up an estimate of the INVERSE Hessian
            # from how the gradient changes, then step to where the quadratic
            # model says the minimum is (plus a line search to be safe).
            d = -H_inv @ g
            if g @ d >= 0:                               # not downhill: reset to I
                H_inv, d = np.eye(P), -g
            a, f_new = line_search(f, w, d, fw, g @ d, 1.0)
            if a == 0:
                return finish(w)
            s = a * d                                     # Eq. (53)
            w = w + s
            g_new = grad(w)
            z = g_new - g                                 # Eq. (52)
            sz = s @ z
            if sz > 1e-12:                                # Eq. (51)
                Hz = H_inv @ z
                H_inv = (H_inv + (1 + z @ Hz / sz) * np.outer(s, s) / sz
                         - (np.outer(s, Hz) + np.outer(Hz, s)) / sz)
            g, fw = g_new, f_new
            if record(w, g):
                break
            continue

        elif method == "oss":
            # One-step secant = BFGS that forgets its matrix each step: plug
            # H^-1 = I into Eq. (51) and apply it to -g. Needs only vectors.
            if s is None:
                d = -g
            else:
                sz = s @ z
                if sz > 1e-12:
                    A = -(1 + z @ z / sz) * (s @ g) / sz + (z @ g) / sz
                    B = (s @ g) / sz
                    d = -g + A * s + B * z
                else:
                    d = -g
            if g @ d >= 0:
                d = -g
            a, f_new = line_search(f, w, d, fw, g @ d, 1.0)
            if a == 0:
                return finish(w)
            s = a * d
            w = w + s
            g_new = grad(w)
            z = g_new - g
            g, fw = g_new, f_new
            if record(w, g):
                break
            continue

        elif method in ("cgf", "cgp", "cgb"):
            # Conjugate gradient, Eqs. (55)-(57): each new direction mixes the
            # new gradient with the previous direction, so we don't undo the
            # progress made along earlier directions.
            slope = g @ d
            if slope >= 0:                                 # lost descent: restart
                d, slope = -g, -(g @ g)
            step0 = a_prev * 2 if epoch > 1 else 0.1
            a, f_new = line_search(f, w, d, fw, slope, step0)
            if a == 0:
                return finish(w)
            a_prev = a
            w = w + a * d
            g_new = grad(w)
            if method == "cgf":
                beta = (g_new @ g_new) / (g @ g)                        # Fletcher-Reeves
            else:
                beta = max(0.0, g_new @ (g_new - g) / (g @ g))          # Polak-Ribiere, Eq. (57)
            restart = (epoch % P == 0)                                  # every P steps
            if method == "cgb":                                         # Powell-Beale test:
                restart |= abs(g_new @ g) >= 0.2 * (g_new @ g_new)      # gradients no longer orthogonal
            d = -g_new if restart else -g_new + beta * d                # Eq. (56)
            g, fw = g_new, f_new
            if record(w, g):
                break
            continue

        elif method == "scg":
            # Scaled conjugate gradient (Moller 1993): CG with NO line search.
            # It estimates the curvature along d from two gradients, and uses an
            # LM-style damping lam (like sigma above) to keep that estimate positive.
            if scg["success"]:
                sig = 5e-5 / np.linalg.norm(d)
                curv = (grad(w + sig * d) - g) / sig            # ~ H d
                scg["delta"] = d @ curv
            delta = scg["delta"] + (scg["lam"] - scg["lam_bar"]) * (d @ d)
            if delta <= 0:                                    # make it positive definite
                scg["lam_bar"] = 2 * (scg["lam"] - delta / (d @ d))
                delta = -delta + scg["lam"] * (d @ d)
                scg["lam"] = scg["lam_bar"]
            mu = -(d @ g)
            alpha = mu / delta
            f_new = f(w + alpha * d)
            ratio = 2 * delta * (fw - f_new) / mu ** 2        # how good the step was
            if ratio >= 0:                                     # accept
                w = w + alpha * d
                g_new = grad(w)
                scg["lam_bar"], scg["success"] = 0.0, True
                scg["k"] += 1
                if scg["k"] % P == 0:
                    d = -g_new
                else:
                    beta = (g_new @ g_new - g_new @ g) / mu
                    d = -g_new + beta * d
                g, fw = g_new, f_new
                if ratio >= 0.75:
                    scg["lam"] = scg["lam"] / 4
            else:                                              # reject, retry
                scg["lam_bar"], scg["success"] = scg["lam"], False
            if ratio < 0.25:
                scg["lam"] = scg["lam"] + delta * (1 - ratio) / (d @ d)
            if record(w, g):
                break
            continue

        else:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

        # Common bookkeeping for the methods that didn't `continue` above.
        g = grad(w)
        fw = f(w)
        if record(w, g):
            break

    return finish(w)
