"""Machine Learning: The High-Interest Credit Card of Technical Debt (Sculley et al., 2014) -- the paper's warnings
turned into small measurable experiments with logistic-regression models.

  entanglement (CACE)   'Changing Anything Changes Everything': shift one input, add or drop a feature, change a
                        hyper-parameter -- and the weights of OTHER features and predictions on many slices move
  hidden feedback loop  a CTR model uses x_week = clicks in the past week; a better model raises clicks, which shifts
                        x_week a week later, which changes the model ...
  underutilized deps    legacy / bundled / epsilon features: little value, but the model leans on them anyway. The
                        paper's example: old and new product numbers both kept as features; a year later someone stops
                        populating the old numbers -> "This will not be a good day"
  correction cascade    model a' = a + correction for a slightly different problem; improving a can make a' WORSE
  fixed thresholds      a hand-picked decision threshold stops meaning the same precision after retraining
  correlations break    two features always co-occur, only one is causal; when the world decouples them, a model that
                        split the credit between them goes wrong

Mitigations from the paper that we also implement: leave-one-feature-out ablation to find underutilized features,
learning thresholds on held-out data, and prediction-bias monitoring.
"""

import numpy as np

# ----------------------------------------------------------------------------------------------- a small learner

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2=1e-3, iters=400, lr=0.5):
    """Plain full-batch gradient descent on L2-regularised logistic loss (with a bias column added)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return w


def predict(w, X):
    return sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w)


def log_loss(w, X, y):
    p = np.clip(predict(w, X), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def accuracy(w, X, y, thr=0.5):
    return float(((predict(w, X) > thr) == y).mean())

# ----------------------------------------------------------------------------------------------- 1. CACE

def cace_data(n=20000, seed=0):
    """Four correlated features feeding one label."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    X = np.column_stack([z[:, 0], z[:, 0] * 0.7 + 0.7 * rng.standard_normal(n), z[:, 1], z[:, 1] * 0.5 + rng.standard_normal(n)])
    y = (rng.random(n) < sigmoid(1.2 * z[:, 0] - 0.8 * z[:, 1] + 0.3 * X[:, 3])).astype(float)
    return X, y


def cace_experiment(seed=0):
    """Train a base model, then (a) drop feature 0, (b) add a new feature, (c) change the L2 strength, (d) shift the
    distribution of feature 0 in training -- report how much the OTHER features' weights and the predictions move."""
    X, y = cace_data(seed=seed)
    base = fit_logreg(X, y)
    p0 = predict(base, X)
    rng = np.random.default_rng(seed + 1)
    out = {}
    w = fit_logreg(X[:, 1:], y)                                      # (a) remove x0
    out["remove feature x0"] = (np.abs(w[:3] - base[1:4]).max(), np.abs(predict(w, X[:, 1:]) - p0).mean())
    extra = (0.9 * X[:, 0] + 0.4 * rng.standard_normal(len(y)))[:, None]      # an informative, correlated feature
    w = fit_logreg(np.hstack([X, extra]), y)                         # (b) add a feature
    out["add a feature x4"] = (np.abs(w[1:4] - base[1:4]).max(), np.abs(predict(w, np.hstack([X, extra])) - p0).mean())
    w = fit_logreg(X, y, l2=0.05)                                    # (c) change a hyper-parameter
    out["change L2 from 0.001 to 0.05"] = (np.abs(w[1:4] - base[1:4]).max(), np.abs(predict(w, X) - p0).mean())
    Xs = X.copy()
    Xs[:, 0] = np.where(rng.random(len(y)) < 0.3, 0.0, Xs[:, 0])    # (d) 30% of x0 values go missing (-> 0)
    w = fit_logreg(Xs, y)
    out["30% of x0 now missing"] = (np.abs(w[1:4] - base[1:4]).max(), np.abs(predict(w, X) - p0).mean())
    return base, out

# ----------------------------------------------------------------------------------------------- 2. legacy feature

def product_number_world(n=20000, n_products=100, seed=0, sample_seed=None, old_populated=True):
    """Each product has a quality q; the model sees it through two one-hot 'product number' schemes: the OLD numbering
    (only for old products) and the NEW numbering (for all products). After the cleanup the old column is all zeros.
    `seed` fixes the products (qualities, which are old); `sample_seed` draws the traffic (train vs test)."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n_products)
    old = rng.random(n_products) < 0.7                                # 70% of products existed before the merger
    rng = np.random.default_rng(seed if sample_seed is None else sample_seed + 1000)
    pid = rng.integers(0, n_products, n)
    new_oh = np.zeros((n, n_products))
    new_oh[np.arange(n), pid] = 1
    old_oh = np.zeros((n, n_products))
    if old_populated:
        rows = np.where(old[pid])[0]
        old_oh[rows, pid[rows]] = 1
    y = (rng.random(n) < sigmoid(2 * q[pid])).astype(float)
    return np.hstack([old_oh, new_oh]), y, old[pid]


def ablation(X, y, groups, Xv, yv, **fit_kw):
    """Leave-one-group-out: retrain without each feature group and report the validation log-loss increase."""
    full = log_loss(fit_logreg(X, y, **fit_kw), Xv, yv)
    out = {}
    for name, cols in groups.items():
        keep = [c for c in range(X.shape[1]) if c not in set(cols)]
        out[name] = log_loss(fit_logreg(X[:, keep], y, **fit_kw), Xv[:, keep], yv) - full
    return full, out

# ----------------------------------------------------------------------------------------------- 3. hidden feedback loop

def feedback_loop(weeks=8, users=4000, items=50, improve_week=3, quality_boost=0.6, seed=0):
    """Weekly CTR system with a hidden loop. The model predicts P(click) from [affinity estimate of the shown headline,
    x_week = the user's clicks last week]. At `improve_week` the recommender improves (better affinity estimates), so
    users click more -- and x_week rises only the FOLLOWING week. Each week we first score last week's model on this
    week's traffic (prediction bias = mean predicted - observed CTR), then retrain. Returns CTR, prediction bias and the
    weight on x_week per week."""
    rng = np.random.default_rng(seed)
    aff = rng.standard_normal((users, items))
    habit = rng.uniform(-1, 1, users)                                 # some users click more than others
    x_week = np.zeros(users)
    model = None
    ctrs, bias, w_week = [], [], []
    for wk in range(weeks):
        noise = 1.0 - quality_boost if wk >= improve_week else 1.0
        est = aff + noise * rng.standard_normal(aff.shape)
        shown = est.argmax(1)
        f_aff = est[np.arange(users), shown]
        clicks = (rng.random(users) < sigmoid(aff[np.arange(users), shown] - 1.5 + habit)).astype(float)
        X = np.column_stack([f_aff, x_week])
        bias.append(float(predict(model, X).mean() - clicks.mean()) if model is not None else float("nan"))
        model = fit_logreg(X, clicks, iters=300)
        ctrs.append(float(clicks.mean()))
        w_week.append(float(model[1]))
        x_week = clicks.copy()                                        # next week's feature: did they click this week
    return ctrs, bias, w_week


# ----------------------------------------------------------------------------------------------- 4. correction cascade

def correction_cascade(seed=0):
    """Problem A has a model a; problem A' (slightly different labels) is solved by learning a correction on top of a's
    score. Then a is 'improved' (retrained with a better feature). The correction is not retrained -- A' degrades."""
    rng = np.random.default_rng(seed)
    n = 20000
    X = rng.standard_normal((n, 3))
    yA = (rng.random(n) < sigmoid(1.5 * X[:, 0] + 1.0 * X[:, 1])).astype(float)
    yA2 = (rng.random(n) < sigmoid(1.5 * X[:, 0] + 1.0 * X[:, 1] + 0.8 * X[:, 2])).astype(float)
    a_old = fit_logreg(X[:, :1], yA)                                  # v1 only sees x0
    score_old = np.log(predict(a_old, X[:, :1]) / (1 - predict(a_old, X[:, :1])))
    corr = fit_logreg(np.column_stack([score_old, X[:, 1:]]), yA2)   # a' = correction on a's logit
    a_new = fit_logreg(X[:, :2], yA)                                  # v2 of a: better (uses x1 too)
    score_new = np.log(predict(a_new, X[:, :2]) / (1 - predict(a_new, X[:, :2])))
    return {"A with a v1": log_loss(a_old, X[:, :1], yA), "A with a v2": log_loss(a_new, X[:, :2], yA),
            "A' with a v1 + correction": log_loss(corr, np.column_stack([score_old, X[:, 1:]]), yA2),
            "A' with a v2 + OLD correction": log_loss(corr, np.column_stack([score_new, X[:, 1:]]), yA2)}

# ----------------------------------------------------------------------------------------------- 5. thresholds, 6. correlations

def precision_at(w, X, y, thr):
    pred = predict(w, X) > thr
    return float(y[pred].mean()) if pred.any() else float("nan")


def threshold_drift(seed=0):
    """Pick a threshold for 90% precision on model v1; retrain (v2, stronger regularisation, rebalanced data) and keep the
    fixed threshold vs re-learn it on held-out data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((30000, 5))
    y = (rng.random(30000) < sigmoid(X @ np.array([1.0, -0.8, 0.6, 0.3, 0.0]) - 1.0)).astype(float)
    tr, va = slice(0, 20000), slice(20000, None)

    def learn_thr(w, target=0.9):
        for t in np.linspace(0.05, 0.99, 200):
            if precision_at(w, X[va], y[va], t) >= target:
                return float(t)
        return 0.99
    v1 = fit_logreg(X[tr], y[tr])
    t1 = learn_thr(v1)
    idx = np.where(y[tr] == 1)[0]                                     # v2: positives up-sampled 3x, more L2
    Xu, yu = np.vstack([X[tr], X[tr][idx], X[tr][idx]]), np.concatenate([y[tr], y[tr][idx], y[tr][idx]])
    v2 = fit_logreg(Xu, yu, l2=0.02)
    return {"v1 threshold": t1, "v1 precision": precision_at(v1, X[va], y[va], t1),
            "v2 with v1's fixed threshold": precision_at(v2, X[va], y[va], t1),
            "v2 re-learned threshold": learn_thr(v2), "v2 precision (re-learned)": precision_at(v2, X[va], y[va], learn_thr(v2))}


def correlation_break(seed=0):
    """x_cause drives the label; x_proxy always equalled x_cause in training. In production the proxy decouples."""
    rng = np.random.default_rng(seed)
    n = 20000
    c = rng.standard_normal(n)
    proxy = c + 0.05 * rng.standard_normal(n)
    y = (rng.random(n) < sigmoid(2 * c)).astype(float)
    w = fit_logreg(np.column_stack([c, proxy]), y, l2=1e-4)
    c2 = rng.standard_normal(n)
    proxy2 = rng.standard_normal(n)                                  # the world stops making them co-occur
    y2 = (rng.random(n) < sigmoid(2 * c2)).astype(float)
    w_causal = fit_logreg(c[:, None], y)
    return {"weights (cause, proxy)": w[:2].tolist(),
            "accuracy before": accuracy(w, np.column_stack([c, proxy]), y),
            "accuracy after decoupling": accuracy(w, np.column_stack([c2, proxy2]), y2),
            "causal-only model after": accuracy(w_causal, c2[:, None], y2)}


def prediction_bias(w, X, y):
    """Monitoring (Section 5.3): mean predicted probability minus observed rate -- should be ~0 for a calibrated model."""
    return float(predict(w, X).mean() - y.mean())


REPORTED = {
    "debts named": "boundary erosion (entanglement / CACE, hidden feedback loops, undeclared consumers), data "
                   "dependencies (unstable, underutilized: legacy / bundled / epsilon features; static analysis), "
                   "correction cascades, system-level spaghetti (glue code, pipeline jungles, dead experimental "
                   "codepaths, configuration debt), changes in the external world (fixed thresholds, correlations that "
                   "stop correlating, monitoring and testing)",
    "glue code": "a mature system might end up being (at most) 5% machine learning code and (at least) 95% glue code",
    "mitigations": "isolate models / ensembles, slice-level metrics, versioned copies of unstable signals, regular "
                   "leave-one-feature-out evaluation, automated feature-dependency tooling, thresholds learned on "
                   "held-out data, monitoring (prediction bias, action limits)",
    "message": "shipping v1.0 of an ML system is easy; making subsequent improvements is unexpectedly difficult",
}
