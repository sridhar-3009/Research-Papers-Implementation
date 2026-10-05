"""Data Distribution Shifts and Monitoring (Chip Huyen, blog post, Feb 2022) -- the post's ideas as small, measurable
experiments.

  shift types        P(X, Y) = P(Y|X) P(X) = P(X|Y) P(Y)
                       covariate shift  P(X) changes,   P(Y|X) fixed
                       label shift      P(Y) changes,   P(X|Y) fixed
                       concept drift    P(Y|X) changes, P(X) fixed
                     -> which monitor (inputs / predictions / accuracy) can see each one?
  adapting           importance weighting for covariate shift (weights P_target(x) / P_source(x) from a domain
                     classifier); re-estimating the class prior under label shift WITHOUT labels (confusion-matrix
                     method, as in black-box shift estimation)
  time scale         cumulative vs sliding-window statistics; window length = detection delay vs false alarms;
                     alert policies with a duration ("condition holds for k windows") against alert fatigue
  retraining         from scratch on all data vs fine-tune on new data vs train only from the drift point
  feedback loops     a recommender trained on its own clicks concentrates on popular items; measure aggregate
                     diversity / popularity of what it recommends; randomising part of the traffic breaks the loop
"""

import numpy as np
from scipy import stats


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2=1e-3, iters=400, lr=0.5, weights=None, w0=None):
    """(Weighted) L2 logistic regression by gradient descent; `w0` warm-starts (fine-tuning)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    sw = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    w = np.zeros(Xb.shape[1]) if w0 is None else w0.copy()
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (sw * (p - y)) / sw.sum() + l2 * np.r_[w[:-1], 0])
    return w


def predict(w, X):
    return sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w)


def log_loss(p, y, weights=None):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(l, weights=weights))

# ----------------------------------------------------------------------------------------------- 1. the three shifts

def cancer_world(n, kind="source", seed=0):
    """The post's running example: predict a disease from age (+ a biomarker).
      source          ages ~ N(50, 10); P(Y|X) = sigmoid(0.08 (age - 60) + 1.2 marker)
      covariate       older patients come in (ages ~ N(65, 8)), P(Y|X) unchanged
      label           a preventive drug lowers the disease rate for everyone: P(Y) falls, P(X|Y) unchanged
                      (implemented by resampling source data with fewer positives)
      concept         same people, but the relationship changes (the marker stops mattering, age matters more)"""
    r = np.random.default_rng(seed)
    if kind == "label":
        X, y = cancer_world(4 * n, "source", seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        k = int(0.4 * n * y.mean())                                   # positives cut to 40% of their share
        idx = np.r_[r.choice(pos, k, replace=False), r.choice(neg, n - k, replace=False)]
        return X[idx], y[idx]
    age = r.normal(65, 8, n) if kind == "covariate" else r.normal(50, 10, n)
    marker = r.normal(0, 1, n)
    if kind == "concept":
        z = 0.15 * (age - 55) + 0.0 * marker
    else:
        z = 0.08 * (age - 60) + 1.2 * marker
    y = (r.random(n) < sigmoid(z)).astype(float)
    return np.column_stack([(age - 50) / 10, marker]), y


def monitor_signals(w, Xs, ys, Xt, yt):
    """What each monitor sees: KS on each input feature (Bonferroni over 2), KS on the predicted probabilities,
    and the change in accuracy (needs labels)."""
    p_in = min(1.0, 2 * min(stats.ks_2samp(Xs[:, j], Xt[:, j]).pvalue for j in range(Xs.shape[1])))
    ps, pt = predict(w, Xs), predict(w, Xt)
    p_pred = stats.ks_2samp(ps, pt).pvalue
    acc_s, acc_t = float(((ps > 0.5) == ys).mean()), float(((pt > 0.5) == yt).mean())
    return {"inputs KS p": float(p_in), "predictions KS p": float(p_pred), "accuracy source": acc_s,
            "accuracy target": acc_t, "mean prediction": float(pt.mean()), "true rate": float(yt.mean())}


def shift_table(n=3000, seed=0):
    Xs, ys = cancer_world(20000, "source", seed)
    w = fit_logreg(Xs, ys)
    Xv, yv = cancer_world(n, "source", seed + 1)
    return {k: monitor_signals(w, Xv, yv, *cancer_world(n, k, seed + 2)) for k in ("source", "covariate", "label",
                                                                                     "concept")}

# ----------------------------------------------------------------------------------------------- 2. adapting without labels

def importance_weighting(seed=0, n=6000):
    """Covariate shift with a MIS-specified model: the true P(Y|X) is quadratic in x, the model is linear. Train on
    source (a) plainly, (b) with weights w(x) = P_t(x)/P_s(x) estimated by a domain classifier on UNLABELLED target
    inputs, (c) with the true density ratio. Evaluate on target."""
    r = np.random.default_rng(seed)
    xs = r.normal(0, 1, n)
    xt = r.normal(1.5, 0.7, n)
    f = lambda x: (r.random(len(x)) < sigmoid(1.5 * x - 0.6 * x ** 2 + 0.5)).astype(float)
    ys, yt = f(xs), f(xt)
    feat = lambda x: x[:, None]
    plain = fit_logreg(feat(xs), ys)
    dom = fit_logreg(np.column_stack([np.r_[xs, xt], np.r_[xs, xt] ** 2]), np.r_[np.zeros(n), np.ones(n)])
    pt = predict(dom, np.column_stack([xs, xs ** 2]))
    w_est = pt / (1 - pt)                                             # odds = P_t(x)/P_s(x) when classes are balanced
    w_true = stats.norm.pdf(xs, 1.5, 0.7) / stats.norm.pdf(xs, 0, 1)
    res = {}
    for name, sw in (("unweighted", None), ("weights from a domain classifier", w_est), ("true density ratio", w_true)):
        m = fit_logreg(feat(xs), ys, weights=sw)
        res[name] = {"target log-loss": log_loss(predict(m, feat(xt)), yt),
                     "target accuracy": float(((predict(m, feat(xt)) > 0.5) == yt).mean())}
    ess = lambda v: float(v.sum() ** 2 / (v ** 2).sum())
    res["effective sample size (of 6000)"] = {"estimated weights": ess(w_est), "true weights": ess(w_true)}
    return res


def label_shift_prior(seed=0, n=4000):
    """Label shift: estimate the target class prior from UNLABELLED target predictions using the source confusion
    matrix C[i, j] = P(pred = i | y = j):  mu_pred = C q  ->  q = C^-1 mu_pred. Then correct the model's
    probabilities by Bayes' rule with the new prior."""
    Xs, ys = cancer_world(20000, "source", seed)
    w = fit_logreg(Xs, ys)
    Xv, yv = cancer_world(n, "source", seed + 1)
    Xt, yt = cancer_world(n, "label", seed + 2)
    pv = (predict(w, Xv) > 0.5).astype(int)
    C = np.array([[np.mean(pv[yv == j] == i) for j in (0, 1)] for i in (0, 1)])
    mu = np.array([np.mean((predict(w, Xt) > 0.5) == i) for i in (0, 1)])
    q = np.clip(np.linalg.solve(C, mu), 1e-3, 1)
    q = q / q.sum()
    p_src = ys.mean()
    pt = predict(w, Xt)
    ratio1, ratio0 = q[1] / p_src, q[0] / (1 - p_src)
    pt_adj = pt * ratio1 / (pt * ratio1 + (1 - pt) * ratio0)
    return {"source positive rate": float(p_src), "true target positive rate": float(yt.mean()),
            "naive estimate (mean prediction)": float(pt.mean()), "confusion-matrix estimate": float(q[1]),
            "log-loss before": log_loss(pt, yt), "log-loss after prior correction": log_loss(pt_adj, yt),
            "accuracy before": float(((pt > 0.5) == yt).mean()), "accuracy after": float(((pt_adj > 0.5) == yt).mean())}

# ----------------------------------------------------------------------------------------------- 3. time scale

def stream(hours=24 * 14, base=0.20, dip_start=24 * 9, dip_hours=6, dip=0.12, seed=0, per_hour=200):
    """Hourly conversion rate with a daily cycle; an outage lowers it for a few hours."""
    r = np.random.default_rng(seed)
    t = np.arange(hours)
    rate = base + 0.04 * np.sin(2 * np.pi * t / 24)
    rate = np.where((t >= dip_start) & (t < dip_start + dip_hours), rate - dip, rate)
    return r.binomial(per_hour, rate) / per_hour, rate


def window_alerts(series, window, ref_hours=24 * 7, z=4.0, duration=1, seasonal=True, per_hour=200):
    """Compare the mean of the last `window` hours with the same hours one week earlier (seasonal) or with the mean of
    the reference week (non-seasonal). The noise scale is the binomial standard error of a window mean (times sqrt 2
    for a difference of two windows). Alert when |difference| / noise > z for `duration` consecutive hours."""
    s = np.asarray(series)
    alerts, streak = [], 0
    se = np.sqrt(s[:ref_hours].mean() * (1 - s[:ref_hours].mean()) / (per_hour * window))
    for h in range(ref_hours + window, len(s)):
        cur = s[h - window + 1:h + 1].mean()
        if seasonal:
            ref, noise = s[h - 24 * 7 - window + 1:h - 24 * 7 + 1].mean(), np.sqrt(2) * se
        else:
            ref, noise = s[:ref_hours].mean(), se
        streak = streak + 1 if abs(cur - ref) / noise > z else 0
        if streak >= duration:
            alerts.append(h)
    return alerts


def cumulative_vs_sliding(series, start=24 * 7):
    s = np.asarray(series)
    cum = np.array([s[:h + 1].mean() for h in range(len(s))])
    slide = np.array([s[max(0, h - 5):h + 1].mean() for h in range(len(s))])
    return cum, slide

# ----------------------------------------------------------------------------------------------- 4. retraining

def drifting_days(days=30, drift_day=20, per_day=1500, seed=0):
    """Daily data; on `drift_day` the concept changes (the sign of feature 1's effect flips)."""
    r = np.random.default_rng(seed)
    out = []
    for d in range(days):
        X = r.standard_normal((per_day, 3))
        b = np.array([1.0, 0.8, -0.5]) if d < drift_day else np.array([1.0, -0.8, -0.5])
        y = (r.random(per_day) < sigmoid(X @ b)).astype(float)
        out.append((X, y))
    return out


def retraining_strategies(days=30, drift_day=20, eval_days=(22, 25, 29), seed=0):
    """On each evaluation day d, build the model from data up to day d - 1:
      stale        the model trained before the drift, never updated
      scratch      retrain from scratch on ALL data so far
      fine-tune    warm-start from the stale model, a few steps on the last 2 days only
      from drift   train from scratch only on data since the drift started (needs the drift to be detected)"""
    D = drifting_days(days, drift_day, seed=seed)
    stack = lambda a, b: (np.vstack([D[i][0] for i in range(a, b)]), np.concatenate([D[i][1] for i in range(a, b)]))
    stale = fit_logreg(*stack(0, drift_day))
    res = {}
    for d in eval_days:
        Xe, ye = D[d]
        ft = fit_logreg(*stack(d - 2, d), iters=60, w0=stale)
        res[d] = {"stale": log_loss(predict(stale, Xe), ye),
                  "scratch, all data": log_loss(predict(fit_logreg(*stack(0, d)), Xe), ye),
                  "fine-tune on last 2 days": log_loss(predict(ft, Xe), ye),
                  "from the drift point": log_loss(predict(fit_logreg(*stack(drift_day, d)), Xe), ye)}
    return res

# ----------------------------------------------------------------------------------------------- 5. degenerate feedback loops

def feedback_recommender(items=200, users_per_round=500, rounds=40, explore=0.0, seed=0):
    """Each round the system shows each user the item with the highest estimated CTR (with probability `explore` a
    random item instead, the TikTok-style randomised exposure). Estimates are click counts / views with a prior that
    favours items that were popular at launch. Returns the share of distinct items recommended in the last round
    (aggregate diversity), the share of impressions on the single most-shown item, the realised CTR, and where the
    truly best item ranks by the system's own estimates."""
    r = np.random.default_rng(seed)
    quality = r.beta(2, 20, items)                                    # true CTRs, mean ~0.09
    launch_pop = r.pareto(1.5, items)
    clicks = 1 + 5 * launch_pop / launch_pop.max()
    views = clicks / 0.05
    for t in range(rounds):
        noise = r.normal(0, 0.01, (users_per_round, items))          # per-user taste noise
        est = clicks / views
        pick = (est[None, :] + noise).argmax(1)
        rand = r.random(users_per_round) < explore
        pick[rand] = r.integers(0, items, rand.sum())
        c = r.random(users_per_round) < quality[pick]
        np.add.at(views, pick, 1)
        np.add.at(clicks, pick, c)
    return {"distinct items shown (last round)": len(np.unique(pick)) / items,
            "share of impressions on the most-shown item": float(np.bincount(pick).max() / len(pick)),
            "CTR (last round)": float(c.mean()), "best possible CTR": float(quality.max()),
            "rank of true best item by estimate": int((np.argsort(-clicks / views) == quality.argmax()).nonzero()[0][0])}


REPORTED = {
    "author / date": "Chip Huyen, 7 Feb 2022 (material from her book Designing Machine Learning Systems)",
    "failures": "a Google study of 96 ML pipeline outages over 15 years: 60 of the 96 were due to causes not directly "
                "related to ML (software failures: dependencies, deployment, hardware, downtime)",
    "ML-specific failures": "production data differing from training data, edge cases, degenerate feedback loops",
    "shift types": "covariate shift (P(X) changes, P(Y|X) fixed), label shift (P(Y) changes, P(X|Y) fixed), concept "
                   "drift (P(Y|X) changes); also feature change and label schema change",
    "detection": "summary statistics; two-sample tests such as Kolmogorov-Smirnov (1-D only), MMD, learned-kernel MMD, "
                 "LSDD; alibi-detect implements many; time scale matters: cumulative vs sliding statistics, shorter "
                 "windows detect faster but alert more",
    "addressing shift": "train on massive datasets; adapt without new labels (domain adaptation); retrain -- from "
                        "scratch on old + new data or fine-tune on new data; which data to use depends on when drift "
                        "started",
    "monitoring": "operational metrics plus ML metrics on accuracy, predictions, features, raw inputs; logs, "
                  "dashboards, alerts (policy, notification channels, description); alert fatigue; observability = "
                  "inferring internal state from outputs without shipping new code",
    "feedback loops": "detect via popularity diversity / hit rate by popularity bucket; fix with randomisation "
                      "(e.g. TikTok gives new videos random initial traffic) and positional features",
}
