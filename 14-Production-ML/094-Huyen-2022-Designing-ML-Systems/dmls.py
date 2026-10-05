"""Designing Machine Learning Systems (Chip Huyen, O'Reilly 2022) -- the book's techniques as small experiments.

The book is paid; this file is built from its public table of contents / chapter summaries and the standard
definitions of each technique, without quoting the book. Chapter 8 (distribution shifts and monitoring) is covered by
paper 092, and the production-debt material by 087-089, so the focus here is chapters 4-7 and 9.

  ch. 4 training data      reservoir sampling; stratified vs simple random sampling; weak supervision (labelling
                           functions -> majority vote / accuracy-weighted label model); class imbalance (accuracy
                           paradox, ROC-AUC vs PR-AUC, class weights); active learning (uncertainty sampling)
  ch. 5 feature eng.       data leakage (feature selection on ALL data before splitting; duplicates across splits);
                           the hashing trick and its collisions
  ch. 6 offline eval.      baselines; calibration (reliability, expected calibration error, Platt scaling);
                           behavioural tests (invariance, directional expectation)
  ch. 7 deployment         batch vs online prediction: freshness of features vs serving cost
  ch. 9 test in production A/B-test sample size; interleaving vs A/B sensitivity; Thompson-sampling bandit vs A/B
                           regret; stateless retraining vs stateful fine-tuning (compute vs accuracy)
"""

import numpy as np
from scipy import stats


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2=1e-3, iters=300, lr=0.5, weights=None, w0=None):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    sw = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    w = np.zeros(Xb.shape[1]) if w0 is None else w0.copy()
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (sw * (p - y)) / sw.sum() + l2 * np.r_[w[:-1], 0])
    return w


def predict(w, X):
    return sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w)


def log_loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def roc_auc(s, y):
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def pr_auc(s, y):
    """Average precision: mean of the precision at each positive, ranking by score."""
    order = np.argsort(-s, kind="stable")
    yy = y[order]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / max(yy.sum(), 1))

# ----------------------------------------------------------------------------------------------- ch. 4 sampling

def reservoir_sample(stream, k, rng):
    """Algorithm R: keep the first k; for the i-th item (i >= k, 0-based) replace a random slot with prob k / (i+1).
    Every item ends up in the sample with probability k / n, without knowing n in advance."""
    res = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = rng.integers(0, i + 1)
            if j < k:
                res[j] = x
    return res


def reservoir_uniformity(n=100, k=10, trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.zeros(n)
    for _ in range(trials):
        counts[reservoir_sample(range(n), k, rng)] += 1
    return counts / trials


def stratified_vs_random(n_pop=100000, sample=1000, trials=500, seed=0):
    """Estimate the overall positive rate of a population with 3 strata of very different rates (one rare stratum).
    Stratified sampling allocates the sample proportionally to stratum sizes -> lower variance of the estimate."""
    rng = np.random.default_rng(seed)
    strata = rng.choice(3, n_pop, p=[0.7, 0.25, 0.05])
    y = rng.random(n_pop) < np.array([0.02, 0.10, 0.60])[strata]
    truth = y.mean()
    est_r, est_s = [], []
    idx_by = [np.where(strata == s)[0] for s in range(3)]
    shares = np.array([len(i) for i in idx_by]) / n_pop
    for _ in range(trials):
        est_r.append(y[rng.choice(n_pop, sample, replace=False)].mean())
        est_s.append(sum(shares[s] * y[rng.choice(idx_by[s], max(1, int(round(sample * shares[s]))), replace=False)].mean()
                         for s in range(3)))
    return {"truth": float(truth), "random: std of estimate": float(np.std(est_r)),
            "stratified: std of estimate": float(np.std(est_s))}

# ----------------------------------------------------------------------------------------------- ch. 4 weak supervision

def weak_supervision(n=6000, seed=0, gold=100):
    """A text-classification-like task with 60 features (4 informative). Labelling functions (LFs) are cheap
    heuristics that vote +1 / -1 or abstain (0), each with its own coverage and accuracy. Compare classifiers trained
    on: majority vote of the LFs, an accuracy-weighted vote (accuracies estimated WITHOUT labels from agreement with the
    majority), only `gold` hand labels, and all true labels (oracle)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 60))                                  # 60 features, 4 of them matter
    y = (rng.random(n) < sigmoid(2.5 * X[:, 0] + 2.0 * X[:, 1] - 2.0 * X[:, 2] + 1.0 * X[:, 3])).astype(int)
    specs = [(0.6, 0.85), (0.5, 0.80), (0.4, 0.75), (0.7, 0.62), (0.3, 0.90), (0.5, 0.58)]   # (coverage, accuracy)
    L = np.zeros((n, len(specs)), int)
    for j, (cov, acc) in enumerate(specs):
        fire = rng.random(n) < cov
        correct = rng.random(n) < acc
        vote = np.where(correct, 2 * y - 1, 1 - 2 * y)
        L[:, j] = np.where(fire, vote, 0)
    mv = np.sign(L.sum(1))
    # estimate each LF's accuracy from agreement with the majority vote (no labels used)
    acc_hat = np.array([np.mean(L[(L[:, j] != 0) & (mv != 0), j] == mv[(L[:, j] != 0) & (mv != 0)])
                        for j in range(L.shape[1])])
    acc_hat = np.clip(acc_hat, 0.51, 0.99)
    wv = np.sign((L * np.log(acc_hat / (1 - acc_hat))).sum(1))
    tr, te = np.arange(n) < n // 2, np.arange(n) >= n // 2
    def train_eval(labels, mask):
        m = mask & (labels != -2)
        w = fit_logreg(X[m], labels[m].astype(float))
        return float(((predict(w, X[te]) > 0.5) == y[te]).mean())
    covered = tr & (mv != 0)
    res = {"LF coverage (any LF fires)": float((L != 0).any(1).mean()),
           "majority-vote label accuracy": float(((mv[mv != 0] > 0) == y[mv != 0]).mean()),
           "weighted-vote label accuracy": float(((wv[wv != 0] > 0) == y[wv != 0]).mean()),
           "estimated LF accuracies": acc_hat.round(2).tolist(), "true LF accuracies": [a for _, a in specs]}
    res["classifier on majority vote"] = train_eval(np.where(mv > 0, 1, np.where(mv < 0, 0, -2)), covered)
    res["classifier on weighted vote"] = train_eval(np.where(wv > 0, 1, np.where(wv < 0, 0, -2)), tr & (wv != 0))
    g = np.where(tr)[0][:gold]
    gm = np.zeros(n, bool)
    gm[g] = True
    res[f"classifier on {gold} hand labels"] = train_eval(y, gm)
    res["classifier on all true labels"] = train_eval(y, tr)
    return res

# ----------------------------------------------------------------------------------------------- ch. 4 class imbalance

def class_imbalance(n=40000, rate=0.01, seed=0):
    """1% positives (e.g. fraud). Report accuracy of 'always negative', ROC-AUC vs PR-AUC of a real model, and recall
    at threshold 0.5 with and without class weights (and what weighting does to calibration)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (rng.random(n) < sigmoid(X @ np.array([1.2, -0.8, 0.6, 0.0]) + np.log(rate) - 0.8)).astype(int)
    tr = np.arange(n) < n // 2
    w = fit_logreg(X[tr], y[tr])
    cw = np.where(y[tr] == 1, (1 - y[tr].mean()) / y[tr].mean(), 1.0)
    ww = fit_logreg(X[tr], y[tr], weights=cw)
    p, pw = predict(w, X[~tr]), predict(ww, X[~tr])
    yt = y[~tr]
    rec = lambda q: float(((q > 0.5) & (yt == 1)).sum() / max(yt.sum(), 1))
    prec = lambda q: float(((q > 0.5) & (yt == 1)).sum() / max((q > 0.5).sum(), 1))
    return {"positive rate": float(yt.mean()), "accuracy of 'always negative'": float(1 - yt.mean()),
            "model ROC-AUC": roc_auc(p, yt), "model PR-AUC": pr_auc(p, yt),
            "random-scorer PR-AUC (= positive rate)": float(yt.mean()),
            "recall @0.5 unweighted": rec(p), "precision @0.5 unweighted": prec(p),
            "recall @0.5 class-weighted": rec(pw), "precision @0.5 class-weighted": prec(pw),
            "mean prediction unweighted": float(p.mean()), "mean prediction class-weighted": float(pw.mean())}

# ----------------------------------------------------------------------------------------------- ch. 4 active learning

def active_learning(pool=5000, start=20, step=20, rounds=10, seed=0):
    """Label `step` more examples per round, chosen at random or by uncertainty (prediction closest to 0.5)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((pool + 5000, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(len(X)) > 1.2).astype(float)   # boundary off-centre
    Xp, yp, Xt, yt = X[:pool], y[:pool], X[pool:], y[pool:]
    out = {}
    for strat in ("random", "uncertainty"):
        r = np.random.default_rng(seed + 1)
        lab = list(r.choice(pool, start, replace=False))
        curve = []
        for _ in range(rounds):
            w = fit_logreg(Xp[lab], yp[lab], iters=500, lr=1.0)
            curve.append(float(((predict(w, Xt) > 0.5) == yt).mean()))
            rest = np.setdiff1d(np.arange(pool), lab)
            if strat == "random":
                new = r.choice(rest, step, replace=False)
            else:
                new = rest[np.argsort(np.abs(predict(w, Xp[rest]) - 0.5))[:step]]
            lab += list(new)
        out[strat] = curve
    out["labels"] = [start + step * i for i in range(rounds)]
    return out

# ----------------------------------------------------------------------------------------------- ch. 5 leakage, hashing

def leakage_feature_selection(n=200, d=5000, k=20, seed=0):
    """Labels are PURE NOISE. Select the k features most correlated with y using (a) all data -- the leak -- or
    (b) the training split only, then cross-validate a classifier. Honest accuracy is 50%."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (rng.random(n) < 0.5).astype(float)
    tr = np.arange(n) < n // 2
    def top_k(mask):
        Xc = X[mask] - X[mask].mean(0)
        yc = y[mask] - y[mask].mean()
        return np.argsort(-np.abs(Xc.T @ yc))[:k]
    res = {}
    for name, feats in (("selected on ALL data (leak)", top_k(np.ones(n, bool))), ("selected on training data", top_k(tr))):
        w = fit_logreg(X[tr][:, feats], y[tr])
        res[name] = float(((predict(w, X[~tr][:, feats]) > 0.5) == y[~tr]).mean())
    return res


def leakage_duplicates(n=3000, dup_share=0.3, seed=0):
    """A dataset where 30% of rows are near-duplicates (the same item scraped twice). A random split puts copies on
    both sides, so a memorising model (1-nearest neighbour) looks far better than it is."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 10))
    yb = (rng.random(n) < sigmoid(base[:, 0])).astype(int)          # noisy labels: honest accuracy is modest
    dup = rng.choice(n, int(dup_share * n), replace=False)
    X = np.vstack([base, base[dup] + 0.01 * rng.standard_normal((len(dup), 10))])
    y = np.r_[yb, yb[dup]]
    group = np.r_[np.arange(n), dup]
    def nn_acc(tr, te):
        d = ((X[te][:, None, :] - X[tr][None, :, :]) ** 2).sum(-1)
        return float((y[tr][d.argmin(1)] == y[te]).mean())
    r = rng.permutation(len(y))
    tr_r, te_r = r[: int(0.8 * len(y))], r[int(0.8 * len(y)):][:600]
    g_te = rng.random(n) < 0.2
    tr_g, te_g = np.where(~g_te[group])[0], np.where(g_te[group])[0][:600]
    return {"random split (copies leak)": nn_acc(tr_r, te_r), "split by item (no leak)": nn_acc(tr_g, te_g)}


def hashing_collisions(n_values=10000, buckets=(2 ** 10, 2 ** 14, 2 ** 18), seed=0):
    """The hashing trick maps an unbounded vocabulary into B buckets. Share of values that share a bucket with at
    least one other value: measured vs the formula 1 - (1 - 1/B)^(n - 1)."""
    rng = np.random.default_rng(seed)
    out = {}
    for B in buckets:
        h = rng.integers(0, B, n_values)                              # an ideal hash = uniform random bucket
        counts = np.bincount(h, minlength=B)
        out[B] = {"measured": float((counts[h] > 1).mean()), "formula": float(1 - (1 - 1 / B) ** (n_values - 1))}
    return out

# ----------------------------------------------------------------------------------------------- ch. 6 evaluation

def expected_calibration_error(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            rows.append((float(lo), float(hi), int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return float(ece), rows


def naive_bayes_gaussian(X, y):
    """Gaussian naive Bayes: assumes features independent given the class -- overconfident when they are correlated."""
    mu = [X[y == c].mean(0) for c in (0, 1)]
    sd = [X[y == c].std(0) + 1e-9 for c in (0, 1)]
    prior = y.mean()
    def proba(Z):
        l1 = stats.norm.logpdf(Z, mu[1], sd[1]).sum(1) + np.log(prior)
        l0 = stats.norm.logpdf(Z, mu[0], sd[0]).sum(1) + np.log(1 - prior)
        return sigmoid(l1 - l0)
    return proba


def calibration_demo(n=20000, seed=0):
    """Six strongly correlated copies of one signal: naive Bayes counts the same evidence 6 times -> overconfident.
    Platt scaling (a logistic regression on the model's logit, fitted on a held-out set) repairs calibration without
    changing the ranking."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    y = (rng.random(n) < sigmoid(1.5 * z)).astype(int)
    X = z[:, None] + 0.3 * rng.standard_normal((n, 6))
    a, b, c = np.arange(n) < n // 2, (np.arange(n) >= n // 2) & (np.arange(n) < 3 * n // 4), np.arange(n) >= 3 * n // 4
    nb = naive_bayes_gaussian(X[a], y[a])
    logit = lambda p: np.log(np.clip(p, 1e-12, 1) / np.clip(1 - p, 1e-12, 1))
    platt = fit_logreg(logit(nb(X[b]))[:, None], y[b].astype(float), iters=500, lr=0.1)
    p_raw = nb(X[c])
    p_cal = predict(platt, logit(p_raw)[:, None])
    e_raw, rows_raw = expected_calibration_error(p_raw, y[c])
    e_cal, rows_cal = expected_calibration_error(p_cal, y[c])
    return {"ECE raw": e_raw, "ECE after Platt": e_cal, "log-loss raw": log_loss(p_raw, y[c]),
            "log-loss after Platt": log_loss(p_cal, y[c]), "AUC raw": roc_auc(p_raw, y[c]), "AUC after Platt": roc_auc(p_cal, y[c]),
            "reliability raw": rows_raw, "reliability after Platt": rows_cal, "Platt slope": float(platt[0])}


def baselines_and_behaviour(n=20000, seed=0):
    """A loan-approval-like model trained on HISTORICAL decisions that were biased against group g = 1, with features
    (income, debt, the protected attribute g, which is also correlated with income).
    Baselines: random, majority class, a one-feature heuristic. Behavioural tests: INVARIANCE -- flipping the protected
    attribute should not change predictions; DIRECTIONAL -- raising income should not lower approval probability."""
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < 0.4).astype(float)
    income = rng.normal(0, 1, n) + 0.6 * g
    debt = rng.normal(0, 1, n)
    y = (rng.random(n) < sigmoid(1.4 * income - 1.0 * debt - 0.8 * g)).astype(int)   # historical decisions were biased
    X_with = np.column_stack([income, debt, g])
    tr = np.arange(n) < n // 2
    res = {"baseline: random": 0.5, "baseline: majority class": float(max(y[~tr].mean(), 1 - y[~tr].mean())),
           "baseline: income > 0": float(((income[~tr] > 0) == y[~tr]).mean())}
    for name, X in (("model with protected attribute", X_with), ("model without it", X_with[:, :2])):
        w = fit_logreg(X[tr], y[tr])
        p = predict(w, X[~tr])
        res[name] = float(((p > 0.5) == y[~tr]).mean())
        if X.shape[1] == 3:
            Xf = X[~tr].copy()
            Xf[:, 2] = 1 - Xf[:, 2]
            res["invariance: mean |change| when the protected attribute flips"] = float(np.abs(predict(w, Xf) - p).mean())
            res["invariance: decisions that flip"] = float(((predict(w, Xf) > 0.5) != (p > 0.5)).mean())
        Xi = X[~tr].copy()
        Xi[:, 0] += 0.5
        res[f"directional ({name}): share where +income lowers approval"] = float((predict(w, Xi) < p - 1e-12).mean())
    return res

# ----------------------------------------------------------------------------------------------- ch. 7 batch vs online

def batch_vs_online(users=2000, hours=48, seed=0, cost_per_pred=1.0):
    """A user's purchase intent changes during the day (session activity). Batch prediction scores every user once a
    night with the previous day's features; online prediction scores at request time with fresh features. Compare
    accuracy on actual requests and the number of predictions computed."""
    rng = np.random.default_rng(seed)
    base = rng.normal(-1, 1, users)
    acc_b, acc_o, n_req = [], [], 0
    nightly = base.copy()
    for h in range(hours):
        if h % 24 == 0:
            nightly = base.copy()                                     # features snapshot at midnight
        base = base + rng.normal(0, 0.35, users)                      # intent drifts hour by hour
        req = rng.random(users) < 0.1                                 # 10% of users show up each hour
        n_req += req.sum()
        y = (rng.random(req.sum()) < sigmoid(base[req])).astype(int)
        acc_b.append(((sigmoid(nightly[req]) > 0.5) == y).mean())
        acc_o.append(((sigmoid(base[req]) > 0.5) == y).mean())
    days = hours / 24
    return {"batch accuracy": float(np.mean(acc_b)), "online accuracy": float(np.mean(acc_o)),
            "batch predictions computed": int(users * days), "online predictions computed": int(n_req),
            "requests": int(n_req)}

# ----------------------------------------------------------------------------------------------- ch. 9 test in production

def ab_sample_size(p0=0.10, lift=0.01, alpha=0.05, power=0.8):
    """Users per arm to detect an absolute CTR lift with a two-sided two-proportion z-test:
    n = (z_{1-a/2} sqrt(2 pbar(1-pbar)) + z_{power} sqrt(p0(1-p0) + p1(1-p1)))^2 / lift^2."""
    p1 = p0 + lift
    pbar = (p0 + p1) / 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / lift ** 2)


def ab_vs_interleaving(users=1000, trials=40, seed=0, items=20, k=5, noise_a=0.6, noise_b=0.3):
    """Two rankers, B slightly better. A/B: half the users see A, half B; compare click rates. Team-draft
    interleaving: every user sees ONE merged list (alternately drafted from A and B), clicks are credited to the ranker
    that contributed the item; B wins a user if it gets more credited clicks. Report how often each method correctly
    declares B better at p < 0.05."""
    rng = np.random.default_rng(seed)
    wins_ab = wins_il = 0
    for _ in range(trials):
        rel = rng.random((users, items))                              # each user's true relevance per item
        sa = rel + rng.normal(0, noise_a, (users, items))             # ranker A: noisier
        sb = rel + rng.normal(0, noise_b, (users, items))             # ranker B: better
        ra, rb = np.argsort(-sa, 1), np.argsort(-sb, 1)
        pos_w = 1 / np.arange(1, k + 1)                               # attention by position
        click = lambda items_shown, u: rng.random(k) < 0.4 * pos_w * rel[u, items_shown]
        half = rng.random(users) < 0.5
        ca = [click(ra[u, :k], u).sum() for u in np.where(half)[0]]
        cb = [click(rb[u, :k], u).sum() for u in np.where(~half)[0]]
        wins_ab += stats.ttest_ind(cb, ca).pvalue < 0.05 and np.mean(cb) > np.mean(ca)
        pref = []
        for u in range(users):
            shown, team, ia, ib = [], [], 0, 0
            while len(shown) < k:                                     # team draft: random coin decides who picks first
                first_a = rng.random() < 0.5
                for who in (("a", "b") if first_a else ("b", "a")):
                    lst, idx = (ra[u], ia) if who == "a" else (rb[u], ib)
                    while lst[idx] in shown:
                        idx += 1
                    if len(shown) < k:
                        shown.append(lst[idx]); team.append(who)
                    if who == "a":
                        ia = idx
                    else:
                        ib = idx
            c = click(np.array(shown), u)
            ta, tb = sum(c[i] for i in range(k) if team[i] == "a"), sum(c[i] for i in range(k) if team[i] == "b")
            if ta != tb:
                pref.append(1 if tb > ta else 0)
        if pref:
            wins_il += stats.binomtest(sum(pref), len(pref), 0.5).pvalue < 0.05 and np.mean(pref) > 0.5
    return {"A/B: share of trials detecting B": wins_ab / trials, "interleaving: share detecting B": wins_il / trials}


def bandit_vs_ab(arms=(0.10, 0.11, 0.13), users=20000, seed=0, reps=20):
    """Thompson sampling (Beta posteriors) vs an A/B/n test that splits traffic evenly then ships the winner at the
    halfway point. Regret = clicks lost vs always showing the best arm."""
    arms = np.array(arms)
    reg_ab, reg_ts, pick_ok = [], [], []
    for s in range(reps):
        rng = np.random.default_rng(seed + s)
        clicks = np.zeros(len(arms)); views = np.zeros(len(arms))
        total = 0
        for t in range(users):
            if t < users // 2:
                a = t % len(arms)
            else:
                a = int(np.argmax(clicks / np.maximum(views, 1)))
            c = rng.random() < arms[a]
            clicks[a] += c; views[a] += 1; total += arms[a]
        reg_ab.append(arms.max() * users - total)
        al, be = np.ones(len(arms)), np.ones(len(arms))
        total = 0
        for t in range(users):
            a = int(np.argmax(rng.beta(al, be)))
            c = rng.random() < arms[a]
            al[a] += c; be[a] += 1 - c; total += arms[a]
        reg_ts.append(arms.max() * users - total)
        pick_ok.append(int(np.argmax(al / (al + be))) == int(np.argmax(arms)))
    return {"A/B/n then ship: regret (clicks lost)": float(np.mean(reg_ab)),
            "Thompson sampling: regret": float(np.mean(reg_ts)),
            "Thompson picks the best arm at the end": float(np.mean(pick_ok))}


def stateless_vs_stateful(days=30, per_day=2000, seed=0):
    """Daily updates on a slowly drifting world. Stateless: retrain from scratch every day on the last 14 days (14x
    the data each day). Stateful: fine-tune yesterday's model on today's data only. Report average next-day log-loss
    and examples processed."""
    rng = np.random.default_rng(seed)
    data = []
    for d in range(days):
        b = np.array([1.0 + 0.03 * d, -0.8, 0.5 - 0.04 * d])
        X = rng.standard_normal((per_day, 3))
        data.append((X, (rng.random(per_day) < sigmoid(X @ b)).astype(float)))
    w_state = fit_logreg(*data[0])
    ll_sl, ll_sf, ex_sl, ex_sf = [], [], 0, 0
    for d in range(1, days - 1):
        lo = max(0, d - 13)
        Xs = np.vstack([data[i][0] for i in range(lo, d + 1)])
        ys = np.concatenate([data[i][1] for i in range(lo, d + 1)])
        w_sl = fit_logreg(Xs, ys)
        ex_sl += len(ys) * 300                                       # examples x passes
        w_state = fit_logreg(*data[d], iters=50, w0=w_state)
        ex_sf += per_day * 50
        Xn, yn = data[d + 1]
        ll_sl.append(log_loss(predict(w_sl, Xn), yn))
        ll_sf.append(log_loss(predict(w_state, Xn), yn))
    return {"stateless (14-day window from scratch) log-loss": float(np.mean(ll_sl)),
            "stateful (fine-tune on today) log-loss": float(np.mean(ll_sf)),
            "stateless example-passes": int(ex_sl), "stateful example-passes": int(ex_sf)}


REPORTED = {
    "book": "Chip Huyen, Designing Machine Learning Systems, O'Reilly 2022 (paid; built from the public table of "
            "contents and chapter summaries in github.com/chiphuyen/dmls-book, no quotations)",
    "ch. 4": "sampling; labelling (natural labels, hand labels, weak supervision, semi-supervision, transfer and active "
             "learning); class imbalance (metrics, resampling, loss changes); data augmentation",
    "ch. 5": "feature engineering, data leakage (e.g. splitting time-correlated data randomly, scaling before splitting), "
             "feature importance and removal",
    "ch. 6": "model selection, ensembles, experiment tracking, distributed training, offline evaluation with baselines "
             "and sanity checks",
    "ch. 7": "batch vs online prediction, edge vs cloud, inference latency and cost",
    "ch. 8": "data distribution shifts and monitoring (covered by paper 092)",
    "ch. 9": "continual learning (stateless vs stateful updates, how often to update) and testing in production",
}
