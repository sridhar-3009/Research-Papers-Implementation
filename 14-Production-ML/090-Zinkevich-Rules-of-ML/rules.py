"""Rules of Machine Learning: Best Practices for ML Engineering (Martin Zinkevich, Google) -- the rules that make a
measurable claim, each turned into a small experiment with logistic regression.

  Rule  1   don't be afraid to launch without ML: a heuristic gets you a good part of the way
  Rule 10   watch for silent failures: feature coverage dropping (90% -> 60%) with no error anywhere
  Rule 21   the number of feature weights you can learn is roughly proportional to the amount of data
  Rule 24   measure the delta between models (position-weighted symmetric difference of their rankings)
  Rule 30   importance-weight sampled data (sampled with probability 30% -> weight 10/3), don't drop it
  Rule 33   train on data up to Jan 5, test on Jan 6 onwards (random splits flatter you)
  Rule 34   for filtering, hold out a small share of traffic unfiltered to get clean labels
  Rule 36   avoid feedback loops with positional features: train WITH position, serve without it
  Rule 37   measure training/serving skew as three gaps: train->holdout, holdout->next-day, next-day->live
            (with Rule 29 log features at serving time, Rule 31 joined tables change, Rule 32 reuse code)
  Rule 40   keep ensembles simple: a higher base-model score must never lower the ensemble score
"""

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2=1e-3, iters=300, lr=0.5, weights=None):
    """Full-batch gradient descent on (weighted) L2-regularised log-loss; a bias column is added and not penalised."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    sw = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (sw * (p - y)) / sw.sum() + l2 * np.r_[w[:-1], 0])
    return w


def predict(w, X):
    return sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w)


def log_loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc(p, y):
    order = np.argsort(p, kind="stable")
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def one_hot(idx, k):
    out = np.zeros((len(idx), k))
    out[np.arange(len(idx)), idx] = 1
    return out

# ----------------------------------------------------------------------------------------------- Rule 1

def rule1_heuristic_vs_ml(users=3000, apps=40, seed=0):
    """An app store: each user installs an app with probability depending on the app's quality AND the user's taste
    (category match). Compare the install rate of the top-1 recommendation for: random apps, the heuristic 'rank by
    overall install rate', and a model with user-category features."""
    rng = np.random.default_rng(seed)
    quality = rng.normal(0, 1, apps)
    cat = rng.integers(0, 4, apps)
    taste = rng.integers(0, 4, users)
    def p_install(u, a):
        return sigmoid(-2.5 + 0.8 * quality[a] + 1.5 * (taste[u] == cat[a]))
    # logged data from random exposure
    lu, la = rng.integers(0, users, 60000), rng.integers(0, apps, 60000)
    ly = (rng.random(60000) < p_install(lu, la)).astype(float)
    rate = np.array([ly[la == a].mean() for a in range(apps)])
    X = np.hstack([one_hot(la, apps), (taste[lu] == cat[la])[:, None].astype(float)])
    w = fit_logreg(X, ly, iters=400, lr=1.0)
    u = np.arange(users)
    random_pick = rng.integers(0, apps, users)
    heuristic_pick = np.full(users, int(np.argmax(rate)))
    scores = w[:apps][None, :] + w[apps] * (taste[:, None] == cat[None, :])
    ml_pick = scores.argmax(1)
    best = np.array([np.argmax(p_install(np.full(apps, i), np.arange(apps))) for i in u])
    res = {k: float(p_install(u, pick).mean()) for k, pick in
           (("random", random_pick), ("heuristic: most-installed app", heuristic_pick), ("ML model", ml_pick),
            ("oracle", best))}
    res["heuristic share of the ML gain"] = ((res["heuristic: most-installed app"] - res["random"])
                                             / (res["ML model"] - res["random"]))
    return res

# ----------------------------------------------------------------------------------------------- Rule 10

def rule10_coverage_monitor(days=30, drop_day=18, seed=0):
    """A feature is populated for 90% of examples until an implementation change drops it to 60%. Nothing errors;
    the daily coverage monitor (alert if coverage moves > 5 points from its 7-day median) catches it."""
    rng = np.random.default_rng(seed)
    cov, loss, alerts = [], [], []
    X0 = rng.standard_normal((20000, 2))
    y0 = (rng.random(20000) < sigmoid(2 * X0[:, 0] + X0[:, 1])).astype(float)
    w = fit_logreg(X0, y0)
    for d in range(days):
        X = rng.standard_normal((5000, 2))
        y = (rng.random(5000) < sigmoid(2 * X[:, 0] + X[:, 1])).astype(float)
        rate = 0.9 if d < drop_day else 0.6
        missing = rng.random(5000) > rate
        X[missing, 0] = 0.0                                          # missing -> default value 0
        c = 1 - missing.mean()
        cov.append(float(c))
        loss.append(log_loss(predict(w, X), y))
        if d >= 7 and abs(c - np.median(cov[d - 7:d])) > 0.05:
            alerts.append(d)
    return cov, loss, alerts

# ----------------------------------------------------------------------------------------------- Rule 21

def rule21_features_vs_data(sizes=(300, 3000, 30000), seed=0, test=20000):
    """Three categorical columns with 10 values each; the label depends on PAIRS of values. Simple model: 30 one-hot
    features. Crossed model: + 300 pair crosses. Few examples -> the small model wins; many -> the crosses win."""
    rng = np.random.default_rng(seed)
    pair_w = rng.normal(0, 0.6, (3, 10, 10))
    main_w = rng.normal(0, 0.8, (3, 10))

    def gen(n, r):
        c = r.integers(0, 10, (n, 3))
        z = sum(main_w[j, c[:, j]] for j in range(3))
        z = z + sum(pair_w[k, c[:, a], c[:, b]] for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))))
        y = (r.random(n) < sigmoid(z)).astype(float)
        simple = np.hstack([one_hot(c[:, j], 10) for j in range(3)])
        cross = np.hstack([simple] + [one_hot(c[:, a] * 10 + c[:, b], 100) for a, b in ((0, 1), (0, 2), (1, 2))])
        return simple, cross, y
    Ts, Tc, Ty = gen(test, np.random.default_rng(seed + 99))
    out = {}
    for n in sizes:
        S, C, y = gen(n, np.random.default_rng(seed + n))
        ws = fit_logreg(S, y, l2=1e-3, iters=1500, lr=2.0)
        wc = fit_logreg(C, y, l2=1e-3, iters=1500, lr=2.0)
        out[n] = {"30 features": log_loss(predict(ws, Ts), Ty), "330 features": log_loss(predict(wc, Tc), Ty)}
    return out

# ----------------------------------------------------------------------------------------------- Rule 24

def rule24_delta(scores_a, scores_b, k=10):
    """Symmetric difference of the top-k lists, weighted by position (an item at rank r counts 1/(r+1)), normalised
    to [0, 1]: 0 = identical top-k, 1 = disjoint."""
    ta, tb = list(np.argsort(-scores_a)[:k]), list(np.argsort(-scores_b)[:k])
    wa = {d: 1 / (r + 1) for r, d in enumerate(ta)}
    wb = {d: 1 / (r + 1) for r, d in enumerate(tb)}
    diff = sum(wa[d] for d in ta if d not in wb) + sum(wb[d] for d in tb if d not in wa)
    return float(diff / (2 * sum(1 / (r + 1) for r in range(k))))

# ----------------------------------------------------------------------------------------------- Rule 30

def rule30_importance_weighting(n=60000, keep=0.3, seed=0):
    """Negatives are down-sampled to 30% to save storage. Train (a) on all data, (b) on the sample with NO weights,
    (c) on the sample with weight 1/0.3 = 10/3 for kept negatives. Report the mean prediction vs the true rate."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = (rng.random(n) < sigmoid(X @ np.array([1.0, -0.7, 0.4]) - 2.0)).astype(float)
    kept = (y == 1) | (rng.random(n) < keep)
    w_all = fit_logreg(X, y)
    w_drop = fit_logreg(X[kept], y[kept])
    sw = np.where(y[kept] == 1, 1.0, 1 / keep)
    w_imp = fit_logreg(X[kept], y[kept], weights=sw)
    T = rng.standard_normal((n, 3))
    Ty = (rng.random(n) < sigmoid(T @ np.array([1.0, -0.7, 0.4]) - 2.0)).astype(float)
    return {name: {"mean prediction": float(predict(w, T).mean()), "true rate": float(Ty.mean()),
                   "log-loss": log_loss(predict(w, T), Ty), "AUC": auc(predict(w, T), Ty)}
            for name, w in (("all data", w_all), ("sampled, dropped (no weights)", w_drop),
                            ("sampled, weight 10/3", w_imp))}

# ----------------------------------------------------------------------------------------------- Rule 33

def daily_world(days=20, per_day=2000, seed=0):
    """Each day has its own effect (news, holidays, outages): something a model can memorise for days it has seen
    but cannot know for tomorrow."""
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(days), per_day)
    X = rng.standard_normal((len(day), 3))
    day_effect = rng.normal(0, 0.8, days)
    y = (rng.random(len(day)) < sigmoid(1.0 * X[:, 0] - 0.5 * X[:, 1] + day_effect[day])).astype(float)
    return X, one_hot(day, days), day, y


def rule33_temporal_split(seed=0, cutoff=15, valid_days=3):
    """A model with day-identifier features. Before launch we only have days < cutoff and must ESTIMATE how the model
    will do on days >= cutoff. (a) random 80/20 split of the past; (b) train on the past minus its last `valid_days`,
    validate on those days. Each estimate is compared with the model's ACTUAL log-loss / AUC on the future days."""
    X, D, day, y = daily_world(seed=seed)
    F = np.hstack([X, D])
    past, future = day < cutoff, day >= cutoff
    rng = np.random.default_rng(seed + 1)
    rnd = rng.random(len(y)) < 0.8
    out = {}
    for name, tr, va in (("random split of the past", past & rnd, past & ~rnd),
                         ("train on earlier days, validate on the latest", day < cutoff - valid_days,
                          (day >= cutoff - valid_days) & past)):
        w = fit_logreg(F[tr], y[tr])
        out[name] = {"estimate log-loss": log_loss(predict(w, F[va]), y[va]), "estimate AUC": auc(predict(w, F[va]), y[va]),
                     "actual log-loss on future days": log_loss(predict(w, F[future]), y[future]),
                     "actual AUC on future days": auc(predict(w, F[future]), y[future])}
    return out

# ----------------------------------------------------------------------------------------------- Rule 34

def rule34_filter_holdout(n=200000, holdout=0.01, seed=0):
    """Spam filtering. Filter v1 uses sender reputation (hidden from the new content model) and blocks 75% of spam.
    Users label only messages they SEE. Train content model v2 on (a) the shown messages, (b) the 1% held-out
    unfiltered traffic, (c) all traffic (oracle). Evaluate on all traffic: calibration and AUC."""
    rng = np.random.default_rng(seed)
    content = rng.standard_normal((n, 2))
    reputation = rng.standard_normal(n)
    z = 1.2 * content[:, 0] + 0.6 * content[:, 1] + 3.0 * reputation - 1.5
    spam = (rng.random(n) < sigmoid(z)).astype(float)
    v1_score = reputation + 0.1 * rng.standard_normal(n)
    thr = np.quantile(v1_score[spam == 1], 0.25)                     # blocks 75% of spam
    blocked = v1_score > thr
    held = rng.random(n) < holdout                                   # held out: never filtered
    shown = ~blocked | held
    out = {"v1 blocks share of spam": float(blocked[spam == 1].mean()),
           "v1 blocks share of good mail": float(blocked[spam == 0].mean()),
           "blocking with 1% held out": float((blocked & ~held)[spam == 1].mean())}
    Tn = rng.standard_normal((n, 2))
    Trep = rng.standard_normal(n)
    Ty = (rng.random(n) < sigmoid(1.2 * Tn[:, 0] + 0.6 * Tn[:, 1] + 3.0 * Trep - 1.5)).astype(float)
    for name, m in (("trained on shown (filtered) traffic", shown & ~held), ("trained on 1% held-out traffic", held),
                    ("oracle: all traffic labelled", np.ones(n, bool))):
        w = fit_logreg(content[m], spam[m])
        p = predict(w, Tn)
        out[name] = {"examples": int(m.sum()), "mean prediction": float(p.mean()), "true spam rate": float(Ty.mean()),
                     "AUC": auc(p, Ty), "log-loss": log_loss(p, Ty)}
    return out

# ----------------------------------------------------------------------------------------------- Rule 36

def rule36_position(docs=50, impressions=200000, seed=0):
    """Clicks = relevance x position bias. The old ranker put already-popular docs on top. Model A learns one score
    per doc; model B also has a position feature and is served with position fixed at slot 0. Which recovers the
    true relevance ordering of the docs?"""
    rng = np.random.default_rng(seed)
    rel = rng.normal(0, 1, docs)
    pop = 0.3 * rel + rng.normal(0, 1, docs)                          # old ranking signal: popularity, weakly relevant
    pos_bias = np.array([1.5, 0.8, 0.3, 0.0, -0.4, -0.8, -1.2, -1.5, -1.8, -2.0])
    d = rng.integers(0, docs, impressions)
    rank_score = pop[d] + 0.5 * rng.standard_normal(impressions)
    pos = np.clip(np.floor((-rank_score + 2) * 2.5), 0, 9).astype(int)  # popular -> high slots
    y = (rng.random(impressions) < sigmoid(-1.0 + rel[d] + pos_bias[pos])).astype(float)
    A = one_hot(d, docs)
    B = np.hstack([A, one_hot(pos, 10)[:, 1:]])                       # slot 0 is the reference
    wa = fit_logreg(A, y, iters=600, lr=2.0)
    wb = fit_logreg(B, y, iters=600, lr=2.0)
    return {"doc-only model": spearman(wa[:docs], rel), "doc + position model (served at slot 0)":
            spearman(wb[:docs], rel), "popularity (old ranker)": spearman(pop, rel),
            "learned position effects": wb[docs:docs + 9].tolist()}

# ----------------------------------------------------------------------------------------------- Rule 37 (+29, 31, 32)

def rule37_skew(seed=0, log_at_serving=False):
    """Decompose the train->live gap. A feature 'merchant_rating' is JOINED from a table at training time (Rule 31);
    by serving time the table has been refreshed and ratings have changed. With Rule 29 (log the features used at
    serving and train on those) training and serving see the same values."""
    rng = np.random.default_rng(seed)
    merchants = 200
    rating_then = rng.normal(0, 1, merchants)                         # table when the logs were joined
    rating_now = rating_then + rng.normal(0, 1.0, merchants)          # table at serving time (refreshed)
    quality = rating_then                                             # the label truly depended on the old rating

    def day(n, s, table):
        r = np.random.default_rng(s)
        m = r.integers(0, merchants, n)
        X = np.column_stack([r.standard_normal(n), table[m]])
        z = 1.0 * X[:, 0] + 1.2 * quality[m] - 0.5
        return X, (r.random(n) < sigmoid(z)).astype(float), m
    X, y, m = day(20000, seed + 1, rating_then)
    if log_at_serving:                                                # features as the server saw them
        X = np.column_stack([X[:, 0], rating_now[m]])
    tr = np.arange(len(y)) < 16000
    w = fit_logreg(X[tr], y[tr])
    Xn, yn, mn = day(8000, seed + 2, rating_then if not log_at_serving else rating_now)
    Xl, yl, ml = day(8000, seed + 3, rating_now)                      # live: the server joins the NEW table
    ll = lambda XX, yy: log_loss(predict(w, XX), yy)
    return {"train": ll(X[tr], y[tr]), "holdout": ll(X[~tr], y[~tr]), "next day": ll(Xn, yn), "live": ll(Xl, yl)}

# ----------------------------------------------------------------------------------------------- Rule 40

def rule40_ensemble(seed=0, n=20000):
    """Two calibrated base models feed a logistic ensemble on their logits. With non-negative ensemble weights,
    raising a base model's score can never lower the ensemble's -- check by perturbing scores upward."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (rng.random(n) < sigmoid(X @ np.array([1.0, 0.8, -0.6, 0.5]))).astype(float)
    wa = fit_logreg(X[:, :2], y)
    wb = fit_logreg(X[:, 2:], y)
    la = np.hstack([X[:, :2], np.ones((n, 1))]) @ wa
    lb = np.hstack([X[:, 2:], np.ones((n, 1))]) @ wb
    S = np.column_stack([la, lb])
    we = fit_logreg(S, y)
    base = predict(we, S)
    bumped = predict(we, S + np.column_stack([np.full(n, 0.5), np.zeros(n)]))
    return {"ensemble weights": we[:2].tolist(), "non-negative": bool((we[:2] >= 0).all()),
            "share of examples whose ensemble score went DOWN when base A went up": float((bumped < base - 1e-12).mean()),
            "AUC base A": auc(predict(wa, X[:, :2]), y), "AUC base B": auc(predict(wb, X[:, 2:]), y),
            "AUC ensemble": auc(base, y)}


RULES = {
    "Before Machine Learning": {1: "Don't be afraid to launch a product without machine learning.",
                                2: "First, design and implement metrics.",
                                3: "Choose machine learning over a complex heuristic."},
    "ML Phase I: Your First Pipeline": {
        4: "Keep the first model simple and get the infrastructure right.",
        5: "Test the infrastructure independently from the machine learning.",
        6: "Be careful about dropped data when copying pipelines.",
        7: "Turn heuristics into features, or handle them externally.",
        8: "Know the freshness requirements of your system.", 9: "Detect problems before exporting models.",
        10: "Watch for silent failures.", 11: "Give feature columns owners and documentation.",
        12: "Don't overthink which objective you choose to directly optimize.",
        13: "Choose a simple, observable and attributable metric for your first objective.",
        14: "Starting with an interpretable model makes debugging easier.",
        15: "Separate Spam Filtering and Quality Ranking in a Policy Layer."},
    "ML Phase II: Feature Engineering": {
        16: "Plan to launch and iterate.",
        17: "Start with directly observed and reported features as opposed to learned features.",
        18: "Explore with features of content that generalize across contexts.",
        19: "Use very specific features when you can.",
        20: "Combine and modify existing features to create new features in human-understandable ways.",
        21: "The number of feature weights you can learn in a linear model is roughly proportional to the amount of "
            "data you have.",
        22: "Clean up features you are no longer using.", 23: "You are not a typical end user.",
        24: "Measure the delta between models.",
        25: "When choosing models, utilitarian performance trumps predictive power.",
        26: "Look for patterns in the measured errors, and create new features.",
        27: "Try to quantify observed undesirable behavior.",
        28: "Be aware that identical short-term behavior does not imply identical long-term behavior.",
        29: "The best way to make sure that you train like you serve is to save the set of features used at serving "
            "time.",
        30: "Importance-weight sampled data, don't arbitrarily drop it!",
        31: "Beware that if you join data from a table at training and serving time, the data in the table may change.",
        32: "Re-use code between your training pipeline and your serving pipeline whenever possible.",
        33: "If you produce a model based on the data until January 5th, test the model on the data from January 6th "
            "and after.",
        34: "In binary classification for filtering, make small short-term sacrifices in performance for very clean "
            "data.",
        35: "Beware of the inherent skew in ranking problems.", 36: "Avoid feedback loops with positional features.",
        37: "Measure Training/Serving Skew."},
    "ML Phase III: Slowed Growth, Optimization Refinement, and Complex Models": {
        38: "Don't waste time on new features if unaligned objectives have become the issue.",
        39: "Launch decisions are a proxy for long-term product goals.", 40: "Keep ensembles simple.",
        41: "When performance plateaus, look for qualitatively new sources of information to add rather than refining "
            "existing signals.",
        42: "Don't expect diversity, personalization, or relevance to be as correlated with popularity as you think "
            "they are.",
        43: "Your friends tend to be the same across different products. Your interests tend not to be."},
}

REPORTED = {
    "author / motto": "Martin Zinkevich (Google): do machine learning like the great engineer you are, not like the "
                      "great machine learning expert you aren't",
    "Rule 1": "if ML would give a 100% boost, a heuristic gets you 50% of the way there",
    "Rule 10": "a table stale for 6 months -- refreshing it alone gave +2% install rate on Play; a feature column "
               "populated in 90% of examples can suddenly drop to 60%",
    "Rule 21": "~1000 examples -> a dozen features; ~10 million examples -> ~a hundred thousand features; billions "
               "of examples -> ~10 million features (with feature selection and regularization)",
    "Rule 29": "YouTube home page switched to logging features at serving time: significant quality improvements and "
               "less code complexity",
    "Rule 30": "sample X with 30% probability -> give it weight 10/3",
    "Rule 34": "a filter blocking 75% of negatives: hold out 1% of traffic unfiltered -> still blocks at least 74%; "
               "at 95%+ blocking use an even smaller sample (0.1% or 0.001%)",
    "Rule 37": "skew = training vs holdout, holdout vs next-day, next-day vs live",
}
