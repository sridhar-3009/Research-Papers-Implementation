"""Hidden Technical Debt in Machine Learning Systems (Sculley et al., NeurIPS 2015) -- the parts that are NEW relative to
the 2014 workshop paper (087), turned into small working tools and measurable experiments.

  direct feedback loop   a model picks which item to show and only learns from what it showed -> a greedy
                         supervised learner can lock onto a bad item; some randomisation (epsilon-greedy, a bandit)
                         or an isolated random slice of traffic breaks the loop
  hidden feedback loop   two DISJOINT systems on one page (product picker A, review picker B) influence each other
                         through the users: improving A changes what B learns and how big B's own A/B win is
  configuration debt     configs as small diffs of a parent, visual diffs, automatic assertions (unknown / deprecated
                         / unavailable-in-serving / badly-logged-dates / mutually-exclusive features, transitive
                         closure of data dependencies), detection of unused settings
  plain-old-data smell   a raw float does not know if it is a probability or a log-odds score; typed values do
  monitoring             prediction bias SLICED by a dimension, input-data tests against a schema, action limits,
                         up-stream producer checks
  reproducibility debt   float summation order and data order change a "deterministic" training run
  glue code              only a small fraction of a real ML pipeline is ML code (Figure 1) -- measured on our own
                         tiny end-to-end pipeline
"""

import inspect

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2=1e-3, iters=300, lr=0.5):
    """Full-batch gradient descent on L2-regularised log-loss (bias column added, bias not penalised)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return w


def predict(w, X):
    return sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w)

# ----------------------------------------------------------------------------------------------- 1. direct feedback loop

def direct_feedback_loop(policy="greedy", items=10, rounds=1000, users_per_round=20, eps=0.1, holdout=0.0,
                         initial_views=30, seed=0):
    """Each round the system shows ONE item to a batch of users and records their clicks -- only for the shown item.
    It starts from a small random log (`initial_views` per item, as if from the launch period) and estimates each
    item's CTR from its own logged data.
      greedy     always show the item with the highest estimate (plain supervised learning on its own logs)
      eps        epsilon-greedy: a random item with probability eps
      ucb        upper confidence bound bandit: estimate + sqrt(2 * estimate * ln t / n)
    `holdout` > 0 sends that fraction of each batch to a uniformly random item (an isolated slice the model does not
    control). Returns the CTR achieved, the share of rounds spent on the best item, and the final estimates."""
    rng = np.random.default_rng(seed)
    true = rng.uniform(0.02, 0.12, items)
    views = np.full(items, float(initial_views))
    clicks = rng.binomial(initial_views, true).astype(float)          # the launch-period log
    total_clicks = total_views = best_rounds = 0
    for t in range(1, rounds + 1):
        est = clicks / views
        if policy == "ucb":
            choice = int(np.argmax(est + np.sqrt(2 * est * np.log(t + 1) / views)))   # variance-aware width
        elif policy == "eps" and rng.random() < eps:
            choice = int(rng.integers(items))
        else:
            choice = int(np.argmax(est))
        n_hold = int(round(holdout * users_per_round))
        n_main = users_per_round - n_hold
        c = rng.binomial(n_main, true[choice])
        clicks[choice] += c
        views[choice] += n_main
        total_clicks += c
        total_views += n_main
        if n_hold:                                                   # the isolated random slice
            r = rng.integers(items, size=n_hold)
            cr = rng.random(n_hold) < true[r]
            np.add.at(clicks, r, cr)
            np.add.at(views, r, 1)
            total_clicks += int(cr.sum())
            total_views += n_hold
        best_rounds += choice == int(np.argmax(true))
    return {"ctr": total_clicks / total_views, "best possible ctr": float(true.max()),
            "share of rounds on best item": best_rounds / rounds, "estimates": clicks / views, "true": true}

# ----------------------------------------------------------------------------------------------- 2. hidden loop: two systems

def page_clicks(rng, n, product_q, review_q):
    """A user clicks 'buy' with probability depending on the product AND the review shown next to it. Good products
    need less persuasion: the review matters less when the product is already good (the interaction term)."""
    z = -2.0 + 1.5 * product_q + 1.2 * review_q * (1.0 - 0.8 * sigmoid(3 * product_q))
    return (rng.random(n) < sigmoid(z)).astype(float)


def two_systems(product_quality=0.0, n=40000, seed=0):
    """System A (product picker) is summarised by the mean quality of the products it shows; system B (review picker)
    is untouched. We measure what B 'learns' (its fitted weight on review quality) and the lift of B's own A/B test
    (show the best review vs a random review) -- both depend on A, a team B never talks to."""
    rng = np.random.default_rng(seed)
    pq = product_quality + 0.5 * rng.standard_normal(n)
    rq = rng.standard_normal(n)                                    # random reviews (B's logging policy)
    y = page_clicks(rng, n, pq, rq)
    w = fit_logreg(rq[:, None], y)                                  # B's model of review quality -> clicks
    ctr_random = y.mean()
    y_best = page_clicks(rng, n, pq, np.full(n, 1.5))               # B's treatment arm: a top review
    return {"B's learned review weight": float(w[0]), "B's A/B lift": float(y_best.mean() - ctr_random),
            "page CTR": float(ctr_random)}

# ----------------------------------------------------------------------------------------------- 3. configuration debt

FEATURES = {
    # name: annotations a feature-management system would hold
    "query":        {"deps": ["raw_logs"], "serving": True},
    "user_country": {"deps": ["geo_db"], "serving": True},
    "ctr_7d":       {"deps": ["click_logs"], "serving": True},
    "feature_A":    {"deps": ["click_logs"], "serving": True, "bad_dates": (914, 917)},   # logged wrong 9/14-9/17
    "feature_B":    {"deps": ["new_logs"], "serving": True, "available_from": 1007},      # no data before 10/7
    "feature_D":    {"deps": ["offline_join"], "serving": False, "substitutes": ["feature_D1", "feature_D2"]},
    "feature_D1":   {"deps": ["raw_logs"], "serving": True},
    "feature_D2":   {"deps": ["geo_db"], "serving": True},
    "feature_Q":    {"deps": ["slow_service"], "serving": True, "excludes": ["feature_R"]},  # latency budget
    "feature_R":    {"deps": ["slow_service"], "serving": True},
    "feature_Z":    {"deps": ["big_lookup_table"], "serving": True, "needs_memory_gb": 32},
    "old_product_id": {"deps": ["legacy_db"], "serving": True, "deprecated": True},
    "click_logs":   {"deps": ["raw_logs"]}, "new_logs": {"deps": ["raw_logs"]}, "offline_join": {"deps": ["raw_logs", "geo_db"]},
    "big_lookup_table": {"deps": ["raw_logs"]}, "legacy_db": {"deps": []}, "geo_db": {"deps": []},
    "raw_logs": {"deps": []}, "slow_service": {"deps": ["geo_db"]},
}

KNOWN_SETTINGS = {"parent", "features", "l2", "learning_rate", "train_from", "train_to", "memory_gb", "threshold"}


def resolve(name, configs):
    """A config is a small diff of its parent: follow the parent chain, children override, and 'features+' / 'features-'
    add or remove features from the inherited list."""
    cfg = configs[name]
    base = resolve(cfg["parent"], configs) if "parent" in cfg else {}
    out = {k: v for k, v in base.items()}
    for k, v in cfg.items():
        if k == "features+":
            out["features"] = list(out.get("features", [])) + [f for f in v if f not in out.get("features", [])]
        elif k == "features-":
            out["features"] = [f for f in out.get("features", []) if f not in v]
        elif k != "parent":
            out[k] = v
    return out


def diff(a, b):
    """Visual side-by-side difference of two resolved configs."""
    lines = []
    for k in sorted(set(a) | set(b)):
        if k == "features":
            fa, fb = set(a.get(k, [])), set(b.get(k, []))
            lines += [f"+ feature {f}" for f in sorted(fb - fa)] + [f"- feature {f}" for f in sorted(fa - fb)]
        elif a.get(k) != b.get(k):
            lines.append(f"~ {k}: {a.get(k)} -> {b.get(k)}")
    return lines


def closure(names, registry=FEATURES):
    """Transitive closure of data dependencies: everything upstream of the given features."""
    seen, stack = set(), list(names)
    while stack:
        n = stack.pop()
        for d in registry.get(n, {}).get("deps", []):
            if d not in seen:
                seen.add(d)
                stack.append(d)
    return seen


def validate(cfg, registry=FEATURES):
    """Automatic assertions about a resolved config. Dates are month*100 + day (914 = 9/14)."""
    problems = []
    feats = cfg.get("features", [])
    lo, hi = cfg.get("train_from", 0), cfg.get("train_to", 9999)
    for f in feats:
        a = registry.get(f)
        if a is None:
            problems.append(f"unknown feature {f}")
            continue
        if a.get("deprecated"):
            problems.append(f"{f} is deprecated")
        if a.get("serving") is False:
            problems.append(f"{f} is not available in serving (use {' + '.join(a.get('substitutes', []))})")
        if "bad_dates" in a and lo <= a["bad_dates"][1] and hi >= a["bad_dates"][0]:
            problems.append(f"{f} was logged incorrectly {a['bad_dates']} inside the training window")
        if a.get("available_from", 0) > lo:
            problems.append(f"{f} has no data before {a['available_from']} but training starts at {lo}")
        for x in a.get("excludes", []):
            if x in feats:
                problems.append(f"{f} precludes {x} (latency)")
        if a.get("needs_memory_gb", 0) > cfg.get("memory_gb", 8):
            problems.append(f"{f} needs {a['needs_memory_gb']} GB but the job has {cfg.get('memory_gb', 8)} GB")
    if len(set(feats)) != len(feats):
        problems.append("duplicate features")
    unused = sorted(set(cfg) - KNOWN_SETTINGS)
    if unused:
        problems.append(f"unused / misspelled settings: {unused}")
    return problems


def consumers_of(source, configs, registry=FEATURES):
    """Who breaks if `source` is turned off? (Static analysis of data dependencies across all configs.)"""
    out = []
    for name in configs:
        cfg = resolve(name, configs)
        if source in closure(cfg.get("features", []), registry) | set(cfg.get("features", [])):
            out.append(name)
    return out

# ----------------------------------------------------------------------------------------------- 4. plain-old-data smell

class Probability(float):
    """A float that knows it is a probability in [0, 1]."""

    def __new__(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"probability out of range: {v}")
        return super().__new__(cls, v)

    def _check(self, other):
        if isinstance(other, LogOdds):
            raise TypeError("comparing a Probability with a LogOdds")

    def __lt__(self, o):
        self._check(o)
        return float(self) < float(o)

    def __gt__(self, o):
        self._check(o)
        return float(self) > float(o)


class LogOdds(float):
    """A float that knows it is a log-odds score (any real number)."""

    def _check(self, other):
        if isinstance(other, Probability):
            raise TypeError("comparing a LogOdds with a Probability")

    def __lt__(self, o):
        self._check(o)
        return float(self) < float(o)

    def __gt__(self, o):
        self._check(o)
        return float(self) > float(o)

    def to_probability(self):
        return Probability(float(sigmoid(float(self))))


def pod_bug(seed=0, n=10000):
    """A team switches its model output from probabilities to raw log-odds; a downstream consumer still compares it
    with its 0.8 'probability' threshold. Returns the share of decisions that silently flip."""
    rng = np.random.default_rng(seed)
    logit = rng.normal(0, 2, n)
    meant = sigmoid(logit) > 0.8
    actual = logit > 0.8
    return float((meant != actual).mean()), float(meant.mean()), float(actual.mean())

# ----------------------------------------------------------------------------------------------- 5. monitoring

def world(n=30000, slices=("US", "IN", "BR", "DE"), broken=None, seed=0):
    """Traffic from several countries; feature 0 comes from an up-stream producer. `broken` = (country, kind) makes
    that producer misbehave for one slice: 'zeros' (stops populating) or 'units' (switches to x100 units)."""
    rng = np.random.default_rng(seed)
    country = rng.integers(0, len(slices), n)
    X = rng.standard_normal((n, 3))
    y = (rng.random(n) < sigmoid(1.5 * X[:, 0] + 0.7 * X[:, 1] - 0.5)).astype(float)
    if broken:
        m = country == slices.index(broken[0])
        X[m, 0] = 0.0 if broken[1] == "zeros" else X[m, 0] * 100
    return X, y, country, list(slices)


def sliced_bias(w, X, y, country, names):
    """Prediction bias overall and per slice (mean prediction - observed rate)."""
    p = predict(w, X)
    out = {"ALL": float(p.mean() - y.mean())}
    for i, s in enumerate(names):
        m = country == i
        out[s] = float(p[m].mean() - y[m].mean())
    return out


def make_schema(X):
    """Learn a simple schema from training data: per-feature range, mean, std, missing (zero) rate."""
    return {"lo": np.percentile(X, 0.1, axis=0), "hi": np.percentile(X, 99.9, axis=0), "mean": X.mean(0),
            "std": X.std(0), "zero_rate": (X == 0).mean(0)}


def data_tests(schema, X, tol=0.05):
    """Input-data tests: out-of-range share, mean shift (in std units), zero-rate change. Returns failures."""
    fails = []
    for j in range(X.shape[1]):
        oor = float(((X[:, j] < schema["lo"][j]) | (X[:, j] > schema["hi"][j])).mean())
        shift = float(abs(X[:, j].mean() - schema["mean"][j]) / schema["std"][j])
        zr = float((X[:, j] == 0).mean() - schema["zero_rate"][j])
        if oor > 0.002 + tol / 10:
            fails.append(f"feature {j}: {oor:.1%} out of range")
        if shift > 0.1:
            fails.append(f"feature {j}: mean shifted by {shift:.2f} std")
        if zr > tol:
            fails.append(f"feature {j}: zero rate +{zr:.1%}")
    return fails


def action_limit(scores, threshold, limit):
    """An action limit for e.g. a spam marker: if more than `limit` of the batch would be acted on, stop and alert."""
    rate = float((scores > threshold).mean())
    return rate, rate > limit

# ----------------------------------------------------------------------------------------------- 6. reproducibility

def float_order_sums(seed=0, n=100000):
    """The same float32 numbers summed in three orders (as different parallel reductions would)."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) * 10 ** rng.uniform(-3, 3, n)).astype(np.float32)
    seq = np.float32(0)
    for v in x[:20000]:                                             # a sequential loop over the first 20k
        seq = np.float32(seq + v)
    a = np.sum(x[:20000], dtype=np.float32)                          # numpy's pairwise sum
    b = np.sum(np.sort(x[:20000]), dtype=np.float32)                 # a different order
    return float(seq), float(a), float(b), float(np.sum(x[:20000].astype(np.float64)))


def sgd_runs(seeds=(0, 0, 1), n=4000, epochs=3, lr=0.1):
    """Plain SGD logistic regression; the data-shuffle seed is the only thing that changes between runs."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((n, 5))
    y = (rng.random(n) < sigmoid(X @ np.array([1.0, -1.0, 0.5, 0.0, 0.3]))).astype(float)
    out = []
    for s in seeds:
        r = np.random.default_rng(s)
        w = np.zeros(5)
        for _ in range(epochs):
            for i in r.permutation(n):
                w -= lr * (sigmoid(X[i] @ w) - y[i]) * X[i]
        out.append(w)
    return out

# ----------------------------------------------------------------------------------------------- 7. glue code, measured

def pipeline_ingest(raw):
    rows = []
    for line in raw:
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return np.array(rows)


def pipeline_validate(data):
    data = data[np.isfinite(data).all(1)]
    data = data[(data[:, 3] == 0) | (data[:, 3] == 1)]
    if len(data) < 10:
        raise ValueError("too little data after validation")
    return data


def pipeline_features(data, config):
    X = data[:, :3].copy()
    if config.get("log_feature_2"):
        X[:, 2] = np.sign(X[:, 2]) * np.log1p(np.abs(X[:, 2]))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, data[:, 3], (mu, sd)


def pipeline_train(X, y, config):
    return fit_logreg(X, y, l2=config.get("l2", 1e-3))


def pipeline_threshold(w, Xv, yv, target=0.8):
    for t in np.linspace(0.05, 0.95, 91):
        pred = predict(w, Xv) > t
        if pred.any() and yv[pred].mean() >= target:
            return float(t)
    return 0.95


def pipeline_serve(w, stats, thr, line, config):
    x = np.array([float(p) for p in line.split(",")[:3]])
    if config.get("log_feature_2"):
        x[2] = np.sign(x[2]) * np.log1p(abs(x[2]))
    x = (x - stats[0]) / stats[1]
    return bool(predict(w, x[None])[0] > thr)


def pipeline_monitor(w, X, y, history):
    bias = float(predict(w, X).mean() - y.mean())
    history.append(bias)
    if len(history) > 3 and abs(bias - np.mean(history[:-1])) > 0.05:
        return f"ALERT: prediction bias moved to {bias:+.3f}"
    return "ok"


def run_pipeline(seed=0, n=3000):
    """A tiny end-to-end pipeline: raw text -> validated rows -> features -> model -> threshold -> serve -> monitor."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = (rng.random(n) < sigmoid(X @ np.array([1.5, -1.0, 0.5]))).astype(int)
    raw = [",".join(f"{v:.4f}" for v in X[i]) + f",{y[i]}" for i in range(n)] + ["garbage", "1,2,3,nan"]
    config = {"l2": 1e-3, "log_feature_2": True}
    data = pipeline_validate(pipeline_ingest(raw))
    Xf, yf, stats = pipeline_features(data, config)
    w = pipeline_train(Xf[: n // 2], yf[: n // 2], config)
    thr = pipeline_threshold(w, Xf[n // 2:], yf[n // 2:])
    decision = pipeline_serve(w, stats, thr, raw[0], config)
    status = pipeline_monitor(w, Xf, yf, [])
    return {"rows": len(data), "threshold": round(thr, 2), "first decision": decision, "monitor": status}


def code_fraction():
    """Lines of code in the pipeline that are 'machine learning' (the learner + training call) vs everything else."""
    ml = [fit_logreg, predict, sigmoid, pipeline_train]
    other = [pipeline_ingest, pipeline_validate, pipeline_features, pipeline_threshold, pipeline_serve,
             pipeline_monitor, run_pipeline]
    count = lambda fs: sum(len([ln for ln in inspect.getsource(f).splitlines() if ln.strip() and
                                not ln.strip().startswith(("#", '"""'))]) for f in fs)
    return count(ml), count(other)


REPORTED = {
    "Figure 1": "only a small fraction of real-world ML systems is composed of the ML code, as shown by the small "
                "black box in the middle; the required surrounding infrastructure is vast and complex",
    "glue code": "a mature system might end up being (at most) 5% machine learning code and (at least) 95% glue code",
    "feedback loops": "direct loops (a model influences its own future training data; bandits are the theoretically "
                      "correct solution but may not scale; mitigate with some randomization or isolating part of the "
                      "data) and hidden loops between two systems through the world (products vs related reviews; "
                      "two stock-market prediction models)",
    "smells": "plain-old-data type, multiple-language, prototype; abstraction debt (Map-Reduce is a poor abstraction "
              "for iterative ML; the parameter server is more robust)",
    "configuration principles": "small change from a previous configuration; hard to make manual errors; easy to see "
                                "the diff between two models; automatic assertions (number of features, transitive "
                                "closure of data dependencies); detect unused or redundant settings; full code review",
    "monitoring": "prediction bias (sliced), action limits, up-stream producers; automated response",
    "other debts": "data testing, reproducibility, process management, cultural debt",
    "measuring debt": "how easily can a new approach be tested at full scale? transitive closure of data dependencies? "
                      "how precisely can a change's impact be measured? does improving one model degrade others? how "
                      "quickly can new team members be brought up to speed?",
}
