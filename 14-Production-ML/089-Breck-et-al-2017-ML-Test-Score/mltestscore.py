"""The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction (Breck et al., IEEE Big Data
2017).

  the rubric   28 tests in 4 sections of 7: Data, Model, Infrastructure, Monitoring
  scoring      per test: 0 = not done, 0.5 = run MANUALLY with results documented and distributed, 1 = run
               AUTOMATICALLY on a repeated basis. Sum each section; the ML Test Score is the MINIMUM of the 4 sums
               (all four areas matter, so the weakest one sets the score). Table V: 0 research project, (0,1] not
               totally untested, (1,2] first pass at productionization, (2,3] reasonably tested, (3,5] strong,
               >5 exceptional

This file has the rubric and the scorer, plus a toy production ML system (logistic regression on a drifting world with
countries, a spend feature computed by separate training and serving code, a model registry and a server) and an
AUTOMATED implementation of most of the 28 tests against it -- so we can score the toy system, then inject the kinds
of bugs the paper describes and see which tests catch them.
"""

import copy
import time

import numpy as np

# ----------------------------------------------------------------------------------------------- the rubric

RUBRIC = {
    "Data": ["Feature expectations are captured in a schema.", "All features are beneficial.",
             "No feature's cost is too much.", "Features adhere to meta-level requirements.",
             "The data pipeline has appropriate privacy controls.", "New features can be added quickly.",
             "All input feature code is tested."],
    "Model": ["Model specs are reviewed and submitted.", "Offline and online metrics correlate.",
              "All hyperparameters have been tuned.", "The impact of model staleness is known.",
              "A simpler model is not better.", "Model quality is sufficient on important data slices.",
              "The model is tested for considerations of inclusion."],
    "Infra": ["Training is reproducible.", "Model specs are unit tested.", "The ML pipeline is Integration tested.",
              "Model quality is validated before serving.", "The model is debuggable.",
              "Models are canaried before serving.", "Serving models can be rolled back."],
    "Monitor": ["Dependency changes result in notification.", "Data invariants hold for inputs.",
                "Training and serving are not skewed.", "Models are not too stale.", "Models are numerically stable.",
                "Computing performance has not regressed.", "Prediction quality has not regressed."],
}

INTERPRETATION = [(0, "More of a research project than a productionized system."),
                  (1, "Not totally untested, but it is worth considering the possibility of serious holes in reliability."),
                  (2, "There's been first pass at basic productionization, but additional investment may be needed."),
                  (3, "Reasonably tested, but it's possible that more of those tests and procedures may be automated."),
                  (5, "Strong levels of automated testing and monitoring, appropriate for mission-critical systems."),
                  (float("inf"), "Exceptional levels of automated testing and monitoring.")]


def ml_test_score(status):
    """status: {section: [s_1 ... s_7]} with s in {'none', 'manual', 'automated'} (or 0 / 0.5 / 1).
    Returns the per-section sums, the final score (their minimum) and Table V's interpretation."""
    pts = {"none": 0.0, "manual": 0.5, "automated": 1.0}
    sums = {sec: sum(pts.get(s, s) for s in vals) for sec, vals in status.items()}
    final = min(sums.values())
    text = next(t for ub, t in INTERPRETATION if final <= ub)
    return sums, final, text

# ----------------------------------------------------------------------------------------------- a toy production system

COUNTRIES = ["US", "IN", "BR", "NG"]
FEATURES = ["spend", "visits", "tenure", "noise", "age"]          # 'age' is forbidden by policy (Data 4)


def raw_world(n=8000, t=0.0, seed=0, shares=(0.5, 0.3, 0.15, 0.05)):
    """Users at time t (weeks). Label: will the user buy? The effect of visits drifts with t (staleness), the
    smallest country NG behaves differently (a slice), and 'noise' is useless (Data 2)."""
    rng = np.random.default_rng(seed)
    country = rng.choice(len(COUNTRIES), n, p=shares)
    spend_raw = np.exp(rng.normal(3, 1, n))                        # dollars, heavy-tailed
    visits = rng.poisson(5, n).astype(float)
    tenure = rng.uniform(0, 10, n)
    noise = rng.standard_normal(n)
    age = rng.integers(18, 80, n).astype(float)
    z = (0.9 * (np.log1p(spend_raw) - 3) + (0.3 + 0.03 * t) * (visits - 5) - 0.15 * (tenure - 5)
         + np.where(country == 3, -0.8 * (np.log1p(spend_raw) - 3), 0.0) - 0.3)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return {"user_id": rng.integers(0, 10 ** 9, n), "country": country, "spend_raw": spend_raw, "visits": visits,
            "tenure": tenure, "noise": noise, "age": age, "y": y}


def feature_spend_training(spend_raw):
    """Training-time feature code: log(1 + dollars)."""
    return np.log1p(spend_raw)


def feature_spend_serving(spend_raw, bug=False):
    """Serving-time feature code, separately optimised. With bug=True it reads CENTS instead of dollars -- the kind
    of training/serving skew the paper calls out (Monitor 3)."""
    return np.log1p(spend_raw * (100.0 if bug else 1.0))


def featurize(w, cols, serving=False, serving_bug=False):
    out = []
    for c in cols:
        if c == "spend":
            out.append(feature_spend_serving(w["spend_raw"], serving_bug) if serving else feature_spend_training(w["spend_raw"]))
        else:
            out.append(w[c])
    return np.column_stack(out)


class Model:
    """Standardised logistic regression; `spec` is the model specification (features, l2, iters, lr, seed)."""

    def __init__(self, spec):
        self.spec = dict(spec)

    def fit(self, X, y, monitor=None):
        rng = np.random.default_rng(self.spec.get("seed", 0))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        self.w = rng.normal(0, 0.01, Z.shape[1])
        with np.errstate(over="ignore", invalid="ignore"):          # a diverging run should be CAUGHT, not crash
            for i in range(self.spec.get("iters", 200)):
                p = 1 / (1 + np.exp(-np.clip(Z @ self.w, -500, 500)))
                g = Z.T @ (p - y) / len(y) + self.spec.get("l2", 1e-3) * np.r_[self.w[:-1], 0]
                self.w = self.w - self.spec.get("lr", 0.5) * g
                if monitor is not None:
                    monitor(i, self.w, p, y)
        self.version_ops = {"logistic": self.spec.get("op_version", 1)}
        return self

    def logits(self, X):
        return np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))]) @ self.w

    def predict(self, X):
        return 1 / (1 + np.exp(-np.clip(self.logits(X), -30, 30)))

    def explain(self, x):
        """Infra 5: the step-by-step computation for ONE example."""
        z = (x - self.mu) / self.sd
        contrib = z * self.w[:-1]
        logit = contrib.sum() + self.w[-1]
        return {"standardised": z, "contributions": contrib, "bias": self.w[-1], "logit": logit,
                "probability": 1 / (1 + np.exp(-logit))}


def log_loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc(p, y):
    order = np.argsort(p)
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


BASE_SPEC = {"features": ["spend", "visits", "tenure", "noise"], "l2": 1e-3, "iters": 200, "lr": 0.5, "seed": 0}


def train(spec, world):
    return Model(spec).fit(featurize(world, spec["features"]), world["y"])

# ----------------------------------------------------------------------------------------------- Data tests

def make_schema(world, cols=("spend_raw", "visits", "tenure", "noise")):
    """Data 1: statistics from training data, which a human then adjusts with domain knowledge."""
    return {c: {"min": float(np.percentile(world[c], 0.1)), "max": float(np.percentile(world[c], 99.9)),
                "mean": float(world[c].mean()), "std": float(world[c].std()),
                "missing": float(np.isnan(world[c]).mean())} for c in cols}


def check_schema(schema, world, max_out=0.01, max_shift=0.25):
    fails = []
    for c, s in schema.items():
        x = world[c]
        out = float(((x < s["min"]) | (x > s["max"]) | np.isnan(x)).mean())
        shift = abs(float(np.nanmean(x)) - s["mean"]) / s["std"]
        if out > max_out:
            fails.append(f"{c}: {out:.1%} outside schema range")
        if shift > max_shift:
            fails.append(f"{c}: mean moved {shift:.2f} std")
    return fails


def feature_value(spec, train_w, val_w, min_gain=0.002):
    """Data 2: leave-one-feature-out -- validation log-loss increase when each feature is removed."""
    full = log_loss(train(spec, train_w).predict(featurize(val_w, spec["features"])), val_w["y"])
    out = {}
    for f in spec["features"]:
        s = dict(spec, features=[g for g in spec["features"] if g != f])
        out[f] = log_loss(train(s, train_w).predict(featurize(val_w, s["features"])), val_w["y"]) - full
    return out, [f for f, v in out.items() if v < min_gain]


FEATURE_COST = {"spend": {"latency_ms": 0.4, "deps": 2}, "visits": {"latency_ms": 0.1, "deps": 1},
                "tenure": {"latency_ms": 0.1, "deps": 1}, "noise": {"latency_ms": 1.5, "deps": 3},
                "age": {"latency_ms": 0.1, "deps": 1}}


def feature_cost_check(gains, max_ms_per_gain=200.0):
    """Data 3: cost (latency, upstream dependencies) per unit of predictive gain."""
    return [f for f, g in gains.items() if FEATURE_COST[f]["latency_ms"] / max(g, 1e-6) > max_ms_per_gain]


POLICY = {"forbidden": {"age"}, "deprecated": set()}


def meta_requirements(spec):
    """Data 4: programmatically enforce project rules on features."""
    return [f for f in spec["features"] if f in POLICY["forbidden"] | POLICY["deprecated"]]


def privacy_deletion_check(world, deleted_ids, trained_ids):
    """Data 5: user-requested deletions must propagate to the training data (and hence to retrained models)."""
    return sorted(set(deleted_ids) & set(trained_ids))


def unit_test_feature_code():
    """Data 7: unit tests for feature code, with known input -> output pairs."""
    cases = [(0.0, 0.0), (np.e - 1, 1.0), (99.0, np.log(100.0))]
    return all(abs(feature_spend_training(np.array([x]))[0] - want) < 1e-9 for x, want in cases)

# ----------------------------------------------------------------------------------------------- Model tests

def offline_online_correlation(spec, train_w, seed=0, levels=(0.0, 0.2, 0.4, 0.7, 1.0), draws=5):
    """Model 2: intentionally degrade the model (Gaussian noise on its weights, `draws` per level) and compare the
    offline metric (log-loss) with an 'online' one -- the purchase rate among users the model targets (its top 30%)."""
    m = train(spec, train_w)
    rng = np.random.default_rng(seed)
    online_w = raw_world(n=8000, seed=seed + 50)
    X_on = featurize(online_w, spec["features"])
    off, on = [], []
    for lv in levels:
        o1, o2 = [], []
        for _ in range(draws):
            d = copy.deepcopy(m)
            d.w = m.w + lv * rng.standard_normal(len(m.w))
            p = d.predict(X_on)
            o1.append(log_loss(p, online_w["y"]))
            o2.append(float(online_w["y"][p >= np.quantile(p, 0.7)].mean()))
        off.append(float(np.mean(o1)))
        on.append(float(np.mean(o2)))
    return off, on, float(np.corrcoef(off, on)[0, 1])


def tune_hyperparameters(spec, train_w, val_w, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Model 3: a grid search over the L2 strength."""
    res = {l2: log_loss(train(dict(spec, l2=l2), train_w).predict(featurize(val_w, spec["features"])), val_w["y"])
           for l2 in grid}
    return res, min(res, key=res.get)


def staleness_curve(spec, ages=(0, 4, 12, 26, 52), seed=0):
    """Model 4: train at week 0, evaluate at later weeks vs a model retrained at that week."""
    m = train(spec, raw_world(seed=seed))
    out = {}
    for a in ages:
        w = raw_world(t=a, seed=seed + 100 + a)
        fresh = train(spec, raw_world(t=a, seed=seed + 200 + a))
        X = featurize(w, spec["features"])
        out[a] = (log_loss(m.predict(X), w["y"]), log_loss(fresh.predict(X), w["y"]))
    return out


def baseline_comparison(spec, train_w, val_w):
    """Model 5: compare with a very simple baseline (one feature)."""
    full = log_loss(train(spec, train_w).predict(featurize(val_w, spec["features"])), val_w["y"])
    base = dict(spec, features=["visits"])
    simple = log_loss(train(base, train_w).predict(featurize(val_w, ["visits"])), val_w["y"])
    return full, simple


def slice_quality(model, world, cols, prev=None, abs_max=0.62, incr_max=0.02):
    """Model 6 (and the release test of Infra 4): per-country log-loss with an ABSOLUTE bound and an INCREMENTAL bound
    relative to the previous model."""
    X = featurize(world, cols)
    p = model.predict(X)
    pp = prev.predict(X) if prev is not None else None
    rep, fails = {}, []
    for i, c in enumerate(COUNTRIES):
        m = world["country"] == i
        rep[c] = log_loss(p[m], world["y"][m])
        if rep[c] > abs_max:
            fails.append(f"{c}: log-loss {rep[c]:.3f} > {abs_max}")
        if pp is not None:
            d = rep[c] - log_loss(pp[m], world["y"][m])
            if d > incr_max:
                fails.append(f"{c}: log-loss +{d:.3f} vs previous model")
    rep["ALL"] = log_loss(p, world["y"])
    return rep, fails


def inclusion_check(model, world, cols):
    """Model 7: does any input correlate strongly with a protected attribute (age here), and do predictions differ by
    age group?"""
    X = featurize(world, cols)
    corr = {c: float(abs(np.corrcoef(X[:, j], world["age"])[0, 1])) for j, c in enumerate(cols)}
    p = model.predict(X)
    young, old = world["age"] < 40, world["age"] >= 40
    return corr, float(p[young].mean() - world["y"][young].mean()), float(p[old].mean() - world["y"][old].mean())

# ----------------------------------------------------------------------------------------------- Infra tests

def reproducible(spec, world):
    """Infra 1: train twice on the same data -> identical weights."""
    a, b = train(spec, world), train(spec, world)
    return bool(np.array_equal(a.w, b.w))


def spec_unit_tests(spec):
    """Infra 2: random inputs; one gradient step lowers the loss; the model can overfit a tiny set; a checkpoint
    restores to the same predictions."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, len(spec["features"])))
    y = (X[:, 0] > 0).astype(float)
    m0 = Model(dict(spec, iters=0)).fit(X, y)
    m1 = Model(dict(spec, iters=1)).fit(X, y)
    one_step = log_loss(m1.predict(X), y) < log_loss(m0.predict(X), y)
    over = Model(dict(spec, iters=2000, l2=0.0, lr=2.0)).fit(X, y)
    overfit = float(((over.predict(X) > 0.5) == y).mean()) == 1.0
    ckpt = {"w": over.w.copy(), "mu": over.mu.copy(), "sd": over.sd.copy(), "spec": dict(over.spec)}
    rest = Model(ckpt["spec"])
    rest.w, rest.mu, rest.sd = ckpt["w"], ckpt["mu"], ckpt["sd"]
    restore = bool(np.allclose(rest.predict(X), over.predict(X)))
    return {"one step lowers loss": bool(one_step), "can overfit 16 examples": overfit, "checkpoint restores": restore}


class Registry:
    """Model versions, bless/veto, and rollback (Infra 4, Infra 7)."""

    def __init__(self):
        self.versions, self.serving = [], None

    def push(self, model, val_w, abs_max=0.60, rel_max=0.01):
        X = featurize(val_w, model.spec["features"])
        ll = log_loss(model.predict(X), val_w["y"])
        reasons = []
        if ll > abs_max:                                             # loose absolute threshold: slow degradation
            reasons.append(f"validation log-loss {ll:.3f} > {abs_max}")
        if self.serving is not None:                                 # tight threshold vs the previous version
            prev = self.versions[self.serving]
            prev_ll = log_loss(prev.predict(featurize(val_w, prev.spec["features"])), val_w["y"])
            if ll > prev_ll + rel_max:
                reasons.append(f"log-loss {ll:.3f} worse than serving model {prev_ll:.3f} by > {rel_max}")
        self.versions.append(model)
        if not reasons:
            self.serving = len(self.versions) - 1
        return ("BLESSED" if not reasons else "VETOED"), reasons

    def rollback(self):
        if self.serving:
            self.serving -= 1
        return self.serving


SERVER_OPS = {"logistic": {1}}                                      # the old serving binary supports op version 1


def canary(model, server_ops=SERVER_OPS, live=None, traffic=(0.01, 0.05, 0.25, 1.0), max_bias=0.05):
    """Infra 6: (1) the model must LOAD in the serving binary (op versions, Figure 2), (2) inference on production
    inputs must work, (3) traffic ramps up while prediction bias stays sane."""
    for op, v in model.version_ops.items():
        if v not in server_ops.get(op, set()):
            return False, f"serving binary cannot load op {op} v{v} (has {sorted(server_ops.get(op, []))})"
    if live is not None:
        X = featurize(live, model.spec["features"], serving=True, serving_bug=live.get("_bug", False))
        p = model.predict(X)
        if not np.all(np.isfinite(p)):
            return False, "non-finite predictions on production inputs"
        for frac in traffic:
            k = max(1, int(frac * len(p)))
            bias = float(p[:k].mean() - live["y"][:k].mean())
            if abs(bias) > max_bias:
                return False, f"halted at {frac:.0%} traffic: prediction bias {bias:+.3f}"
    return True, "ok"

# ----------------------------------------------------------------------------------------------- Monitoring tests

DEPENDENCIES = {"spend_raw": {"owner": "payments", "version": "v3"}, "visits": {"owner": "web-logs", "version": "v7"}}


def dependency_notifications(current):
    """Monitor 1: compare the dependency versions we were built against with what upstream now reports."""
    return [f"{k}: {DEPENDENCIES[k]['version']} -> {v} (owner {DEPENDENCIES[k]['owner']})"
            for k, v in current.items() if DEPENDENCIES[k]["version"] != v]


def training_serving_skew(world, cols, serving_bug=False, tol=1e-9):
    """Monitor 3: log serving examples with their ids, recompute the features with the training code, and count the
    skewed features and examples."""
    tr = featurize(world, cols)
    sv = featurize(world, cols, serving=True, serving_bug=serving_bug)
    bad = np.abs(tr - sv) > tol
    return {c: int(bad[:, j].sum()) for j, c in enumerate(cols) if bad[:, j].any()}


def staleness_alert(model_age_weeks, tolerable_weeks):
    """Monitor 4: age of the serving model vs the tolerable age found in Model 4."""
    return model_age_weeks > tolerable_weeks


def numeric_monitor(alerts, bound=50.0):
    """Monitor 5: a training hook that records the first NaN / inf and weights outside plausible bounds."""
    def hook(i, w, p, y):
        if not np.all(np.isfinite(w)) and not any(a.startswith("non-finite") for a in alerts):
            alerts.append(f"non-finite weights first seen at step {i}")
        elif np.all(np.isfinite(w)) and np.abs(w).max() > bound and not any(a.startswith("weight") for a in alerts):
            alerts.append(f"weight magnitude {np.abs(w).max():.1f} > {bound} at step {i}")
    return hook


def dead_relu_fraction(H):
    """Monitor 5 for networks: the fraction of ReLU units that are zero for every example in a batch."""
    return float((H <= 0).all(0).mean())


def perf_regression(series, dramatic=1.5, leak_limit=1.3, window=5):
    """Monitor 6: a dramatic jump vs the previous window, or a slow leak past a fixed limit relative to launch."""
    s = np.asarray(series, float)
    alerts = []
    for i in range(window, len(s)):
        if s[i] > dramatic * np.median(s[i - window:i]):
            alerts.append((i, "dramatic"))
            break
    over = np.where(s > leak_limit * s[:window].mean())[0]
    if len(over):
        alerts.append((int(over[0]), "slow leak"))
    return alerts


def calibration_by_slice(model, world, cols, bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Monitor 7: 'models should have zero bias, in aggregate and on slices (e.g. 90% of predictions of probability
    0.9 should in fact be positive)' -- mean prediction vs observed rate per probability bin."""
    p = model.predict(featurize(world, cols))
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() > 30:
            rows.append((lo, hi, int(m.sum()), float(p[m].mean()), float(world["y"][m].mean())))
    return rows

# ----------------------------------------------------------------------------------------------- run the whole suite

def run_suite(spec=BASE_SPEC, bugs=(), seed=0):
    """Run every automated test against the toy system, with optional injected bugs:
      'skew'       serving code reads cents          'schema'     upstream starts sending spend in cents
      'forbidden'  spec adds the 'age' feature        'nondeterministic' a fresh random seed every training run
      'stale'      the serving model is 40 weeks old  'newop'      the model needs op v2 the server lacks
      'nan'        learning rate 1e6                  'deletion'   a deleted user is still in the training data
      'depchange'  an upstream dependency upgraded    'slowleak'   serving latency creeps up
      'badmodel'   the candidate model is degraded
    Returns {section: [(test, passed or None if not automated here, detail)]}."""
    bugs = set(bugs)
    spec = dict(spec)
    if "forbidden" in bugs:
        spec["features"] = spec["features"] + ["age"]
    if "newop" in bugs:
        spec["op_version"] = 2
    if "nan" in bugs:
        spec["lr"] = 1e6
    tr, va = raw_world(seed=seed), raw_world(seed=seed + 1)
    live = raw_world(seed=seed + 2)
    if "schema" in bugs:
        live["spend_raw"] = live["spend_raw"] * 100
    if "skew" in bugs:
        live["_bug"] = True
    cols = spec["features"]
    R = {s: [] for s in RUBRIC}
    # Data
    schema = make_schema(tr)
    f = check_schema(schema, live)
    R["Data"].append((RUBRIC["Data"][0], not f, "; ".join(f) or "live inputs match the schema"))
    gains, useless = feature_value(spec, tr, va)
    R["Data"].append((RUBRIC["Data"][1], not useless, f"no-value features: {useless}" if useless else "all features add value"))
    costly = feature_cost_check(gains)
    R["Data"].append((RUBRIC["Data"][2], not costly, f"too costly for their gain: {costly}" if costly else "ok"))
    v = meta_requirements(spec)
    R["Data"].append((RUBRIC["Data"][3], not v, f"policy violations: {v}" if v else "ok"))
    deleted = list(tr["user_id"][:3]) if "deletion" in bugs else [12345]
    trained_ids = tr["user_id"] if "deletion" in bugs else tr["user_id"][tr["user_id"] != 12345]
    leak = privacy_deletion_check(tr, deleted, trained_ids)
    R["Data"].append((RUBRIC["Data"][4], not leak, f"{len(leak)} deleted users still in training data" if leak else "ok"))
    R["Data"].append((RUBRIC["Data"][5], None, "process metric (idea -> production time), not automatable here"))
    R["Data"].append((RUBRIC["Data"][6], unit_test_feature_code(), "known input/output pairs for log1p(spend)"))
    # Model
    R["Model"].append((RUBRIC["Model"][0], None, "code review is a process, not automatable here"))
    off, on, r = offline_online_correlation(spec, tr) if "nan" not in bugs else ([], [], float("nan"))
    R["Model"].append((RUBRIC["Model"][1], bool(r < -0.8), f"corr(offline log-loss, online buy rate) = {r:.2f}"))
    res, best = tune_hyperparameters(spec, tr, va) if "nan" not in bugs else ({}, None)
    R["Model"].append((RUBRIC["Model"][2], best is not None and best == spec["l2"] or (best is not None and
                       res[spec["l2"]] - res[best] < 0.002), f"best l2 {best}, spec l2 {spec['l2']}"))
    curve = staleness_curve(spec) if "nan" not in bugs else {}
    tolerable = max([a for a, (s, fr) in curve.items() if s - fr < 0.01], default=0)
    R["Model"].append((RUBRIC["Model"][3], bool(curve), f"tolerable staleness ~{tolerable} weeks" if curve else "n/a"))
    full, simple = baseline_comparison(spec, tr, va)
    R["Model"].append((RUBRIC["Model"][4], full < simple, f"full {full:.3f} vs one-feature baseline {simple:.3f}"))
    model = train(spec, tr)
    if "badmodel" in bugs:
        model.w = model.w + np.random.default_rng(9).normal(0, 0.4, len(model.w))
    rep, fails = slice_quality(model, va, cols)
    R["Model"].append((RUBRIC["Model"][5], not fails, "; ".join(fails) or
                       ", ".join(f"{k} {v:.3f}" for k, v in rep.items())))
    corr, bias_young, bias_old = inclusion_check(model, va, cols)
    worst = max(corr, key=corr.get)
    ok = corr[worst] < 0.3 and abs(bias_young - bias_old) < 0.03
    R["Model"].append((RUBRIC["Model"][6], ok, f"max |corr with age| {corr[worst]:.2f} ({worst}); bias young "
                       f"{bias_young:+.3f}, old {bias_old:+.3f}"))
    # Infra
    seeds = (time.time_ns() % 997, 1000 + time.time_ns() % 991) if "nondeterministic" in bugs else (spec["seed"],) * 2
    rep_ok = bool(np.array_equal(train(dict(spec, seed=int(seeds[0])), tr).w, train(dict(spec, seed=int(seeds[1])), tr).w))
    R["Infra"].append((RUBRIC["Infra"][0], rep_ok, "two trainings give identical weights" if rep_ok else "weights differ"))
    ut = spec_unit_tests(spec)
    R["Infra"].append((RUBRIC["Infra"][1], all(ut.values()), str(ut)))
    try:
        small = {k: (v[:1000] if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
        m_int = train(dict(spec, iters=50), small)
        p = m_int.predict(featurize(small, cols))
        integ = bool(np.all(np.isfinite(p))) and auc(p, small["y"]) > 0.6
    except Exception as e:                                           # noqa: BLE001
        integ = False
    R["Infra"].append((RUBRIC["Infra"][2], integ, "subset run of ingest -> features -> train -> predict"))
    reg = Registry()
    good = train(BASE_SPEC, tr)
    reg.push(good, va)
    verdict, why = reg.push(model, va)
    R["Infra"].append((RUBRIC["Infra"][3], verdict == "BLESSED", f"{verdict} {'; '.join(why)}"))
    ex = model.explain(featurize(va, cols)[0])
    R["Infra"].append((RUBRIC["Infra"][4], bool(np.isfinite(ex["logit"])), f"one example: logit {ex['logit']:.3f}"))
    ok, why = canary(model, live=live)
    R["Infra"].append((RUBRIC["Infra"][5], ok, why))
    reg.push(train(BASE_SPEC, tr), va)
    before = reg.serving
    after = reg.rollback()
    R["Infra"].append((RUBRIC["Infra"][6], after == before - 1, f"serving v{before} -> rolled back to v{after}"))
    # Monitoring
    n = dependency_notifications({"spend_raw": "v4" if "depchange" in bugs else "v3", "visits": "v7"})
    R["Monitor"].append((RUBRIC["Monitor"][0], not n, "; ".join(n) or "no upstream changes"))
    f = check_schema(schema, live)
    R["Monitor"].append((RUBRIC["Monitor"][1], not f, "; ".join(f) or "serving inputs match the schema"))
    sk = training_serving_skew(live, cols, serving_bug="skew" in bugs)
    R["Monitor"].append((RUBRIC["Monitor"][2], not sk, f"skewed features (examples): {sk}" if sk else "identical"))
    age = 40 if "stale" in bugs else 1
    R["Monitor"].append((RUBRIC["Monitor"][3], not staleness_alert(age, max(tolerable, 4)),
                         f"model age {age} weeks, tolerable {max(tolerable, 4)}"))
    alerts = []
    Model(spec).fit(featurize(tr, cols), tr["y"], monitor=numeric_monitor(alerts))
    R["Monitor"].append((RUBRIC["Monitor"][4], not alerts, "; ".join(alerts) or "finite, bounded weights"))
    lat = 10 + np.random.default_rng(seed).normal(0, 0.3, 40)
    if "slowleak" in bugs:
        lat = lat * np.linspace(1, 1.6, 40)
    pa = perf_regression(lat)
    R["Monitor"].append((RUBRIC["Monitor"][5], not pa, f"latency alerts {pa}" if pa else "latency stable"))
    Xl = featurize(live, cols, serving=True, serving_bug="skew" in bugs)
    bias = float(model.predict(Xl).mean() - live["y"].mean())
    R["Monitor"].append((RUBRIC["Monitor"][6], abs(bias) < 0.03, f"served prediction bias {bias:+.3f}"))
    return R


def score_suite(R, manual=("New features can be added quickly.", "Model specs are reviewed and submitted.")):
    """Turn a suite run into a rubric status: an automated test that exists counts as 'automated' (1 point) whether it
    passes or fails today -- the rubric scores having the test, not the system being perfect; process tests we can
    only do by hand count as 'manual' (0.5)."""
    status = {}
    for sec, rows in R.items():
        status[sec] = ["manual" if (passed is None and name in manual) else "none" if passed is None else "automated"
                       for name, passed, _ in rows]
    return ml_test_score(status)


REPORTED = {
    "tests": "28 specific tests and monitoring needs: 7 Data, 7 Model, 7 ML Infrastructure, 7 Monitoring",
    "scoring": "half a point for executing a test manually with results documented and distributed; a full point for "
               "a system that runs it automatically on a repeated basis; sum per section; final score = MINIMUM of "
               "the 4 section scores",
    "Table V": "0 research project; (0,1] not totally untested; (1,2] first pass at basic productionization; (2,3] "
               "reasonably tested; (3,5] strong levels of automated testing and monitoring; >5 exceptional",
    "survey": "36 teams at Google interviewed; none of the tests was implemented by more than 80% of teams; most tests "
              "had a nonzero score for at least half of the teams",
    "findings": "integration testing (Infra 3) had much lower adoption than most; training/serving skew (Monitor 3) is "
                "perhaps the most important and least implemented test; a hypothetical TFX system already scored as "
                "'reasonably tested'",
    "examples": "slices: 'global accuracy improved by 1% but accuracy for one country dropped by 50%'; Infra 2: a unit "
                "test that trains for a single step of gradient descent on random input data catches many library "
                "mistakes; avoid 'golden tests'",
}
