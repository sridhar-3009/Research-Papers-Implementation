"""Model Cards for Model Reporting (Mitchell, Wu, Zaldivar, Barnes, Vasserman, Hutchinson, Spitzer, Raji, Gebru;
FAT* 2019).

  a model card    a short document shipped with a trained model, with 9 sections (Figure 1): Model Details, Intended
                  Use, Factors, Metrics, Evaluation Data, Training Data, Quantitative Analyses, Ethical
                  Considerations, Caveats and Recommendations
  the core idea   DISAGGREGATED evaluation: report metrics per group ("unitary": men, women, young, old) and per
                  INTERSECTION of groups (older men), with confidence intervals -- a single aggregate number hides
                  who the model fails
  metrics         classification: false positive / false negative / false discovery / false omission rate (FPR,
                  FNR, FDR, FOR) at a stated threshold; equal FNR across groups = equality of opportunity, equal FPR
                  and FNR = equalized odds. Scores: compare distributions, e.g. per-subgroup AUC / pinned AUC

This file implements the card itself (a dataclass rendered to Markdown), the disaggregated evaluation with bootstrap
95% confidence intervals and a threshold sweep, and the paper's two worked examples on synthetic stand-ins: a smiling
classifier evaluated by gender x age (Figure 2), and a toxicity scorer evaluated per identity term, version 1 vs a
bias-mitigated version (Figure 3).
"""

from dataclasses import dataclass, field

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(X, y, l2=1e-3, iters=500, lr=0.5):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return w


def predict(w, X):
    return sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w)

# ----------------------------------------------------------------------------------------------- the card

SECTIONS = ["Model Details", "Intended Use", "Factors", "Metrics", "Evaluation Data", "Training Data",
            "Quantitative Analyses", "Ethical Considerations", "Caveats and Recommendations"]


@dataclass
class ModelCard:
    """The 9 sections of Figure 1; each is a list of bullet strings (Quantitative Analyses holds rendered tables)."""
    name: str
    model_details: list = field(default_factory=list)
    intended_use: list = field(default_factory=list)
    factors: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    evaluation_data: list = field(default_factory=list)
    training_data: list = field(default_factory=list)
    quantitative_analyses: list = field(default_factory=list)
    ethical_considerations: list = field(default_factory=list)
    caveats_and_recommendations: list = field(default_factory=list)

    def missing(self):
        """Sections left empty -- a card with gaps should say so rather than silently omit them."""
        return [s for s in SECTIONS if not getattr(self, s.lower().replace(" ", "_"))]

    def to_markdown(self):
        out = [f"# Model Card: {self.name}", ""]
        for s in SECTIONS:
            out.append(f"## {s}")
            items = getattr(self, s.lower().replace(" ", "_"))
            if not items:
                out.append("- (not provided)")
            for it in items:
                out.append(it if it.startswith("|") or it.startswith("```") else f"- {it}")
            out.append("")
        return "\n".join(out)

# ----------------------------------------------------------------------------------------------- disaggregated metrics

def confusion_rates(y, yhat):
    """FPR = FP/(FP+TN), FNR = FN/(FN+TP), FDR = FP/(FP+TP), FOR = FN/(FN+TN)."""
    tp = np.sum((yhat == 1) & (y == 1))
    fp = np.sum((yhat == 1) & (y == 0))
    tn = np.sum((yhat == 0) & (y == 0))
    fn = np.sum((yhat == 0) & (y == 1))
    div = lambda a, b: float(a / b) if b else float("nan")
    return {"FPR": div(fp, fp + tn), "FNR": div(fn, fn + tp), "FDR": div(fp, fp + tp), "FOR": div(fn, fn + tn),
            "n": int(len(y))}


def bootstrap_ci(y, yhat, reps=1000, seed=0):
    """95% confidence intervals for the four rates by resampling the examples with replacement."""
    r = np.random.default_rng(seed)
    n = len(y)
    idx = r.integers(0, n, (reps, n))
    Y, P = y[idx], yhat[idx]
    tp = ((P == 1) & (Y == 1)).sum(1)
    fp = ((P == 1) & (Y == 0)).sum(1)
    tn = ((P == 0) & (Y == 0)).sum(1)
    fn = ((P == 0) & (Y == 1)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = {"FPR": fp / (fp + tn), "FNR": fn / (fn + tp), "FDR": fp / (fp + tp), "FOR": fn / (fn + tn)}
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in rates.items()}


def disaggregate(y, score, groups, threshold=0.5, reps=1000):
    """Metrics for ALL data, each unitary group of each factor, and each intersection of two factors.
    `groups` = {factor name: array of group labels}."""
    yhat = (score >= threshold).astype(int)
    rows = {"all": (confusion_rates(y, yhat), bootstrap_ci(y, yhat, reps))}
    names = list(groups)
    for f in names:
        for g in np.unique(groups[f]):
            m = groups[f] == g
            rows[f"{g}"] = (confusion_rates(y[m], yhat[m]), bootstrap_ci(y[m], yhat[m], reps))
    if len(names) >= 2:
        a, b = groups[names[0]], groups[names[1]]
        for ga in np.unique(a):
            for gb in np.unique(b):
                m = (a == ga) & (b == gb)
                rows[f"{gb} {ga}"] = (confusion_rates(y[m], yhat[m]), bootstrap_ci(y[m], yhat[m], reps))
    return rows


def fairness_gaps(rows, keys):
    """Largest difference across the given groups in FNR (equality of opportunity) and in max(FPR gap, FNR gap)
    (equalized odds)."""
    fnr = [rows[k][0]["FNR"] for k in keys]
    fpr = [rows[k][0]["FPR"] for k in keys]
    return {"equality of opportunity gap (FNR)": float(max(fnr) - min(fnr)),
            "equalized odds gap (max of FPR, FNR gaps)": float(max(max(fnr) - min(fnr), max(fpr) - min(fpr)))}


def threshold_sweep(y, score, groups_mask, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """The 'threshold slider' the paper suggests for digital cards: rates per group at several thresholds."""
    return {t: {g: confusion_rates(y[m], (score[m] >= t).astype(int)) for g, m in groups_mask.items()}
            for t in thresholds}


def table_markdown(rows, metrics=("FPR", "FNR", "FDR", "FOR")):
    out = ["| group | n | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for k, (v, ci) in rows.items():
        out.append(f"| {k} | {v['n']} | " + " | ".join(f"{v[m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]" for m in metrics)
                   + " |")
    return out

# ----------------------------------------------------------------------------------------------- example 1: smiling

def smiling_data(n=20000, seed=0):
    """A synthetic stand-in for CelebA 'smiling': two image cues -- mouth curvature (the real smile cue) and
    'face lines' (creases that appear with smiling, but older faces have them anyway). Men's smiles are annotated on
    subtler cues in this toy (weaker mouth signal). Gender and age are annotated like CelebA (binary, by annotators)."""
    r = np.random.default_rng(seed)
    male = r.random(n) < 0.42
    old = r.random(n) < np.where(male, 0.35, 0.2)                     # fewer older faces, especially women
    smile = (r.random(n) < 0.48).astype(int)
    mouth = smile * np.where(male, 1.3, 2.0) + r.normal(0, 1, n)
    lines = 1.2 * smile + 1.6 * old * np.where(male, 1.0, 0.6) + r.normal(0, 1, n)
    X = np.column_stack([mouth, lines])
    return X, smile, np.where(male, "male", "female"), np.where(old, "old", "young")


def smiling_card(seed=0, reps=1000):
    X, y, gender, age = smiling_data(seed=seed)
    tr = np.arange(len(y)) < 12000
    w = fit_logreg(X[tr], y[tr])
    s = predict(w, X[~tr])
    rows = disaggregate(y[~tr], s, {"age": age[~tr], "gender": gender[~tr]}, reps=reps)
    card = ModelCard(
        name="Smiling Detection (synthetic CelebA stand-in)",
        model_details=["Developed in this repository as a teaching example, 2026, v1.",
                       "Logistic regression on two image cues (mouth curvature, face lines).",
                       f"Weights: mouth {w[0]:.2f}, lines {w[1]:.2f}, bias {w[2]:.2f}."],
        intended_use=["Fun / assistive applications such as finding smiling photos.",
                      "Not for emotion or affect detection: smiles are labelled on appearance, not emotion."],
        factors=["Relevant: gender, age, skin type, camera, lighting.",
                 "Evaluated: gender (male/female) and age (young/old), binary annotations as in CelebA."],
        metrics=["FPR, FNR, FDR, FOR at threshold 0.5.", f"95% confidence intervals by bootstrap ({reps} resamples)."],
        evaluation_data=["Held-out synthetic split (8,000 faces)."],
        training_data=["Synthetic training split (12,000 faces); 42% male; older faces 35% of men, 20% of women."],
        quantitative_analyses=table_markdown(rows),
        ethical_considerations=["Gender and age are annotated, binary and coarse; no information is inferred beyond them."],
        caveats_and_recommendations=["Does not capture skin type or race.",
                                     "Older men get many false 'smiling' predictions: fine-tune with more older faces."],
    )
    return card, rows, (y[~tr], s, gender[~tr], age[~tr])

# ----------------------------------------------------------------------------------------------- example 2: toxicity

IDENTITY = ["lesbian", "gay", "homosexual", "straight", "christian", "muslim", "jewish", "american", "young", "old",
            "tall", "teacher"]
TARGETED = {"lesbian", "gay", "homosexual", "muslim"}                  # terms over-represented in toxic training text
TOXIC = ["stupid", "disgusting", "awful", "idiotic", "pathetic", "worthless"]
NICE = ["wonderful", "kind", "great", "smart", "lovely", "friendly"]
NEUTRAL = ["people", "the", "are", "is", "a", "person", "i", "am", "you", "my", "friend", "neighbor", "work", "today",
           "went", "home", "book", "music", "city", "food"]
VOCAB = IDENTITY + TOXIC + NICE + NEUTRAL
WID = {w: i for i, w in enumerate(VOCAB)}


def bow(sentences):
    X = np.zeros((len(sentences), len(VOCAB)))
    for i, s in enumerate(sentences):
        for t in s.split():
            if t in WID:
                X[i, WID[t]] = 1.0
    return X


def comment_corpus(n=12000, mitigate=False, seed=0):
    """Training comments. Most toxic ones contain an insult, but 35% are implicit (no insult word) and 8% of non-toxic
    ones use an insult (banter, quotes). Identity terms appear in comments at rates that differ by
    toxicity -- the TARGETED terms show up mostly inside toxic comments (as on the real web), which is the bias.
    `mitigate` adds non-toxic comments that use every identity term (the data-balancing fix of Dixon et al.)."""
    r = np.random.default_rng(seed)
    sents, ys = [], []
    for _ in range(n):
        tox = r.random() < 0.3
        words = list(r.choice(NEUTRAL, 5))
        if tox:                                                       # 35% of toxic comments are implicit (no insult)
            words.append(r.choice(TOXIC) if r.random() < 0.65 else r.choice(NEUTRAL))
        else:                                                         # some banter / quoting uses insults
            words.append(r.choice(TOXIC) if r.random() < 0.08 else (r.choice(NICE) if r.random() < 0.5 else
                                                                      r.choice(NEUTRAL)))
        if r.random() < 0.35:
            pool = [t for t in IDENTITY if (t in TARGETED) == tox] if r.random() < 0.85 else IDENTITY
            words.append(r.choice(pool))
        r.shuffle(words)
        sents.append(" ".join(words))
        ys.append(int(tox))
    if mitigate:
        for _ in range(n // 3):
            words = list(r.choice(NEUTRAL, 5)) + [r.choice(IDENTITY)]
            if r.random() < 0.5:
                words.append(r.choice(NICE))
            r.shuffle(words)
            sents.append(" ".join(words))
            ys.append(0)
    return sents, np.array(ys)


def template_set():
    """Synthetic identity-phrase templates (the evaluation idea of Dixon et al.): each term in toxic and non-toxic
    sentence frames."""
    frames_ok = ["i am {t}", "{t} people are wonderful", "my friend is {t}", "{t} people are kind", "i am a {t} person",
                 "{t} people are smart"]
    frames_bad = ["{t} people are disgusting", "{t} people are stupid", "{t} people are pathetic",
                  "{t} people are worthless", "you are {t} and idiotic", "{t} people are awful"]
    sents, ys, terms = [], [], []
    for t in IDENTITY:
        for f in frames_ok:
            sents.append(f.format(t=t)); ys.append(0); terms.append(t)
        for f in frames_bad:
            sents.append(f.format(t=t)); ys.append(1); terms.append(t)
    return sents, np.array(ys), np.array(terms)


def auc(score, y):
    order = np.argsort(score, kind="stable")
    rk = np.empty(len(score))
    rk[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):                                       # average ranks for ties
        m = score == v
        rk[m] = rk[m].mean()
    pos = y == 1
    return float((rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def toxicity_versions(seed=0, pinned_draws=20):
    """Train v1 on the biased corpus and v2 on the balanced one. Per identity term report:
      subgroup AUC   toxic vs non-toxic templates of that term only
      pinned AUC     that term's templates + an equal-size random sample of all templates (Dixon et al.), averaged
                     over `pinned_draws` samples
      BPSN AUC       background-positive, subgroup-negative: toxic templates of OTHER terms vs non-toxic templates of
                     this term -- low when innocent sentences mentioning the term score like toxic ones
    plus the mean score of the term's non-toxic templates and the term's learned weight."""
    Xt_s, yt, terms = template_set()
    Xt = bow(Xt_s)
    out = {}
    for name, mit in (("v1 (biased training data)", False), ("v2 (identity terms balanced)", True)):
        s, y = comment_corpus(mitigate=mit, seed=seed)
        w = fit_logreg(bow(s), y, iters=800, lr=1.0)
        sc = predict(w, Xt)
        r = np.random.default_rng(seed)
        per = {}
        for t in IDENTITY:
            m = terms == t
            pins = []
            for _ in range(pinned_draws):
                bg = r.choice(len(yt), m.sum(), replace=False)
                pin = np.r_[np.where(m)[0], bg]
                pins.append(auc(sc[pin], yt[pin]))
            bpsn = np.r_[np.where(m & (yt == 0))[0], np.where(~m & (yt == 1))[0]]
            per[t] = {"subgroup AUC": auc(sc[m], yt[m]), "pinned AUC": float(np.mean(pins)),
                      "BPSN AUC": auc(sc[bpsn], yt[bpsn]),
                      "mean score of non-toxic templates": float(sc[m & (yt == 0)].mean()),
                      "identity weight": float(w[WID[t]])}
        out[name] = {"overall AUC": auc(sc, yt), "per term": per}
    return out


REPORTED = {
    "sections (Figure 1)": "Model Details; Intended Use; Factors; Metrics; Evaluation Data; Training Data; "
                           "Quantitative Analyses (unitary and intersectional results); Ethical Considerations; "
                           "Caveats and Recommendations",
    "metrics": "for classifiers, false positive / false negative / false discovery / false omission rates; equal FNR "
               "across groups = Equality of Opportunity, equal FNR and FPR = Equality of Odds; for scores, compare "
               "distributions (e.g. pinned AUC); report confidence intervals",
    "smiling example (Figure 2)": "CelebA, evaluated by gender and age; 95% CIs by bootstrap; threshold 0.5 where all "
                                  "error types are within 0.04-0.14; false discovery rate much higher for older men; "
                                  "men in aggregate have a higher false negative rate",
    "toxicity example (Figure 3)": "Perspective API TOXICITY v1 vs v5 on synthetic Identity Phrase Templates; v1 performs "
                                   "poorly for several terms, especially 'lesbian', 'gay', 'homosexual'; v5 after bias "
                                   "mitigation is more equitable",
}
