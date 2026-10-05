# Data distribution shifts and monitoring, explained simply

**Source:** Chip Huyen, *Data Distribution Shifts and Monitoring*, blog post, 7 February 2022 (huyenchip.com). It draws on her book *Designing Machine Learning Systems* (094).

**In one sentence:** deploying a model is not the end. Production data drifts away from training data in different ways (covariate shift, label shift, concept drift), and you need to know which kind you face, how to **detect** it (two-sample tests, the right time windows, sensible alerts), and how to **respond** (reweighting, re-estimating priors, retraining on the right data).

The post is a practical overview, not a research paper. Below, its ideas are explained in our own words and each one is tested with a small experiment.

---

## 1. Why ML systems fail

- **Many failures aren't about ML at all.** The post cites a Google study of **96** ML pipeline outages over 15 years: **60** were caused by things not directly related to ML (dependencies, deployment, hardware, downtime).
- **ML-specific failures** are the subtle ones:
  1. **Production data differs from training data** (distribution shift). The post notes that much of what *looks* like shift is actually caused by internal bugs: broken pipelines, missing values, inconsistent features.
  2. **Edge cases:** rare inputs where the model fails catastrophically.
  3. **Degenerate feedback loops:** the model's predictions influence the data it later learns from (e.g. recommendations shape clicks), which amplifies biases such as popularity.

---

## 2. The three kinds of shift, with the math

The joint distribution of inputs X and labels Y can be factored two ways:
```
P(X, Y) = P(Y | X) · P(X)        = P(X | Y) · P(Y)
```

| Shift | What changes | What stays fixed | The post's example |
|---|---|---|---|
| **Covariate shift** | P(X) | P(Y \| X) | more older patients come in; the risk at each age is unchanged |
| **Label shift** (prior shift) | P(Y) | P(X \| Y) | a preventive drug lowers the disease rate; sick people still look like sick people |
| **Concept drift** | P(Y \| X) | P(X) | after COVID, the same house features map to different prices |

The post also mentions **feature changes** (features added or removed, value ranges changed) and **label-schema changes** (new classes, e.g. a sentiment label split into finer categories).

### A worked example of label shift
- Before the drug: 36% of patients are sick. Our model outputs P(sick | x).
- After the drug: only 14% are sick, but a sick patient's test results look the same as before (P(X | Y) fixed).
- **Bayes' rule** says the correct new probability is the old one, reweighted by the ratio of priors:
  ```
  P_new(sick | x) ∝ P_old(sick | x) · (0.14 / 0.36)
  P_new(healthy | x) ∝ P_old(healthy | x) · (0.86 / 0.64)
  ```
- **Example:** old P(sick | x) = 0.5.
  - sick: 0.5 · 0.389 = 0.194;
  - healthy: 0.5 · 1.344 = 0.672;
  - normalised: P_new(sick | x) = 0.194 / (0.194 + 0.672) = **0.22**.
- **The catch:** you need the new prior 0.14, and you usually don't have labels. Section 4 shows how to estimate it anyway.

---

## 3. Detecting shift

- **Simple statistics:** compare the mean, median, variance and missing rate of features or predictions with a reference period.
- **Two-sample tests** (see paper 091 for the details):
  - **Kolmogorov–Smirnov** for one dimension at a time;
  - **MMD** (and learned-kernel MMD) and **LSDD** (least-squares density difference) for many dimensions.
  - The post recommends the open-source **alibi-detect** library, which implements many of them.
- **What to monitor**, from easiest to hardest:
  - **predictions:** low-dimensional and always available;
  - **features:** structured, schema checks, but many false alarms;
  - **raw inputs:** often not even accessible to ML engineers;
  - **accuracy-related metrics:** the most direct, but they need labels, which are often delayed.

### Time scale matters
- **Cumulative statistics** (the average since the start) barely move when something breaks today. **Sliding windows** (the last k hours) react.
- **Window length** trades detection speed against false alarms. Shorter windows react faster but are noisier.
- **Seasonality:** comparing Monday 9 a.m. with Sunday 3 a.m. raises false alarms. Compare with the **same time last week**.

### The noise of a window, step by step
- An hourly conversion rate of 0.20 measured on 200 users has standard error √(0.2 · 0.8 / 200) = **0.028**.
- **A 6-hour window** averages 1,200 users: standard error √(0.2 · 0.8 / 1200) = **0.0115**.
- **Comparing two such windows** (now vs last week) multiplies the noise by √2: **0.016**.
- **The size of the drop matters too.** A 0.12 drop is 0.12 / 0.016 = **7.5 standard errors**, easy to catch at a z = 4 threshold. With 1-hour windows it is only 0.12 / (√2 · 0.028) = **3.0**, which misses at z = 4.

### Alerts
- **An alert has three parts:**
  1. a **policy:** a threshold plus a duration, e.g. "for 3 hours in a row";
  2. **notification channels:** email, Slack, PagerDuty;
  3. a **description**, ideally with a runbook.
- **Too many trivial alerts cause alert fatigue:** people learn to ignore them, including the critical ones.
- **Observability** (vs monitoring) means logging enough fine-grained, tagged data that you can answer new questions without shipping new code, e.g. "show users with wrong predictions in the last hour, grouped by zip code."

---

## 4. Responding to shift

The post lists three broad approaches:
1. **Train on massive data**, hoping the training distribution already covers what production will look like.
2. **Adapt without new labels** (domain adaptation). The post notes this is under-explored in industry.
3. **Retrain** (the common case), either from scratch on old plus new data, or by **fine-tuning** on new data. Which data to use depends on **when the drift started**.

### Importance weighting for covariate shift
- **The idea:** if P(X) changed but P(Y | X) didn't, train on source data with each example weighted by
  ```
  w(x) = P_target(x) / P_source(x)
  ```
  so the training loss imitates the loss on target data.
- **Estimating w(x) without labels:**
  1. train a **domain classifier** to tell source inputs (0) from target inputs (1);
  2. if the two groups are the same size, w(x) = P(target | x) / P(source | x), the classifier's odds.
- **Example:** the domain classifier says 0.8 for some x. Then w = 0.8 / 0.2 = **4**: this source example looks like target data, so count it 4 times.
- **The price is variance.** A few examples get huge weights. The **effective sample size** (Σw)² / Σw² shows how many equally weighted examples the weighted set is "worth".
- **When it helps:** if the model is already correct for all x (well-specified), plain training is already fine. Weighting helps when the model is **mis-specified**, because it decides *where* the model's limited capacity is spent.

### Estimating a new class prior under label shift (no labels)
- **On source validation data,** measure the confusion matrix C[i, j] = P(predict i | true class j).
- **On target data,** the share of each *predicted* class μ satisfies μ = C · q, where q is the unknown target prior.
- **So solve q = C⁻¹ μ, then correct probabilities with Bayes' rule** as in section 2.
- **Example (2 classes):**
  - C = [[0.9, 0.3], [0.1, 0.7]] (column j = true class j);
  - target predictions are 75% class 0, 25% class 1;
  - from 0.9 q₀ + 0.3 q₁ = 0.75 and q₀ + q₁ = 1: 0.9 − 0.6 q₁ = 0.75, so q₁ = **0.25**.
  - (The naive guess, "25% of predictions are class 1", happens to match here, but in general the two differ.)

---

## 5. What our code found

**Which monitor sees which shift** (logistic model trained on the source "cancer" data; 3,000 target samples):

| Target data | Inputs KS p | Predictions KS p | Accuracy (source → target) | Mean prediction vs true rate |
|---|---|---|---|---|
| no shift | 0.22 | 0.59 | 0.764 → 0.754 | 0.366 vs 0.367 |
| covariate shift | ~0 | ~0 | 0.764 → 0.725 | 0.578 vs **0.586** (still calibrated) |
| label shift | 7·10⁻⁶ | 3·10⁻¹¹ | 0.764 → **0.812** | **0.309 vs 0.144** |
| concept drift | **0.22** | **0.59** | 0.764 → **0.668** | 0.366 vs 0.369 |

- **Concept drift is invisible** to input and prediction monitors: the inputs are exactly the same. Only accuracy, which needs labels, reveals it.
- **Under label shift, accuracy went up** (fewer sick patients, and the model mostly predicts "healthy"), while the probabilities became badly miscalibrated (over-predicting disease about 2×). Accuracy alone can hide a broken model.
- **Over 2 seeds** (`experiments.py` E1 smoke run), the pattern held: covariate and label shift always raised input and prediction alarms; concept drift never did; accuracy dropped by 0.104 on average for concept drift and rose by 0.072 for label shift.

**Adapting without labels:**
- **Importance weighting** (mis-specified linear model, quadratic truth): target log-loss **0.713 → 0.567** with domain-classifier weights, the same as with the true density ratio (0.567). The effective sample size is 973 of 6,000 (1,162 with the true weights). Accuracy barely changed (0.747 → 0.743): weighting fixed the probabilities, not the decisions.
- **Label-shift prior:** true target rate 0.143, naive mean prediction 0.301, confusion-matrix estimate **0.167**. After the Bayes correction, log-loss 0.428 → **0.342** and accuracy 0.812 → **0.865**.

**Time scale** (hourly conversion with a daily cycle; a 0.12 drop for 6 hours on day 9):
- **Three hours into the outage,** the cumulative mean had moved only −0.0016, while a 6-hour sliding mean moved −0.037.
- **Alert policies:**

| Window | Compared with | Alert after | False alarms (7 days) | Delay |
|---|---|---|---|---|
| 1 h | same hours last week | 1 h | 0 | missed |
| 3 h | same hours last week | 1 h | 0 | **2 h** |
| 6 h | same hours last week | 1 h | 0 | 4 h |
| 6 h | last week's mean | 1 h | **11** | 3 h |
| 6 h | last week's mean | 3 h | 2 | 5 h |
| 24 h | same hours last week | 1 h | 0 | missed |

- **Comparing with the same hours last week removes the daily-cycle false alarms** (11 → 0).
- **Requiring 3 hours in a row cuts false alarms,** at the cost of delay.
- **Windows that are too short (noise) or too long (dilution) miss the outage.**

**Retraining after concept drift on day 20** (log-loss on the evaluation day):

| Day | Stale | Scratch on all data | Fine-tune on last 2 days | From the drift point |
|---|---|---|---|---|
| 22 | 0.816 | 0.761 | **0.552** | **0.552** |
| 25 | 0.762 | 0.668 | 0.546 | 0.546 |
| 29 | 0.810 | 0.671 | 0.571 | 0.572 |

- **Retraining from scratch on everything dilutes the new concept** with 20 days of old data.
- **Fine-tuning on recent data matches training from the drift point,** without needing to know exactly when the drift began.

**Degenerate feedback loop** (200 items; the recommender estimates CTR from its own logs):

| Exploration | Distinct items shown | Top item's share | CTR (best possible 0.278) | True best item ranked |
|---|---|---|---|---|
| 0% | 0.5% | 100% | 0.254 | #7 |
| 10% | 22.5% | 91% | 0.258 | **#2** |

- **Without exploration,** the system collapses onto one item and its estimates of everything else freeze.
- **Randomising 10% of traffic** keeps learning going. It is better, but after 40 rounds still not perfect.

**`experiments.py`:**
- **E1:** monitors × shift strengths × seeds;
- **E2:** when weighting helps;
- **E3:** the false-alarm vs delay curve over window, z and duration;
- **E4:** retraining cadence;
- **E5:** exploration rates.
- Only the E1 smoke run (2 seeds) was done here.

---

## 6. Why it matters

- **Monitoring and drift detection are now standard parts of ML platforms** (Evidently, WhyLabs, Arize, alibi-detect, the cloud providers' model monitors).
- **This post is a common entry point to them.**
- **Its central warning holds up in our experiments:** the most damaging shift, concept drift, is exactly the one that input monitoring cannot see. Getting labels, even delayed or sampled, is irreplaceable.

---

## 7. Check yourself

1. Define covariate shift, label shift and concept drift in terms of P(X), P(Y), P(Y|X), P(X|Y).
<details><summary>Answer</summary>Covariate: P(X) changes, P(Y|X) fixed. Label: P(Y) changes, P(X|Y) fixed. Concept drift: P(Y|X) changes (here with P(X) fixed).</details>

2. Which kind of shift can't be detected by monitoring inputs or predictions alone, and why?
<details><summary>Answer</summary>Concept drift with unchanged P(X): the inputs, and therefore the model's predictions, have exactly the same distribution. Only the relationship to the labels changed, so you need labels (accuracy metrics) to see it.</details>

3. Under label shift, accuracy rose from 0.764 to 0.812 in our toy. Is the model fine?
<details><summary>Answer</summary>No. Its probabilities are badly miscalibrated (mean prediction 0.309 vs a true rate of 0.144). Accuracy rose only because the new population is mostly healthy and the model mostly predicts "healthy". Calibration and log-loss reveal the problem.</details>

4. A domain classifier gives P(target | x) = 0.6 for a source example (balanced training). What is its importance weight?
<details><summary>Answer</summary>0.6 / 0.4 = 1.5.</details>

5. What is the effective sample size of weights (1, 1, 1, 9)?
<details><summary>Answer</summary>(Σw)² / Σw² = 12² / (1 + 1 + 1 + 81) = 144 / 84 ≈ 1.7. Four examples, but worth fewer than two.</details>

6. The old model says P(sick | x) = 0.3 with a source rate of 0.4; the target rate is 0.2. What is the corrected probability?
<details><summary>Answer</summary>Sick: 0.3 · (0.2/0.4) = 0.15. Healthy: 0.7 · (0.8/0.6) = 0.933. Corrected = 0.15 / (0.15 + 0.933) ≈ 0.14.</details>

7. Why do cumulative statistics hide an outage?
<details><summary>Answer</summary>The cumulative mean averages over all past data, so a few bad hours change it very little (−0.0016 in our toy). A sliding window only averages recent data, so it moves a lot (−0.037).</details>

8. Why compare a window with the same hours last week rather than with last week's overall mean?
<details><summary>Answer</summary>Metrics have daily cycles. Comparing a quiet hour with the overall mean looks like a drop every night, which causes false alarms (11 vs 0 in our toy). Comparing like with like removes the seasonal effect.</details>

9. After concept drift, why can retraining from scratch on all data be worse than fine-tuning on recent data?
<details><summary>Answer</summary>Most of the data still follows the old concept, so the fit is pulled toward it (log-loss 0.67–0.76 vs 0.55–0.57 for fine-tuning in our toy). Recent data reflects the new concept.</details>

10. What is a degenerate feedback loop, and how can randomisation help?
<details><summary>Answer</summary>A system whose predictions shape its future training data, e.g. recommending already-popular items, which then get more clicks. Showing a small random share of items gives unbiased feedback on items the model would otherwise never show (diversity 0.5% → 22.5% in our toy).</details>
