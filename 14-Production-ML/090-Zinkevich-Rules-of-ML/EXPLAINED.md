# Rules of Machine Learning, explained simply

**Source:** Martin Zinkevich (Google), *Rules of Machine Learning: Best Practices for ML Engineering*. A web guide (no PDF): https://developers.google.com/machine-learning/guides/rules-of-ml

**In one sentence:** 43 practical rules from Google's experience building ML products, summed up by its motto: "do machine learning like the great engineer you are, not like the great machine learning expert you aren't." Most of the gains come from good features, clean data and solid pipelines, not clever algorithms.

**Basic approach** (from the guide's opening):
1. make sure your pipeline is solid end to end;
2. start with a reasonable objective;
3. add common-sense features in a simple way;
4. make sure that your pipeline stays solid.

---

## 1. Vocabulary the guide uses

| Term | Meaning |
|---|---|
| Instance | the thing you make a prediction about (e.g. a web page) |
| Label | the answer: from the system, or the true answer in training data |
| Feature | a property of an instance used to predict |
| Feature column | a set of related features (e.g. all possible user countries) |
| Example | an instance with its features, plus a label |
| Model | a statistical representation of the prediction task |
| Metric | a number you care about (may or may not be optimised directly) |
| Objective | the metric your algorithm optimises |
| Pipeline | the infrastructure around the ML algorithm |
| Click-through rate (CTR) | the percentage of visitors who click a link in an ad |

---

## 2. The rules, by phase

### Before machine learning (1–3)
1. **Don't be afraid to launch without ML.** "If you think that machine learning will give you a 100% boost, then a heuristic will get you 50% of the way there." Examples: rank apps by install rate; blacklist spammers.
2. **Design and implement metrics first,** so you have history before you need it.
3. **Choose ML over a complex heuristic.** Once heuristics get complicated, a model is easier to update and maintain.

### Phase I: your first pipeline (4–15)
- **4–5:** keep the first model **simple** and get the **infrastructure** right; test the infrastructure separately from the ML (check what goes in, check that the model scores the same in training and serving).
- **6:** be careful about **dropped data when copying pipelines.** Example: Google Plus "What's Hot" dropped older posts; the pipeline was copied to Stream, where old posts mattered.
- **7:** turn heuristics into **features** (or handle them outside the model).
- **8–11 (monitoring):**
  - **8:** know your **freshness** needs (ads models need daily updates);
  - **9:** **detect problems before exporting** a model (check held-out quality, e.g. AUC, before it touches users);
  - **10:** watch for **silent failures:**
    - Play had a table that was stale for **6 months**, and refreshing it alone gave **+2% install rate**;
    - a feature column can drop from **90% to 60%** coverage without any error;
  - **11:** give feature columns **owners and documentation.**
- **12–15 (first objective):**
  - **12–13:** don't overthink the objective; choose a simple, **observable and attributable** metric;
  - **14:** start with an **interpretable** model, which makes debugging easier;
  - **15:** separate spam filtering and quality ranking in a **policy layer.**

### Phase II: feature engineering (16–37)
- **16–22:**
  - **16:** plan to launch and iterate;
  - **17:** start with directly observed features rather than learned ones;
  - **18:** explore content features that generalise across contexts;
  - **19:** use very specific features when you can;
  - **20:** combine features in human-understandable ways;
  - **21:** the number of feature weights you can learn is **roughly proportional to your data**. About 1,000 examples → a dozen features; 10 million → a hundred thousand; billions → around 10 million (with feature selection and regularisation);
  - **22:** clean up unused features.
- **23–28 (human analysis):**
  - **23:** you are not a typical end user;
  - **24:** **measure the delta** between models;
  - **25:** utilitarian performance beats predictive power;
  - **26:** look for patterns in errors and create features;
  - **27:** quantify undesirable behaviour;
  - **28:** identical short-term behaviour ≠ identical long-term behaviour. A model keyed on doc_id and exact query matched the old system in A/B tests but would never surface new apps.
- **29–37 (training/serving skew):**
  - **29:** **log the features used at serving time** and train on them. The YouTube home page did this with "significant quality improvements and a reduction in code complexity";
  - **30:** **importance-weight** sampled data, don't drop it: sampled with probability 30% → weight **10/3**;
  - **31:** tables joined at training and serving time **may change** in between;
  - **32:** reuse code between training and serving;
  - **33:** train on data up to **January 5**, test on **January 6** and after;
  - **34:** for filtering, make small short-term sacrifices for **clean data**. A filter blocking 75% of negatives can hold out **1% of traffic** unfiltered and still block at least **74%**; at 95%+ blocking, use 0.1% or 0.001%;
  - **35:** beware the inherent skew in ranking;
  - **36:** avoid feedback loops with **positional features** (train with position, serve without it);
  - **37:** **measure training/serving skew** as three parts: training vs holdout, holdout vs next-day, next-day vs live.

### Phase III: slowed growth, refinement, complex models (38–43)
- **38:** don't waste time on new features if **unaligned objectives** are the problem.
- **39:** launch decisions are a proxy for long-term product goals.
- **40:** **keep ensembles simple.** "Each model should either be an ensemble only taking the input of other models, or a base model taking many features, but not both." Increasing a base model's score should not decrease the ensemble's.
- **41:** when performance plateaus, look for **qualitatively new** information sources.
- **42:** diversity, personalisation and relevance are less correlated with popularity than you think.
- **43:** your friends tend to be the same across products; your interests don't.

The full list of 43 rule titles is printed by `demo.py` (`rules.RULES`).

---

## 3. The math behind the measurable rules

### Rule 30: why weight 10/3?
- **The setup:** suppose negatives (y = 0) are kept with probability 0.3, and all positives are kept.
- **Without weights,** each kept negative stands for 1/0.3 = 3.33 original negatives, so the model sees too few negatives. It learns a base rate that is too high.
- **Example:** the true data has 176 positives and 824 negatives per 1,000 (rate 17.6%). After sampling, 176 positives and 0.3 · 824 ≈ 247 negatives remain, so the apparent rate is 176 / 423 = **41.6%**. (Our model's mean prediction rose from 17.6% to 35.3%; it's below 41.6% because the model fits per-example probabilities, not one global rate.)
- **With weight w = 1/0.3 = 10/3 on each kept negative,** the weighted negative count is 247 · 10/3 ≈ 824, and the original proportion returns. The weighted log-loss is an unbiased estimate of the full-data log-loss.

### Rule 21: why more weights need more data
- A one-hot feature for a value seen only k times has a weight estimated from k examples. A **cross** of two 10-value columns has 100 weights, each seen about n/100 times.
- With n = 300, each cross weight sees about **3 examples**, so it mostly fits noise. With n = 30,000, about 300 examples each, enough to estimate real interactions.

### Rule 24: the delta between two models
- Take each model's top-k list. Every item that is in one list but not the other adds 1/(rank + 1), so differences near the top count most.
- We divide by the largest possible value (two disjoint lists), giving 0 for the same set and 1 for disjoint sets.
- **Example (k = 3):** A = (a, b, c), B = (a, b, d). Then c contributes 1/3 and d contributes 1/3. The maximum is 2 · (1 + 1/2 + 1/3) = 3.67. Delta = 0.667 / 3.67 = **0.18**.
- **Note:** reordering within the same set gives 0 under this measure. A finer measure would also compare ranks.

### Rule 33: why random splits flatter you
- If each day has its own effect (news, holidays, outages) and the model has per-day features, a random split puts examples from **every** day in training. The model learns each day's effect and is scored on the same days.
- After launch, tomorrow's effect is unknown. A validation set made of the **latest days** mimics that.

### Rule 37: reading the three gaps
| Gap | Normal cause | Big gap means |
|---|---|---|
| training → holdout | overfitting (always some) | too little data or regularisation |
| holdout → next day | the world changes daily | tune regularisation / freshness here |
| next day → live | should be ≈ 0 | **an engineering bug**: features differ between training and serving |

---

## 4. What our code found

| Rule | Toy | Result |
|---|---|---|
| 1 | app store: users like apps of their category | random 0.129, **most-installed heuristic 0.294**, ML 0.506 install rate: the heuristic gets **44%** of the way (E1 smoke run over 3 seeds: 44–70%, mean 58%), close to the guide's "50%" |
| 10 | coverage drops 90% → 60% on day 18 | nothing errors; log-loss rises 0.459 → 0.527; the coverage monitor alerts on day 18 |
| 21 | 3 categorical columns, label depends on pairs | 300 examples: 30 features **0.623** vs 330 features 0.690 (overfit); 3,000: 0.584 vs **0.544**; 30,000: 0.580 vs **0.524** |
| 24 | position-weighted top-10 difference | same 0.000, small noise 0.000, big change 0.709, reversed 1.000 |
| 30 | negatives kept at 30% | unweighted: mean prediction **0.353** vs true 0.176; weight 10/3: **0.176**; AUC 0.794 in all three. Sampling breaks probabilities, not ranking |
| 33 | per-day effects, day-id features | random split **estimates** log-loss 0.544 but the **actual** future value is 0.594 (AUC 0.791 vs 0.763); a time-based validation estimates 0.572 / 0.758, close to the actual 0.592 / 0.763 |
| 34 | old filter uses sender reputation; the new content model learns from user labels | trained on shown traffic: mean prediction **0.135** vs true 0.344 (log-loss 0.779); trained on the **1% held-out** (1,955 examples): 0.350, log-loss **0.604**, matching the oracle 0.603. Blocking falls only from 75.0% to 74.3% |
| 36 | clicks = relevance × slot; the old ranker put popular docs on top | rank correlation with true relevance: doc-only model 0.766, **doc + position model 0.950** |
| 37 | a merchant-rating table refreshed between training and serving | train 0.509 → holdout 0.517 → next day 0.516 → **live 0.631**: the jump is at live, so it is an engineering skew. Training on features logged at serving gives live **0.575** |
| 40 | stacking two calibrated base models | weights 1.12 and 1.28 (non-negative); raising base A never lowered the ensemble (0% of examples); ensemble AUC 0.814 vs 0.762 / 0.659 |

**Honest notes:**
- **Rule 37:** with logged serving features, the **offline** log-loss looks worse (0.568 vs 0.509), because the refreshed ratings are noisier than the ones the labels depended on. Yet live is much better (0.575 vs 0.631). Offline numbers computed on non-serving features are misleading.
- **Rule 40:** our unconstrained stacking already gave positive weights. We tried correlated and mixed setups and found no case where monotonicity broke. We show the property holding, not a failure.
- **Rule 34:** our old filter also blocks 15.8% of good mail, a crude filter, which is why it is worth replacing.
- **`experiments.py`:**
  - **E1:** the heuristic's share over seeds and taste strengths;
  - **E2:** the Rule 21 crossover;
  - **E3:** sampling rates and held-out shares;
  - **E4:** split optimism and skew over seeds.
  - Only the E1 smoke run (3 seeds) was done here.

---

## 5. Why it matters

- **The guide is one of the most widely shared pieces of practical ML advice.** Its themes appear in the technical-debt papers (087, 088), the ML Test Score (089) and Huyen's book (094).
- **Several rules became standard tools:**
  - Rule 29's "log features at serving time" is how feature stores and feature logging work;
  - Rule 36's "train with position, serve without" is the standard way to handle **position bias** in ranking;
  - Rule 30's importance weighting is the standard fix for negative down-sampling in ad click prediction.

---

## 6. Check yourself

1. Negatives were kept with probability 5%. What weight should each kept negative get?
<details><summary>Answer</summary>1 / 0.05 = 20.</details>

2. Why didn't down-sampling negatives change AUC in our Rule 30 experiment?
<details><summary>Answer</summary>Dropping negatives at random mainly shifts the model's intercept (the base rate), which moves every score by the same amount and keeps their order. AUC only depends on order, so it didn't change. The probabilities, and hence log-loss and calibration, did.</details>

3. What are the three components of training/serving skew in Rule 37, and which one points to an engineering bug?
<details><summary>Answer</summary>Training vs holdout, holdout vs next-day, next-day vs live. A large next-day → live gap means the served features differ from the training features: an engineering problem.</details>

4. Why does Rule 29 (log features at serving time) fix skew from a changing joined table (Rule 31)?
<details><summary>Answer</summary>Training then uses exactly the values the server saw, rather than re-joining the table later when its contents may have changed.</details>

5. Compute the Rule 24 delta for k = 2: A = (x, y), B = (y, z).
<details><summary>Answer</summary>x is only in A (rank 0: weight 1). z is only in B (rank 1: weight 1/2). Sum 1.5. Maximum 2 · (1 + 1/2) = 3. Delta = 0.5.</details>

6. Why can a random train/test split overestimate quality (Rule 33)?
<details><summary>Answer</summary>Examples from the same day (or user, or session) share effects. A random split puts some in training and some in test, so the model is tested on days it has seen. After launch it faces unseen days. Testing on later days mimics that.</details>

7. What is the point of holding out 1% of traffic unfiltered in Rule 34?
<details><summary>Answer</summary>Labels from filtered traffic are biased: messages the old filter blocked are never seen or labelled. The held-out slice gives unbiased labels for training and evaluation, at the cost of letting a little more spam through (75% → 74% blocking).</details>

8. In Rule 36, what does "train with position, serve without it" mean in practice?
<details><summary>Answer</summary>Include the display position as a feature during training so it absorbs the position effect. At serving time, set it to one fixed value (e.g. the top slot) for every candidate, so documents are compared on their own merits.</details>

9. With 300 examples, why did 330 features do worse than 30?
<details><summary>Answer</summary>Each of the 300 cross weights saw only about 3 examples, so it fit noise. With thousands of examples the cross weights become reliable and beat the simple model (Rule 21).</details>

10. What is the guide's advice when performance plateaus (Rule 41)?
<details><summary>Answer</summary>Look for qualitatively new sources of information (new kinds of signals) rather than refining the existing ones.</details>
