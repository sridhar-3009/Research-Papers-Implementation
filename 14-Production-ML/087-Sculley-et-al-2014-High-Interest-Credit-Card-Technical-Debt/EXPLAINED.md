# Technical debt in machine learning, explained simply

**Paper:** D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young (Google), *Machine Learning: The High-Interest Credit Card of Technical Debt*, NIPS 2014 workshop on Software Engineering for Machine Learning (SE4ML).

**In one sentence:** machine learning lets you build complex systems quickly, but those quick wins are **borrowed**: ML systems pile up hidden maintenance costs ("technical debt") at the **system level** rather than in the code. The paper names the specific ways this happens and the patterns that help pay the debt down.

This paper has no equations, no model and no results table. It is a list of warnings drawn from experience at Google. To make each warning concrete, our code turns it into a tiny experiment with logistic regression and **measures** the damage.

---

## 1. What "technical debt" means

- **The metaphor (from Ward Cunningham, 1992):** moving fast in software is like borrowing money. You ship sooner, but you then pay "interest" on every later change, until you "pay down" the debt by refactoring, adding tests, deleting dead code and so on.
- **The paper's claim:** ML systems have all the ordinary code debt **plus** an ML-specific kind that is harder to see. It lives in **data dependencies, feedback with the world, and configuration**, not in lines of code. Code-level tools (compilers, linters, unit tests) don't find it.
- **The memorable line:** "shipping the first version of a machine learning system is easy, but … making subsequent improvements is unexpectedly difficult."

---

## 2. A two-minute refresher: logistic regression

Our experiments all use the same small model, so it helps to know exactly what it does.

- **Inputs:** a row of numbers x = (x₁, …, xₙ), called **features**.
- **Weights:** one number wᵢ per feature, plus a bias b.
- **Score (the "logit"):** z = w₁x₁ + … + wₙxₙ + b.
- **Probability:** p = σ(z) = 1 / (1 + e^(−z)). The sigmoid σ squashes any score into (0, 1).
- **Training:** choose w to minimise the **log-loss** −[y·log p + (1 − y)·log(1 − p)], averaged over examples (y is 0 or 1). We use plain gradient descent plus a small L2 penalty λ·Σwᵢ².

**Worked example.** Say w = (1.2, −0.8), b = 0, and an example has x = (1, 0.5).
- z = 1.2·1 − 0.8·0.5 = 0.8.
- p = 1 / (1 + e^(−0.8)) = 1 / (1 + 0.449) = 0.69.
- If the true label is y = 1, its log-loss is −log 0.69 = 0.37. If y = 0, it is −log 0.31 = 1.17.
- **A log-loss of 0.693 = log 2** is what you get by always predicting 0.5. Anything above that is worse than guessing.

**Why weights "share credit."** If two features carry the same information, the model can put the weight on either one or split it between them. Training doesn't care which, because the predictions come out the same. It is the **world** that will later care, when the two features stop being identical. Most of the paper's warnings come from this one fact.

---

## 3. The debts, one by one

### 3.1 Entanglement: CACE, "Changing Anything Changes Everything"
- **The idea:** a model mixes all its inputs together. If you change the distribution of x₁, or add a feature, or remove one, or change a hyper-parameter (regularisation, learning rate, sampling, convergence thresholds), then **all the other weights can change**, and predictions change on various slices of the data.
- **Why:** weights are chosen jointly. If x₁ is removed, any feature correlated with x₁ will start carrying x₁'s information, so its weight jumps.
- **Mitigations the paper suggests:**
  - isolate models and serve ensembles (when sub-problems really decompose);
  - tools that show prediction changes across many dimensions and slices, and slice-by-slice metrics;
  - regularisation that penalises changes in predictions, which "may add more debt … than is reduced".

### 3.2 Hidden feedback loops
- **Example from the paper:** a model predicts click-through rate (CTR) of news headlines. One input is x_week = how many headlines this user clicked in the past week.
- **The loop:** improve the model, so users get better recommendations and click more. A week later, x_week has risen for everyone. The model, retrained on the new data, then changes its opinion of x_week. So the system changes behaviour **slowly**, "over a time scale much longer than a week."
- **Why it is debt:** quick A/B tests miss these slow effects, so even simple improvements become hard to evaluate.
- **Advice:** look for hidden feedback loops and remove them where feasible.

### 3.3 Undeclared consumers
- **The problem:** other systems quietly read model A's predictions (at runtime or from logs) and use them as their inputs.
- **Why it is debt:** every change to A now silently affects them. And if a consumer influences A's own training data (the paper's example: a module that sizes headline fonts starts consuming the CTR prediction, and font size affects whether users click), it creates **another** hidden feedback loop.
- **Mitigation:** access controls, so consumers have to be declared.

### 3.4 Data dependencies cost more than code dependencies
- **Unstable data dependencies:** an input signal that changes over time, for example the output of another ML model, a TF-IDF lookup table, or a signal owned by another team that rolls out "improvements" without telling you. By CACE, an "improvement" upstream can hurt you downstream.
  - **Mitigation:** a **versioned (frozen) copy** of the signal until a new version is vetted. This costs staleness and the work of maintaining versions.
- **Underutilized data dependencies:** features that add little accuracy but still make the system vulnerable.
  - **Legacy features:** added early, later made redundant by newer features, never removed.
  - **Bundled features:** a group was added together under deadline pressure, hiding members that add nothing.
  - **ε-features:** a tiny accuracy gain bought with a lot of complexity.
  - **The paper's example:** after a team merger, both old and new product numbers are kept as features. Old products have both; new ones only the new number. "The machine learning algorithm knows of no reason to reduce its reliance on the old numbers." A year later someone, "acting with good intent," stops populating the old numbers. No regression test notices. "This will not be a good day for the maintainers of the machine learning system."
  - **Mitigation:** regularly evaluate the effect of **removing each feature**, and act on it.
- **Static analysis of data dependencies:** code has compilers and build systems to track who uses what; data usually has nothing. The paper describes an automated feature-management tool (annotations plus dependency checks) that lets one Google team "safely delete thousands of lines of feature-related code per quarter."

### 3.5 Correction cascades
- **The setup:** model a solves problem A. For a slightly different problem A′, it is tempting to learn a small model a′ that takes a's output and **corrects** it. Then maybe a″ on top of a′, and so on.
- **Why it is debt:** a′ depends on the exact behaviour of a. "Improving the accuracy of a actually leads to system-level detriments." The coupled system can get stuck in a poor local optimum where **no component can be improved on its own** (a deadlock).

### 3.6 System-level spaghetti
- **Glue code:** code that pushes data into and out of general-purpose ML packages. "A mature system might end up being (at most) 5% machine learning code and (at least) 95% glue code." The paper argues that re-implementing an algorithm inside the system can be cheaper than gluing to a clumsy API.
- **Pipeline jungles:** data preparation that grew organically into a tangle of scrapes, joins, sampling steps and intermediate files. The fix is a clean-slate redesign, which is expensive but cuts ongoing costs.
- **Dead experimental codepaths:** quick experiments left behind as conditional branches. Each one is a combination nobody tests. (The paper cites Knight Capital, which lost $465 million in 45 minutes partly from obsolete experimental code.)
- **Configuration debt:** "feature A was incorrectly logged from 9/14 to 9/17", "feature D is not available in production, so substitutes D′ and D″ must be used", and so on. The number of lines of configuration can exceed the ML code. Treat configuration like code: assertions, diffs, code review.

### 3.7 Changes in the external world
- **Fixed thresholds in dynamic systems:** a decision threshold (spam / not spam, show the ad or not) is often set by hand to hit a precision or recall target. After the model is retrained, the old threshold no longer gives that target. **Mitigation:** learn thresholds on held-out validation data.
- **Correlations that no longer correlate:** if two features always co-occur but only one is causal, the model may split credit between them. When the world stops making them co-occur, predictions change.
- **Monitoring and testing:** unit tests and end-to-end tests aren't enough in a changing world; live monitoring is critical. Two starting points:
  - **Prediction bias:** the distribution of predicted labels should equal the distribution of observed labels. A null model passes this test too, so it isn't sufficient, but changes in it are "often indicative of an issue that requires attention", such as the world suddenly changing.
  - **Action limits:** a sanity cap on how many real-world actions the system takes; hitting it fires an alert.

### 3.8 The conclusion
- Some debt is reasonable for speed, but it must be recognised and accounted for.
- "Research solutions that provide a tiny accuracy benefit at the cost of massive increases in system complexity are rarely wise practice."

---

## 4. Prediction bias, step by step

Prediction bias is the one monitoring formula in the paper, so here it is by hand.

```
prediction bias = (average predicted probability) − (observed rate of positive labels)
```

**Example.** Last week the model predicted these click probabilities for 4 users: 0.5, 0.4, 0.6, 0.5.
- The average prediction is (0.5 + 0.4 + 0.6 + 0.5) / 4 = 2.0 / 4 = 0.50.
- This week 3 of the 4 users actually clicked, so the observed rate is 3/4 = 0.75.
- Prediction bias = 0.50 − 0.75 = **−0.25**. The model **under-predicts** by 25 points.

A well-calibrated logistic regression has bias ≈ 0 on data like its training data, because at the optimum (with a bias term and no penalty on it) the gradient for b is Σ(pᵢ − yᵢ) = 0, which says exactly that the average prediction equals the average label. So a sudden non-zero bias means **the data changed**, not that the model got unlucky.

---

## 5. How our code turns each warning into an experiment

| Warning | Our toy | What we measure |
|---|---|---|
| CACE | 4 features; x1 is a noisy copy of x0's cause | how much the **other** weights move when we remove x0, add a feature, change L2, or blank out 30% of x0 |
| Legacy feature | 100 products; 70% "old" with both an old and a new one-hot product number | log-loss before and after the old numbers stop being populated, versus a model that only ever used new numbers, plus leave-one-group-out ablation |
| Hidden feedback loop | weekly CTR simulator; inputs are the shown headline's estimated affinity and x_week = clicked last week; the recommender improves in week 3 | CTR, the prediction bias of last week's model on this week's traffic, and the weight on x_week |
| Correction cascade | a v1 uses x0; a′ = logistic correction on a's logit plus x1, x2; a v2 also uses x1 | log-loss of A′ with the old correction on top of a v1 vs a v2 |
| Fixed threshold | threshold chosen for 90% precision on v1; v2 is retrained with positives up-sampled 3× and more L2 | precision of v2 with v1's threshold vs a re-learned threshold |
| Correlations | x_proxy = x_cause + small noise in training; independent in production | accuracy before and after, versus a causal-only model |

---

## 6. What our code found

**CACE** (max change in the weights of the **other** features; mean change in predicted probability):

| Change | Other weights move by | Predictions move by |
|---|---|---|
| remove x0 | **0.714** | 0.124 |
| add x4 (noisy copy of x0) | 0.003 | 0.004 |
| L2 0.001 → 0.05 | 0.248 | 0.050 |
| 30% of x0 missing (set to 0) | 0.340 | 0.036 |

- **Removing x0** made x1, which had weight 0.002 because it was redundant, take over x0's credit.
- **Honest note on "add x4":** the new feature took its credit from x0 itself, which this column doesn't count, so x1..x3 barely moved. Not every change moves every weight; correlated ones do.
- **`experiments.py` E1** (a quick smoke run, 3 seeds): when x0 is removed, the other weights move by 0.15, 0.28, 0.73 and 0.97 as the correlation between x0 and x1 rises through 0, 0.3, 0.7 and 0.95. Even at zero correlation they move by 0.15. That is because logistic regression is non-collapsible: dropping any real cause changes the scale of every other coefficient.

**Legacy feature** (test log-loss):

| Model | Before cleanup | After old numbers stop being populated |
|---|---|---|
| old + new product numbers | 0.485 | **0.527** |
| new product numbers only | 0.497 | 0.497 |

- **"Not a good day" reproduced:** after the cleanup, the model that used both schemes is **worse than a model that never used the old numbers** (0.527 vs 0.497). The new-only model doesn't notice the cleanup at all.
- **Honest note:** leave-one-group-out ablation says the old numbers **help** a little (removing them costs +0.012 log-loss). So in our toy, the paper's suggested mitigation would *not* have flagged them. Their offline value is small but positive; the danger is that their upstream source is unstable. Ablation tells you what a feature is worth, not how risky it is.

**Hidden feedback loop:**

| Week | CTR | Bias of last week's model | Weight on x_week |
|---|---|---|---|
| 0 | 0.527 | — | 0.000 |
| 1 | 0.518 | +0.013 | 0.311 |
| 2 | 0.514 | +0.005 | 0.309 |
| 3 (recommender improves) | 0.628 | **−0.170** | 0.287 |
| 4 | 0.635 | +0.004 | 0.259 |
| 5 | 0.619 | +0.019 | 0.211 |
| 6 | 0.636 | −0.014 | 0.384 |
| 7 | 0.632 | +0.009 | 0.355 |

- **Prediction-bias monitoring catches the change:** in the week the world changes, last week's model under-predicts by 17 points, while normal weeks stay within ±0.02.
- **The x_week weight keeps moving after the improvement** (0.29 → 0.26 → 0.21 → 0.38), because its own input distribution shifted a week later.
- **Honest note:** in our toy the drift is noisy rather than a clean slow trend. Week-to-week sampling noise in a 4,000-user simulation is about as large as the loop effect. The paper's "time scale much longer than a week" needs a richer model of user behaviour to show cleanly.

**Correction cascade** (log-loss):

| System | Log-loss |
|---|---|
| A with a v1 | 0.561 |
| A with a v2 (improved) | **0.492** |
| A′ with a v1 + correction | 0.470 |
| A′ with a v2 + the old correction | **0.529** |

Improving a helps A and **hurts A′** until someone retrains the correction, exactly the "improving a leads to system-level detriments" warning.

**Fixed threshold:**
- v1's threshold (0.820) gives 90.6% precision.
- After retraining (v2, rebalanced), the same threshold gives **77.5%**.
- A threshold re-learned on held-out data (0.905) restores **90.7%**.

**Correlations:**
- The model gave cause and proxy almost equal weights (1.02 and 0.95).
- Accuracy fell from 0.777 to **0.692** when the world decoupled them; a causal-only model kept 0.781.

**`experiments.py`:**
- **E1:** entanglement vs correlation;
- **E2:** legacy features over product counts and old-product share;
- **E3:** feedback-loop bias spikes over improvement sizes and 20 weeks;
- **E4:** 20 seeds of cascade, threshold and correlation damage and their mitigations.
- Only quick smoke runs of E1 and E3 were done here.

---

## 7. Why it matters

- **This paper (and its NeurIPS 2015 sequel, "Hidden Technical Debt in Machine Learning Systems," paper 088)** named the problems that the field of **MLOps** grew up to solve: feature stores and data versioning (unstable dependencies), lineage and dependency tracking (static analysis of data), monitoring of prediction bias and drift (changes in the external world), and the "ML Test Score" checklist (paper 089).
- **The 5% / 95% observation** changed how people talk about ML engineering: most of the work is around the model, not in it.

---

## 8. Check yourself

1. What does CACE stand for, and what does it mean?
<details><summary>Answer</summary>Changing Anything Changes Everything: because the model mixes all inputs together, changing one feature's distribution, adding or removing a feature, or changing a hyper-parameter can change the weights of all the other features and the predictions on many slices.</details>

2. In our CACE experiment, why did x1's weight jump when x0 was removed?
<details><summary>Answer</summary>x1 is correlated with x0's cause. While x0 was present, x1 was redundant (weight 0.002). Without x0, x1 is the best remaining carrier of that information, so the retrained model gives it x0's credit.</details>

3. Explain the hidden feedback loop in the paper's CTR example.
<details><summary>Answer</summary>x_week counts the user's clicks last week. A better model makes users click more, which raises x_week a week later, and the retrained model then changes its weight on x_week. The system's behaviour drifts slowly, over longer than a week, so quick experiments can't measure the true effect of a change.</details>

4. Compute the prediction bias: predictions 0.2, 0.3, 0.1, 0.4; observed labels 0, 1, 0, 0.
<details><summary>Answer</summary>Mean prediction = 1.0 / 4 = 0.25. Observed rate = 1/4 = 0.25. Bias = 0. (A null model predicting 0.25 for everyone would also get 0, which is why bias alone isn't a sufficient test.)</details>

5. Why is a legacy feature dangerous even if it adds almost no accuracy?
<details><summary>Answer</summary>The model still gives it some weight, sharing credit with the features that made it redundant. If the feature's source changes or stops being populated, those predictions shift. The model is vulnerable to a signal nobody watches.</details>

6. In our legacy-feature toy, ablation said the old numbers help (+0.012). What does that teach?
<details><summary>Answer</summary>Leave-one-feature-out ablation measures offline value, not risk. A redundant feature can still look slightly useful, so the decision to remove it also needs to consider how stable its source is.</details>

7. What is a correction cascade, and why does it create a deadlock?
<details><summary>Answer</summary>A model a′ learned on top of model a's output to solve a related problem. a′ depends on a's exact behaviour, so improving a can hurt a′ (in our toy A′'s log-loss rose from 0.470 to 0.529). If many models are chained, no single one can be improved without breaking the others.</details>

8. Why does a fixed precision threshold stop working after retraining, and what is the fix?
<details><summary>Answer</summary>A threshold means a certain precision only for a particular model's score distribution. Retraining (here with rebalanced data and more regularisation) changes the scores, so the same number now gives a different precision (90.6% → 77.5%). The fix is to learn the threshold on held-out validation data each time (back to 90.7%).</details>

9. What fraction of a mature ML system does the paper say may be "machine learning code"?
<details><summary>Answer</summary>At most 5%; at least 95% is glue code. The paper uses this to argue that re-implementing an algorithm inside the system can be cheaper than gluing to a general-purpose package's API.</details>

10. Two features always co-occurred in training, but only one is causal. What happens when the world decouples them?
<details><summary>Answer</summary>The model split credit between them (weights 1.02 and 0.95 in our toy). When they decouple, half its evidence now comes from a feature with no real effect, so accuracy drops (0.777 → 0.692), while a model using only the causal feature keeps its accuracy (0.781).</details>
