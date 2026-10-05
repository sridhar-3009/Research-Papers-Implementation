# The ML Test Score, explained simply

**Paper:** Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (Google), *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction*, IEEE International Conference on Big Data, 2017.

**In one sentence:** the technical-debt papers (087, 088) named the problems; this one gives a **checklist of 28 concrete tests** (7 for data, 7 for the model, 7 for infrastructure, 7 for monitoring) plus a **scoring rule**, so a team can measure how production-ready its ML system is and see what to fix next.

---

## 1. Why ML systems need special tests

- **Ordinary software:** the behaviour is written in code, so you test the code.
- **ML systems:** the behaviour is **learned from data**, so it "cannot be strongly specified a priori".
- **The paper's analogy:** training is like **compilation**, "where the source is both code and training data". So:
  - training data needs testing **like code**;
  - a trained model needs the production practices **of a binary**: debuggability, rollbacks, monitoring.
- **"Doesn't this go without saying?"** In a survey of several dozen teams at Google, **no test was implemented by more than 80% of teams**. Yet most tests had a nonzero score for at least half the teams, so teams do find them worth doing.

Each test is written as an **assertion** ("all features are beneficial"). The advice: check that it's true, "the more frequently the better", and fix the system when it isn't.

---

## 2. The 28 tests

### Data (Table I)
| # | Test | How, in one line |
|---|---|---|
| 1 | Feature expectations are captured in a **schema** | compute statistics from training data, adjust with domain knowledge (an adult is between 1 and 10 feet tall) |
| 2 | **All features are beneficial** | correlations, models with one or two features, or leave-one-feature-out |
| 3 | No feature's **cost** is too much | count latency, RAM, upstream dependencies, instability, not just accuracy |
| 4 | Features adhere to **meta-level requirements** | programmatically enforce rules, e.g. no features from user data, no `age`, nothing deprecated |
| 5 | The data pipeline has appropriate **privacy controls** | control access like raw user data; user-requested **deletions propagate** to training data and models |
| 6 | **New features can be added quickly** | highly efficient teams take as little as 1–2 months, even at global scale; privacy takes precedence |
| 7 | All input **feature code is tested** | unit tests; feature bugs are nearly impossible to detect once in both training and test data |

### Model (Table II)
| # | Test | How |
|---|---|---|
| 1 | Model specs are **reviewed and checked in** | code review and version control, so you know exactly what code made a model |
| 2 | **Offline and online metrics correlate** | small A/B experiments with an **intentionally degraded model** |
| 3 | All **hyperparameters** have been tuned | grid search or smarter search |
| 4 | The impact of **model staleness** is known | A/B tests with older models give an **age-versus-quality curve** |
| 5 | **A simpler model is not better** | compare regularly against a very simple baseline (e.g. linear with few features) |
| 6 | Quality is sufficient on all important **data slices** | release tests with **absolute** thresholds (error on slice x < 5%) and **incremental** ones (change < 1% vs the previous model) |
| 7 | The model is tested for considerations of **inclusion** | check input features for strong correlation with protected categories; slice predictions by user group |

### ML infrastructure (Table III)
| # | Test | How |
|---|---|---|
| 1 | Training is **reproducible** | train twice, get identical models; seed RNGs, fix initialisation order, beware thread ordering; ensembling helps |
| 2 | Model specs are **unit tested** | random input data plus **one step of gradient descent**; restore from a checkpoint; check that loss decreases; deliberately overfit. Avoid "golden tests" |
| 3 | The full ML pipeline is **integration tested** | an automated end-to-end run, with fast subset versions for quick feedback |
| 4 | Model quality is **validated before serving** | automatically bless or veto: loose thresholds for slow degradation, tight comparison with the previous version for sudden drops |
| 5 | The model is **debuggable** | feed one example and inspect every step of the computation |
| 6 | Models are **canaried** | check the model loads in the production serving binary (Figure 2: a model needing op v0.2 on a server with only v0.1), then ramp up traffic gradually |
| 7 | Serving models can be **rolled back** | practise rolling back in normal times, not only in emergencies |

### Monitoring (Table IV)
| # | Test | How |
|---|---|---|
| 1 | **Dependency changes** result in notification | subscribe to upstream announcement lists; make sure upstream knows you use their data |
| 2 | **Data invariants** hold for inputs | check serving inputs against the Data 1 schema; tune alert thresholds |
| 3 | Training and serving are **not skewed** | log serving examples with ids, recompute features with the training code, count skewed features and examples |
| 4 | Models are not too **stale** | alert on model age (and the age of each pipeline stage and lookup table), using Model 4's curve |
| 5 | Models are **numerically stable** | alert on the first NaN or infinity, implausible weights, too many dead ReLUs |
| 6 | **Computing performance** has not regressed | training speed, latency, throughput, RAM; catch dramatic jumps **and slow leaks** |
| 7 | **Prediction quality** has not regressed | prediction bias in aggregate and on slices ("90% of predictions of probability 0.9 should in fact be positive"), fast labels, human raters |

---

## 3. The scoring rule, step by step

For each of the 28 tests:
- **0 points** if it isn't done;
- **½ point** if it's run **manually**, with the results documented and distributed;
- **1 point** if a system runs it **automatically**, on a repeated basis.

Then:
1. Add up the points **within each section** (so each section scores between 0 and 7).
2. The **ML Test Score is the minimum** of the four section sums.

**Why the minimum?** "We believe all four sections are important, and so a system must consider all in order to raise the score." Great infrastructure tests can't compensate for having no monitoring.

**Worked example.** A team has:
- **Data:** 3 tests automated, 2 manual, 2 missing → 3·1 + 2·0.5 = **4.0**
- **Model:** 2 automated, 2 manual, 3 missing → 2 + 1 = **3.0**
- **Infra:** 5 automated → **5.0**
- **Monitoring:** 3 manual, 4 missing → 3·0.5 = **1.5**

ML Test Score = min(4.0, 3.0, 5.0, 1.5) = **1.5**, which Table V reads as "first pass at basic productionization".

**Table V: what the score means**

| Score | Meaning |
|---|---|
| 0 | More of a research project than a productionized system |
| (0, 1] | Not totally untested, but there may be serious holes in reliability |
| (1, 2] | First pass at basic productionization; more investment may be needed |
| (2, 3] | Reasonably tested; more tests and procedures could be automated |
| (3, 5] | Strong automated testing and monitoring, appropriate for mission-critical systems |
| > 5 | Exceptional levels of automated testing and monitoring |

All tests are worth the same, on purpose: priorities differ by team, and "choosing any test to implement will raise the score."

---

## 4. What the paper learned from 36 teams

- **Checklists help even experts:**
  - one team found a **thousand-line, completely untested** feature file;
  - another had **no way to detect poor predictions in a single country**;
  - a speech team said their system couldn't be biased because "we just get vectors of numbers", until asked about African American Vernacular English and the diversity of their human raters.
- **Dependency issues:** teams assumed a bigger upstream (or downstream) system's monitoring would catch their problems. But a small system's errors can be masked in the noise of a big one.
- **Frameworks matter:**
  - **integration testing (Infra 3)** had much lower adoption than most tests, because training is often ad hoc scripts;
  - **training/serving skew (Monitor 3)** is "perhaps the most important and least implemented test";
  - canarying was common mainly where the release framework made it easy;
  - a hypothetical system using **TFX** with its standard recommendations already scored "reasonably tested".
- **Limitations:** image and audio teams found many Data tests less applicable, and teams with expensive labels struggle with Model 4 and Infra 4.

(Figure 4 of the paper shows the average score per test across the 36 systems. Those numbers are only in the figure image, so we don't quote them.)

---

## 5. How our code implements it

`mltestscore.py` contains:
1. **The rubric (`RUBRIC`) and the scoring rule (`ml_test_score`)**, including Table V.
2. **A toy production ML system:**
   - users from 4 countries (NG is small and behaves differently);
   - features: spend (heavy-tailed dollars, log-transformed), visits, tenure, a useless `noise`, and a forbidden `age`;
   - the effect of visits **drifts over time**;
   - the spend feature is computed by **separate training and serving code**;
   - a model registry, and a server that supports only op version 1.
3. **Automated versions of 26 of the 28 tests.** Data 6 (speed of adding features) and Model 1 (code review) are processes, so they are scored as manual.
4. **`run_suite(bugs=...)`,** which runs everything and can inject 11 kinds of production bug.

### Two pieces of math used by the tests

**Log-loss** (the offline metric): −[y log p + (1 − y) log(1 − p)], averaged. A perfect model gives 0; always predicting 0.5 gives log 2 ≈ 0.693.

**Pearson correlation** (Model 2): for paired lists a and b,
```
r = Σ (aᵢ − ā)(bᵢ − b̄) / sqrt( Σ (aᵢ − ā)² · Σ (bᵢ − b̄)² )
```
- r = −1 means a perfect inverse straight-line relationship.
- For us, a model with **higher** log-loss (worse) should give a **lower** online buy rate, so we want r close to −1.

**Small example.** Offline log-losses (0.57, 0.65, 0.78) with buy rates (0.73, 0.69, 0.62):
- means: ā = 0.667, b̄ = 0.68;
- deviations: a: (−0.097, −0.017, +0.113); b: (+0.05, +0.01, −0.06);
- numerator: −0.00485 − 0.00017 − 0.00678 = −0.0118;
- denominator: √[(0.0094 + 0.0003 + 0.0128) · (0.0025 + 0.0001 + 0.0036)] = √(0.0225 · 0.0062) = 0.0118;
- **r ≈ −1.0**: the offline metric tracks the online one.

---

## 6. What our code found

**The clean toy system:**

| Section | Result |
|---|---|
| Data | schema, policy, privacy and feature-code tests pass. **Data 2 / Data 3 fail: `noise` adds no value** but costs 1.5 ms. Data 6 is manual |
| Model | offline/online correlation **−0.97**; L2 already optimal; staleness curve known; full model 0.579 vs one-feature baseline 0.652. **Model 6 fails: NG log-loss 0.686 > 0.62** while the global log-loss is 0.579; inclusion check passes |
| Infra | all 7 pass (reproducible, unit tests, integration, blessed, explainable, canary, rollback) |
| Monitoring | all 7 pass |

- **Score:** section sums Data 6.5, Model 6.5, Infra 7, Monitoring 7 → **ML Test Score 6.5, "exceptional"**.
- **Honest caveats:**
  1. The rubric scores **having** a test, not passing it. Our suite gets full points while exposing two real problems: the useless feature, and the bad slice, which is the paper's "global accuracy improved but one country dropped" story.
  2. "Automated" here means a script a scheduler **could** run repeatedly. Nothing in this folder runs on a schedule, so by the paper's strict definition a real team would only claim these points once the scheduling exists.

**The bug-injection matrix** (which tests newly fail):

| Injected bug | Caught by |
|---|---|
| serving code reads cents (training/serving skew) | Infra 6, **Monitor 3**, Monitor 7 |
| upstream sends spend in cents | **Data 1**, Infra 6, **Monitor 2**, Monitor 7 |
| forbidden `age` feature added | **Data 4**, Model 7 |
| fresh random seed every training run | **Infra 1** |
| serving model 40 weeks old | **Monitor 4** |
| model needs op v2, server has v1 | **Infra 6** (canary refuses to load, as in Figure 2) |
| learning rate 1e6 (training diverges) | 12 tests, including **Monitor 5** (first non-finite weight) and Infra 2 |
| deleted users still in training data | **Data 5** |
| an upstream dependency upgraded | **Monitor 1** |
| latency creeps up 60% over 40 days | **Monitor 6** (slow-leak alarm) |
| degraded candidate model | **Infra 4** (vetoed), Infra 6, Monitor 7 |

- **Every bug is caught by at least one test, usually by several layers.**
- **Two tests tell apart bugs that look alike:**
  - a units bug in the **serving code** trips Monitor 3 (skew) but not the schema test;
  - the same change made **upstream** trips the schema tests but **not** Monitor 3, because training and serving code still agree with each other.
- **Honest note:** in the divergence case Infra 1 also "fails". NaN ≠ NaN, so two diverged runs never compare equal. It's a side effect, not a reproducibility bug.

**Staleness curve** (Model 4):

| Age (weeks) | Stale model log-loss | Fresh model log-loss | Gap |
|---|---|---|---|
| 0 | 0.578 | 0.577 | +0.000 |
| 4 | 0.560 | 0.556 | +0.004 |
| 12 | 0.538 | 0.502 | +0.035 |
| 26 | 0.512 | 0.388 | +0.124 |
| 52 | 0.511 | 0.272 | +0.238 |

- **The gap is the signal, not the stale model's own number.** The stale model's log-loss *falls* with age because, in our toy world, buying gets more predictable over time (the visits effect grows). Looking only at the deployed model's log-loss would suggest things are improving.
- **From the gap,** about 4 weeks of staleness is tolerable (gap < 0.01), and that becomes Monitor 4's alert threshold.

**Calibration (Monitor 7):** predicted vs observed in each probability bin: 0.136 / 0.130, 0.300 / 0.306, 0.499 / 0.497, 0.694 / 0.685, 0.866 / 0.860.

**`experiments.py`:**
- **E1:** detection rate vs bug size;
- **E2:** false alarms on healthy seeds;
- **E3:** threshold trade-offs;
- **E4:** staleness at three drift speeds.
- Only quick smoke runs of E2 (5 healthy seeds, no false alarms) and E4 were done here.

---

## 7. Why it matters

- **It turned "technical debt" into a checklist** a team can act on and a number it can improve.
- **Its tests map onto later tooling:**
  - **TFX** and TensorFlow Data Validation (schemas, skew detection);
  - **model registries** with blessing and rollback;
  - **canary deployments;**
  - **ML monitoring products.**
- **Training/serving skew,** "perhaps the most important and least implemented test", is a core motivation for **feature stores**, which compute a feature once for both training and serving.

---

## 8. Check yourself

1. How is the final ML Test Score computed?
<details><summary>Answer</summary>Each test earns 0, ½ (manual, documented and distributed) or 1 (automated, repeated). Sum within each of the 4 sections; the final score is the minimum of the 4 sums.</details>

2. A team scores Data 5, Model 4.5, Infra 2, Monitoring 6. What is its score, and what does it mean?
<details><summary>Answer</summary>min = 2, which is in (1, 2]: "first pass at basic productionization, but additional investment may be needed". Infra is the area to improve.</details>

3. Why the minimum rather than the sum?
<details><summary>Answer</summary>All four areas matter. With a sum, a team could ignore monitoring entirely and still score well; the minimum forces attention on the weakest area.</details>

4. What is training/serving skew, and how does Monitor 3 detect it?
<details><summary>Answer</summary>The same feature computed differently at training and serving time (e.g. a separately optimised serving path). Log serving examples with ids, recompute their features with the training code, and count the features and examples whose values differ.</details>

5. How does the paper suggest checking that offline metrics match online impact (Model 2)?
<details><summary>Answer</summary>Run small A/B experiments with an intentionally degraded model and measure how much the online metric drops as the offline metric gets worse.</details>

6. Give the paper's quick unit test for a model specification (Infra 2).
<details><summary>Answer</summary>Generate random input data and train for a single step of gradient descent. It catches many library mistakes quickly. Also: restore from a checkpoint, check that loss decreases over a few steps, or deliberately overfit a tiny dataset.</details>

7. What does a canary protect against that offline tests can't (Infra 6)?
<details><summary>Answer</summary>Mismatches between the model and the serving infrastructure (e.g. a model needing an op version the old server lacks, Figure 2), and non-stationarity in live traffic. Traffic is ramped up gradually while behaviour is checked.</details>

8. What two kinds of thresholds does the paper suggest for slice-quality release tests?
<details><summary>Answer</summary>Absolute (e.g. error on slice x must be below 5%) to catch large drops, and incremental (e.g. the change in error on slice x must be below 1% vs the previously released model).</details>

9. In our staleness table the stale model's log-loss improved from 0.578 to 0.511. Why is it still stale?
<details><summary>Answer</summary>The world became easier to predict, so every model's loss fell. A freshly trained model reaches 0.272 at 52 weeks, so the stale model is 0.238 worse than it could be. Model 4 measures that gap, not the stale model's raw number.</details>

10. Which two tests did the paper find had particularly low adoption?
<details><summary>Answer</summary>Integration testing of the full pipeline (Infra 3), and training/serving skew monitoring (Monitor 3), which the paper calls perhaps the most important and least implemented test.</details>
