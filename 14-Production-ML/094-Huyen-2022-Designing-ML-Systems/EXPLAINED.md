# Designing Machine Learning Systems, explained simply

**Book:** Chip Huyen, *Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications*, O'Reilly, 2022.

**A note on the source:** the book is paid. This folder is built from its **public** table of contents and chapter summaries (github.com/chiphuyen/dmls-book) and the standard definitions of the techniques it covers. Nothing here quotes the book, and the experiments and numbers are our own.

**In one sentence:** an ML system is much more than a model. The book walks through the whole lifecycle: data, labels, features, evaluation, deployment, monitoring, continual learning, testing in production, infrastructure and people. It treats ML engineering as an **iterative process**.

---

## 1. The book at a glance

| Chapter | Topic | Where it's covered in this repo |
|---|---|---|
| 1–2 | ML systems overview; designing an ML system (objectives, requirements, iteration) | 090 (Rules of ML) |
| 3 | Data engineering fundamentals (formats, data models, batch vs stream processing) | — |
| 4 | **Training data:** sampling, labelling, class imbalance, augmentation | **here** |
| 5 | **Feature engineering:** missing values, scaling, hashing, crosses, **data leakage** | **here** |
| 6 | **Model development and offline evaluation:** baselines, calibration, slices, behavioural tests | **here** (also 093) |
| 7 | **Deployment:** batch vs online prediction, edge vs cloud, compression | **here** (compression: 074–078) |
| 8 | Data distribution shifts and monitoring | **092** |
| 9 | **Continual learning and testing in production** | **here** |
| 10 | Infrastructure and tooling for MLOps | 088, 089 |
| 11 | The human side of ML (UX, team structure, responsible AI) | 093 |

---

## 2. Chapter 4: training data

### Sampling
- **Simple random sampling:** every item is equally likely to be picked. A rare group may get very few samples.
- **Stratified sampling:** split the population into groups (strata) and sample from each, e.g. in proportion to its size. This lowers variance when strata differ.
- **Reservoir sampling:** keep a uniform random sample of k items from a stream whose length is unknown.
  - Keep the first k items.
  - For item number i (counting from 0, with i ≥ k), draw j uniformly from 0..i; if j < k, replace slot j.
  - **Why it's uniform:** item i enters with probability k/(i + 1), and then survives each later step m with probability 1 − 1/(m + 1) = m/(m + 1). The product telescopes:
    ```
    k/(i+1) · (i+1)/(i+2) · … · (n−1)/n = k/n
    ```
    So every item is kept with probability k/n.

### Labelling
- **Hand labels** are accurate but slow and expensive.
- **Weak supervision:** write **labelling functions (LFs)**, cheap heuristics such as "contains the word refund → complaint" that vote or abstain. Combine their votes into noisy labels. A **label model** weights each LF by its estimated accuracy:
  ```
  weight_j = log( acc_j / (1 − acc_j) )
  ```
  - **Example:** an LF with 90% accuracy gets weight log 9 = 2.2; one with 60% gets log 1.5 = 0.41. The good LF counts about five times as much.
- **Active learning:** ask humans to label the examples the model is **least sure** about (prediction nearest 0.5). These are the most informative.

### Class imbalance
- **The accuracy paradox:** with 1% positives, "always negative" is 99% accurate and useless.
- **Better metrics:** precision, recall, and **PR-AUC** (area under the precision–recall curve).
  - A random scorer's ROC-AUC is 0.5 at any imbalance.
  - Its PR-AUC equals the **positive rate** (0.01 here), so PR-AUC shows how hard the rare class really is.
- **Fixes:**
  - resample the data, or weight the classes (e.g. positives × 99);
  - or simply **move the threshold**.
  - Weighting changes what the model's probabilities mean: they no longer match real frequencies.

---

## 3. Chapter 5: features and leakage

- **Data leakage:** information about the label sneaks into training or evaluation in a way that won't exist at prediction time. Common forms:
  1. **Preprocessing using all the data:** choosing features, or fitting scalers, before splitting.
  2. **Random splits of time-correlated data** (see Rule 33 in 090).
  3. **Duplicates across train and test.**
- **Why feature selection on all data leaks:**
  - With 5,000 random features and 200 labels, some features correlate with the labels **by chance**.
  - Choosing them using the test rows too means the test rows helped pick the features. The test score then reflects that luck, not real skill.
- **The hashing trick:** map an unbounded set of category values into B buckets with a hash function. This saves memory and handles unseen values, at the cost of **collisions**.
  - With n values and an ideal hash, the chance that a given value shares its bucket with at least one other is
    ```
    1 − (1 − 1/B)^(n − 1)
    ```
  - **Example:** n = 10,000 and B = 2¹⁴ = 16,384: 1 − (1 − 1/16384)^9999 ≈ 1 − e^(−0.61) = **0.46**. Nearly half the values collide with something.

---

## 4. Chapter 6: offline evaluation

- **Baselines:** always compare against random, majority-class and a simple heuristic. A model that barely beats "income > 0" may not be worth its complexity.
- **Calibration:** a model is calibrated if, among the items it gives probability 0.8, about 80% are positive.
  - **Expected calibration error (ECE):** bin the predictions, and take the size-weighted average gap between mean prediction and observed rate:
    ```
    ECE = Σ_bins (n_bin / n) · | mean prediction_bin − observed rate_bin |
    ```
  - **Example:** half the items predicted 0.9 with 40% actually positive, half predicted 0.1 with 0% positive. ECE = 0.5 · 0.5 + 0.5 · 0.1 = **0.30**.
  - **Platt scaling:** fit a small logistic regression p′ = σ(a · logit(p) + b) on held-out data. It fixes calibration without changing the **ranking** (AUC), as long as a > 0.
- **Behavioural tests:**
  - **Invariance test:** changing something that should not matter (a protected attribute, a name) should not change the prediction.
  - **Directional expectation test:** raising income should not lower loan approval.
- **Slice-based evaluation** (see 093): check metrics per group.

---

## 5. Chapter 7: deployment

| | Batch prediction | Online prediction |
|---|---|---|
| When computed | periodically (e.g. nightly) for everyone | on request |
| Features | possibly stale | fresh |
| Cost | computes predictions nobody may request | only for requests, but needs low latency |
| Example | daily recommendations email | search ranking, fraud check at checkout |

Edge vs cloud deployment and model compression (quantisation, pruning, distillation) are covered in papers 074–078.

---

## 6. Chapter 9: continual learning and testing in production

- **Stateless retraining** trains from scratch on a window of data every time. **Stateful training** fine-tunes the previous model on new data only. The second is much cheaper and often just as good.
- **A/B test sample size** for comparing click rates p₀ and p₁ (two-sided, significance α, power 1 − β):
  ```
  n per arm = ( z_{1−α/2} · √(2 p̄ (1 − p̄))  +  z_{1−β} · √(p₀(1 − p₀) + p₁(1 − p₁)) )² / (p₁ − p₀)²
  ```
  - **Example:** p₀ = 0.10, p₁ = 0.11, p̄ = 0.105, z values 1.96 and 0.84:
    - numerator: (1.96 · √0.188 + 0.84 · √0.1879)² = (0.850 + 0.364)² = 1.474;
    - n = 1.474 / 0.0001 ≈ **14,700 users per arm**.
  - Small lifts need lots of traffic.
- **Interleaving** (for rankers): each user sees **one** list that mixes results from rankers A and B (team draft: the rankers take turns picking). Clicks are credited to the ranker that contributed the item. Each user compares both rankers directly, which removes between-user noise, so fewer users are needed.
- **Bandits** (e.g. Thompson sampling) shift traffic to better variants **while** testing, so they lose fewer clicks than a fixed 50/50 test. Thompson sampling keeps a Beta(clicks + 1, non-clicks + 1) belief per arm, draws one sample from each, and shows the arm with the highest draw.
- **Shadow deployment and canary releases** (run the new model silently, or for a small slice of traffic) are covered in 089 (Infra 6).

---

## 7. What our code found

| Technique | Result |
|---|---|
| Reservoir sampling (k = 10 of 100) | each item kept in 9.1–11.0% of 5,000 runs (target 10%) |
| Stratified vs random (rate 0.0695, 1,000 samples) | std of the estimate 0.0070 vs 0.0081: modestly better |
| Weak supervision (6 LFs, 60 features) | majority-vote labels 85.5% correct; classifier on them **0.857** vs **0.747** with 100 hand labels vs 0.866 with all true labels |
| Class imbalance (1.4% positives) | "always negative" 98.6% accurate; model ROC-AUC 0.863 but PR-AUC **0.155** (random 0.014); recall at 0.5: 0.00 → **0.82** with class weights, but the mean prediction jumps 0.018 → 0.344 (calibration broken) |
| Active learning | uncertainty sampling slightly ahead from 60 labels (0.946 vs 0.940 at 100); random ahead at 40. Small gain on an easy problem |
| Leakage: feature selection | **pure-noise labels**: 0.84 test accuracy when features are chosen on all data, 0.43 when chosen on training data only (true value 0.5) |
| Leakage: duplicates | 1-NN accuracy 0.732 with a random split vs 0.557 splitting by item |
| Hashing (10,000 values) | collision share 1.000 / 0.463 / 0.038 for B = 2¹⁰ / 2¹⁴ / 2¹⁸; the formula gives 1.000 / 0.457 / 0.037 |
| Calibration (naive Bayes, 6 correlated copies) | ECE **0.204 → 0.010** with Platt scaling; log-loss 1.163 → 0.533; AUC unchanged (0.807). Platt slope 0.18 ⇒ log-odds about 5.5× too extreme |
| Baselines and behaviour (biased historical loans) | random 0.50, majority 0.51, "income > 0" 0.68, model 0.75; **invariance test fails: 15.9% of decisions flip** when only the protected attribute changes; directional test passes (0% violations) |
| Batch vs online | accuracy 0.729 (nightly batch) vs **0.789** (online, fresh features) |
| A/B sample size (10% → 11%) | 14,751 users per arm |
| A/B vs interleaving | the better ranker detected in 13% vs **23%** of trials (200 users), 27% vs 40% (500), 43% vs 47% (1,000) |
| Bandit vs A/B/n (arms 10 / 11 / 13%) | regret 167 clicks (A/B/n then ship) vs **81** (Thompson); Thompson ends on the best arm in 90% of runs |
| Stateless vs stateful | log-loss 0.5091 vs **0.5024**, with **67× less** compute for stateful |

**Honest notes:**
- **Weak supervision:** our accuracy-weighted vote did not beat plain majority vote (label accuracy 83.4% vs 85.5%). Estimating LF accuracy from agreement with the majority vote is biased upward. Real label models (e.g. Snorkel's) estimate it more carefully.
- **Interleaving:** its advantage in our toy is modest and shrinks with more users. Published industrial results report much larger gains; our click model is simple.
- **Batch vs online:** online computed **more** predictions than batch here (9,714 vs 4,000), because active users return several times a day. Batch is wasteful when most users never show up.
- **`experiments.py`:**
  - **E1:** label budgets;
  - **E2:** imbalance levels;
  - **E3:** active learning;
  - **E4:** interleaving and bandits;
  - **E5:** continual learning.
  - Only an E3 smoke run was done here (2 seeds: random 0.950 vs uncertainty 0.953).

---

## 8. Check yourself

1. Why does reservoir sampling keep every item with probability k/n?
<details><summary>Answer</summary>Item i enters with probability k/(i+1), and at each later step m it survives with probability m/(m+1). The product telescopes to k/n, the same for every item.</details>

2. Two labelling functions are 90% and 70% accurate. What are their log-odds weights?
<details><summary>Answer</summary>log(0.9/0.1) = log 9 ≈ 2.20 and log(0.7/0.3) ≈ 0.85.</details>

3. With 1% positives, what PR-AUC does a random scorer get, and why report PR-AUC?
<details><summary>Answer</summary>About 0.01 (the positive rate). ROC-AUC can look high when negatives are plentiful, while PR-AUC focuses on how well the rare positives are found among the top predictions.</details>

4. Why did choosing features on all the data give 84% accuracy on pure-noise labels?
<details><summary>Answer</summary>With 5,000 random features, some correlate with the labels by chance in this particular sample, including the test rows. Selecting them using the test rows makes the test score reflect that chance correlation. Selecting on training data only gives about 50%, the truth.</details>

5. Compute the collision probability for n = 1,000 values in B = 10,000 buckets.
<details><summary>Answer</summary>1 − (1 − 1/10000)^999 ≈ 1 − e^(−0.0999) ≈ 0.095, so about 9.5%.</details>

6. Predictions: half at 0.8 with 60% observed positives, half at 0.2 with 20% observed. What is the ECE?
<details><summary>Answer</summary>0.5 · |0.8 − 0.6| + 0.5 · |0.2 − 0.2| = 0.1.</details>

7. Why doesn't Platt scaling change AUC?
<details><summary>Answer</summary>It applies the same increasing function σ(a · logit(p) + b) (with a > 0) to every score, so the order of the scores, and hence AUC, is unchanged.</details>

8. What does an invariance test check, and why did our loan model fail it?
<details><summary>Answer</summary>That changing an attribute that shouldn't matter (here the protected attribute) doesn't change predictions. The model was trained on historically biased decisions and used the attribute directly, so flipping it changed 15.9% of decisions.</details>

9. Roughly how many users per arm are needed to detect a CTR change from 10% to 11%?
<details><summary>Answer</summary>About 14,700 (α = 0.05 two-sided, 80% power).</details>

10. Why can stateful training be much cheaper than stateless retraining?
<details><summary>Answer</summary>Stateful training fine-tunes the existing model on only the newest data (a few passes), instead of retraining from scratch on a long window. Our toy needed 67× fewer example-passes for slightly better log-loss.</details>
