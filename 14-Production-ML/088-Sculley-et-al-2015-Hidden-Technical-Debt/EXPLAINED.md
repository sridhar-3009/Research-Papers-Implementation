# Hidden Technical Debt in ML Systems, explained simply

**Paper:** D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (Google), *Hidden Technical Debt in Machine Learning Systems*, NIPS 2015.

**In one sentence:** "developing and deploying ML systems is relatively fast and cheap, but maintaining them over time is difficult and expensive". This paper is the expanded, main-conference version of the 2014 workshop paper ([087](../087-Sculley-et-al-2014-High-Interest-Credit-Card-Technical-Debt/)). It is famous for **Figure 1**, where the ML code is a tiny black box inside a huge amount of surrounding infrastructure.

Like 087, this paper has no equations and no experiments. Everything 087 already covers (CACE, legacy features, correction cascades, glue code, pipeline jungles, fixed thresholds, prediction bias) is only summarised here. This folder implements what is **new in 2015**:
- direct vs hidden feedback loops;
- configuration debt and its six principles;
- ML "smells" (plain-old-data type, multiple-language, prototype) and abstraction debt;
- monitoring of up-stream producers and action limits;
- data-testing, reproducibility, process-management and cultural debt;
- the "measuring debt" questions.

---

## 1. What carries over from 087 (short recap)

| Debt | One line |
|---|---|
| Entanglement / CACE | changing any input, feature or hyper-parameter changes everything |
| Correction cascades | a model learned on top of another's output: improving the base can hurt the system |
| Undeclared consumers | other systems silently use your predictions |
| Unstable / underutilized data dependencies | legacy, bundled, ε-features; in 2015 also **correlated features** ("credit the two features equally, or may even pick the non-causal one"); fix with "exhaustive leave-one-feature-out evaluations" run regularly |
| Static analysis of data dependencies | annotate features and resolve dependency trees automatically |
| Glue code, pipeline jungles, dead experimental codepaths | "(at most) 5% machine learning code and (at least) 95% glue code" |
| Fixed thresholds | learn them on held-out data |
| Prediction bias | the distribution of predicted labels should equal that of observed labels; slice it |

---

## 2. Feedback loops, now two kinds

"Live ML systems … often end up influencing their own behavior if they update over time." The paper calls the result **analysis debt**: it is hard to predict what a model will do before it is released.

### 2.1 Direct feedback loops
- **What it is:** a model influences the selection of its **own future training data**. A recommender shows item X, so it only ever gets click data about item X.
- **The theoretically correct fix:** **bandit algorithms**, which balance exploiting what looks best with exploring what is uncertain. The paper notes they "do not necessarily scale well" to real action spaces.
- **Practical mitigations:** "some amount of randomization", or "isolating certain parts of data from being influenced by a given model".

**A small worked example of the trap.** Two items with true click rates 10% and 8%.
- In the launch log, item 1 was shown 30 times and got 2 clicks, an estimate of 6.7%. Item 2 was shown 30 times and got 3 clicks, an estimate of 10%.
- A greedy model now always shows item 2, so item 1 **never gets another view**, and its 6.7% estimate is never corrected.
- The model will keep believing item 2 is better forever, even though it's wrong. Its own choices starve it of the data that would prove it wrong.

**How a bandit fixes it: UCB (Upper Confidence Bound).** Instead of showing the item with the highest estimate p̂, show the one with the highest **optimistic** estimate:
```
score = p̂ + sqrt(2 · p̂ · ln t / n)
```
- p̂ is the estimated click rate;
- n is how many times the item was shown;
- t is the round number.

The second term is a confidence width: it is large when n is small. **Example** at round t = 100 (ln 100 = 4.6):
- item 1: p̂ = 0.067, n = 30 → width = √(2 · 0.067 · 4.6 / 30) = √0.0205 = 0.143, score 0.210;
- item 2: p̂ = 0.10, n = 500 → width = √(2 · 0.10 · 4.6 / 500) = √0.00184 = 0.043, score 0.143.

UCB shows item 1 again because its estimate is still uncertain, and the estimate gets corrected. (Our version scales the width by p̂ because click rates are small. That is a common variance-aware variant; textbook UCB1 uses √(2 ln t / n).)

### 2.2 Hidden feedback loops between systems
- **What it is:** "two systems influence each other indirectly through the world."
- **The paper's examples:**
  - one system picks the **products** shown on a page, another picks the related **reviews**. Improving one changes how users click on the other;
  - two stock-market prediction models from **different companies**: "improvements (or, more scarily, bugs) in one may influence the bidding and buying behavior of the other."
- **Why it is the harder case:** these loops can exist "between completely disjoint systems", where no code, data pipeline or team connects them.

---

## 3. Anti-patterns new in 2015

### Abstraction debt
- There is "a distinct lack of strong abstractions to support ML systems". Nothing compares to the relational database. What is the right interface for "a stream of data, or a model, or a prediction"?
- Map-Reduce was widely used for ML, but there is broad agreement it is "a poor abstraction for iterative ML algorithms". The parameter server "seems much more robust", but has competing specifications.

### Common smells
A *smell* is a hint that something may be wrong, not a hard rule.
- **Plain-old-data type smell:** information is passed around as raw floats and integers. "A model parameter should know if it is a log-odds multiplier or a decision threshold, and a prediction should know … about the model that produced it and how it should be consumed."
- **Multiple-language smell:** each extra language raises testing cost and makes ownership harder to transfer.
- **Prototype smell:** relying on a separate prototyping environment suggests the real system is hard to change. There is also a danger that the prototype becomes production. "Results found at small scale rarely reflect the reality at full scale."

### Why log-odds vs probability matters (step by step)
- A model's raw score is a **log-odds** z. The probability is p = σ(z) = 1 / (1 + e^(−z)).
- **Example:** z = 1.2 gives p = 1 / (1 + e^(−1.2)) = 1 / (1 + 0.301) = 0.769.
- A consumer with the rule "act if score > 0.8" means p > 0.8, so it should **not** act on this item.
- If the producer quietly starts sending z instead of p, then 1.2 > 0.8 and the consumer **acts**. Nothing crashes; decisions just change.
- The two thresholds aren't equivalent: p > 0.8 is the same as z > ln(0.8 / 0.2) = ln 4 = 1.386. Every item with 0.8 < z < 1.386 flips.

---

## 4. Configuration debt

The paper's examples of messy configuration:
- "Feature A was incorrectly logged from 9/14 to 9/17."
- "Feature B is not available on data before 10/7."
- "Feature D is not available in production, so substitute features D′ and D″ must be used."
- "If feature Z is used, then jobs for training must be given extra memory."
- "Feature Q precludes the use of feature R because of latency constraints."

**The six principles of a good configuration system:**
1. It should be easy to specify a configuration as a **small change** from a previous one.
2. It should be **hard to make manual errors**, omissions or oversights.
3. It should be easy to **see the difference** between two models' configurations.
4. It should be easy to **automatically assert and verify** basic facts: number of features, transitive closure of data dependencies, and so on.
5. It should be possible to **detect unused or redundant settings**.
6. Configurations should undergo **full code review** and be checked into a repository.

**What "transitive closure of data dependencies" means.** If feature D is computed from `offline_join`, and `offline_join` is built from `raw_logs` and `geo_db`, then D depends on all three, directly or indirectly. The closure is everything upstream. It answers "if `geo_db` goes down, which models break?"

---

## 5. Dealing with changes in the external world

The monitoring list grows to three items:
- **Prediction bias:** as in 087, and slicing it "isolate[s] issues quickly".
- **Action limits:** for systems that act (bidding, marking spam), set a sanity limit on how often they act; hitting it fires alerts.
- **Up-stream producers (new):** the data producers feeding a model should be "thoroughly monitored, tested, and routinely meet a service level objective". Their alerts must reach the ML system's control plane, and the ML system's own failures must be propagated to its consumers.
- **Automated response:** "Because external changes occur in real-time, response must also occur in real-time." Paging a human is brittle for time-sensitive issues.

### Sliced prediction bias, by hand
Say a model serves two countries with 900 and 100 users.
- **Country 1:** predictions average 0.30; observed rate 0.30. Bias 0.
- **Country 2:** an upstream bug makes predictions average 0.50, but the observed rate is 0.30. Bias **+0.20**.
- **Overall:** (900 · 0.30 + 100 · 0.50) / 1000 = 0.32, against an observed 0.30, so the overall bias is only **+0.02**. That is easy to miss.
- **Sliced by country,** the problem is obvious: +0.20 in country 2.

---

## 6. Other areas of debt

| Debt | What the paper says |
|---|---|
| **Data testing** | "If data replaces code in ML systems, and code should be tested," then input data needs tests too: sanity checks and monitoring of input distributions |
| **Reproducibility** | strict reproducibility is hard because of randomized algorithms, non-determinism in parallel learning, reliance on initial conditions, and interactions with the world |
| **Process management** | mature systems run dozens or hundreds of models: updating many configs safely, assigning resources by priority, seeing blockages in data flow, recovering from incidents; avoid processes with many manual steps |
| **Cultural** | reward deleting features, reducing complexity, and improving reproducibility, stability and monitoring as much as accuracy; this works best in heterogeneous teams with both research and engineering strengths |

### Why floating-point order matters (reproducibility)
- Computers round every addition. In float32, adding a tiny number to a huge one can lose the tiny number entirely: 100,000,000 + 1 = 100,000,000 in float32, which has about 7 significant digits.
- So (a + b) + c and a + (b + c) can differ. A parallel sum on 8 GPUs adds in a different order than one CPU, and the result differs in the last digits.
- Training multiplies these tiny differences over millions of steps.

---

## 7. Measuring debt

The paper admits that technical debt "does not provide a strict metric". "Simply noting that a team is still able to move quickly is not in itself evidence of low debt." It suggests these questions instead:
1. How easily can an entirely new algorithmic approach be tested at full scale?
2. What is the transitive closure of all data dependencies?
3. How precisely can the impact of a new change to the system be measured?
4. Does improving one model or signal degrade others?
5. How quickly can new members of the team be brought up to speed?

---

## 8. What our code found

**Direct feedback loop** (10 items with true CTRs between 2% and 12%, a 30-view-per-item launch log, 1,000 rounds of 20 users, 20 seeds):

| Policy | Mean CTR | Best possible | Rounds on the best item | Seeds stuck (< 10% on best) |
|---|---|---|---|---|
| Greedy supervised | 0.1067 | 0.1128 | 0.25 | **75%** |
| ε-greedy (ε = 0.1) | 0.1064 | 0.1128 | 0.57 | 30% |
| UCB bandit | 0.1065 | 0.1128 | 0.55 | **5%** |
| Greedy + 10% isolated random slice | 0.1036 | 0.1128 | 0.52 | 40% |

- **Greedy locks in:** it gets stuck on a non-best item in three seeds out of four. The bandit almost never does.
- **Honest note:** the **CTR** cost of greedy is tiny here (0.1067 vs 0.1065), because several items are nearly as good as the best. The real debt is what the system **can no longer learn**. Its estimates of unshown items are frozen, so if the world changed, it would never notice. The isolated random slice costs CTR directly (0.1036), because some users see random items.

**Hidden loop between two systems** (B's code and data never change; only A's product quality does):

| A's product quality | B's learned review weight | B's A/B lift from a top review | Page CTR |
|---|---|---|---|
| −1 | 0.930 | +0.095 | 0.056 |
| 0 | 0.546 | +0.131 | 0.160 |
| +1 | 0.271 | +0.104 | 0.397 |

- When A improves, B's model changes a lot (0.93 → 0.27), because users who see better products need less persuasion from reviews.
- B's measured A/B win also changes, so B's past experiment results go stale when another team ships.

**Configuration system:**
- **Inheritance and diffs:** configs are small diffs from a parent ("features+" / "features−"), with visual diffs.
- **Validation caught every one of the paper's examples:**
  - feature A's bad dates inside the training window;
  - feature B with no data before 10/7;
  - feature D unavailable in serving (suggests D1 + D2);
  - Q precludes R;
  - Z needs 32 GB on an 8 GB job;
  - a deprecated old product ID;
  - a misspelled setting `learing_rate`, caught as unused.
- **Dependency queries:**
  - the closure for experiment B is {click_logs, geo_db, new_logs, offline_join, raw_logs};
  - turning off `geo_db` breaks all 5 configs;
  - turning off `legacy_db` breaks only exp_D.

**Plain-old-data smell:**
- Switching the producer from probabilities to log-odds made the consumer's "score > 0.8" act on 35.0% of items instead of 24.9%; **10.1% of decisions silently flipped**.
- With typed `LogOdds` / `Probability` values, the same comparison raises a `TypeError`.

**Monitoring** (model trained on 4 countries; feature 0 of country IN broken by its producer):

| Producer state | Overall bias | IN bias | Data tests |
|---|---|---|---|
| healthy | +0.001 | −0.001 | pass |
| IN stops populating (zeros) | **−0.005** | **−0.025** | zero rate +24.9% |
| IN switches units (×100) | +0.022 | **+0.081** | 24.5% out of range; mean shifted 0.19 std |

- **Overall bias can hide a broken slice;** sliced bias and data tests catch it.
- **Action limit:** a spam marker at threshold 0.5 would act on 38% of messages, which trips a 25% limit; at 0.9 it acts on 5.1%, which is fine.

**Reproducibility:**
- The same 20,000 float32 numbers summed sequentially vs pairwise give 52766.5586 vs 52766.5625.
- SGD with the same shuffle seed is bit-identical; a different seed moves the weights by up to **0.58**.

**Glue code (Figure 1):** in our own tiny end-to-end pipeline (ingest → validate → features → train → threshold → serve → monitor), the learner and the training call are **13 of 67 lines (19%)**, even with no real infrastructure.

**`experiments.py`:**
- **E1:** policies × launch-log size × 50 seeds;
- **E2:** A's quality from −2 to 2;
- **E3:** detection curves for overall bias vs sliced bias vs data tests;
- **E4:** reproducibility spreads.
- Only quick smoke runs of E3 and E4 were done here. In E3 the overall bias **never** fired (0 of 3 seeds) for any bug size, while sliced bias and data tests fired in 67–100% of seeds.

---

## 9. Why it matters

- **Figure 1 is one of the most reproduced diagrams in ML engineering.** The paper is the founding text of **MLOps**. Feature stores, data validation (TFX Data Validation, Great Expectations), model registries, config systems, experiment tracking and production monitoring all address debts named here.
- **Its "measuring debt" questions led directly to Google's ML Test Score rubric** (paper 089).

---

## 10. Check yourself

1. What is the difference between a direct and a hidden feedback loop?
<details><summary>Answer</summary>Direct: a model influences its own future training data (it only sees outcomes of what it chose). Hidden: two systems influence each other indirectly through the world, e.g. a product picker and a review picker on the same page, possibly owned by different teams with no shared code.</details>

2. Why can a greedy recommender trained on its own logs stay wrong forever?
<details><summary>Answer</summary>It only shows the item it currently believes is best, so items it underestimates are never shown again and their estimates are never corrected. Its own choices remove the data that would fix its mistake.</details>

3. Compute the UCB score for p̂ = 0.05, n = 20 at t = 100 (ln 100 ≈ 4.6), using p̂ + √(2 p̂ ln t / n).
<details><summary>Answer</summary>√(2 · 0.05 · 4.6 / 20) = √0.023 = 0.152, so the score is 0.05 + 0.152 = 0.202.</details>

4. What does the paper offer as practical mitigations for direct loops when bandits don't scale?
<details><summary>Answer</summary>Some amount of randomization, or isolating certain parts of the data from being influenced by the model (e.g. a random slice of traffic).</details>

5. A probability threshold is 0.8. What is the equivalent log-odds threshold?
<details><summary>Answer</summary>ln(0.8 / 0.2) = ln 4 ≈ 1.386. Comparing a log-odds score with 0.8 instead flips every item with 0.8 < z < 1.386.</details>

6. Name four of the six configuration principles.
<details><summary>Answer</summary>Specify a config as a small change from a previous one; make manual errors hard; make diffs between two models' configs easy to see; automatically assert basic facts (number of features, transitive closure of dependencies); detect unused or redundant settings; full code review and version control.</details>

7. Overall prediction bias is +0.02. Why might that hide a serious problem?
<details><summary>Answer</summary>A small slice can be badly broken while the large healthy slices dilute it. E.g. 10% of traffic with bias +0.20 contributes only +0.02 overall. Slicing by country, device and so on reveals it.</details>

8. Why does summing the same numbers in a different order give a different float32 result?
<details><summary>Answer</summary>Each addition rounds to about 7 significant digits, so addition isn't associative: (a + b) + c can differ from a + (b + c). Parallel hardware sums in a different order than sequential code.</details>

9. What is cultural debt, and what does the paper recommend?
<details><summary>Answer</summary>A hard line between research and engineering that rewards only accuracy gains. Teams should reward deleting features, reducing complexity, and improving reproducibility, stability and monitoring as much as accuracy. Heterogeneous teams with both strengths do this best.</details>

10. In our two-system toy, why did B's learned review weight fall when A improved?
<details><summary>Answer</summary>Users who see better products are already likely to buy, so the review shown next to the product sways them less (the interaction term). B's data, and therefore B's model and its measured A/B gain, changed even though B's code didn't.</details>
