# Bayesian Clustered Tensor Factorization, explained simply

**Paper:** Ilya Sutskever (Toronto), Ruslan Salakhutdinov (MIT), Joshua B. Tenenbaum (MIT), *Modelling Relational Data using Bayesian Clustered Tensor Factorization*, NIPS 2009. ([proceedings page](https://proceedings.neurips.cc/paper_files/paper/2009/hash/5705e1164a8394aace6018e27d20d237-Abstract.html))

**In one sentence:** to model a database of facts "(object, relation, object) is true or false", combine two kinds of model:
- a **tensor factorization**, which gives every object vectors and every relation a matrix, and predicts well;
- a **clustering**, which groups similar objects and similar relations, and is easy to interpret.

BCTF clusters the *vectors*, does full Bayesian inference, and gets both: good predictions (especially when data are sparse) and readable clusters.

---

## 1. Relational data

- **A fact is a triple** (a, r, b) with a truth value: (cup, can-contain, coffee) = true.
- **Everything is a big 3-D table (tensor)** of truth values: objects × relations × objects.
- **Most entries are unknown,** and the task is to predict them.
- **The paper's datasets:**
  - **Animals:** 50 animals × 85 attributes;
  - **Kinship:** 104 people, 26 kinship relations;
  - **UML:** 135 medical terms, 49 relations;
  - **MovieLens:** 1,000,209 ratings;
  - **ConceptNet:** 7,000 objects, 19 relations, 82,062 true common-sense facts.

---

## 2. The factorization part

- **Each object a gets two d-dimensional vectors:** a_L (when it is the *left* argument) and a_R (when it is the *right* one).
- **Each relation r gets a d × d matrix** R.
- **The model's guess** for a triple is
  ```
  t(a, r, b) ≈ a_Lᵀ R b_R        and the data are modelled as  t ~ N(a_Lᵀ R b_R, σ²)
  ```
- **The Gaussian likelihood** is a poor fit for 0/1 data, but it is chosen because it makes the conditionals Gaussian, so Gibbs sampling is easy.

**Worked example (d = 2):**
- a_L = (1, 0), b_R = (0.5, 1), R = [[2, 0], [1, 3]];
- R b_R = (2 · 0.5 + 0, 1 · 0.5 + 3 · 1) = (1, 3.5);
- a_Lᵀ (R b_R) = 1 · 1 + 0 · 3.5 = **1.0**: a strongly "true" prediction (the data are centred, so 0 is the average).

**Why two vectors per object?** With only a_L given everything else, the conditional distribution of the a_L's factorises over objects. So all of them can be sampled at once (even in parallel). With one shared vector, each object's vector would appear on both sides of the product, which would require slower sequential sampling.

---

## 3. The clustering part

- **A Chinese Restaurant Process (CRP) partitions the objects** (and separately the relations) into clusters. The number of clusters is not fixed in advance.
  - **The rule:** object number n + 1 joins an existing cluster with probability ∝ its size, or starts a new one with probability ∝ α_DP.
  - **Example:** with clusters of sizes 5 and 3 and α_DP = 1, the probabilities are 5/9, 3/9 and 1/9 (new cluster).
- **Each cluster has its own mean μ and diagonal variance Σ,** drawn from a Normal–Inverse-Gamma prior (μ | Σ ~ N(0, Σ), each σ² ~ IG(α, 1)). Members' vectors (a_L, a_R) are drawn from N(μ, Σ).
- **So objects in the same cluster get *similar* vectors:**
  - predictions depend mostly on which clusters a, r, b are in, which is interpretable;
  - but each object can still differ a little, which helps prediction.

---

## 4. Inference

1. **MAP first:** fit the factorization by conjugate gradient (penalised least squares), ignoring clusters, and use it to initialise.
2. **Then MCMC.** Each sweep:
   - **cluster assignments:** collapsed Gibbs sampling (the paper adds split-merge moves);
   - **cluster means and variances:** sampled from their conjugate posterior;
   - **every a_L, then every a_R:** exact Gaussian draws (Bayesian linear regression for each object);
   - **every R:** the paper uses hybrid Monte Carlo, because d² = 400 or 1,600 dimensions makes exact sampling slow;
   - **hyperparameters.**
3. **Predict** by averaging a_Lᵀ R b_R over the samples.

**The Gaussian conditional, by hand (one dimension).**
- **Setup:** an object's a_L is a single number v with prior N(μ = 0, s² = 1). Two observations t = v · x: (x = 2, t = 1.2) and (x = 1, t = 0.4), with noise variance σ² = 0.5.
- **Posterior precision** = 1/s² + Σx²/σ² = 1 + 5/0.5 = **11**.
- **Posterior mean** = (μ/s² + Σ x t/σ²) / 11 = (0 + (2.4 + 0.4)/0.5) / 11 = 5.6/11 = **0.51**.
- **Draw** v ~ N(0.51, 1/11). With few observations the variance is large, and averaging over draws protects against overfitting.

---

## 5. The paper's results (Table 1, RMSE / area under the precision–recall curve)

| Dataset | MAP (d=20) | BTF (d=20) | **BCTF** (d=20) | IRM | MRC |
|---|---|---|---|---|---|
| Animals | 0.467 / 0.78 | 0.337 / 0.85 | **0.331 / 0.86** | 0.382 / 0.75 | – / 0.81 |
| Kinship (d=40 for MAP, BCTF) | 0.110 / 0.90 | – | **0.108 / 0.90** | 0.140 / 0.66 | – / 0.85 |
| UML (d=40) | 0.024 / 0.98 | – | **0.024 / 0.98** | 0.054 / 0.70 | – / 0.98 |
| MovieLens (RMSE) | 0.899 | 0.835 | 0.836 | – | – |
| ConceptNet (d=40) | 0.614 / 0.48 | 0.267 / 0.94 | **0.260 / 0.94** | – | – |

(BTF = Bayesian Tensor Factorization: BCTF with everything in a single cluster. IRM = Infinite Relational Model, a pure clustering model. MRC = Multiple Relational Clustering.)

- **BCTF beats the cluster-only IRM and MRC.**
- **On dense Kinship and UML, MAP is as good** as the Bayesian models, because there are many more observations than parameters.
- **On the small Animals dataset, BTF beats MAP and BCTF beats BTF.**
- **On sparse MovieLens and ConceptNet, MAP overfits badly** (ConceptNet AUC 0.48–0.57), while the Bayesian models reach 0.93–0.94.
- **The clusters are meaningful:** e.g. whales/seals/dolphins with the features "flippers, swims, ocean".

---

## 6. What our code found

- **No real data:** the paper's datasets aren't downloaded. We **plant** structure instead:
  - 60 objects in 4 clusters, 6 relations in 2 clusters;
  - true 3-dim vectors = cluster means + small jitter;
  - truth = a_Lᵀ R b_R + noise above a threshold (30% true);
  - 10% of the 21,600 facts held out.
- **Everything is implemented in numpy:** MAP by conjugate gradient, the full Gibbs sampler (exact Gaussian draws for R too, since d = 3), CRP collapsed Gibbs, and an IRM-like block model (k-means clusters on observed profiles plus block means).

| Observed training facts | MAP | BTF | **BCTF** | Block model |
|---|---|---|---|---|
| 19,440 (all) | 0.228 / 0.971 | 0.228 / 0.970 | 0.228 / 0.970 | 0.278 / 0.902 |
| 1,944 (10%) | 0.278 / 0.922 | 0.255 / 0.938 | **0.250 / 0.943** | 0.306 / 0.842 |
| 583 (3%) | **0.792 / 0.384** | **0.295 / 0.882** | 0.322 / 0.830 | 0.424 / 0.566 |

(test RMSE / area under the precision–recall curve)

**Cluster recovery** (adjusted Rand index vs the planted partition; 1 = perfect):
- **All data:** BCTF found 4 object clusters (ARI **0.79**) and 2 relation clusters (ARI **1.00**); the block model found objects with ARI 0.75.
- **10%:** 5 object clusters (ARI 0.72), but the two relation clusters merged.
- **3%:** ARI 0.42 for objects and 0.35 for relations.

**What matches the paper:**
1. **Dense data:** MAP = BTF = BCTF for prediction (like Kinship and UML), and BCTF additionally recovers the clusters.
2. **Sparse data:** MAP overfits catastrophically (PR-AUC 0.38), while the Bayesian models keep 0.83–0.88 (like ConceptNet).
3. **At 10% density BCTF is slightly better than BTF** (like Animals).
4. **The cluster-only model is worst at every density.**

**Honest notes:**
- **At 3% density, BTF beat BCTF,** and the clusters were only partly recovered.
- **Two simplifications:** our sampler starts from many small random clusters instead of using split-merge moves, and our cluster-variance prior scale is tied to the size of the MAP vectors (0.05 × their variance) rather than sampled. With the paper's literal IG(α, 1) prior, our small vectors all merged into 1–2 clusters.
- **Over a second seed** (during development), BCTF recovered the partitions perfectly on dense data (ARI 1.00 / 1.00).

**`experiments.py`:**
- **E1:** a density sweep × 5 seeds;
- **E2:** dimensionality;
- **E3:** cluster recovery vs density and prior scale;
- **E4:** sampler length.
- Only E2 and E4 smoke runs were done here (d = 3 at 10%: BCTF 0.251 RMSE; 20 sweeps: object ARI 0.70).

---

## 7. Why it matters

- **BCTF is part of the line of work on relational and knowledge-graph learning** that runs from the Infinite Relational Model through tensor factorizations (RESCAL) to modern knowledge-graph embeddings.
- **Its specific contributions:**
  - showing that **full Bayesian inference beats MAP on sparse relational data**, an early large-scale demonstration;
  - showing that **clustering the embeddings** gives interpretability without sacrificing accuracy.
- **The "a_L and a_R" trick** for making conditionals factorise is a standard design choice in Gibbs samplers for factorization models.

---

## 8. Check yourself

1. Compute a_Lᵀ R b_R for a_L = (0, 1), R = [[1, 2], [3, 4]], b_R = (1, 1).
<details><summary>Answer</summary>R b_R = (1 + 2, 3 + 4) = (3, 7); a_Lᵀ (3, 7) = 7.</details>

2. Why does BCTF use a Gaussian likelihood for binary data?
<details><summary>Answer</summary>It makes the model conjugate. The conditional distributions of a_L, a_R and R are Gaussian and can be sampled exactly, which makes Gibbs sampling easy and fast, even though a logistic likelihood would fit 0/1 data better.</details>

3. Under a CRP with α_DP = 2 and clusters of sizes 3 and 1, what is the probability that the next object starts a new cluster?
<details><summary>Answer</summary>2 / (3 + 1 + 2) = 1/3.</details>

4. Why does each object have two vectors a_L and a_R?
<details><summary>Answer</summary>Given the right vectors and the relation matrices, the conditional of every left vector is an independent Gaussian (and vice versa), so all objects can be sampled in parallel. With a single vector per object the conditionals would be coupled.</details>

5. Compute the posterior mean for one-dimensional v with prior N(0, 1), noise variance 1, and observations (x = 1, t = 2), (x = 1, t = 0).
<details><summary>Answer</summary>Precision = 1 + 2 = 3; mean = (2 + 0)/3 = 0.67.</details>

6. Why does MAP overfit on sparse data while the Bayesian model doesn't?
<details><summary>Answer</summary>MAP picks the single parameter setting that best fits the few observations, which can be far from the truth. The Bayesian model averages predictions over all plausible parameters (weighted by the posterior), which hedges against fitting noise. In our toy, 3% density gave MAP a PR-AUC of 0.38 vs 0.88 for BTF.</details>

7. When does MAP do as well as the Bayesian models?
<details><summary>Answer</summary>When there are many more observations than parameters (dense Kinship and UML in the paper; our fully observed setting). The posterior is then sharply concentrated and averaging changes little.</details>

8. What does the cluster structure add if the predictions are the same?
<details><summary>Answer</summary>Interpretability: you can read off which objects and relations behave alike (e.g. sea mammals with "flippers, swims, ocean"), which helps exploratory analysis.</details>

9. What does the adjusted Rand index measure?
<details><summary>Answer</summary>Agreement between two partitions, corrected for chance: 1 means identical (up to relabelling), about 0 means no better than random.</details>

10. Why did our BCTF need a smaller cluster-variance prior than IG(α, 1)?
<details><summary>Answer</summary>Our learned vectors are small (variance about 0.15). A prior expecting within-cluster variances around 1 makes every cluster broad enough to swallow all objects, so everything merged. Tying the scale to the vectors' variance (the paper instead samples its hyperparameters) let the sampler separate the clusters.</details>
