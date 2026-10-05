# Failing Loudly, explained simply

**Paper:** Stephan Rabanser, Stephan Günnemann, Zachary C. Lipton, *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*, NeurIPS 2019.

**In one sentence:** ML systems "tend to fail silently" when the incoming data stops looking like the training data. The paper compares many ways to **detect** such dataset shift from unlabelled incoming data, by squeezing the data into a few dimensions and running a **statistical two-sample test**. The winner is surprisingly simple: run a KS test on each output of the **model's own softmax** (BBSDs) and combine them with a Bonferroni correction.

---

## 1. The problem

- **Two distributions:**
  - **Source** distribution p: where the training data came from. We have labelled samples.
  - **Target** distribution q: what the deployed model sees now. We have **unlabelled** samples x′₁, …, x′ₘ.
- **The question:** has the distribution changed? As a hypothesis test:
  - **H₀ (null):** p(x) = q(x′), no shift;
  - **H_A (alternative):** p(x) ≠ q(x′), shift.
- **The paper's three goals:**
  1. **detect** shift from as few samples as possible;
  2. **characterise** it (which samples typify it);
  3. judge whether it is **harmful** ("malignant") or harmless ("benign").

**Why not just test the raw inputs?** Images have hundreds or thousands of dimensions, and two-sample tests lose power badly as the dimension grows. So the pipeline (Figure 1) is:

```
source x, target x'  →  dimensionality reduction (D → K)  →  two-sample test(s)  →  p-value  →  "shift" if p < α = 0.05
```

---

## 2. Statistics refresher: p-values and two-sample tests

- **A p-value** is the probability, *if there were no shift*, of seeing a difference at least as large as the one observed. If p < α (here 0.05), we reject H₀ and say "shift detected".
- **Even with no shift,** a correct test rejects 5% of the time. That is the **false-positive rate**, and it should be about α.
- **Detection accuracy (power)** is how often the test rejects when there really **is** a shift. The paper reports this, averaged over many random draws.

---

## 3. The dimensionality-reduction (DR) methods

| Name | Representation | Dimensions |
|---|---|---|
| **NoRed** | raw pixels | D |
| **PCA** | projection on the top-K principal components | K |
| **SRP** | sparse random projection (eq. 1) | K |
| **UAE** | an **untrained** (random) autoencoder's encoder | K |
| **TAE** | a **trained** autoencoder's encoder | K |
| **BBSDs** | the label classifier's **softmax** vector ("black box shift detection", soft) | C (number of classes) |
| **BBSDh** | the label classifier's **hard prediction** (argmax) | 1 (categorical) |
| **Classif** | a **domain classifier** trained to tell source from target | 1 (its accuracy) |

**Sparse random projection, eq. (1).** Each entry of the D×K matrix R is:
```
+√(v/K)  with probability 1/(2v)
   0     with probability 1 − 1/v
−√(v/K)  with probability 1/(2v)
```
with v = √D (the usual choice for this construction, from Li et al.). For our D = 64, v = 8: each entry is non-zero with probability 1/8, and the non-zero values are ±√(8/16) = ±0.707 when K = 16. That is cheap, and distances are roughly preserved.

**Why BBSD works.** Lipton et al. showed that if the label classifier f has an invertible confusion matrix and the shift is a **label shift** (class proportions change, while each class's images stay the same), then p ≠ q exactly when the distribution of f(x) changes. The paper finds BBSD works well **even when this assumption doesn't hold**.

---

## 4. The tests, with worked examples

### Kolmogorov–Smirnov (KS), one dimension at a time
- **The statistic** is the largest gap between the two empirical CDFs:
  ```
  Z = sup_z | F_p(z) − F_q(z) |
  ```
- **Example:**
  - source values {1, 2, 3, 4}, target values {3, 4, 5, 6};
  - at z = 2: F_p = 2/4 = 0.5, F_q = 0;
  - at z = 4: F_p = 1, F_q = 2/4 = 0.5;
  - the largest gap is **Z = 0.5**.
  - Under H₀, Z follows the Kolmogorov distribution, which gives the p-value.

### Bonferroni: combining K tests
- **The problem:** run K = 10 tests at α = 0.05 with no shift, and the chance that **at least one** rejects by luck is up to 1 − 0.95¹⁰ ≈ 40%.
- **The fix:** reject only if the **smallest** p-value is below **α / K**.
- **Example:** K = 10 softmax dimensions give p-values 0.30, 0.004, 0.6, …
  - The minimum is 0.004 < 0.05 / 10 = 0.005, so reject.
  - If it were 0.008, then even though 0.008 < 0.05, it is not < 0.005, so don't reject.
- **This is conservative,** so the false-positive rate is at most α (often well below).

### Maximum Mean Discrepancy (MMD), all dimensions at once
- **The idea:** compare the average of a kernel "similarity" within and across the two samples:
  ```
  MMD² = mean κ(xᵢ, xⱼ) over source pairs (i ≠ j)  +  mean κ(x′ᵢ, x′ⱼ) over target pairs  −  2 · mean κ(xᵢ, x′ⱼ) across
  κ(x, x̃) = exp(−‖x − x̃‖² / σ),   σ = median distance in the pooled sample
  ```
- If both samples come from the same distribution, cross-pair similarity equals within-pair similarity, so MMD² ≈ 0.
- **Example:**
  - Suppose within-source pairs average 0.6, within-target pairs 0.6, and cross pairs 0.6. Then MMD² = 0.6 + 0.6 − 1.2 = **0**.
  - If cross pairs only average 0.4, then MMD² = 1.2 − 0.8 = **0.4**: the samples are less similar to each other than to themselves.
- **The p-value** comes from a **permutation test:** shuffle which points are "source" and "target" many times, and count how often the shuffled MMD² is at least the observed one.

### Chi-squared (for BBSDh)
- **Build a 2 × C table** of predicted-class counts (source row, target row).
- **Expected count** for each cell: (row total · column total) / grand total.
- **Statistic:** X² = Σ (O − E)² / E, with C − 1 degrees of freedom.
- **Example with 2 classes:**
  - source predicts (50, 50); target predicts (70, 30);
  - column totals are 120 and 80, grand total 200, so E = 60 and 40 in each row;
  - X² = (50−60)²/60 + (50−40)²/40 + (70−60)²/60 + (30−40)²/40 = 1.67 + 2.5 + 1.67 + 2.5 = **8.33**;
  - with 1 degree of freedom, p ≈ 0.004, so reject.

### Binomial test (for the domain classifier)
- **Setup:** split source and target in half. Train a classifier to tell them apart on the first halves, and test its accuracy on the second halves.
- **Under H₀,** the two domains are indistinguishable, so accuracy ~ Binomial(N, 0.5)/N.
- **Example:** 100 held-out samples, 62 correct. P(≥ 62 or ≤ 38 correct | p = 0.5) ≈ 0.02, so reject.

---

## 5. Beyond detection

- **Most anomalous samples:** rank target samples by how confidently the domain classifier calls them "target". The top ones show **what** changed. In the paper's surprise finding, the original MNIST train/test split is **not** i.i.d.:
  - the most anomalous test samples were mostly **6s**;
  - training 6s are slightly rotated right, test 6s more open and centred;
  - a KS test gave p = 2.7·10⁻¹⁰, far below the Bonferroni threshold of 6.3·10⁻⁵;
  - the shift was judged harmless.
- **Malignancy:** get true labels for those top anomalous samples (e.g. from a human) and measure the model's accuracy on them. If it's much lower than on source data, the shift is **harmful**.

---

## 6. The paper's setup and results

- **Data:** MNIST and CIFAR-10.
- **Models:** ResNet-18 label and domain classifiers.
- **Settings:**
  - K = 32 for PCA / SRP / UAE / TAE;
  - α = 0.05;
  - 10 to 10,000 target samples (kernel tests only up to 1,000);
  - 5 random splits.
- **Shifts:**
  - Gaussian noise σ ∈ {1, 10, 100} (s/m/l);
  - image transforms (rotation, translation, zoom: s/m/l);
  - knock-out (remove a fraction δ of class 0);
  - adversarial (FGSM);
  - m img + ko;
  - only-zero + m img;
  - each applied to a fraction δ ∈ {0.1, 0.5, 1.0} of the target.
- Also the original dataset splits, and the domain-adaptation pairs MNIST→USPS and COIL-100 (rotation angles).

**Table 1a** (univariate tests, averaged over all shifts):

| Method | 10 | 100 | 1,000 | 10,000 samples |
|---|---|---|---|---|
| NoRed (KS) | 0.03 | 0.36 | 0.54 | 0.72 |
| **BBSDs (KS)** | **0.19** | **0.47** | **0.70** | **0.79** |
| BBSDh (χ²) | 0.03 | 0.22 | 0.46 | 0.57 |
| Classif (binomial) | 0.01 | 0.21 | 0.51 | 0.67 |

**Key findings:**
1. **BBSDs is best overall** among univariate tests; UAE is best among multivariate tests.
2. **Univariate KS + Bonferroni is comparable to multivariate MMD,** despite the heavy correction.
3. **The domain classifier is bad with ≤ 100 samples but catches up.**
4. **Shift size matters:**
   - large shifts (l gn, l img, m img+ko, oz+m img) are detectable with about 20 samples;
   - small ones (s gn, ko) need orders of magnitude more;
   - shifts affecting only 10% of samples are hard to detect, while 50% / 100% can be detected with 50 samples.
5. **The COIL-100 rotation partition is detected but harmless.**

---

## 7. What our code found

**Setup:**
- **Data:** scikit-learn's 8×8 digits (1,797 images, D = 64), split 900 for training / 450 source / 447 target.
- **Label classifier:** an MLP (64-64-10) with 97.1% source accuracy; K = 16 (PCA explains 85% of the variance).
- **Sampling:** each number below is the share of 10 random draws that rejected H₀ at α = 0.05.
- **Settings we chose:** the noise levels (σ = 0.05 / 0.2 / 1.0 on [0, 1] pixels) and the FGSM step (0.15).

**False positives (no shift):** the mean rejection rate over all 14 method/test pairs and 3 sample sizes is **0.024**, below α = 0.05, so the tests are valid (KS + Bonferroni is conservative).

**Detection averaged over 7 shifts** (s gn, l gn, m img, ko, adv, m img+ko, oz+m img):

| Method / test | n = 10 | n = 50 | n = 200 |
|---|---|---|---|
| NoRed KS | 0.04 | 0.51 | **0.86** |
| PCA KS | 0.06 | 0.50 | 0.61 |
| SRP KS | 0.04 | 0.47 | 0.60 |
| UAE KS | 0.04 | 0.31 | 0.71 |
| TAE KS | 0.10 | 0.44 | 0.74 |
| **BBSDs KS** | **0.16** | 0.50 | 0.73 |
| BBSDh χ² | 0.06 | 0.33 | 0.57 |
| Classif binomial | 0.10 | 0.37 | 0.80 |
| NoRed MMD | 0.27 | 0.61 | 0.69 |
| UAE MMD | 0.11 | 0.54 | 0.64 |
| BBSDs MMD | 0.17 | 0.43 | 0.57 |

- **Agrees with the paper:**
  - BBSDs is the best univariate method at the smallest sample size;
  - univariate KS + Bonferroni is comparable to MMD;
  - the domain classifier starts weak (0.10) and catches up (0.80);
  - BBSDh (hard labels) is weaker than BBSDs (soft).
- **Honest difference:** at n = 200, raw pixels (NoRed) win. Our images have only 64 pixels, not 784 or 3,072, so the high-dimension problem that motivates DR is mild. Also, the digits' **border pixels are always 0**, so any added noise is trivially visible to a per-pixel KS test.

**Per shift with BBSDs + KS** (and the label classifier's accuracy on the shifted target):

| Shift (δ) | Accuracy | n = 10 | n = 50 | n = 200 |
|---|---|---|---|---|
| s gn (0.5) | 0.975 | 0.00 | 0.10 | 0.00 |
| l gn (0.5) | 0.626 | 0.10 | 0.40 | 1.00 |
| m img (0.5) | 0.566 | 0.10 | 0.90 | 1.00 |
| ko (1.0) | 0.972 | 0.00 | 0.00 | 0.10 |
| adv (0.5) | 0.566 | 0.00 | 0.10 | 1.00 |
| m img+ko (0.5) | 0.567 | 0.30 | 1.00 | 1.00 |
| oz+m img (0.5) | 0.571 | **0.60** | 1.00 | 1.00 |

- **The same easy/hard pattern as Table 1b:**
  - the "only zeros" target and m img+ko are caught with 10–50 samples;
  - small noise and knock-out are almost never caught.
- **The hard shifts are also the harmless ones** (accuracy 0.975 / 0.972), while every detectable shift cuts accuracy to about 0.57–0.63.

**Most anomalous samples and malignancy** (domain classifier, 200 source vs 200 target):

| Shift | Domain classifier p | Accuracy on its top-20 "most target-like" samples | Verdict |
|---|---|---|---|
| m img | 2.1·10⁻¹¹ | **0.30** (source 0.97) | malignant |
| s gn | 0.10 | 1.00 | benign (and not even detected) |
| ko | 0.10 | 0.90 | benign (not detected) |

**`experiments.py`:**
- **E1:** the full grid of 14 methods × 10 shifts × 3 δ × 6 sizes × 5 splits;
- **E2:** per-shift and per-δ tables;
- **E3:** latent size K;
- **E4:** whether top-k anomalous accuracy tracks true target accuracy;
- **E5:** optional MNIST (download).
- Only an E4 smoke run was done here: on 2 shifts the correlation was 1.00, which is trivially true for 2 points.

---

## 8. Why it matters

- **It turned "monitor for drift" into a concrete, tested recipe:**
  1. reduce the inputs with the model you already have (its softmax);
  2. run cheap univariate tests with a Bonferroni correction;
  3. use a domain classifier to explain the shift and check whether it hurts.
- **Drift-monitoring libraries** (e.g. Alibi Detect, Evidently) implement these detectors: KS, MMD, chi-squared, classifier-based, and BBSD-style "classifier uncertainty" drift.
- **It also showed** that a famous benchmark split (MNIST) isn't i.i.d.

---

## 9. Check yourself

1. What are H₀ and H_A in shift detection?
<details><summary>Answer</summary>H₀: the source and target input distributions are equal, p(x) = q(x′). H_A: they differ. We reject H₀ (declare shift) when the p-value is below α = 0.05.</details>

2. Compute the KS statistic for source {1, 2, 3} and target {2, 3, 4}.
<details><summary>Answer</summary>At z = 1: F_p = 1/3, F_q = 0 (gap 1/3). At z = 2: 2/3 vs 1/3. At z = 3: 1 vs 2/3. The maximum gap is Z = 1/3.</details>

3. With K = 10 softmax dimensions and α = 0.05, the smallest KS p-value is 0.007. Is shift declared?
<details><summary>Answer</summary>No. The Bonferroni threshold is 0.05 / 10 = 0.005, and 0.007 > 0.005.</details>

4. Why reduce dimensionality before testing?
<details><summary>Answer</summary>Two-sample tests (especially kernel tests) lose power as the dimension grows, and kernel tests scale badly with sample size. A low-dimensional representation that keeps the relevant variation makes shifts easier to detect.</details>

5. What is BBSD, and why can it detect shift?
<details><summary>Answer</summary>Black Box Shift Detection: test whether the distribution of the trained label classifier's outputs changed. Under label shift with an invertible confusion matrix, p ≠ q exactly when the output distribution changes. Empirically it also works for many other shifts.</details>

6. How does the domain classifier give a p-value?
<details><summary>Answer</summary>Train it to separate source from target on half the data, then measure its accuracy on the other half. Under no shift its accuracy is Binomial(N, 0.5)/N, and the binomial test gives the probability of an accuracy this far from 0.5.</details>

7. Why was knock-out of class 0 hard to detect in both the paper and our toy?
<details><summary>Answer</summary>It only changes the class proportions a little (removing class 0 changes about 10% of the data), and each remaining image looks exactly like source data. Only a test on class frequencies with many samples can notice.</details>

8. Why did raw pixels (NoRed) do best for us at n = 200, unlike in the paper?
<details><summary>Answer</summary>Our images have only 64 dimensions, so high dimension hurts little, and the always-zero border pixels make any added noise easy to see per pixel. On MNIST/CIFAR (784 / 3,072 dims) raw-pixel tests are much weaker.</details>

9. How does the paper decide whether a shift is malignant?
<details><summary>Answer</summary>Take the target samples the domain classifier is most confident are "target", obtain their true labels, and compare the model's accuracy on them with its source accuracy. A large drop means the shift is harmful.</details>

10. What surprising thing did the paper find about MNIST?
<details><summary>Answer</summary>Its original train/test split shows a statistically significant shift (mostly in the 6s: training 6s rotated slightly right, test 6s more open and centred), with KS p = 2.7·10⁻¹⁰. The shift was judged harmless.</details>
