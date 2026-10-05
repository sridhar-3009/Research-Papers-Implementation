# Model Cards, explained simply

**Paper:** Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru (Google and University of Toronto), *Model Cards for Model Reporting*, FAT* 2019.

**In one sentence:** every trained model should ship with a short document, a **model card**, that says what it is for, who it was tested on, and how well it works **for each relevant group of people and conditions** (not just on average), so users can tell whether it fits their situation.

---

## 1. The problem

- **Models are reused far from where they were built,** including in law enforcement, medicine, education and employment.
- **Release notes usually give one aggregate number** ("94% accuracy"). That number can hide a model that fails badly for some group.
- **The paper's analogy:** electronic components come with datasheets that state their operating conditions. Model cards do the same for models. (The sister proposal for datasets is "Datasheets for Datasets".)

---

## 2. The nine sections (Figure 1)

| Section | What goes in it |
|---|---|
| **Model Details** | developer, date, version, type, training algorithm, fairness constraints, paper, citation, license, contact |
| **Intended Use** | primary intended uses and users; **out-of-scope** uses |
| **Factors** | groups (demographic, phenotypic), instrumentation (camera type), environment (lighting); which are **relevant**, and which were actually **evaluated** |
| **Metrics** | which performance measures and why; **decision thresholds**; how uncertainty was estimated (e.g. bootstrap, cross-validation) |
| **Evaluation Data** | datasets, why chosen, preprocessing |
| **Training Data** | ideally like Evaluation Data; at minimum the distribution over groups |
| **Quantitative Analyses** | results **disaggregated** by factor: **unitary** (per group) and **intersectional** (per combination of groups), with confidence intervals |
| **Ethical Considerations** | sensitive data, effects on human life, mitigations, risks and harms, fraught use cases |
| **Caveats and Recommendations** | what wasn't tested, groups missing from evaluation, recommended next steps |

### Unitary vs intersectional
- **Unitary groups** are defined by one factor: "men", "women", "older people".
- **Intersectional groups** combine factors: "older men", "younger women".
- **A model can look fine for "men" and fine for "older people"** and still fail badly for "older men". The paper ties this to intersectionality theory, and to the Gender Shades finding that commercial face classifiers failed most on darker-skinned women.

---

## 3. The error rates, step by step

For a binary classifier (smiling = positive), the confusion matrix has four counts:

|  | predicted smiling | predicted not smiling |
|---|---|---|
| **really smiling** | TP (true positive) | FN (false negative) |
| **really not smiling** | FP (false positive) | TN (true negative) |

The paper's four error rates:
```
False Positive Rate  FPR = FP / (FP + TN)   — of the non-smilers, how many did we call smiling?
False Negative Rate  FNR = FN / (FN + TP)   — of the smilers, how many did we miss?
False Discovery Rate FDR = FP / (FP + TP)   — of our "smiling" predictions, how many were wrong?
False Omission Rate  FOR = FN / (FN + TN)   — of our "not smiling" predictions, how many were wrong?
```

**Worked example.** 8 photos with truth (1, 1, 1, 0, 0, 0, 0, 1) and predictions (1, 1, 0, 1, 0, 0, 0, 0):
- counts: TP = 2, FN = 2, FP = 1, TN = 3;
- FPR = 1 / 4 = 0.25; FNR = 2 / 4 = 0.50; FDR = 1 / 3 = 0.33; FOR = 2 / 5 = 0.40.
- (This is our test `test_confusion_rates_by_hand`.)

**Which rate matters depends on who you are.** In surveillance, the operator wants a low FNR (don't miss anyone), while the people being watched want a low FPR (don't flag the innocent). The paper recommends **reporting all of them**, and saying which were prioritised.

### Fairness criteria as equal error rates
- **Equality of opportunity:** equal FNR across groups (every group's true positives are equally likely to be found).
- **Equalized odds:** equal FNR **and** equal FPR across groups.
- A card that reports these rates per group shows directly which criteria hold.

### Score-based models
- **For models that output a score** (risk, toxicity), compare score **distributions** across groups: means, quantiles, or AUC within each group.
- **The paper mentions pinned AUC:** the AUC on a group's examples mixed with an equal-sized sample of all data.

### Confidence intervals by bootstrap
- **Per-group numbers come from smaller samples,** so they are noisier.
- **The bootstrap:**
  1. resample the group's examples **with replacement** (same size), say 1,000 times;
  2. recompute the rate each time;
  3. the 2.5th and 97.5th percentiles form a **95% interval**.
- **Example:** a group of 50 with FNR 0.20 might give an interval of [0.05, 0.40], which is almost useless. With 1,000 examples it narrows to about [0.17, 0.23]. The card should show this, so readers don't over-interpret small groups.

---

## 4. The paper's two examples

**Smiling classifier on CelebA (Figure 2):**
- **Setup:** evaluated by gender and age; 95% confidence intervals by bootstrap; threshold 0.5, where all four error types lie within 0.04–0.14.
- **Finding 1:** the **false discovery rate for older men is much higher** than for other groups (older men wrongly called smiling).
- **Finding 2:** **men in aggregate have a higher false negative rate** (smiling men missed).
- **Advice:** it suits uses where finding smiles matters more than missing a few ("fun moments"), and fine-tuning with images of older men may help. It is **not for emotion detection**, since smiles were labelled by appearance.

**Toxicity scorer (Figure 3):**
- **Setup:** Perspective API's TOXICITY model, evaluated on synthetic Identity Phrase Templates (sentences like "I am a ___ person").
- **Finding:** **version 1 performs poorly for several identity terms, especially "lesbian", "gay", "homosexual"**, because the model learned to associate those words with toxicity. **Version 5,** after bias mitigation, is more equitable.
- **Lesson:** models change over time, so the card must be updated with every release.

---

## 5. What our code found

**Smiling (synthetic CelebA stand-in):**
- **The setup:** two image cues.
  - **Mouth curvature,** the real smile cue, is weaker for men in our toy.
  - **Face lines** appear when smiling, but older faces, especially men's in our toy, have them anyway.
- **The model:** logistic regression trained on 12,000 faces, evaluated on 8,000.

| Group | n | FPR | FNR | FDR | FOR |
|---|---|---|---|---|---|
| all | 8,000 | 0.149 | 0.170 | 0.164 | 0.156 |
| female | 4,781 | 0.131 | 0.122 | 0.139 | 0.114 |
| male | 3,219 | 0.177 | **0.243** | 0.203 | 0.214 |
| female old | 970 | 0.207 | 0.060 | 0.205 | 0.061 |
| **male old** | 1,065 | **0.315** | 0.098 | **0.276** [0.242, 0.308] | 0.116 |
| female young | 3,811 | 0.111 | 0.136 | 0.120 | 0.126 |
| male young | 2,154 | 0.109 | **0.315** | 0.147 | 0.245 |

- **Both of the paper's findings are reproduced:** older men have the highest FDR (0.276 vs 0.120 for young women), and men in aggregate have a higher FNR (0.243 vs 0.122). In our toy this is by construction: we built the cues that cause it. The point is that the disaggregated table **reveals** it, while the "all" row (all rates around 0.15–0.17) does not.
- **Over 2 seeds** (`experiments.py` E2 smoke run), "older men have the highest FDR" held both times.
- **Fairness gaps:** the FNR gap is 0.122 between genders but **0.255 across the four intersections**. Intersectional analysis finds larger disparities.
- **Threshold slider:** raising the threshold from 0.3 to 0.7 cuts older men's FPR (0.469 → 0.192) but raises younger men's FNR (0.190 → 0.492). No single threshold equalises them, which is why the card must state its threshold.
- **Bootstrap intervals** (E1 smoke run): with 50 examples, a 95% interval is about 0.30–0.40 wide; with 500, about 0.06–0.10.

**Toxicity (synthetic corpus and identity-phrase templates):**
- **The training corpus:** 30% of comments are toxic.
  - 35% of the toxic ones contain no insult word (implicit), and 8% of non-toxic ones contain an insult (banter).
  - **The bias:** the "targeted" terms (lesbian, gay, homosexual, muslim) appear mostly inside toxic comments.
- **v2 (mitigated)** adds non-toxic sentences using every identity term, the data-balancing approach of Dixon et al.

| Term | v1 subgroup AUC | v1 BPSN AUC | v1 innocent-sentence score | v2 BPSN AUC | v2 innocent-sentence score |
|---|---|---|---|---|---|
| lesbian | 1.00 | **0.64** | **0.45** | 0.88 | 0.19 |
| gay | 1.00 | **0.64** | **0.46** | 0.95 | 0.18 |
| homosexual | 1.00 | **0.64** | **0.44** | 0.97 | 0.17 |
| muslim | 1.00 | **0.64** | **0.48** | 0.96 | 0.18 |
| straight, christian, jewish, … | 1.00 | 1.00 | 0.04–0.06 | 1.00 | 0.03 |

- **Overall AUC** is 0.889 (v1) vs 0.981 (v2).
- **BPSN AUC** ("background positive, subgroup negative", from the follow-up work of Borkan et al.) compares a term's **innocent** sentences with **other terms' toxic** sentences. It exposes exactly the bias the paper reports: innocent mentions of targeted terms score like toxic text.
- **Subgroup AUC is 1.00 everywhere,** so measuring within one term alone would miss the bias completely.
- **Honest notes:**
  1. Mitigation helped (0.64 → 0.88–0.97) but didn't fully fix it.
  2. Our pinned AUC (the metric the paper names) is noisy: it depends on which random background is drawn. In v1 even non-targeted terms got pinned AUC of 0.90–0.96, because the background contains targeted-term sentences.

**`experiments.py`:**
- **E1:** CI width vs group size;
- **E2:** the smiling finding over 20 seeds;
- **E3:** toxicity over 5 seeds;
- **E4:** write cards to disk.
- Only E1 and E2 smoke runs were done here.

---

## 6. Why it matters

- **Model cards are now standard practice:** Hugging Face model cards (every model page), Google's Model Card Toolkit, and cards for large language models.
- **The idea of disaggregated, intersectional evaluation with uncertainty** is central to fairness auditing, and regulations increasingly ask for documentation of this kind.

---

## 7. Check yourself

1. Name the nine sections of a model card.
<details><summary>Answer</summary>Model Details, Intended Use, Factors, Metrics, Evaluation Data, Training Data, Quantitative Analyses, Ethical Considerations, Caveats and Recommendations.</details>

2. What is the difference between unitary and intersectional results?
<details><summary>Answer</summary>Unitary results are per group defined by one factor (e.g. men, older people). Intersectional results are per combination (e.g. older men). Problems can appear only at intersections.</details>

3. A group has TP = 30, FP = 10, TN = 50, FN = 10. Compute FPR, FNR, FDR, FOR.
<details><summary>Answer</summary>FPR = 10/60 = 0.167; FNR = 10/40 = 0.25; FDR = 10/40 = 0.25; FOR = 10/60 = 0.167.</details>

4. Which equality of error rates corresponds to "equality of opportunity", and which to "equalized odds"?
<details><summary>Answer</summary>Equality of opportunity: equal false negative rates across groups. Equalized odds: equal false negative and equal false positive rates.</details>

5. Why should a card report confidence intervals?
<details><summary>Answer</summary>Disaggregated groups can be small, so their rates are noisy. Without intervals, readers may treat random differences as real (or miss real ones). With 50 examples a 95% interval can be about 0.3 wide.</details>

6. How does the bootstrap give a 95% interval?
<details><summary>Answer</summary>Resample the examples with replacement many times (e.g. 1,000), recompute the metric each time, and take the 2.5th and 97.5th percentiles of those values.</details>

7. What did the paper's smiling card find?
<details><summary>Answer</summary>A much higher false discovery rate for older men (often wrongly predicted to be smiling), and a higher false negative rate for men overall (smiling men missed).</details>

8. Why should the "Intended Use" section list out-of-scope uses?
<details><summary>Answer</summary>To stop the model being used where it wasn't designed or tested to work. E.g. the smiling model is not for emotion detection, because smiles were labelled by appearance, not feeling.</details>

9. In our toxicity toy, why was subgroup AUC 1.00 even for the biased terms?
<details><summary>Answer</summary>Within one term, all templates share that term's bias, so toxic sentences still outrank innocent ones. The bias only shows when the term's innocent sentences are compared with other sentences (BPSN AUC 0.64).</details>

10. Why does the paper stress updating the card with each model version?
<details><summary>Answer</summary>Behaviour changes between versions (TOXICITY v1 vs v5 differed drastically for identity terms), so a card describes one specific version. Users need the card for the version they actually use.</details>
