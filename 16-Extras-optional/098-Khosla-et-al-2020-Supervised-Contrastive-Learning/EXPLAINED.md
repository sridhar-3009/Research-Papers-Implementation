# Supervised Contrastive Learning, explained simply

**Paper:** Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan (Google Research, Boston University, MIT), *Supervised Contrastive Learning*, NeurIPS 2020.

**In one sentence:** self-supervised contrastive learning (like SimCLR) pulls two augmented views of the **same image** together and pushes everything else apart. **SupCon** uses the labels too: it pulls together **all images of the same class** in a batch. Trained this way and then given a linear classifier, a ResNet beats ordinary cross-entropy training on ImageNet (81.4% top-1 with ResNet-200) and is more robust to corruptions.

---

## 1. The training recipe (two stages)

1. **Representation stage:**
   - take a batch of N images and augment each **twice**, giving 2N "views";
   - pass each through an **encoder** (a ResNet) to get r, normalised to length 1;
   - pass r through a small **projection network** to get z, also normalised to length 1;
   - compute the contrastive loss on z.
2. **Classification stage:** throw away the projection network, **freeze** the encoder, and train a linear classifier on r with cross-entropy.

Normalising to the unit sphere means the dot product z_i · z_j is the **cosine similarity** (−1 to 1).

---

## 2. The losses, step by step

In each formula below:
- i is the **anchor**;
- A(i) is every other view in the batch;
- τ is the **temperature** (0.1 in the paper).

### Self-supervised (Eq. 1, SimCLR): exactly one positive, the other view of the same image j(i)
```
L_i^self = − log  exp(z_i · z_j(i) / τ) / Σ_{a ∈ A(i)} exp(z_i · z_a / τ)
```
This is a softmax classification: "which of the other 2N − 1 views is my twin?"

### Supervised, "outside" (Eq. 2, SupCon): P(i) = every other view with the same label
```
L_i^out = − (1/|P(i)|) Σ_{p ∈ P(i)}  log  exp(z_i · z_p / τ) / Σ_{a ∈ A(i)} exp(z_i · z_a / τ)
```
The average over positives sits **outside** the log.

### Supervised, "inside" (Eq. 3)
```
L_i^in = − log  (1/|P(i)|) Σ_{p ∈ P(i)}  exp(z_i · z_p / τ) / Σ_{a ∈ A(i)} exp(z_i · z_a / τ)
```
The average sits **inside** the log.

### Worked example (τ = 1, to keep the numbers simple)
- **The setup:** an anchor has two positives with similarities 0.9 and 0.1, and one negative with similarity −0.5.
- **Denominator:** e^0.9 + e^0.1 + e^−0.5 = 2.460 + 1.105 + 0.607 = 4.172.
- **The two positive probabilities:** 2.460 / 4.172 = **0.590** and 1.105 / 4.172 = **0.265**.
- **L_out** = −½ (ln 0.590 + ln 0.265) = −½ (−0.528 − 1.328) = **0.928**.
- **L_in** = −ln(½ (0.590 + 0.265)) = −ln 0.4275 = **0.850**.
- **Jensen check:** L_in ≤ L_out, because the log of an average is at least the average of the logs.

**Which is better?** On ImageNet (ResNet-50, batch 6144), L_out gave **78.7%** vs **67.4%** for L_in. The authors explain:
- in L_in the 1/|P(i)| factor becomes an additive constant inside the log, so it doesn't affect the gradient;
- in L_out it properly balances the positives.

When there is only one positive (|P(i)| = 1), all three losses are identical.

---

## 3. Why it works: implicit hard mining (Eq. 4)

- **The gradient** with respect to the anchor's embedding has the form
  ```
  ∂L_i/∂z_i = (1/τ) [ Σ_p z_p (P_ip − X_ip) + Σ_n z_n P_in ]
  ```
  where P_ix is the softmax probability of view x for anchor i.
- **Backpropagating through the normalisation** z = w/‖w‖ multiplies by (I − z zᵀ)/‖w‖. This removes the component along z.
- **The consequence:**
  - **easy positives** (already almost aligned, z_i · z_p ≈ 1) contribute almost **zero** gradient;
  - **hard positives** (nearly orthogonal) contribute a lot.
- So the loss automatically focuses on hard examples, without the delicate hard-negative mining that triplet losses need.

---

## 4. The paper's results

**Table 2** (ResNet-50, top-1 %):

| Dataset | SimCLR | Cross-entropy | Max-margin | **SupCon** |
|---|---|---|---|---|
| CIFAR-10 | 93.6 | 95.0 | 92.4 | **96.0** |
| CIFAR-100 | 70.7 | 75.3 | 70.5 | **76.5** |
| ImageNet | 70.2 | 78.2 | 78.0 | **78.7** |

- **ResNet-200 with Stacked RandAugment:** **81.4%** vs 80.9% for cross-entropy (their implementation). That is 0.8 points above the best previously reported number for this architecture (80.6%).
- **Robustness on ImageNet-C** (mean corruption error, lower is better): ResNet-50 **68.6 → 67.2**, ResNet-200 **52.4 → 50.6**.
- **Other findings:**
  - a memory bank of 8,192 with batch 256 gives 79.1%;
  - cross-entropy with batch 12,288 (to match SupCon's two views) only reached 77.5%;
  - the N-pairs loss reached 57.4%;
  - SupCon was **less sensitive to hyperparameters** than cross-entropy.

---

## 5. What our code found

- **The setup:** everything is in numpy on scikit-learn's 8×8 digits (1,300 train / 497 test).
  - **augmentations:** shift, intensity scaling and noise;
  - **encoder:** 64 → 128 → 64, normalised;
  - **projection:** 64 → 64 → 32, normalised;
  - **settings:** τ = 0.1, batch 128 (256 views).

**Losses and gradients:**
- Analytic gradients of L_out and L_in match finite differences to **6·10⁻¹⁰**.
- **L_in (2.636) ≤ L_out (2.944)** on a random batch, as Jensen requires.
- With one positive per anchor the two losses are identical (a test).

**Implicit hard mining:** the gradient on the un-normalised positive vs its cosine to the anchor:

| Cosine | Gradient norm |
|---|---|
| 0.00 (hard) | **9.82** |
| 0.50 | 4.56 |
| 0.90 | 0.171 |
| 0.99 (easy) | **0.035** |

**Two-stage training** (100 epochs each, learning rate 3·10⁻³, one seed):

| Method | Test accuracy |
|---|---|
| Cross-entropy (same encoder + linear head, end to end) | 0.962 |
| **SupCon L_out** + linear probe | **0.982** |
| SupCon L_in + linear probe | 0.970 |
| Self-supervised (Eq. 1) + linear probe | 0.944 |

- **Using labels in the contrastive loss clearly helps** over the self-supervised version.
- **Honest notes:**
  1. SupCon's 2-point lead over cross-entropy is about 10 test images. With 200 epochs each (tuned during development) both reached **97.8%**, a tie.
  2. We found **no consistent L_out > L_in ordering.** L_in reached 98.8% at τ = 0.3. The paper's large gap (78.7 vs 67.4) is on ImageNet with huge batches, where each anchor has many positives.

**Robustness** (test images corrupted; cross-entropy vs SupCon L_out):

| Corruption | Cross-entropy | SupCon |
|---|---|---|
| noise 0.2 / 0.4 / 0.6 | 0.875 / 0.642 / 0.392 | **0.936 / 0.726 / 0.473** |
| blur 0.5 / 0.8 / 1.2 | 0.948 / 0.859 / 0.586 | **0.974 / 0.926 / 0.710** |
| shift 0.5 / 1.0 / 1.5 | **0.958 / 0.861 / 0.193** | 0.909 / 0.765 / 0.157 |

- **SupCon is much more robust to noise and blur,** in line with the paper's ImageNet-C finding.
- **But it is *less* robust to translations here,** so robustness gains depend on the type of corruption.

**`experiments.py`:**
- **E1:** methods × 5 seeds × epochs;
- **E2:** temperature;
- **E3:** batch size;
- **E4:** robustness over seeds;
- **E5:** learning-rate sensitivity.
- Only an E5 smoke run was done here (30 epochs, two learning rates): the cross-entropy spread was 0.018 and SupCon's 0.034. That is the opposite of the paper's "less sensitive", though only two settings were tried.

---

## 6. Why it matters

- **SupCon showed that the contrastive machinery of self-supervised learning** (augmented views, normalised embeddings, temperature-scaled softmax over many negatives) can replace cross-entropy even when labels are available.
- **Its loss is widely used** for representation learning with labels, for fine-tuning, and in long-tailed and noisy-label settings.
- **It made the role of normalisation and temperature in implicit hard mining explicit.**

---

## 7. Check yourself

1. In a batch of N images with 2 views each, how many terms does the denominator of Eq. 1 have?
<details><summary>Answer</summary>2N − 1: every view except the anchor itself (the positive plus 2N − 2 negatives).</details>

2. What is P(i) in SupCon, and how big is it on average for a random batch?
<details><summary>Answer</summary>The set of other views in the batch with the same label as the anchor (including its own second view). With C classes it averages about 2N/C − 1.</details>

3. With positive probabilities 0.5 and 0.2, compute L_out and L_in for one anchor.
<details><summary>Answer</summary>L_out = −½(ln 0.5 + ln 0.2) = −½(−0.693 − 1.609) = 1.151. L_in = −ln(0.35) = 1.050. L_in ≤ L_out.</details>

4. Why does L_in's 1/|P(i)| factor not affect its gradient?
<details><summary>Answer</summary>Inside the log, −log((1/|P|)·S) = log|P| − log S, so the factor becomes an additive constant with zero gradient.</details>

5. Why do easy positives get almost no gradient?
<details><summary>Answer</summary>The gradient through the normalisation removes the component along the embedding, and an easy positive's embedding is already nearly parallel to the anchor, so what remains is tiny (0.035 at cosine 0.99 vs 9.8 at cosine 0 in our toy).</details>

6. What happens to the projection network after training?
<details><summary>Answer</summary>It is discarded. A linear classifier is trained on the frozen, normalised encoder output r.</details>

7. When are Eqs. 1, 2 and 3 identical?
<details><summary>Answer</summary>When each anchor has exactly one positive (|P(i)| = 1); then the average over positives disappears.</details>

8. What was SupCon's ImageNet result on ResNet-200?
<details><summary>Answer</summary>81.4% top-1, 0.8 points above the best reported cross-entropy number (80.6%) for that architecture.</details>

9. Why does the paper use temperature τ = 0.1?
<details><summary>Answer</summary>The paper used τ = 0.1 for all results. Smaller temperatures sharpen the softmax and help training (stronger gradients, which scale with 1/τ), but extremely low temperatures are numerically unstable to train.</details>

10. In our toy, where was SupCon more and less robust than cross-entropy?
<details><summary>Answer</summary>More robust to pixel noise and blur (e.g. 0.926 vs 0.859 at blur 0.8), less robust to shifts (0.765 vs 0.861 at a 1-pixel shift).</details>
