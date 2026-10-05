# Encoding vs training in feature learning, explained simply

**Paper:** Adam Coates and Andrew Y. Ng (Stanford), *The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization*, ICML 2011.

**In one sentence:** before deep networks took over, image features were learned in two steps: **train** a dictionary of basis patterns, then **encode** each image patch with it. This paper separates the two steps and finds that the **encoder** is what matters. A dictionary of **randomly chosen image patches** works about as well as a carefully learned sparse-coding dictionary, and a one-line **soft-threshold** encoder works about as well as expensive sparse coding.

---

## 1. The setting: a 2011 feature pipeline

1. **Cut many small patches** from images (6×6 pixels on CIFAR).
2. **Normalise** each patch (subtract its mean, divide by its standard deviation), then **ZCA-whiten** (decorrelate the pixels).
3. **Training:** learn a dictionary D of d unit-length basis vectors ("atoms").
4. **Encoding:** map each patch x to a feature vector f using D.
5. **Pool:** average the features over the 4 quadrants of the image.
6. **Classify:** train a linear classifier (an L2-SVM in the paper).

**Whitening, briefly:**
- if C = U Λ Uᵀ is the covariance of the patches, ZCA whitening maps x ↦ U diag(1/√(λ + ε)) Uᵀ (x − μ);
- the result has (approximately) identity covariance, which removes the strong correlations between neighbouring pixels.

---

## 2. The training algorithms (how D is made)

| Name | How |
|---|---|
| **SC**, sparse coding | minimise Σᵢ ‖D sᵢ − xᵢ‖² + λ‖sᵢ‖₁ over D and the codes s, alternating, with ‖Dⱼ‖ = 1 |
| **OMP-k** | the same, but codes have at most k non-zeros (found by orthogonal matching pursuit) |
| **OMP-1** | "gain-shape vector quantization": each patch uses only its single best-matching atom, almost K-means |
| **RBM, SAE** | sparse RBMs or autoencoders; their weights become D |
| **RP**, random patches | D = randomly chosen (normalised) training patches; **no training at all** |
| **R**, random weights | D = random Gaussian vectors |

---

## 3. The encoders (how f is made from D)

Every encoder **splits polarity**, making separate features for positive and negative parts (2d features in all).

| Name | Features |
|---|---|
| **SC** | solve the lasso s = argmin ‖Ds − x‖² + λ‖s‖₁ with D fixed; f = [max(0, s), max(0, −s)] |
| **OMP-k** | s from OMP with k non-zeros; same split |
| **T**, soft threshold | f = [max(0, Dᵀx − α), max(0, −Dᵀx − α)]: one matrix multiply and a threshold |

**Worked example of the soft threshold.**
- **Setup:** D is the identity, x = (0.6, −0.2, −1.0), α = 0.25, so Dᵀx = x.
- **Positive half:** max(0, x − 0.25) = (0.35, 0, 0).
- **Negative half:** max(0, −x − 0.25) = (0, 0, 0.75).
- **Result:** f = (0.35, 0, 0, 0, 0, 0.75). Small responses are zeroed and large ones kept, which is a cheap imitation of sparsity. (This is our test `test_soft_threshold_and_polarity_split`.)

**How sparse coding computes s** (in our code, by FISTA):
- **One update** takes a gradient step on ‖Ds − x‖², then applies the **shrinkage** sign(s) · max(0, |s| − λ/L).
- **Repeat with momentum.** The soft threshold encoder is exactly **one** such shrinkage step from s = 0. That is why it behaves so much like sparse coding.

**OMP-k, step by step:**
1. Start with residual r = x.
2. Repeat k times: pick the atom most correlated with r, refit all chosen coefficients by least squares, and update r.
- OMP-1 picks one atom, with coefficient Dⱼᵀx: a soft version of "which cluster is this patch in?".

---

## 4. The paper's results

**Table 1:** CIFAR-10, 5-fold cross-validation accuracy (%), best hyperparameters for each pair. Rows are dictionaries, columns are encoders.

| Dictionary | Natural | SC | OMP-1 | OMP-10 | **T** |
|---|---|---|---|---|---|
| R | 70.5 | 74.0 | 65.8 | 68.6 | 73.2 |
| RP | 76.0 | 76.6 | 70.1 | 71.6 | **78.1** |
| RBM | 74.1 | 76.7 | 69.5 | 72.9 | 78.3 |
| SAE | 74.8 | 76.5 | 68.8 | 71.5 | 76.7 |
| SC | **77.9** | 78.5 | 70.8 | 75.3 | 78.5 |
| OMP-1 | 71.4 | 78.7 | 71.4 | 76.0 | 78.9 |
| OMP-2 | 73.8 | 78.5 | 71.0 | 75.8 | 79.0 |
| OMP-5 | 75.4 | 78.8 | 71.0 | 76.1 | 79.1 |
| OMP-10 | 75.3 | 79.0 | 70.7 | 75.3 | **79.4** |

**How to read it:**
- **The "natural" column** (each algorithm with its own encoder) suggests sparse coding is best (77.9 vs 71.4 for OMP-1).
- **But read across the rows:** with the SC or T encoder, every dictionary except random Gaussian reaches about 77–79%.
- **So sparse coding's advantage came from its encoder, not its dictionary.**

**Other results:**
- **CIFAR-10 test set:**
  - random patches + soft threshold: **79.1%**, with no learning beyond choosing α;
  - OMP-10 + T: 80.1%;
  - OMP-1 + T with **d = 6,000** atoms: **81.5%**, the best known CIFAR-10 result at the time.
- **NORB:** random patches + SC encoder: **95.0%**, better than published results (a conv net had 94.4%). The soft threshold reached 93.6% in **1 hour** vs sparse coding's 7 hours on 40 cores.
- **Caltech 101** (SC encoder): random patches 72.6%, the same as an SC dictionary (72.6%).

---

## 5. What our code found

**Setup** (scikit-learn's 8×8 digits, the only image data used here):
- 4×4 patches, stride 1 (25 per image), normalised and whitened;
- d = 48 atoms;
- 4 overlapping quadrants, giving 384 features;
- logistic regression on **20 labelled images per class**, tested on the other ~1,600 images (mean of 3 label draws).
- Raw pixels score **0.923**.

| Dictionary \ encoder | Natural | SC | OMP-1 | OMP-5 | **T** |
|---|---|---|---|---|---|
| R (random Gaussian) | 0.957 | 0.954 | 0.876 | 0.907 | 0.959 |
| RP (random patches) | 0.962 | 0.961 | 0.901 | 0.929 | **0.963** |
| OMP-1 (gain-shape VQ) | 0.911 | 0.961 | 0.911 | 0.938 | **0.963** |
| OMP-5 | 0.920 | 0.958 | 0.894 | 0.920 | 0.962 |
| SC (sparse coding) | 0.959 | 0.959 | 0.891 | 0.931 | 0.962 |

- **The dictionary barely matters:** with the SC encoder every dictionary scores 0.954–0.961, and with the soft threshold 0.959–0.963.
- **The encoder matters:** the hard OMP-1 encoder scores 0.876–0.911, 5–8 points below the soft threshold whatever the dictionary.
- **The "natural" trap reproduces:** SC + SC 0.959 vs OMP-1 + OMP-1 0.911 makes sparse coding look better. But the VQ dictionary with a soft threshold reaches 0.963.
- **Cost** (E4 smoke run, 10,000 patches):

| Step | Time |
|---|---|
| sparse-coding dictionary training | 1,350 ms |
| OMP-1 dictionary training | 15 ms |
| SC encoding | 226 ms |
| soft-threshold encoding | 1 ms |

- **Honest difference:** on CIFAR, random Gaussian dictionaries were clearly worse (73.2 with T). On our 16-dimensional patches they are nearly as good (0.959). With so few dimensions, random directions cover the space well.
- **Labels** (E3 smoke run): with 5 labels per class, features from random patches + T reach 0.889 vs raw pixels 0.825; with 20 labels, 0.962 vs 0.923.

**`experiments.py`:**
- **E1:** the table with the paper's hyperparameter grids;
- **E2:** dictionary size;
- **E3:** labels per class;
- **E4:** timing;
- **E5:** CIFAR-10, if the data is on disk.
- Only E3 and E4 smoke runs were done here.

---

## 6. Why it matters

- **This paper was part of the "simple single-layer networks are surprisingly strong" line of work** (Coates, Lee and Ng 2011, the "K-means features" paper).
- **Its message,** that the nonlinearity and architecture matter more than how carefully the first layer is pretrained, foreshadowed deep learning's shift away from layer-wise unsupervised pretraining toward simple ReLU-like nonlinearities trained end to end.
- **The soft threshold** max(0, z − α) is essentially a **ReLU with a bias**.

---

## 7. Check yourself

1. What are the two components of a feature learner in this paper?
<details><summary>Answer</summary>Training (learning the dictionary D) and encoding (mapping an input x to features f using D).</details>

2. Compute the soft-threshold features for Dᵀx = (1.2, −0.4) with α = 0.5.
<details><summary>Answer</summary>Positive half: max(0, 1.2 − 0.5) = 0.7 and max(0, −0.4 − 0.5) = 0. Negative half: max(0, −1.2 − 0.5) = 0 and max(0, 0.4 − 0.5) = 0. So f = (0.7, 0, 0, 0).</details>

3. Why split features into positive and negative parts?
<details><summary>Answer</summary>So the classifier can weight a strong positive response differently from a strong negative one (the paper found it always helped). It is like non-negative sparse coding with dictionary [D, −D].</details>

4. What is OMP-1, and why is it close to K-means?
<details><summary>Answer</summary>Each input uses only its single most correlated atom, with coefficient Dⱼᵀx. With unit-length data and atoms, picking the most correlated atom is picking the nearest centroid, which is (spherical) K-means.</details>

5. Why is the soft threshold related to sparse coding?
<details><summary>Answer</summary>One step of the iterative shrinkage algorithm for the lasso, starting from s = 0, computes exactly shrink(Dᵀx) = sign(Dᵀx) · max(0, |Dᵀx| − α), the soft threshold (up to a step size).</details>

6. In Table 1, why is the "natural" column misleading?
<details><summary>Answer</summary>It changes the dictionary and the encoder together. Changing only the encoder shows that dictionaries from random patches or VQ do as well as sparse coding's. The advantage was the encoder.</details>

7. What did the paper achieve with OMP-1 + soft threshold and d = 6,000?
<details><summary>Answer</summary>81.5% on CIFAR-10, the best known result at the time, using cheap training and encoding that scale to large dictionaries.</details>

8. Why might random Gaussian dictionaries do worse on CIFAR than on our digits?
<details><summary>Answer</summary>CIFAR patches are 108-dimensional colour patches whose structure lies in a small part of that space, and random directions rarely align with it. Our patches are only 16-dimensional and whitened, so 48 random directions already cover the space well.</details>

9. Which encoder was the fastest in our timing, and by how much?
<details><summary>Answer</summary>The soft threshold: about 1 ms for 10,000 patches vs about 226 ms for sparse coding, since it is one matrix multiply rather than an iterative optimisation.</details>

10. What modern component does the soft threshold resemble?
<details><summary>Answer</summary>A ReLU with a (negative) bias: max(0, w·x − α), applied to both w and −w.</details>
