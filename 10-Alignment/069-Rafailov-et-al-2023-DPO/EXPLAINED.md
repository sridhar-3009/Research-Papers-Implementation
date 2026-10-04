# Direct Preference Optimization (DPO), explained simply

**Paper:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (Stanford), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.

**In one sentence:** RLHF trains a reward model and then runs reinforcement learning (PPO) against it. DPO shows that, with a bit of algebra, the **same objective** can be optimised by a **single classification-style loss** on preference pairs. No reward model, no sampling during training, no RL.

---

## 1. Background: what RLHF optimises

From papers 066–067, RLHF has three stages:
1. **SFT:** fine-tune on demonstrations, giving π_SFT.
2. **Reward model:** learn r_φ(x, y) from preference pairs (x, y_w ≻ y_l) with the **Bradley–Terry** model:
```
P(y_w ≻ y_l | x) = σ(r(x, y_w) − r(x, y_l)),   σ(z) = 1/(1 + e^(−z))
```
3. **RL:** maximise reward while staying close to the reference policy π_ref = π_SFT:
```
max_π  E_{x, y~π}[ r(x, y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )                      (Eq. 3)
```
   This is done with PPO, sampling from the policy during training.

**Why this is painful:** it needs **four models** (policy, reference, reward, value), sampling inside the training loop, and finicky PPO hyper-parameters. RL is unstable.

---

## 2. The key derivation, step by step

### Step 1: the optimum of Eq. 3 has a closed form
For a fixed prompt x, the objective over the distribution π(·|x) is:
```
Σ_y π(y) r(y) − β Σ_y π(y) log(π(y)/π_ref(y))
```
- **Solve it:** maximise subject to Σπ = 1 with a Lagrange multiplier. Setting the derivative to zero gives r(y) − β(log π(y)/π_ref(y) + 1) − λ = 0.
- **The solution:**
```
π*(y|x) = π_ref(y|x) · exp(r(x, y)/β) / Z(x),     Z(x) = Σ_y π_ref(y|x) exp(r(x, y)/β)     (Eq. 4)
```
- **In words:** start from the reference and **re-weight** each answer by exp(reward/β).
  - **Small β:** the policy concentrates on high-reward answers.
  - **Large β:** the policy stays near the reference.

**Worked example:** π_ref = [0.5, 0.5], r = [0, 1], β = 1.
- Weights: 0.5·e⁰ = 0.5 and 0.5·e¹ = 1.359; Z = 1.859.
- Result: π* = [0.269, 0.731].
- With β = 0.5, r/β = [0, 2]: weights 0.5 and 3.69, so π* = [0.119, 0.881]. **The more we trust the reward, the further we move.**

**Why we can't use this directly:** Z(x) sums over **every possible sentence**. That is impossible to compute.

### Step 2: flip it around and write the reward in terms of the policy
Take logs of Eq. 4 and rearrange:
```
r(x, y) = β log( π*(y|x) / π_ref(y|x) ) + β log Z(x)                                        (Eq. 5)
```
- **Any reward** can be written through its own optimal policy, **up to the term β log Z(x)**.
- That term depends only on the prompt, **not on the answer**.

### Step 3: Bradley–Terry only uses reward differences, so Z cancels
```
r(x, y_w) − r(x, y_l) = β log π*(y_w|x)/π_ref(y_w|x) − β log π*(y_l|x)/π_ref(y_l|x)    (the β log Z(x) terms cancel!)
```

### Step 4: fit the policy directly by maximum likelihood on the preferences
Replace π* by our trainable policy π_θ and maximise the Bradley–Terry likelihood of the data:
```
L_DPO(θ) = −E_{(x, y_w, y_l)} [ log σ( β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x) ) ]    (Eq. 7)
```
This is the whole method:
- one loss;
- two forward passes per pair through the policy and two through the frozen reference;
- log-probabilities summed over the answer's tokens.

**Why it gives the same optimum:**
- Training a reward model with Bradley–Terry and then solving Eq. 3 exactly gives π* via Eq. 4.
- DPO fits the reward model **parameterised by a policy**, r̂_θ = β log π_θ/π_ref. Its best fit **is** that π*.
- Hence the subtitle: *your language model is secretly a reward model.*

### Worked example of the loss
Take log π_θ(y_w) = −10, log π_θ(y_l) = −12, log π_ref of both = −11, and β = 0.5.
- **Implicit rewards:**
  - r̂_w = 0.5·(−10 + 11) = **+0.5**;
  - r̂_l = 0.5·(−12 + 11) = **−0.5**.
- **Loss:** −log σ(0.5 − (−0.5)) = −log σ(1) = −log 0.731 = **0.313**. (Our demo prints exactly this.)

---

## 3. What the gradient does
```
∇L_DPO = −β E[ σ(r̂_l − r̂_w) · ( ∇log π(y_w|x) − ∇log π(y_l|x) ) ]
```
- **Push up** the probability of the preferred answer and **push down** the dispreferred one.
- **Weight each pair by σ(r̂_l − r̂_w):** how wrong the implicit reward model currently is.
  - Pairs already ranked correctly by a wide margin get little weight.
  - Wrongly ranked pairs get a lot of weight.
  - In the example above the weight is σ(−1) = **0.269**: the pair is already ranked correctly.
- **The weighting is crucial.** The paper reports that a naive version without it makes the language model **degenerate** (Appendix Table 3). That naive version is close to the **Unlikelihood** baseline: maximise log p(y_w), minimise log p(y_l).
  - Without the weight, the model keeps pushing the loser's probability down forever. Driving any sequence's probability to 0 is easy, and it wrecks the rest of the distribution.
  - The σ weight turns this into a **bounded classification** problem that stops once the pair is correctly ranked by the margin the data supports.
  - π_ref keeps the push **relative** to where the model started.

---

## 4. Theory (Section 5)
- **Equivalence classes:** two rewards that differ by a function of x only, r′(x, y) = r(x, y) + f(x), give **the same preferences** and **the same optimal policy**.
- **Theorem 1:** every equivalence class contains a reward of the form β log π(y|x)/π_ref(y|x). So DPO's parameterisation **loses no generality**.
- **Why PPO struggles (their view):**
  - the actor-critic needs to estimate the normaliser (a value baseline);
  - the high-variance reward makes PPO unstable.
  - DPO avoids both because the normaliser cancels analytically.

---

## 5. Experiments and results

**Setup (Appendix B):**
- defaults: β = 0.1, batch size 64, RMSprop with learning rate 1e-6, linear warm-up over 150 steps;
- **β = 0.5 for TL;DR**;
- the paper says they did **not** meaningfully tune β.

| Task | Setup | Result |
|---|---|---|
| **IMDb controlled sentiment** | GPT-2-large SFT; preferences labelled by a sentiment classifier; ground-truth reward known | DPO has **by far the most efficient reward-KL frontier**, dominating PPO and even **PPO-GT** (PPO with the true reward) |
| **TL;DR summarisation** | GPT-J SFT; GPT-4 judges win rate vs reference summaries | DPO **≈ 61%** at temperature 0 vs PPO **57%** at its best temperature; DPO is much more robust to sampling temperature; Preferred-FT barely beats SFT |
| **Human evaluation, TL;DR** | | DPO (temp 0.25) preferred over PPO (temp 0) **58%** of the time |
| **Anthropic HH single-turn dialogue** | Pythia-2.8B, Preferred-FT reference | DPO is the **only** computationally efficient method that improves over the chosen responses; similar to or better than **Best-of-128** |
| **Out-of-distribution: CNN/DailyMail** | TL;DR policies on news | GPT-4 win rate vs reference: **DPO 0.36 / 0.31**, **PPO 0.26 / 0.23** (temp 0 / 0.25) |

**Other findings:**
- GPT-4 judgments agree with humans about as often as humans agree with each other.
- DPO converges relatively quickly.

---

## 6. Why it matters
- **Simplicity:** a few lines of code (the paper prints them in Appendix B), no RL, no reward model, no sampling loop.
- **Adoption:** DPO and its many variants became the default for open-model preference tuning: Zephyr, Tulu 2, Llama 3 post-training (with rejection sampling), and others.
- **Theory:** it showed the reward model and the policy are two views of the same thing.
- **Limits:**
  - DPO is **offline**: it learns only from a fixed dataset of pairs (sampled from the SFT model), with no fresh exploration.
  - Later work found it can overfit, or lower the likelihood of **both** answers.
  - Online or iterative DPO, IPO and KTO address these points.

---

## 7. What our code found

### The bandit: everything is exact
- **Setup:** one prompt, 6 answers, π_ref = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03], r = [0, 1, 2, −1, 3, 0.5].
- **Training:** a softmax policy trained on the **exact expected** DPO loss (all pairs, Bradley–Terry labels).
- **Result: DPO equals the closed form π_ref·e^(r/β)/Z to 1e-16** for β = 0.5, 1 and 2. Equations 4 and 7 agree numerically.

| β = 1 | Policy | KL to π_ref | E[r] |
|---|---|---|---|
| DPO = optimum | [0.076, 0.171, 0.372, 0.014, 0.354, 0.012] | 0.59 | **1.97** |
| Unlikelihood (no σ weight, no π_ref) | [0, 0.481, 0.385, 0, 0.135, 0] | 0.65 | 1.65 |
| Preferred-FT (SFT on winners) | [0.219, 0.279, 0.293, 0.061, 0.121, 0.028] | 0.08 | 1.18 |

- **Unlikelihood moves as far from π_ref as DPO** but gets less reward. It zeroes out every answer that ever loses, including some decent ones, and gives the best answer (r = 3) only 13%.

### Controlled sentiment with a real tiny language model
- **The model:** a GRU over a 16-word vocabulary: 3 positive, 3 negative and 9 neutral words, plus a BOS token.
- **The task:**
  - prompts are 2 words; completions are 6 words;
  - the ground-truth reward is +1 per positive word and −1 per negative word (our "sentiment classifier");
  - the SFT model scores −0.04.
- **The data:** 2000 pairs sampled from SFT and labelled by Bradley–Terry on the true reward, just as the paper labelled IMDb pairs with a classifier.
- **The exact frontier:** because words are independent given the prompt, the best possible reward at every KL can be computed **exactly**.

| Method | β | KL | True reward | Best possible at that KL | Fraction of best |
|---|---|---|---|---|---|
| DPO | 0.5 | 3.74 | +3.93 | +4.00 | 98% |
| RLHF (RM + policy gradient) | 0.5 | 4.34 | +4.30 | +4.28 | 100% |
| DPO | 1 | 1.47 | +2.40 | +2.56 | 94% |
| RLHF | 1 | 1.33 | +2.41 | +2.43 | 99% |
| DPO | 2 | 0.55 | +1.36 | +1.57 | 87% |
| RLHF | 2 | 0.32 | +1.20 | +1.21 | 99% |
| Preferred-FT | | 0.17 | +0.55 | +0.87 | 63% |
| Unlikelihood α = 0.1 | | 0.23 | +0.67 | +1.02 | 65% |
| Unlikelihood α = 1 | | **6.68** | +2.86 | +5.17 | **55%** |
| Best-of-4 | | 0.64 | +1.50 | +1.69 | 89% |
| Best-of-16 | | 1.84 | +2.63 | +2.85 | 92% |

- **DPO works.** With no reward model and no sampling during training, it reaches 87–98% of the best possible reward at its KL. It beats Preferred-FT, Unlikelihood and Best-of-N.
- **Unlikelihood with α = 1 runs far away** from the reference (KL 6.7) for little reward, which is the degeneration the σ weight prevents.
- **Honest negative:** we do **not** reproduce "DPO strictly dominates PPO". Our RLHF baseline is closer still (99–100%). Our toy favours it:
  - the reward model is exactly the right family (bag of words);
  - RL samples fresh completions every step;
  - DPO sees only the 2000 fixed pairs.
  - DPO's gap is largest at large β (small KL), where finite-data noise in the tiny implicit rewards matters most.
- **"Secretly a reward model":** on 2000 held-out pairs with noisy labels, ranking accuracy is:
  - **78.4%** for the true reward (the ceiling);
  - **77.2%** for the explicit reward model;
  - **76.6%** for the DPO policy's implicit reward β log π/π_ref.

**`experiments.py`:**
- **E1:** a multi-seed frontier sweep;
- **E2:** data size;
- **E3:** label noise;
- **E4:** the σ-weight and reference-model ablations;
- **E5:** a real DPO fine-tune of a 0.5B model on Anthropic HH with the paper's hyper-parameters.
- E1–E4 run in minutes; E5 needs a GPU. None were run here.

---

## 8. Check yourself

1. What is the closed-form solution of the KL-constrained RLHF objective?
<details><summary>Answer</summary>π*(y|x) = π_ref(y|x)·exp(r(x, y)/β)/Z(x): the reference re-weighted by exponentiated reward, normalised by Z(x).</details>

2. Why can't we just compute π* from that formula?
<details><summary>Answer</summary>Z(x) sums over all possible responses (every sentence), which is intractable.</details>

3. Why does Z(x) disappear in DPO?
<details><summary>Answer</summary>Writing r = β log π*/π_ref + β log Z(x), the β log Z(x) term depends only on x. Bradley–Terry uses r(y_w) − r(y_l), so it cancels.</details>

4. π_ref = [0.5, 0.5], r = [0, 1], β = 1. What is π*?
<details><summary>Answer</summary>Weights 0.5 and 0.5e ≈ 1.359; π* ≈ [0.269, 0.731].</details>

5. log π_θ(y_w) = −10, log π_θ(y_l) = −12, both reference log-probs are −11, β = 0.5. What are the loss and the gradient weight?
<details><summary>Answer</summary>Implicit rewards +0.5 and −0.5; loss = −log σ(1) ≈ 0.313; weight σ(r̂_l − r̂_w) = σ(−1) ≈ 0.269.</details>

6. What role does β play?
<details><summary>Answer</summary>It is the strength of the KL penalty. A small β lets the policy move far from π_ref toward high reward; a large β keeps it close. In DPO it scales the implicit reward.</details>

7. Why is the σ(r̂_l − r̂_w) weight important?
<details><summary>Answer</summary>It concentrates updates on pairs the implicit reward currently ranks wrongly and fades out once a pair is ranked correctly. Without it (unlikelihood-style updates), the model keeps pushing probabilities down without limit and degenerates.</details>

8. What does "your language model is secretly a reward model" mean?
<details><summary>Answer</summary>Any policy defines a reward β log π(y|x)/π_ref(y|x) (up to a per-prompt constant). The policy that best fits the preference data under that reward is exactly the KL-regularised optimal policy, so training the LM on preferences is training a reward model.</details>

9. Name two practical advantages of DPO over PPO-based RLHF.
<details><summary>Answer</summary>No separate reward or value model and no sampling during training (less compute and memory), and fewer, more stable hyper-parameters (a supervised-style loss).</details>

10. In our toy, RLHF was slightly closer to the optimal frontier than DPO. Give a reason.
<details><summary>Answer</summary>Our reward model is exactly the right functional form, and RL samples fresh completions every step, while DPO only sees 2000 fixed pairs. Finite-data noise hurts its implicit reward most at large β.</details>
