# Learning to Summarize from Human Feedback, explained simply

**Paper:** Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei & Paul Christiano (OpenAI), *Learning to summarize from human feedback*, NeurIPS 2020.

**In one sentence:** instead of training a summariser to **copy** human-written summaries, ask people **which of two summaries is better**, train a **reward model** to predict their choice, and optimise the summariser against that reward with reinforcement learning (PPO), while a **KL penalty** keeps it close to where it started. The result beat summaries written by people.

---

## 1. The big idea

### 1.1 What's wrong with imitation?
- **Supervised fine-tuning (SFT)** trains the model to maximise the likelihood of **reference summaries**: "predict what the human wrote".
- **Problem 1, references vary in quality:** Reddit TL;DRs, written by the posters themselves, are sometimes great and sometimes lazy. Imitation copies the bad habits too.
- **Problem 2, likelihood treats every token equally:** a summary with one wrong fact can still have high likelihood. What we care about is **quality as judged by people**.
- **Problem 3, automatic metrics are poor proxies:** metrics like ROUGE (word overlap with the reference) correlate poorly with human judgement.

### 1.2 Optimise what people actually want
People find it hard to write a perfect summary, or to give a score from 1 to 10, but **easy to compare two**. So:
1. collect **comparisons**;
2. learn a **reward model** that predicts them;
3. **optimise** the policy against it.

This is **RLHF** (reinforcement learning from human feedback), the same recipe that later trained InstructGPT and ChatGPT (paper 067).

---

## 2. The method (Section 3, Figure 2)

### 2.1 Data and models
- **TL;DR:** Reddit posts with the poster's own one-line summary, filtered to about 123k posts.
- **Models:** GPT-3-style Transformers with **1.3B** and **6.7B** parameters, pre-trained, then fine-tuned on the TL;DRs. This SFT model is the starting point.

### 2.2 Step 1: collect human comparisons
- **What labelers see:** a post and two summaries (from the current policies, the SFT model, the references, …); they pick the better one.
- **How many:** **64,832 comparisons** in total.
- **Quality control was intense:**
  - labelers were hired from Upwork, Scale and Lionbridge;
  - they had detailed instructions and frequent feedback from the researchers;
  - **labeler–researcher agreement was 77% ± 2%,** higher than researcher–researcher agreement (73%).

### 2.3 Step 2: train a reward model
- **Architecture:** start from the SFT model and replace the output layer with a single number r_θ(x, y), the quality of summary y for post x.
- **Train on comparisons:**
```
loss(r_θ) = − E [ log σ( r_θ(x, y_preferred) − r_θ(x, y_other) ) ]
```
  This is the Bradley–Terry model (also used in paper 063): P(A preferred) = σ(r_A − r_B).
- **Normalisation:** the outputs are shifted so the **reference summaries average 0**. A positive reward then means "better than the human reference".

**Worked example:**
- r(preferred) = 1.0 and r(other) = −0.5 give σ(1.5) = 0.818, so the loss is −ln 0.818 = 0.20.
- If the reward model has them backwards, σ(−1.5) = 0.182 and the loss is 1.70.

### 2.4 Step 3: optimise the policy with RL
- **Each summary is an "episode";** each token is one action.
- **The reward at the end:**
```
R(x, y) = r_θ(x, y) − β · log[ π_RL(y | x) / π_SFT(y | x) ]
```
- **What the KL term does:** it charges the policy for making y more likely than the SFT model would. Summed over samples, that is the KL divergence between the two policies.
  1. **It acts as an entropy bonus:** it keeps the policy exploring, preventing collapse onto one answer.
  2. **It stops the policy from drifting** to outputs unlike anything the reward model saw in training, where the RM's judgements are unreliable.
- **The algorithm is PPO (Proximal Policy Optimization):**
  - **The objective:** policy-gradient updates whose step size is limited by **clipping** the probability ratio π_new/π_old to [1 − ε, 1 + ε].
  - **Advantages:** estimated with a learned **value function**, i.e. how much better this outcome was than expected.
- **The value function is a separate Transformer, initialised from the reward model.** Sharing it with the policy would let value updates damage the pre-trained policy early on.

---

## 3. Results (Section 4)

### 3.1 Better than human references (Figure 1)
Preference against the reference summaries:

| Model | Preferred to reference |
|---|---|
| 6.7B supervised | **43%** |
| **1.3B human feedback** | **61%** |
| 6.7B human feedback | higher still; ~65% after controlling for length |

- **A 1.3B RLHF model beats a supervised model 10× its size.**
- **Length check:** RLHF summaries are a bit longer, and labelers like length somewhat, so the authors controlled for it. The advantage shrinks by about 5% but remains.
- **On a 7-point scale** (coverage, accuracy, coherence, overall), the 6.7B RLHF model is best on all four axes, especially coverage.

### 3.2 Transfer to news (Figure 4)
- **No retraining:** applied to **CNN/DailyMail** news articles without any news-specific training, the RLHF model nearly matched a model fine-tuned on CNN/DM itself, while writing summaries about half as long.

### 3.3 Optimising the reward model too hard (Figure 5)
- **The experiment:** train policies against the RM with progressively smaller KL penalties (stronger optimisation).
- **The result:**
  - at first, real (human-judged) quality rises as the RM score rises;
  - **with too much optimisation,** true quality falls even as the RM score keeps rising;
  - eventually the RM is **anti-correlated** with people.
- **This is Goodhart's law:** "when a measure becomes a target, it ceases to be a good measure". The same happens when optimising ROUGE.

### 3.4 Understanding the reward model (Figure 6)
- **Scaling (7 RMs, 160M–13B parameters, 8k–64k comparisons):**
  - **doubling the data** gives **+1.1%** accuracy;
  - **doubling the model** gives **+1.8%**;
  - the 6.7B RM trained on all data approaches a single human's agreement rate.
- **Sensitivity:** RMs prefer minimally improved edits 79–83% of the time (humans 84%). They detect swapped roles (who did what to whom) 93–97% of the time.
- **A bias:** RMs prefer **longer** summaries. They favour shortening edits only 62.6% of the time vs 76.4% for humans.
- **Metrics:** the RM predicts human preferences much better than ROUGE, length or copying. Optimising ROUGE gives worse summaries than optimising the RM.

---

## 4. Why it matters
- **It was the first large-scale demonstration of RLHF for language,** and the direct predecessor of InstructGPT (paper 067) and ChatGPT.
- **It made reward-model over-optimisation and the KL penalty** central topics in alignment research.
- **It released 64k+ comparisons** (`openai/summarize_from_feedback`), still used to study reward models and DPO (paper 069).
- **It shows the value of careful human data:** agreement checks, labeler training, and the cost of collecting it.

---

## 5. What our code found
We built the **entire pipeline with real (tiny) neural networks**, on a toy task where the "human" judgement is known.
- **The task:** a "post" is 14 tokens with three topics (appearing 4, 3 and 2 times) plus filler. A good summary names the **two most frequent topics** in order: "T7 T9".
- **The labelers' true utility:**
  - +1 per main topic and +0.3 for the right order;
  - −1 per hallucinated token, −0.5 per filler word;
  - −0.6 per repetition, −0.4 per token beyond 3.
  - Labelers compare noisily: P(A) = σ((q_A − q_B)/0.5).
- **References (like real TL;DRs) are noisy:** half name only the main topic, and others add the right or wrong second topic, or filler.
- **The pipeline (tiny LLaMA-style networks from paper 056):**
  1. **SFT** on the references;
  2. **comparisons:** 3,000 labelled pairs;
  3. **the reward model,** initialised from SFT with a scalar head, trained on the Bradley–Terry loss and normalised so references score 0;
  4. **PPO,** with:
     - the per-token KL penalty to SFT;
     - clipped ratios (ε = 0.2) and GAE (λ = 0.95);
     - a separate value network initialised from the RM.

| Policy (greedy) | True quality | RM score | Preferred to reference | Example (ideal → output) |
|---|---|---|---|---|
| SFT (imitation) | 1.35 | −0.40 | 50.8% | T1 T4 → T1 |
| PPO, β = 0 (no KL) | 1.81 | **1.53** | 66.6% | T1 T4 → **T1 T1** |
| **PPO, β = 0.2** | **2.02** | 1.44 | **73.8%** | T1 T4 → T1 T4 |

- **Imitation copies the references' habit** of naming only one topic, so it is preferred to them only ~51% of the time.
- **Optimising human preferences** learns to name both main topics, and is preferred to the references 74% of the time (the paper: 61% for 1.3B).
- **Without the KL penalty, the policy finds what the reward model over-rates:** repeating the main topic ("T1 T1"). It gets the **highest RM score** but **lower true quality** than with the penalty. That is the start of Figure 5's over-optimisation.
- **Reward-model accuracy is 58%** on held-out comparisons. Our labelers are deliberately noisy, so even a perfect RM couldn't reach 100%. The RM was still good enough to steer PPO.

**`experiments.py` (written, not run on this laptop):**
- **E1:** a β sweep (KL vs RM score vs true quality: the full Figure 5 curve);
- **E2:** RM accuracy vs number of comparisons and model width (Figure 6);
- **E3:** best-of-n vs PPO;
- **E4:** optimising a ROUGE-like overlap with one noisy reference vs the RM;
- **E5:** a **real** reward model on the released `openai/summarize_from_feedback` comparisons.

---

## 6. Check yourself

1. Why compare two summaries instead of rating one?
<details><summary>Answer</summary>Comparisons are easier and more consistent for people than absolute scores. The Bradley–Terry model turns them into a scalar reward: P(A better) = σ(r_A − r_B).</details>

2. Compute the reward-model loss when r(preferred) = 2.0 and r(other) = 1.0.
<details><summary>Answer</summary>−ln σ(1.0) = −ln 0.731 = 0.31.</details>

3. Write the RL reward and explain both terms.
<details><summary>Answer</summary>R = r_θ(x, y) − β log[π_RL(y|x)/π_SFT(y|x)]. The first term is the learned human-preference score. The second penalises moving away from the SFT model: it acts as an entropy bonus and keeps the policy in the region where the reward model is reliable.</details>

4. What happens when β is too small, and why?
<details><summary>Answer</summary>The policy over-optimises the imperfect reward model: it finds outputs the RM scores highly but people don't like. True quality first rises, then falls, while the RM score keeps rising (Figure 5). In our toy, β = 0 learned to repeat the main topic.</details>

5. Why is the value function a separate network initialised from the reward model?
<details><summary>Answer</summary>If it shared parameters with the policy, value-function updates early in training could damage the pre-trained policy. Initialising it from the RM gives a sensible starting estimate of returns, since the RM already predicts quality.</details>

6. A 1.3B RLHF model beat a 6.7B supervised model. What does that suggest?
<details><summary>Answer</summary>The training objective matters more than scale here: optimising what people prefer beats imitating references, even with 5× fewer parameters (61% vs 43% preferred to references).</details>

7. Why normalise the reward model so references score 0?
<details><summary>Answer</summary>The Bradley–Terry loss only fixes reward differences, not their absolute level. Shifting so references average 0 makes the scale interpretable (positive means better than the human reference) and keeps the RL reward centred.</details>

8. What bias did the paper find in its reward models?
<details><summary>Answer</summary>A preference for longer summaries: the 6.7B RM preferred shortening improvements only 62.6% of the time vs 76.4% for humans.</details>

9. In our toy, SFT is preferred to the references only ~51% of the time. Why so low?
<details><summary>Answer</summary>SFT imitates the references, including their most common habit (naming only the main topic). Greedy decoding picks that most likely pattern, so its summaries are about as good as a typical reference.</details>

10. What is Goodhart's law, and where does it show up here?
<details><summary>Answer</summary>"When a measure becomes a target, it ceases to be a good measure." The reward model (or ROUGE) is a measure of human preference. Optimising it too hard exploits its errors, so the measure goes up while real quality goes down.</details>
