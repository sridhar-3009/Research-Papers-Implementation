# GPT-4 Technical Report, explained simply

**Report:** OpenAI, *GPT-4 Technical Report*, arXiv 2303.08774, 2023.

**In one sentence:** GPT-4 is a large multimodal model (text and image in, text out). It scores around the **top 10%** on a simulated bar exam and **86.4%** on MMLU, and OpenAI could **predict parts of its performance before training it**. The report deliberately does **not** say how big it is, what data it saw, or how it was trained.

---

## 1. What is (and isn't) in this report
**Section 2 says it directly:**
> "Given both the competitive landscape and the safety implications of large-scale models like GPT-4, this report contains no further details about the architecture (including model size), hardware, training compute, dataset construction, training method, or similar."

**What we know:**
- **Architecture and training:** a Transformer pre-trained to predict the next token (public internet data plus licensed data), then fine-tuned with **RLHF** (reinforcement learning from human feedback).
- **Evaluation:** a large evaluation on exams and benchmarks.
- **Two methods described in some detail:**
  - **predictable scaling** (forecasting the big model from small ones);
  - **safety training** with **rule-based reward models**.
- **Discussion:** limitations, risks and mitigations, plus a long **system card** (an appendix about risks).

So unlike papers 048–056, there is no model to rebuild. **What we can implement is the methodology:**
- loss and capability prediction;
- calibration measurement;
- contamination checking;
- rubric-based rewards.

---

## 2. Predictable scaling (Section 3)
**Why it matters:**
- A run like GPT-4 is too expensive to tune by trial and error.
- If you can **predict** how it will turn out from runs **1,000–10,000× smaller**, you can:
  - choose its settings safely;
  - know in advance what capabilities to prepare for, which matters for safety.

### 3.1 Predicting the loss
- **The law:** following Kaplan (paper 051) and Henighan et al., the final loss follows a power law in training compute C with an **irreducible** term:
```
L(C) = a · C^b + c        (b < 0)
```
- **The irreducible term c:** the loss you can never go below, because text has inherent randomness (the entropy of language).
- **The decaying term** a·C^b is the part that more compute removes.
- **What OpenAI did:**
  - fitted this law to models using **at most 1/10,000** of GPT-4's compute, on an internal code dataset not in the training set;
  - made the prediction **shortly after the run started**, without using partial results;
  - GPT-4 landed on the curve (Figure 1).

**Why the c term matters, with a worked example.** Suppose the true law is L = 0.5·C^(−0.1) + 1.2, with compute normalised so the big run is C = 1.
- **The truth at C = 1:** L = 0.5 + 1.2 = **1.7**.
- **A pure power law** (no c) fitted to small runs treats log L as a straight line in log C. But L can't go below 1.2, so log L flattens out, and a straight-line fit pointed at the far future misses the flattening.
- **Fitting the c term** recovers the bend. Our test fits runs from C = 10⁻⁸ to 10⁻⁴ and predicts **1.70** at C = 1, a 10,000× extrapolation.

**How to fit it** (our `fit_power_law_offset`):
1. Try many values of c.
2. For each one, log(L − c) = log a + b·log C is a straight line, so fit it by least squares.
3. Keep the c with the smallest error.

### 3.2 Predicting a capability: HumanEval
- **The question:** loss is smooth, but what people care about is **capability**, e.g. "does the code pass the tests?" (paper 054).
- **The difficulty:** the pass rate of a single problem jumps around with scale.
- **The quantity that turned out to be predictable:**
```
−E_P[ log(pass_rate(C)) ]  =  α · C^(−k)
```
  the **average of −log(pass rate)** over a set of problems P.

**Why the log?**
- **It is smooth:** a pass rate of 0.001 → 0.01 → 0.1 is a steady improvement in log space (−log p goes 6.9 → 4.6 → 2.3), even though the raw rate looks like "nothing, then suddenly something".
- **Example:** pass rates 0.5 and 0.25 give −mean log = (0.693 + 1.386)/2 = **1.04**.

**Practical rules:**
- **The log of 0 is infinite,** so only problems that **every** model solves at least once (with a large sample budget) are kept.
- **Buckets:**
  - the problems (except the 15 hardest) were split into **6 difficulty buckets** using small-model performance;
  - Figure 2 shows the 3rd-easiest bucket, extrapolated from models with **at most 1/1,000** of the compute;
  - the predictions were accurate, except GPT-4 **under**-performed the prediction on the easiest bucket.
- **They registered these predictions before training finished.**

### 3.3 Not everything is predictable: inverse scaling
- **Inverse scaling:** the **Inverse Scaling Prize** collected tasks where bigger models do **worse**.
- **Hindsight Neglect** is one: judge a bet by its **expected value**, not by how it happened to turn out.
  - **Example:** "Michael has a 20% chance to win $500 and an 80% chance to lose $500. He takes the bet and wins. Was it a good decision?"
  - The expected value is 0.2 × 500 − 0.8 × 500 = **−$300**, so the answer is **No**, even though he won.
  - Small models answer from the outcome. Accuracy **fell** from ada to babbage to curie (the API models), so extrapolating that trend would predict GPT-4 to be bad.
  - **GPT-4 reverses the trend** (Figure 3), getting it right.
- **Lesson:** extrapolating trends can fail; some abilities appear only beyond a certain scale.

---

## 3. Capabilities (Section 4)

### 3.1 Exams (Table 1)
Simulated real exams, given as they would be to humans (no exam-specific training):

| Exam | GPT-4 | GPT-3.5 |
|---|---|---|
| Uniform Bar Exam | 298/400 (~**90th** percentile) | 213/400 (~10th) |
| LSAT | 163 (~88th) | 149 (~40th) |
| SAT Math | 700/800 (~89th) | 590/800 (~70th) |
| GRE Verbal | 169/170 (~99th) | 154/170 (~63rd) |
| GRE Quantitative | 163/170 (~80th) | 147/170 (~25th) |
| USA Biology Olympiad semifinal | 87/150 (99–100th) | 43/150 (31–33rd) |
| AP Calculus BC | 4 (43–59th) | 1 (0–7th) |
| AP English Literature | 2 (8–22nd) | 2 (8–22nd) |
| Codeforces rating | 392 (below 5th) | 260 (below 5th) |

- **The exam scores come mostly from pre-training:** on multiple-choice questions, the base model and the RLHF model score about the same.
- **Weak spots:** English literature and competitive programming.

### 3.2 Benchmarks (Table 2)
| Benchmark | GPT-4 | GPT-3.5 | Best LM (few-shot) | SOTA (any method) |
|---|---|---|---|---|
| MMLU (5-shot) | **86.4** | 70.0 | 70.7 | 75.2 |
| HellaSwag (10-shot) | **95.3** | 85.5 | 84.2 | 85.6 |
| ARC (25-shot) | **96.3** | 85.2 | 85.2 | 86.5 |
| WinoGrande (5-shot) | **87.5** | 81.6 | 85.1 | 85.1 |
| HumanEval (0-shot) | **67.0** | 48.1 | 26.2 | 65.8 |
| DROP (F1, 3-shot) | 80.9 | 64.1 | 70.8 | **88.4** |
| GSM-8K (5-shot, chain of thought) | **92.0** | 57.1 | 58.8 | 87.3 |

- **GPT-4 beats every previous language model** on these benchmarks, and beats specially-trained state-of-the-art systems on all of them except DROP.
- **Caveat:** part of GSM-8K's *training* set was mixed into GPT-4's pre-training.
- **Multilingual:** on MMLU machine-translated into 26 languages, GPT-4 beats the previous English-language state of the art (GPT-3.5, Chinchilla, PaLM) in 24 of them.

### 3.3 Contamination checks (Appendix C)
**The problem:** if test questions appear in the training data, the score measures memory, not ability.

**OpenAI's check:**
1. Remove all spaces and symbols from both the evaluation example and the training data, keeping only letters and digits.
2. Pick **3 random 50-character substrings** of the evaluation example (or the whole example if it is shorter).
3. If **any** of them appears in the training data, mark the example as contaminated.
4. Report scores with and without the contaminated examples. For most exams it made little difference.

**Limitations:**
- **False negatives:** a paraphrase or a small edit escapes the check.
- **False positives:** the check uses only the question, not the answer.
- **BIG-bench:** portions of it turned out to be in the training data, so it was excluded from the results.

---

## 4. Limitations (Section 5)
- **Hallucination:** it still makes things up.
  - On OpenAI's internal adversarial factuality tests, GPT-4 is **19 percentage points** better than the latest GPT-3.5, but far from perfect.
  - On TruthfulQA, the base model is only slightly better than GPT-3.5; **after RLHF** it is much better.
- **Knowledge cut-off:** mostly September 2021, and it doesn't learn from experience.
- **Calibration** (Figure 8):
  - **Calibrated** means: when the model says it is 70% sure, it is right 70% of the time.
  - **The pre-trained** GPT-4 is very well calibrated on MMLU (**ECE 0.007**).
  - **After RLHF** it is much less so (**ECE 0.074**).

### 4.1 Expected calibration error (ECE)
How to compute it:
1. Put each answer into a bin by its confidence (0–10%, 10–20%, …, 90–100%).
2. In each bin, compare the **average confidence** with the **actual accuracy**.
3. ECE = Σ over bins of (fraction of answers in the bin) × |accuracy − confidence|.

**Worked example:** 10 answers, all at 90% confidence.
- If 9 are right, the accuracy is 90% and **ECE = 0**: perfect calibration.
- If only 5 are right, the accuracy is 50% and **ECE = |0.5 − 0.9| = 0.4**: overconfident.

**Why RLHF hurts calibration:** RLHF optimises for answers that people (or reward models) like, not for probabilities that match reality. That pushes the model toward confident answers.

---

## 5. Making it safer (Section 6)
- **Expert red-teaming:** over 50 experts probed the model for risks (cybersecurity, bio-risk, and more).
- **The model-assisted safety pipeline:**
  - **Problem 1:** after RLHF, models can still be unsafe on some inputs (e.g. giving crime advice) and **over-cautious** on others (refusing innocent requests).
  - **The fix:** extra safety-focused RLHF prompts plus **rule-based reward models (RBRMs)**.

### 5.1 How an RBRM works
An RBRM is a **zero-shot GPT-4 classifier**. It reads three things:
- the prompt;
- the policy model's response;
- a human-written **rubric**.

It then classifies the response, for example as:
- (a) a refusal in the desired style;
- (b) a refusal in an undesired style (evasive, rambling, preachy);
- (c) a response containing disallowed content;
- (d) a safe response that doesn't refuse.

**Reward:**
- On **harmful** prompts, reward (a).
- On prompts known to be **safe**, reward (d), so the model stops refusing innocent requests.

This gives a much more targeted reward signal than general human preferences.

### 5.2 Results
- **82% fewer** responses to requests for disallowed content than GPT-3.5.
- **29% more often** follows policy on sensitive requests (medical advice, self-harm).
- **Toxicity:** 0.73% toxic generations on RealToxicityPrompts, vs 6.48% for GPT-3.5.
- **Jailbreaks** still exist.

---

## 6. Why it matters
- **Scale plus post-training produced professional-level test performance.** The report also marks the move from open science (GPT-2, GPT-3 papers) to closed reports.
- **Predictable scaling** became an explicit goal: **forecast** capabilities before training, and pre-register the predictions.
- **The evaluation toolkit** (exam suites, contamination checks, calibration, red-teaming, model-graded rubrics) became standard practice for frontier models.

---

## 7. What our code found
Everything below runs in ~5 seconds on a laptop.

**Exact checks (tests):**
- **`fit_power_law_offset`** recovers b = −0.1 and c = 1.2 and predicts 1.70 at a **10,000×** extrapolation.
- **The capability metric:** the "solved by every model" filter, −mean log pass rate, the α and k fit, and 6 difficulty buckets with the 15 hardest problems excluded.
- **ECE:** 0 when 90% confidence goes with 90% accuracy, and 0.4 when it goes with 50%; a calibrated simulation has ECE < 0.02 and an overconfident one > 0.1.
- **The contamination check:**
  - it flags a copy with different spacing, punctuation and capitals;
  - it ignores unrelated text;
  - it **misses a paraphrase**, the false negative the report acknowledges.
- **The toy RBRM** rewards refusals of harmful prompts and non-refusals of safe prompts.
- **Hindsight Neglect items:** the label always follows the expected value, and the outcome always points the other way.
- **Table 2:** GPT-4 beats SOTA on 6 of 7, all except DROP.

**Demo: predicting a real model's loss.**
- **Setup:** six tiny byte-level LLaMA-style models (paper 056), 1.7k–56k non-embedding parameters, trained on this repository's text.
- **The fit** on the five smaller runs: L = 0.282·C^(−0.309) + 2.192.
- **The prediction** for the largest run (3.3× more compute than the next): **2.475**; actual **2.361**. That is a **4.8%** over-prediction: the big model did *better* than forecast.
- **The honest comparison:** OpenAI extrapolated 10,000× and hit the curve. Our tiny runs share one hand-picked learning rate and very few steps, so they are less "properly trained", and their curve is less clean.

**Demo: the capability metric** (simulated under the report's own power-law hypothesis):
- 57 of 60 problems survive the "solved by every model" filter.
- The fit gives **k = 0.251** (true 0.25) and predicts **0.067** at C = 1, vs a measured **0.069**.
- Meanwhile the mean pass@1 bends toward 1 (0.26 → 0.71 → 0.94), which is harder to extrapolate.
- **Caveat:** this only shows the *method* works when the hypothesis holds.

**Demo: calibration** (simulated): calibrated ECE 0.014 vs overconfident ECE 0.117.

**`experiments.py` (written, not run on this laptop):**
- **E1:** a real model family across ~1000× compute, with the prediction **written to disk before** the largest run;
- **E2:** −mean log pass rate on addition problems, with digit-count buckets;
- **E3:** a Hindsight Neglect probe across sizes;
- **E4:** ECE before and after a REINFORCE "reward the right answer" fine-tune (an RLHF stand-in);
- **E5:** detection rates of the substring check for verbatim, re-formatted and lightly edited copies, for different substring lengths and sample counts.

---

## 8. Check yourself

1. What does the report refuse to disclose, and why does that matter for reproduction?
<details><summary>Answer</summary>Model size, architecture details, hardware, training compute, dataset construction and training method. Without them nobody can rebuild or independently verify GPT-4. Only its evaluated behaviour and some methods (prediction, safety pipeline, evaluation) can be studied.</details>

2. Why does the loss law include an irreducible term c?
<details><summary>Answer</summary>Text has inherent unpredictability (its entropy), so loss can't go to zero no matter the compute. Without c, a fit to small models treats the curve as an unbounded straight line in log-log and mis-predicts where it flattens. With c, L = a·C^b + c approaches c as C grows.</details>

3. Why predict −mean log(pass rate) instead of the pass rate itself?
<details><summary>Answer</summary>Pass rates can sit near 0 for small models and then shoot up, which is hard to extrapolate. Their logarithm changes smoothly (a power law in compute), so it can be fitted on small models and extrapolated. Problems with zero passes must be excluded because log 0 is infinite.</details>

4. Compute −mean log pass rate for pass rates 0.1, 0.5 and 1.0.
<details><summary>Answer</summary>−(ln 0.1 + ln 0.5 + ln 1)/3 = (2.303 + 0.693 + 0)/3 = 0.999.</details>

5. Why is Hindsight Neglect a warning about prediction?
<details><summary>Answer</summary>Small models got worse with size (inverse scaling), so extrapolating would predict GPT-4 does badly. GPT-4 instead does well: the trend reversed. Some capabilities can't be forecast by simple extrapolation.</details>

6. A model answers 100 questions with 80% confidence each and gets 60 right. What is its ECE?
<details><summary>Answer</summary>All answers fall in one bin, so ECE = |0.60 − 0.80| = 0.20. It is overconfident by 20 points.</details>

7. Why can RLHF make a model less calibrated?
<details><summary>Answer</summary>RLHF optimises a reward (human or model preference), not log-likelihood. It pushes the model toward answers that are preferred and confident, which distorts the probabilities away from the true frequency of being right (ECE 0.007 → 0.074 in the report).</details>

8. How does the contamination check work, and what can it miss?
<details><summary>Answer</summary>Strip spaces and symbols from both texts, sample three 50-character substrings of the eval example, and flag it if any appears in the training data. It misses paraphrases and small edits (false negatives). Because it ignores the answers, it can also flag examples whose question appears but whose answer doesn't (false positives).</details>

9. What problem do rule-based reward models solve that ordinary RLHF struggled with?
<details><summary>Answer</summary>Ordinary RLHF left the model both unsafe on some harmful prompts and over-refusing on safe ones, because labeller instructions were underspecified. RBRMs give a precise, rubric-based reward: refusals are rewarded on harmful prompts, helpful non-refusals on safe prompts, and preachy or evasive refusals are penalised.</details>

10. Our tiny-model loss prediction was off by 4.8% after a 3.3× extrapolation, while GPT-4's was accurate after 10,000×. Give two reasons.
<details><summary>Answer</summary>(1) OpenAI built infrastructure specifically so that models at every scale are trained comparably well ("properly trained"); our runs share one hand-picked learning rate and very short schedules, so their quality varies non-smoothly. (2) Tiny models are in a regime where small changes (steps, LR, noise in a few validation batches) matter a lot; with many more and larger runs, the fit is far more stable.</details>
