# Training Verifiers to Solve Math Word Problems, explained simply

**Paper:** Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Łukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse & John Schulman (OpenAI), *Training Verifiers to Solve Math Word Problems*, 2021.

**In one sentence:** instead of trusting a model's single answer to a maths problem, let it write **many** solutions, train a **second model (a verifier)** to judge which solutions are correct, and return the one the verifier likes best. With enough training data, this gives about the same improvement as making the model **30× bigger**. The paper also introduced **GSM8K**, the grade-school maths benchmark used everywhere since.

---

## 1. The big idea

### 1.1 The problem: one mistake ruins everything
- **Multi-step reasoning is fragile.** A solution with 5 steps has 5 chances to go wrong, and one slip makes the whole answer wrong.
- **Language models generate one token at a time** and cannot go back to fix an earlier mistake.
- **Even GPT-3 175B, fine-tuned on maths solutions,** gets only about a third of grade-school problems right.

### 1.2 Generating is hard, checking is easier
- **For people:** solving a puzzle is often harder than checking a proposed solution.
- **For models, the same may be true:** a model that only sometimes writes a correct solution may still be good at **recognising** one.
- **The recipe:**
  1. **Sample** many candidate solutions (say 100) at a high temperature, so they are diverse.
  2. **Score** each one with a verifier.
  3. **Return** the highest-scoring one.
- **The paper's vocabulary:**
  - **test@1:** accuracy of a single (greedy) answer;
  - **test@N:** the fraction of problems where **at least one** of N samples is right. This is the **coverage**, and it is what a perfect verifier could reach (the same idea as pass@k, paper 054).
  - **The gap** between test@1 and test@100 is what verification can harvest.

---

## 2. GSM8K (Section 2)
- **8,500 problems** written by human problem writers: 7,500 for training, 1,000 for testing.
- **Each problem** needs **2–8 steps** of basic arithmetic (+ − × ÷). They are grade-school level: hard enough to be challenging, easy enough that a bright middle-schooler could solve them all.
- **The solutions are written in natural language,** with **calculator annotations**:
```
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```
- **The final answer** comes after "####".
- **The calculator:** when the model writes "<<48+24=", the actual result 72 is inserted, so the model can't make **arithmetic** slips; only its **reasoning** (which numbers to combine) is its own.
- **Quality:** problems were checked by other workers; only about 1.7% had disagreements.

---

## 3. The baseline: fine-tuning (Section 3)
- **Set-up:** fine-tune GPT-3 (6B or 175B) on the training solutions, then generate one answer at temperature 0.
- **Findings:**
  - **Steps matter:** fine-tuning a 6B model to answer directly, without steps, drops accuracy from **20.6% to 5.2%**.
  - **Data matters too:** performance improves roughly log-linearly with training-set size. Reaching 80% would need far more data or far bigger models.
  - **Overfitting differs by metric:**
    - test@1 keeps improving for many epochs;
    - but **test@100** (coverage) **peaks after only ~2 epochs** and then drops, because the model becomes over-confident and its samples less diverse.
    - So **the generator used for verification is trained for only 2 epochs**: it should produce *diverse* candidates, not one confident answer.

---

## 4. Verification (Section 4, Figure 4)

### 4.1 Training the verifier
1. **Generator:** fine-tune for 2 epochs on the training set.
2. **Samples:** generate **100 solutions** for every **training** problem, at temperature 0.7.
3. **Labels:** mark each solution correct or incorrect, **only** by whether its final answer matches. (Some "correct" ones reached the right answer by flawed reasoning; these are noisy labels.)
4. **Verifier:** train it for **1 epoch** on these labelled solutions. It is a language model with an extra **scalar head** that outputs P(correct).

**At test time:** sample 100 solutions per test problem, score them all, and output the top one.

### 4.2 Design choices
- **Token-level verifier (the default):**
  - **What it does:** it predicts P(correct) **after every token**, not just at the end. All positions of a solution share the same label.
  - **What that means:** it is a **value function**, an estimate of "how likely is this partial solution to end correctly?".
  - **Training behaviour:** it trains more slowly at first, but ends up **better** and overfits less than the **solution-level** verifier (one prediction at the end).
  - **The authors' guess:** it forces the model to judge the reasoning throughout, rather than memorise final answers.
- **A joint objective:** the verifier also keeps the normal **language-modelling loss** on the solutions. This is a strict improvement: understanding the text helps judge it.
- **Separate models:** the generator and verifier are separate, so the generator can't overfit through the verifier's training.

### 4.3 Training loss
For a solution with tokens x₁ … x_T and label y ∈ {0, 1}, with verifier output v_t = P(correct | x₁…x_t):
```
L = Σ_t [ −y log v_t − (1−y) log(1 − v_t) ]   (token-level binary cross-entropy)
  + LM loss (−Σ_t log p(x_t | x_<t))           (joint objective)
```
**Worked example:** a wrong solution (y = 0) whose verifier says v = 0.9 at some token pays −log(1 − 0.9) = **2.30** there. If it said v = 0.1, it would pay only −log 0.9 = 0.105.

---

## 5. Results

### 5.1 Verification vs fine-tuning (Figure 5)
- **With small training sets,** verification is **not** helpful, and can be worse.
  - **The reason:** the verifier overfits to recognising the specific correct answers it saw, faster than it learns general properties of correct reasoning.
- **With the full GSM8K training set,** verification gives a **big boost**:
  - **6B + verifier slightly beats 175B fine-tuning,** roughly the gain of a **30× larger model**;
  - **the curve scales better with data**, so more data would widen the gap;
  - **175B verifiers "take off" earlier** (need fewer training problems) than 6B ones.

### 5.2 Ablations (Figure 6)
- **Token-level beats solution-level** (unless dropout is added; see below).
- **Joint (value + language modelling)** beats verification-only.
- **A big generator with a small verifier** beats a small generator with a big verifier.
  - **The authors' reading:** the verifier often uses relatively **coarse heuristics** rather than truly re-deriving the solution.

### 5.3 Test-time compute (Figure 7)
- **More samples help,** up to about **400** for the 6B verifier.
- **Beyond that, performance drops:** more samples means more chances to find an **adversarial** solution that fools the verifier.
- **Voting:** instead of taking only the top solution, let the **top few vote** on the final answer.
  - With 100 samples, letting the **top 3–5** vote is best.
  - With 3,200 samples, it is the top ~30.

### 5.4 Regularisation (Section 6)
- **20% residual dropout** helps both fine-tuning and verification.
- **With dropout,** solution-level verifiers catch up with token-level ones.

---

## 6. Why it matters
- **GSM8K became the standard maths-reasoning benchmark.** Chain of thought (paper 059) and nearly every LLM report since evaluate on it.
- **"Generate many, then select"** is now everywhere:
  - majority voting (self-consistency, paper 060);
  - **reward models** for best-of-N sampling and for RLHF (papers 066–067);
  - **process reward models**, which judge each *step*: a direct descendant of the token-level verifier;
  - test-time compute scaling in modern reasoning models.
- **The warning about adversarial samples** (too many samples fool the verifier) is the same problem as **reward hacking** in RLHF.

---

## 7. What our code found
Training GPT-3-scale models is impossible on a laptop, so the **whole pipeline** runs on a toy task chosen so that **generating needs search but checking is easy**:
- **The task:** "pick two of these 6 digits that add up to t", e.g. `Q398259t7=` → `2+5;`.
  - **Generating** needs a search over pairs.
  - **Checking** a proposed pair is one sum plus a membership test.
- **Generator:** a tiny Transformer (paper 056's block), trained briefly (400 steps), like the paper's 2-epoch generator.
  - greedy (test@1, T = 0): **58.3%**;
  - one random sample at T = 1: 42.1%;
  - coverage test@1 / 4 / 16: 42.7% / 81.3% / **93.7%**.
- **Verifier:** initialised from the generator, with a token-level value head and the joint LM loss, trained on 16 samples per training problem labelled only by correctness.

| Selection rule | Accuracy (300 test problems) |
|---|---|
| Greedy (fine-tuning baseline) | 58.3% |
| Majority vote of 16 (no verifier) | 55.0% |
| Verifier trained on **100** problems, best of 16 | 44.0% |
| Verifier trained on **2000** problems, best of 16 | **75.0%** |
| Verifier on 2000 problems, **top-3 vote** | **80.7%** |
| Oracle (any of 16 correct) | 93.7% |

- **The paper's central result, in miniature:**
  - **with little data, the verifier is worse than the baseline;**
  - **with enough data, it clearly beats it**, and letting the top 3 vote helps further.
- **Number of samples ranked** (large verifier): N = 1: 42.7%, 2: 59.3%, 4: 72.7%, 8: 76.0%, 16: 75.0%. It climbs, then flattens; the 8 → 16 dip is within noise on 300 problems.
- **What didn't work (an honest record):**
  1. **Running-sum chains.** A verifier on chain-of-thought solutions for "sum of digits mod 10" was *worse than random selection*.
     - **The shortcut:** it learned "short means correct", because short problems are easier.
     - **The deeper problem:** even with fixed lengths, catching an error in a running-sum chain means recomputing the sums, which is as hard as generating them.
  2. **Restated steps** ("9+1=0,0+3=3,…") made checking local, but also made generation trivially easy (100%), leaving nothing to verify.
  - **The lesson:** verification only pays off when **checking is easier than generating**, as the paper's "coarse heuristics" observation hints.

**`experiments.py` (written, not run on this laptop):**
- **E1:** verification vs fine-tuning across training-set sizes (3 seeds);
- **E2:** token- vs solution-level, joint vs verification-only;
- **E3:** generator strength vs verifier strength;
- **E4:** number of samples and top-k voting;
- **E5:** the real GSM8K pipeline with a small open model (2-epoch generator, 20 samples per problem, a token-level joint verifier).

---

## 8. Check yourself

1. What is test@N, and why does it bound what a verifier can achieve?
<details><summary>Answer</summary>test@N is the fraction of problems where at least one of N samples is correct. A verifier can only pick among the samples, so even a perfect verifier cannot beat test@N.</details>

2. Why is the generator trained for only 2 epochs?
<details><summary>Answer</summary>test@100 (coverage) peaks after a couple of epochs and then falls as the model becomes over-confident and its samples less diverse. The verifier needs diverse candidates that include a correct one, not one confident answer.</details>

3. How are training labels for the verifier obtained, and what is their weakness?
<details><summary>Answer</summary>Sample many solutions per training problem and label each by whether its final answer is correct. Some solutions reach the right answer with flawed reasoning, so labels are noisy (false positives).</details>

4. What is a token-level verifier, and why might it be better than a solution-level one?
<details><summary>Answer</summary>It predicts P(correct) after every token: a value function over partial solutions. This forces it to judge the reasoning as it unfolds instead of just memorising which final answers are right, so it overfits less and ends up more accurate.</details>

5. A wrong solution gets verifier score 0.9 at a token. What is the token-level loss there?
<details><summary>Answer</summary>−log(1 − 0.9) = −log 0.1 ≈ 2.30.</details>

6. Why is verification worse than fine-tuning with small training sets?
<details><summary>Answer</summary>With few problems, the verifier quickly memorises which specific answers are correct (overfits) instead of learning general signs of correct reasoning. Our toy shows it: trained on 100 problems, best-of-16 is 44% vs greedy 58%.</details>

7. Why does performance eventually fall as you rank more and more samples?
<details><summary>Answer</summary>The verifier is imperfect. With more samples, there are more chances that some wrong solution happens to fool it with a high score (an adversarial example), and it gets picked. For 6B, the best was about 400 samples.</details>

8. Why does letting the top 3–5 vote help?
<details><summary>Answer</summary>A single top-scored solution might be a lucky fooler. Several high-scoring solutions agreeing on the same final answer is stronger evidence. In our toy, top-3 voting raised accuracy from 75.0% to 80.7%.</details>

9. Our first two toy designs failed. What does that teach about when verifiers help?
<details><summary>Answer</summary>Verifiers help when checking is easier than generating. In the running-sum task, checking a chain required redoing the computation, so the verifier was no better (and found a length shortcut). In the restated-steps version, generation became trivial, so there was nothing to fix. Only in the search task (find a pair; check one sum) did verification pay off.</details>

10. How does the "verifier" idea show up in today's LLM training?
<details><summary>Answer</summary>As reward models for best-of-N and RLHF, process reward models that score each reasoning step, and test-time compute scaling (sample many, select). The danger of selecting against an imperfect judge is the same as reward hacking.</details>
