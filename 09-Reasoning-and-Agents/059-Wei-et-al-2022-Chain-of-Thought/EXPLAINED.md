# Chain-of-Thought Prompting, explained simply

**Paper:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le & Denny Zhou (Google Research, Brain Team), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS 2022.

**In one sentence:** if the few-shot examples in a prompt show the **intermediate reasoning steps** before the answer ("Roger started with 5 balls. 2 cans of 3 is 6. 5 + 6 = 11. The answer is 11."), a large enough language model will also write out its steps, and it then solves multi-step problems far better. PaLM 540B goes from **17.9% to 56.9%** on grade-school maths (GSM8K), with no training at all.

---

## 1. The big idea

### 1.1 Few-shot prompting, recap (paper 050)
- GPT-3 showed you can teach a task by putting a few **examples** in the prompt:
```
Q: <question 1>
A: The answer is 11.
Q: <question 2>
A: The answer is 6.
Q: <new question>
A:
```
- The model continues the pattern. This works well for many tasks.
- **It fails on multi-step reasoning** (maths word problems, logic), and making the model bigger helped surprisingly little: the scaling curve was **flat**.

### 1.2 The change: show the reasoning
Write each example's answer as a **chain of thought**, i.e. natural-language steps that lead to the answer:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls.
   How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11.
   The answer is 11.
```
- **Test question:** "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"
- **With standard prompting,** the model answers "27" (wrong).
- **With chain-of-thought prompting,** it writes "The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 − 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9." That is **correct**.
- **Nothing is trained.** Only the prompt changes.

---

## 2. Why writing steps helps (the intuition, with a bit of maths)
**A Transformer does a fixed amount of computation per token.** It is L layers deep, and every output token comes from one pass through those L layers.

**Direct answering:** the model must do **all** of a problem's steps inside **one** pass. If a problem needs more serial steps than the network can fit in its depth, it has to guess.

**With a chain of thought,** every generated token is one more full pass, and its result is **written down**, so the next token can read it back. The text becomes a **scratchpad**:
- the amount of computation grows with the number of steps written;
- each step only needs to do **one small thing**: read the previous result and the next number, then combine them.

**Our toy shows exactly this.** The task: add k digits, modulo 10. For "Q3729=":
- **Direct:** output "1" immediately. The network must combine 4 digits at once.
- **Chain of thought:** output "302>1". These are the running sums: 3, (3+7) mod 10 = 0, (0+2) = 2, then (2+9) mod 10 = **1**. Each step adds just **one** digit to the previous running sum.

**But the paper's ablations show "more tokens" alone is not the reason** (Section 5):
- **dots** of the same length (more passes, no content) don't help;
- **the steps written after the answer** don't help either.

The written-down **intermediate results, placed before the answer**, are what matter.

---

## 3. Experiments (Sections 3–5)

### 3.1 Set-up
- **Models:**
  - GPT-3 (350M–175B);
  - LaMDA (422M–137B);
  - PaLM (8B, 62B, 540B);
  - UL2 20B;
  - Codex.
- **Decoding:** greedy (always the most likely token).
- **Prompts:** 8 hand-written chain-of-thought exemplars for the maths datasets (4 for multiple-choice AQuA). They were **not** tuned; the exact same 8 were used for every maths benchmark.
- **Arithmetic benchmarks:**
  - **GSM8K** (grade-school maths word problems, paper 061);
  - SVAMP;
  - ASDiv;
  - AQuA;
  - MAWPS.

### 3.2 The main results (Table 2, GSM8K solve rate)
| Model | Standard | Chain of thought |
|---|---|---|
| LaMDA 137B | 6.5% | 14.3% |
| GPT-3 175B | 15.6% | 46.9% |
| Codex | 19.7% | **63.1%** |
| PaLM 540B | 17.9% | **56.9%** |
| Prior best: GPT-3 fine-tuned **with a verifier** (paper 061) | | 55% |

**Three key findings:**
1. **It is emergent with scale.**
   - Below about **100B parameters**, chain of thought **does not help**, and can even **hurt**: small models write fluent but **illogical** steps.
   - The gains only appear in the biggest models, and the curve goes **up** steeply instead of flat.
2. **Harder problems gain more.**
   - GSM8K (many steps) more than doubles.
   - MAWPS SingleOp (one step) barely changes, or gets worse.
3. **It beats fine-tuning.** PaLM 540B with 8 prompted examples beats GPT-3 that was **fine-tuned** on 7,500 problems with a trained verifier.

### 3.3 Are the chains actually right? (Section 3.2, Appendix D)
- **50 correct answers from LaMDA 137B:** 48 had fully correct reasoning. Only **2** got lucky.
- **50 wrong answers:**
  - **46%** had *minor* errors (a calculator slip, a symbol mapping error, one missing step);
  - **54%** had *major* errors in understanding or coherence.
- **Going from PaLM 62B to 540B** fixed many of the "one step missing" and "semantic understanding" errors.

### 3.4 Ablations: what exactly helps? (Figure 5)
| Variant | What the exemplars show | Result |
|---|---|---|
| **Equation only** | just the equation, e.g. "5 + 2 × 3 = 11" | helps a bit on 1–2-step problems, **not on GSM8K**: turning a complicated question into an equation needs the natural-language steps |
| **Variable compute only** | dots "....." as long as the equation | ≈ **no gain**: extra tokens alone don't help |
| **Chain of thought after the answer** | the answer first, then the steps | ≈ **no gain**: the steps must come *before* the answer, so they can be used to produce it |
| **Full chain of thought** | steps, then the answer | big gain |

### 3.5 Robustness (Figure 6)
- **Different writers:** chains written by three different co-authors (annotators A, B and C) all beat standard prompting by a large margin.
- **Different exemplars:** sets taken from the GSM8K training data (α, β) also work.
- **Order:** different exemplar orders work too.
- **Only the size of the gain varies.** Prompt wording matters, but the effect is not a fluke of one lucky prompt.

### 3.6 Beyond arithmetic
- **Commonsense** (CSQA, StrategyQA, date understanding, sports understanding, robot planning with SayCan): gains for large models. PaLM 540B beats an unaided sports enthusiast on sports understanding (95.4% vs 84%).
- **Symbolic tasks:**
  - **last-letter concatenation:** "Amy Brown" → "yn";
  - **coin flip:** "the coin is heads; A flips it; B does not; is it heads?".
  - **Length generalisation:** with chains of thought, large models handle **longer** inputs than the exemplars showed (e.g. 4 names when the exemplars had 2).

---

## 4. Limitations (Section 6)
- **No guarantee that the reasoning is correct.** The model can produce a correct answer with flawed steps, or the reverse.
- **It only works at scale.** At 100B+ parameters, the method is expensive to use.
- **Writing chains for exemplars** is cheap for few-shot prompting, but costly if you want chains for fine-tuning data.
- **Whether the model is "really reasoning"** is left open.

---

## 5. Why it matters
- **It started "prompting for reasoning"**, with many follow-ups:
  - zero-shot "Let's think step by step" (paper 060);
  - self-consistency (sampling many chains and voting);
  - verifiers (paper 061);
  - tool use and agents (paper 062);
  - and the reasoning models trained to think before answering.
- **It changed how we evaluate large models.** Reasoning benchmarks are now always reported with chain-of-thought prompting.

---

## 6. What our code found
Prompting a 100B+ model is not possible on a laptop, so **the core claim is tested in miniature**:
- **The task:** sum of k digits mod 10, a tiny multi-step problem.
- **The models:** tiny Transformers (paper 056's LLaMA block; d = 64, 2 layers), one per answer format, each trained for 600 steps on problems with 2–6 digits.

| Format | k = 2 | k = 4 | k = 6 | k = 8 (never seen) |
|---|---|---|---|---|
| Standard (answer only) | 92.5% | 14.0% | 11.5% | 11.5% |
| **Chain of thought** (running sums, then answer) | **100%** | **100%** | **100%** | 9.0% |
| Variable compute (dots, same length) | 13.0% | 10.5% | 8.5% | 8.5% |
| Reasoning after answer | 99.5% | 15.5% | 9.5% | 9.0% |

(Chance is 10%.)

- **The core claim holds:** writing the steps first turns a task the direct model fails beyond 2 digits into one it solves perfectly.
- **The chains are right too:** at k = 6, 100% of the correct answers came with fully correct chains.
- **Both ablations fail, as in the paper:**
  - same-length dots don't help (in our toy they even *hurt* at k = 2, because the answer ends up farther from the digits);
  - the same steps after the answer don't help.
- **Honest differences from the paper:**
  1. **Training vs prompting.** Our models were **trained** on each format, whereas the paper only **prompts** a frozen model. That makes our toy closer to "scratchpad" training (Nye et al. 2021).
  2. **No length generalisation.** At **k = 8**, longer than anything trained on, chain of thought fails: the model writes too few steps, and its few "correct" answers come with wrong chains (lucky guesses). Large models in the paper did generalise on symbolic tasks; our tiny model does not.
  3. **A failed first attempt.** We first tried the true prompting setting (one model, format chosen by in-context exemplars). With this tiny budget it learned nothing (all formats at chance), so we fell back to one model per format.

**`experiments.py` (written, not run on this laptop):**
- **E1:** the paper's GSM8K experiment on open base models of 0.5B–7B, with the exact 8 exemplars of Table 20: standard vs chain of thought;
- **E2:** the equation-only, dots and after-answer ablations;
- **E3:** exemplar order and chain style;
- **E4:** toy emergence in the true prompting setting, across model sizes;
- **E5:** the gain vs the number of steps, including one-step problems.

---

## 7. Check yourself

1. What is the only difference between standard and chain-of-thought prompting?
<details><summary>Answer</summary>The exemplars' answers. In chain-of-thought prompting, each exemplar's answer is preceded by natural-language intermediate steps. The model, the decoding and the number of exemplars are the same, and nothing is trained.</details>

2. Why might writing intermediate steps help a Transformer compute?
<details><summary>Answer</summary>Each output token gets one fixed-depth forward pass. Writing a step down lets the next tokens read its result, so a k-step problem can be split into k small one-step computations, each with a full forward pass, instead of all steps inside one pass.</details>

3. The "variable compute only" ablation gives the model the same number of extra tokens, filled with dots. Why is it an important control?
<details><summary>Answer</summary>It separates "more computation (more tokens)" from "intermediate results written in language". Dots give extra forward passes but no content, and they don't help. So the benefit comes from the content of the steps, not just the extra computation.</details>

4. Why does "chain of thought after the answer" not help?
<details><summary>Answer</summary>The answer is produced before the steps, so the steps can't be used to compute it. This rules out the idea that chain-of-thought exemplars simply "activate relevant knowledge"; the order (reason, then answer) matters.</details>

5. What does "emergent with scale" mean here?
<details><summary>Answer</summary>Below roughly 100B parameters, chain-of-thought prompting gives no gain or even hurts (fluent but illogical chains). Above that, the gain appears and grows sharply. It is a qualitative change rather than a smooth improvement.</details>

6. Compute the GSM8K improvement factors for GPT-3 175B and PaLM 540B.
<details><summary>Answer</summary>GPT-3: 46.9/15.6 ≈ 3.0×. PaLM: 56.9/17.9 ≈ 3.2×.</details>

7. Why do one-step problems gain little or nothing?
<details><summary>Answer</summary>There is nothing to decompose: the direct answer already needs only one step, so writing a chain adds tokens (and chances to make mistakes) without splitting a hard computation into easier ones.</details>

8. Of 50 correct LaMDA answers, 2 had wrong chains. Why does that matter?
<details><summary>Answer</summary>A correct final answer doesn't guarantee correct reasoning: a few answers are right by luck or compensating errors. When you use chains as explanations, you must check them, not just the answer. (Our toy's k = 8 shows the same: its few correct answers had wrong chains.)</details>

9. In our toy, chain of thought gets 100% at k = 6 but 9% at k = 8. What does that tell you?
<details><summary>Answer</summary>The tiny model learned the step-by-step procedure only for chain lengths it was trained on. It doesn't continue the procedure for longer inputs, so it fails at length generalisation, unlike the large models in the paper's symbolic experiments.</details>

10. How is our toy different from the paper's setting, and why does it still say something useful?
<details><summary>Answer</summary>We trained a model per format; the paper prompts one frozen model. But the toy isolates the mechanism the ablations point to: with the same model and training budget, intermediate results written before the answer make a multi-step task learnable and solvable, while the same number of content-free tokens, or the same steps after the answer, do not.</details>
