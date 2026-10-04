# Large Language Models are Zero-Shot Reasoners, explained simply

**Paper:** Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo & Yusuke Iwasawa (University of Tokyo & Google Research), *Large Language Models are Zero-Shot Reasoners*, NeurIPS 2022.

**In one sentence:** you don't need hand-written reasoning examples (paper 059). Just add **"Let's think step by step."** after the question, let the model write its reasoning, then ask it once more for the answer. On MultiArith, accuracy jumps from **17.7% to 78.7%**; on GSM8K, from **10.4% to 40.7%** (InstructGPT, text-davinci-002).

---

## 1. The big idea

### 1.1 Where we were
- **Few-shot chain of thought** (paper 059) needs **examples written by hand**, each with step-by-step reasoning, chosen per task. That is human effort, and results depend on which examples you write.
- **Plain zero-shot prompting** ("Q: … A: The answer is") gives no examples and does badly on multi-step problems.

### 1.2 One sentence instead of examples
- **The finding:** large models already *can* reason step by step. They just need to be **asked to start**.
- **The prompt:** append the **trigger** "Let's think step by step." to the answer slot, and the model writes its reasoning on its own.
- **Why it works:** a large model has read a lot of text where a phrase like "let's think step by step" is followed by a worked solution. The phrase **puts the model into that mode**, so it continues the way such text usually continues.

**The same single sentence is used for every task:** arithmetic, symbolic puzzles, dates, commonsense. That is why the method is "zero-shot" and "task-agnostic".

---

## 2. The method: two prompts (Section 3, Figure 2)

### Stage 1: reasoning extraction
```
Q: On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes.
   How many punches did he throw?
A: Let's think step by step.
```
The model continues and writes the reasoning **z**:
> In one minute, Joe throws 25 punches. In three minutes, Joe throws 3 × 25 = 75 punches. In five rounds, Joe throws 5 × 75 = 375 punches.

### Stage 2: answer extraction
Feed everything back, plus an **answer trigger** that fits the answer format:
```
[stage-1 prompt] [z]
Therefore, the answer (arabic numerals) is
```
The model writes " 375."

### Why two stages?
- **Free-form text** after "Let's think step by step" can end anywhere, in any format.
- **Few-shot exemplars normally fix the format** ("The answer is X."). Here there are no exemplars, so a second prompt is needed to **pull out the answer in a parseable form**.

**Answer triggers by format:**

| Answer type | Trigger |
|---|---|
| number | "Therefore, the answer (arabic numerals) is" |
| multiple choice | "Therefore, among A through E, the answer is" |
| yes / no | "Therefore, the answer (Yes or No) is" |

### Answer cleansing
Take the **first** part of the output that fits the format:
- " probably 375 and 376" → **375**;
- " B, C, and D" → **B**.

**The fair baseline:** standard zero-shot also gets an answer trigger ("The answer (arabic numerals) is"), just with no reasoning first. The comparison isolates the **reasoning**.

---

## 3. Results

### 3.1 Zero-shot vs zero-shot-CoT (Table 1, text-davinci-002)
| Task | Zero-shot | Zero-shot-CoT |
|---|---|---|
| MultiArith | 17.7 | **78.7** |
| GSM8K | 10.4 | **40.7** |
| AQUA-RAT | 22.4 | 33.5 |
| SVAMP | 58.8 | 62.1 |
| SingleEq (one step) | 74.6 | 78.0 |
| AddSub (one step) | 72.2 | 69.6 |
| Last Letter (4 words) | 0.2 | **57.6** |
| Coin Flip (4 flips) | 12.8 | **91.4** |
| Date Understanding | 49.3 | 67.5 |
| Shuffled Objects | 31.3 | 52.4 |
| CommonsenseQA | 68.8 | 64.6 |
| StrategyQA | 12.7 | 54.8 |

- **Big gains on multi-step tasks:** arithmetic, symbolic and logical tasks.
- **Little or no gain on one-step arithmetic** (SingleEq, AddSub): there is nothing to break into steps.
- **Commonsense QA does not improve,** but the written reasoning is often sensible even when the final choice is wrong. The model sometimes lists several answers when unsure.

### 3.2 Compared with few-shot (Table 2, MultiArith)
| Method | Accuracy |
|---|---|
| Zero-shot | 17.7 |
| Few-shot (8 examples, no reasoning) | 33.8 |
| **Zero-shot-CoT** | **78.7** |
| Few-shot-CoT (8 examples) | 93.0 |

- **One sentence beats 8 plain examples** by far. It is still below 8 hand-written reasoning examples.

**PaLM 540B on GSM8K:**
- zero-shot 12.5 → zero-shot-CoT **43.0** → with **self-consistency** **70.1**;
- for comparison, few-shot-CoT 56.9 → with self-consistency 74.4.

**Self-consistency** (Wang et al. 2022):
1. **Sample** several reasoning paths at a temperature above 0, instead of one greedy path.
2. Extract the answer from each.
3. Take the **majority vote**.
4. The idea: right reasoning paths tend to agree on the same answer, while wrong ones scatter.

### 3.3 Model size (Figure 3)
- **Without the trigger,** accuracy barely grows with model size (a flat curve).
- **With the trigger,** it grows sharply, but only for **large** models. Small models write weak reasoning, as with few-shot CoT (paper 059).

### 3.4 The wording matters (Table 4, MultiArith)
| Category | Trigger | Accuracy |
|---|---|---|
| instructive | **Let's think step by step.** | **78.7** |
| instructive | First, | 77.3 |
| instructive | Let's think about this logically. | 74.5 |
| instructive | Let's solve this problem by splitting it into steps. | 72.2 |
| instructive | The answer is after the proof. | 45.7 |
| misleading | Don't think. Just feel. | 18.8 |
| misleading | Let's think step by step but reach an incorrect answer. | 18.7 |
| misleading | By using the fact that the earth is round, | 9.3 |
| irrelevant | By the way, I found a good restaurant nearby. | 17.5 |
| irrelevant | Abrakadabra! | 15.5 |
| (none) | zero-shot baseline | 17.7 |

- **Every instructive trigger helps,** though by very different amounts.
- **Misleading and irrelevant triggers do nothing, or hurt.**
- **So it is the meaning** ("reason step by step") that matters, not just adding words.

### 3.5 Error analysis
- **Extra steps:** zero-shot-CoT sometimes keeps reasoning **after** reaching the right answer and talks itself into a wrong one.
- **No reasoning:** it sometimes just rephrases the question.
- **Few-shot-CoT fails differently:** it tends to fail on expressions with three numbers, like (3 + 2) × 4.

---

## 4. Why it matters
- **It set a baseline:** "Let's think step by step" became the standard way to measure a model's zero-shot reasoning.
- **It showed latent ability:** the reasoning was already in the model; the prompt **elicits** it rather than **teaching** it. That supports the view that large pre-trained models have broad abilities waiting to be prompted.
- **It led to later work:**
  - self-consistency;
  - plan-and-solve prompting;
  - automatic prompt search, which found "Take a deep breath and work on this problem step-by-step";
  - instruction-tuned models that reason by default.

---

## 5. What our code found
A real 100B+ model can't run on a laptop, so we built a **toy "pre-trained" model** that captures the mechanism:
- **The corpus:** problems "sum of k digits mod 10". After each question, the text continues in one of four ways, each announced by a **trigger**:
  - **30%** "**>**" then the answer directly;
  - **60%** "**T**" then running sums (step-by-step text), then ">" and the answer. This plays the role of "Let's think step by step" text on the web.
  - **5%** "**R**" then filler dots, then the answer (an irrelevant trigger);
  - **5%** "**X**" then random digits and a random answer (a misleading trigger).
- **The model:** one tiny Transformer (paper 056's LLaMA block) trained for 2,000 steps.
- **At test time, there are no exemplars;** only the trigger changes. Zero-shot-CoT uses the paper's two stages: reason after "T" until the model writes ">", then append the answer trigger ">" and read one digit.

| Prompt | k = 2 | k = 4 | k = 6 |
|---|---|---|---|
| Zero-shot (answer trigger only) | 100% | 35.3% | 13.3% |
| **T: "let's think step by step"** | 99.3% | **89.3%** | **82.7%** |
| R: irrelevant | 92.7% | 16.0% | 9.3% |
| X: misleading | 9.3% | 14.0% | 9.3% |

(Chance is 10%.)

- **Table 4's pattern, in miniature:** the instructive trigger makes the **same** model reason and wins on multi-step problems; the irrelevant trigger is no better than the baseline; the misleading trigger is worse (even on easy 2-digit problems).
- **Self-consistency** on 6-digit problems:
  - greedy reasoning: 80.0%;
  - one sampled path at temperature 1: 68.5%;
  - majority vote of 8 sampled paths: 76.7%.
  - **So voting lifted the sampled paths a lot, but did not beat greedy here.** The paper's large gains need individual samples that are nearly as good as greedy, with errors that disagree.
- **Honest caveats:**
  - **We chose the corpus.** 60% reasoning text is our choice, and it makes the trigger work quickly. With a 45% share, 900 or 2,000 steps were not enough for clean chains. `experiments.py` E4 sweeps this share.
  - **The trigger is learned in our toy:** "T" works because the corpus put reasoning after it. That is also the paper's own explanation of why the phrase works in real models; we just control the corpus.

**`experiments.py` (written, not run on this laptop):**
- **E1:** zero-shot vs two-stage zero-shot-CoT with open models on GSM8K, MultiArith, and generated Last Letter and Coin Flip tasks;
- **E2:** all 16 triggers of Table 4;
- **E3:** self-consistency with 1–20 paths;
- **E4:** how much reasoning text the toy corpus needs;
- **E5:** toy emergence across model sizes.

---

## 6. Check yourself

1. What is the difference between few-shot-CoT (paper 059) and zero-shot-CoT?
<details><summary>Answer</summary>Few-shot-CoT puts hand-written examples with reasoning into the prompt; zero-shot-CoT uses no examples, only the trigger "Let's think step by step." The model writes its own reasoning in both cases.</details>

2. Why does zero-shot-CoT need a second prompt?
<details><summary>Answer</summary>Without exemplars, nothing fixes the answer format: the reasoning can end anywhere and in any form. The second prompt ("Therefore, the answer (arabic numerals) is") asks for the answer in a parseable format.</details>

3. The output of stage 2 is " probably 375 and 376." What is the prediction?
<details><summary>Answer</summary>375: answer cleansing takes the first piece that fits the format (the first number).</details>

4. Why did one-step datasets like SingleEq and AddSub barely improve?
<details><summary>Answer</summary>They need only one operation, so there is nothing to decompose. Writing reasoning adds tokens and room for mistakes without making the computation easier.</details>

5. What does Table 4 show about why the trigger works?
<details><summary>Answer</summary>Instructive triggers (encouraging step-by-step reasoning) help; irrelevant ones ("Abrakadabra!") and misleading ones ("Don't think. Just feel.") don't. The effect comes from what the sentence means, which steers the model into reasoning text, not from simply adding words.</details>

6. On MultiArith, zero-shot-CoT gets 78.7, few-shot (8 examples, no reasoning) 33.8, and few-shot-CoT 93.0. What do you conclude?
<details><summary>Answer</summary>Eliciting reasoning matters more than giving examples: one sentence beats 8 examples without reasoning. Good reasoning examples still help further (93.0).</details>

7. How does self-consistency work, and why can it help?
<details><summary>Answer</summary>Sample several reasoning paths at temperature > 0, extract each answer, and take the majority. Correct paths tend to agree on the same answer while wrong paths scatter, so the vote filters out individual mistakes (PaLM GSM8K: 43.0 → 70.1).</details>

8. In our toy, voting 8 sampled paths (76.7%) did not beat greedy decoding (80.0%). Why?
<details><summary>Answer</summary>Sampling at temperature 1 made each path much worse (68.5%), and the tiny model's errors were not diverse enough for the majority to correct them. Self-consistency needs samples that are individually almost as good as greedy, with disagreeing errors.</details>

9. Why does the trigger "T" work in our toy model?
<details><summary>Answer</summary>In its training corpus, "T" was always followed by correct step-by-step running sums, so the model learned to continue a "T" with reasoning. This mirrors the paper's explanation: web text often follows "let's think step by step" with worked solutions.</details>

10. Name one failure mode of zero-shot-CoT found in the error analysis.
<details><summary>Answer</summary>It keeps reasoning after reaching the correct answer and changes it to a wrong one; or it doesn't reason at all and just rephrases the question.</details>
