# Codex: Evaluating Large Language Models Trained on Code, explained simply

**Paper:** Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. (OpenAI), *Evaluating Large Language Models Trained on Code*, 2021.

**In one sentence:** fine-tune GPT-3-style models on 159 GB of Python from GitHub, and judge them by **running** what they write against unit tests on a new hand-written benchmark (**HumanEval**). Report **pass@k**, the chance that at least one of k samples is correct, using an **unbiased estimator**. The 12B model, Codex, solves **28.8%** of problems in one try and **72.3%** with 100 tries. It became the model behind GitHub Copilot.

---

## 1. The big idea

### 1.1 GPT-3 could almost write code
- GPT-3 (paper 050) saw some code on the web and could write simple Python from a description, but it was unreliable. On HumanEval, plain GPT-3 solves **0%**.
- **The question:** what if a language model is trained **specifically on code**?

### 1.2 How do you grade a program?
For text, people used **match-based** metrics such as **BLEU**, which counts how many word sequences of the output also appear in a reference answer. **This fails for code:**
- **A correct program can look nothing like the reference:** `return [x + 1 for x in l]` vs a `for` loop with `append`.
- **A wrong program can look almost identical:** one `-` instead of `+`, or an off-by-one in a loop bound.

**Codex's answer: functional correctness.** Run the program. If it passes the unit tests, it's right. That is how human programmers judge code too (test-driven development).

---

## 2. HumanEval (Section 2.2)
- **164 hand-written problems.** Each one has:
  - a function signature;
  - a docstring (the description, often with examples);
  - a reference solution;
  - on average **7.7 unit tests**.
- **Why hand-written:** GitHub already contains solutions to most published programming puzzles, and the model was trained on GitHub. New problems avoid testing memorisation.
- **What it measures:** language comprehension, reasoning, algorithms and simple maths, at about the level of easy interview questions.

**The model's input** is the function header and docstring, e.g.
```python
def incr_list(l: list):
    """Return list with elements incremented by 1.
    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    """
```
- **Output:** the model continues with the body.
- **Stop sequences:** generation stops at `\nclass`, `\ndef`, `\n#`, `\nif` or `\nprint`. Otherwise the model keeps writing more functions or test code after the one it was asked for.

### 2.1 The sandbox (Section 2.3)
- **The risk:** running untrusted, model-generated code is dangerous; it could delete files or open network connections.
- **OpenAI's sandbox:**
  - **gVisor** containers: a kernel emulator between the program and the host;
  - **eBPF firewall** rules that block network access.
- (Our code only runs each program in a separate **process** with a time limit. That protects against hangs and crashes, but it is **not** a security sandbox.)

---

## 3. pass@k and why the obvious estimator is wrong (Section 2.1, Appendix A)

### 3.1 The definition
**pass@k** = the probability that **at least one of k** independent samples passes all the tests, averaged over problems.
- It answers: "If the user may try k suggestions, how often does one work?"
- If one sample is right with probability p, then pass@k = 1 − (1 − p)^k, since all k must fail for the problem to stay unsolved.

### 3.2 Measuring it
**The obvious way** is to draw exactly k samples, check them, and repeat. That is very noisy. **The paper instead:**
1. draws **n ≥ k** samples per problem (n = 200);
2. counts the **c** that pass;
3. computes
```
pass@k = 1 − C(n − c, k) / C(n, k)
```
where C(a, b) is "a choose b", the number of ways to pick b items from a.

**Meaning:** imagine picking k of your n samples at random **without replacement**.
- C(n, k) is the number of ways to pick any k.
- C(n − c, k) is the number of ways to pick k that are **all failures**.
- Their ratio is the chance that all k picks fail; one minus it is the chance that at least one passes.

**Worked example:** n = 5 samples, c = 2 correct, k = 2.
- C(3, 2) = 3 ways to pick 2 failures; C(5, 2) = 10 ways to pick any 2.
- pass@2 = 1 − 3/10 = **0.70**.

### 3.3 The numerically stable form (Figure 3)
- **The problem:** C(200, 100) has 59 digits, so dividing such numbers directly overflows ordinary floats.
- **The fix:** expand the ratio of factorials, and almost everything cancels:
```
C(n−c, k) / C(n, k) = ∏ over i = n−c+1 … n of (1 − k/i)
```
- **Check with the example:** i runs over 4 and 5, giving (1 − 2/4)(1 − 2/5) = 0.5 × 0.6 = 0.3, so pass@2 = 0.7 ✓.
- If n − c < k, there aren't enough failures to fill k picks, so pass@k = 1.

```python
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

### 3.4 Why not just 1 − (1 − c/n)^k?
- **The tempting estimate:** p̂ = c/n, then pass@k ≈ 1 − (1 − p̂)^k.
- **For the example:** 1 − (1 − 0.4)² = 1 − 0.36 = **0.64**, not 0.70.
- **Why it is wrong:** this formula draws k samples **with replacement**, so it can pick the same failed sample twice, as if your failures were repeated. It **underestimates**, and the paper shows it stays biased even when n > 5k.

**Why the combinatorial estimator is unbiased** (Appendix A):
- c comes from a Binomial(n, p) distribution.
- The paper shows by algebra that the average of C(n−c, k)/C(n, k) over that distribution is exactly (1 − p)^k.
- **Intuition:** "pick k of the n samples without replacement" is just a way of looking at k fresh, independent samples, so on average it gives the right failure probability.
- Our test (and demo section 1) checks this exactly. For p = 0.1, n = 20, k = 10:
  - truth 0.6513;
  - unbiased estimator 0.6513;
  - naive estimator 0.5681.

---

## 4. Training Codex (Section 3)

### 4.1 Data
- **Source:** **54 million** public GitHub repositories (May 2020); 179 GB of unique Python files under 1 MB.
- **Filtered out:**
  - likely auto-generated files;
  - average line length > 100;
  - any line > 1000 characters;
  - files with too few alphanumeric characters.
- **Result:** **159 GB**.

### 4.2 Model and optimisation
- **Starting point:** GPT models (paper 050's architecture) of 12M to 12B parameters, fine-tuned on the code.
- **Starting from GPT-3 weights did not improve the final result** (the code dataset is huge), **but it converged faster**, so they always started from GPT-3.
- **Optimiser:**
  - the same learning rate as the GPT model;
  - 175-step linear warm-up, then cosine decay;
  - **100B tokens**;
  - Adam (β₁ = 0.9, β₂ = 0.95, ε = 10⁻⁸), weight decay 0.1.

### 4.3 Whitespace tokens
- **The problem:** Python code is full of **indentation**, and the GPT-3 text tokenizer (paper 049's BPE) spends many tokens on runs of spaces.
- **The fix:** Codex adds **extra tokens for whitespace runs of different lengths**. "Eight spaces" becomes one token, not several.
- **The effect:** code takes **~30% fewer tokens**. That means more code fits in the context window, and every training token covers more code.

### 4.4 The loss follows a power law (Figure 4)
- **The fit:** test loss on held-out code follows L(N) = (N / 5.92·10⁷)^−0.13 nats per token (N = non-embedding parameters), like Kaplan's laws (paper 051).
- **Examples:**
  - N = 12M: (0.203)^−0.13 = **1.23**;
  - N = 12B: (203)^−0.13 = **0.50**.
- **Reading the fit:** doubling N multiplies the loss by 2^−0.13 = 0.914, an 8.6% reduction.

---

## 5. Sampling: temperature, nucleus, and which sample to show (Section 3.3)

### 5.1 Temperature
**How it works:**
- The model outputs logits z₁ … z_V for the next token.
- With temperature T, the probabilities are softmax(z / T).
- **Example:** logits (2, 1, 0).

| T | probabilities |
|---|---|
| 0.5 | 0.867, 0.117, 0.016 (sharper: almost always the top choice) |
| 1 | 0.665, 0.245, 0.090 |
| 2 | 0.506, 0.307, 0.186 (flatter: more variety) |

T → 0 is **greedy** decoding: always pick the most likely token.

### 5.2 The best temperature depends on k (Figure 5)
- **For pass@1** you want the model's **best guess**, so use a low T. The 679M model's optimum was T* = **0.2**.
- **For pass@100** you want **diversity**. 100 copies of the same wrong answer are useless, while 100 different attempts cover more possibilities. The optimum was T* = **0.8**.
- **Too high** a temperature makes samples degrade into nonsense, so there is an optimum, not "as high as possible".

### 5.3 Nucleus (top-p) sampling
- **The rule:** sort tokens by probability and keep the smallest set whose total is ≥ p (p = **0.95** in Codex). Renormalise and sample only from those.
- **The effect:** this cuts off the long tail of very unlikely tokens, which at high T would otherwise be picked surprisingly often.
- **Example:** probabilities (0.5, 0.3, 0.15, 0.05) with p = 0.9.
  - The first three sum to 0.95 ≥ 0.9, so they stay.
  - The new probabilities are 0.526, 0.316, 0.158, 0.

### 5.4 Choosing one sample without tests (Figure 7)
- **The situation:** in an autocomplete tool you have no unit tests but must show one suggestion. Generate k and pick one by a heuristic:
  - **random:** the baseline;
  - **highest sum of token log-probs**, i.e. the most likely sequence: this **does slightly worse than random**;
  - **highest mean token log-prob**: clearly **better than random**;
  - **oracle:** pick one that passes the tests. This is the upper bound, equal to pass@k.
- **Why sum is bad:** every token adds a negative number, so **shorter sequences have higher sums**.
  - **Example:** A has 3 tokens at −1 each, so sum −3 and mean −1. B has 10 tokens at −0.5 each, so sum −5 and mean −0.5.
  - The sum picks the short A (maybe a truncated, wrong answer); the mean picks B, the more confident one per token.
- **Codex-S-12B:** picking by mean log-prob gives **44.5%**, vs **37.7%** for one random sample.

---

## 6. Results (Table 1, Figure 6)

| Model | pass@1 | pass@10 | pass@100 |
|---|---|---|---|
| GPT-3 (any size) | 0 | 0 | 0 |
| GPT-Neo 2.7B (trained on the Pile, 8% GitHub) | 6.41 | 11.27 | 21.37 |
| GPT-J 6B | 11.62 | 15.74 | 27.74 |
| TabNine (code autocomplete product) | 2.58 | 4.35 | 7.59 |
| Codex-12M | 2.00 | 3.62 | 8.58 |
| Codex-85M | 8.22 | 12.81 | 22.40 |
| Codex-300M | 13.17 | 20.37 | 36.27 |
| Codex-679M | 16.22 | 25.70 | 40.95 |
| Codex-2.5B | 21.36 | 35.42 | 59.50 |
| **Codex-12B** | **28.81** | **46.81** | **72.31** |

- **Equivalences:**
  - Codex-85M ≈ GPT-Neo 2.7B, with **30× fewer** parameters;
  - Codex-300M ≈ GPT-J 6B, with 20× fewer.
- **Scaling:** pass rates rise smoothly, like a sigmoid in log(parameters).
- **BLEU fails (Figure 8):** for 4 random problems, the BLEU distributions of correct and incorrect samples **overlap heavily**. Improving BLEU need not improve correctness.

---

## 7. Codex-S: supervised fine-tuning on standalone functions (Section 4)
- **The motivation:** GitHub Python is mostly classes, scripts and configuration, not "write one function from a docstring". The authors suspected this mismatch was costing performance.
- **They collected:**
  - **~10,000** problems from competitive programming and interview-prep sites;
  - **~40,000** functions **traced from open-source projects with continuous integration**: they ran the projects' tests, recorded the inputs and outputs of function calls, and turned those into unit tests.
- **Quality filtering:** for each problem, Codex-12B generated 100 samples. If none passed, the problem was assumed ambiguous or broken and dropped.
- **Training:**
  - the loss is computed **only on the solution tokens**; the prompt is masked;
  - prompts are **left-padded** so the solutions line up;
  - the learning rate is 1/10 of Codex's.
- **Results:**
  - **Codex-S-12B: 37.7% pass@1**, and **77.5%** with 100 samples;
  - on average **+6.5 points pass@1** and **+15.1 points pass@100** over Codex across sizes.
- **Temperatures:** Codex-S prefers slightly **higher** temperatures (T* = 0 for pass@1, T* = 1 for pass@100), perhaps because it models a narrower distribution.

### 7.1 Codex-D: docstrings from code (Section 5)
- **The idea:** train the reverse direction, function to docstring, which is useful to explain generated code.
- **Results:** hand-graded on 10 samples per problem:
  - Codex-D-12B gets **20.3%** pass@1 and 46.5% pass@10;
  - Codex-S gets 32.2% and 59.5%.
- **Back-translation ranking:** choosing a code sample by P(original docstring | sample) under Codex-D beats random but **loses to mean log-prob**.

---

## 8. Limitations and risks (Sections 6–7)
- **Sample inefficiency:** Codex trains on hundreds of millions of lines of code, yet a strong student after an introductory course should solve more of HumanEval than Codex-12B.
- **Long specifications break it (Figure 11):**
  - **The setup:** 13 simple string operations (Appendix C) — "convert the string to lowercase", "remove every third character", "reverse the order of words" — are chained into one docstring.
  - **The result:** with each extra step, the pass rate drops by roughly a factor of **2–3**.
  - A human who can do two steps can do ten; Codex can't.
- **Variable binding:** it confuses which operation applies to which variable when a docstring has many of both.
- **Misalignment (Section 7.2, Appendix E):**
  - if the prompt contains subtle bugs, Codex writes **worse code than it is capable of**, imitating the buggy style;
  - the gap **grows with model size**;
  - it is a case of a model able to do X but not doing it.
- **Other risks:** over-reliance, insecure code, bias, economic effects, and the environmental cost of training are discussed in the paper's long Broader Impacts section.

---

## 9. Why it works (the intuition)
- **Code is text with very strict rules,** and GitHub has an enormous amount of it. A next-token predictor trained on it learns Python syntax, common idioms, library APIs, and how docstrings relate to the bodies below them.
- **Docstring → body is a natural pattern on GitHub,** so "complete this function" is close to the training distribution. Codex-S makes it even closer.
- **Sampling many times works because execution is a perfect filter.** If you can test, a 30%-per-try model becomes a 72%-in-100-tries system. This is why pass@k, and later AlphaCode-style "sample many and filter", matter.

---

## 10. What our code found
Everything below runs in under a second on a laptop CPU. **No code model is trained in the demo**: training one that can solve problems would take hours (see `experiments.py`).

**Exact checks (tests):**
- **The two pass@k forms agree:** Figure 3's product form equals 1 − C(n−c, k)/C(n, k), and pass@1 = c/n.
- **Unbiasedness:** averaged exactly over c ~ Binomial(n, p), the estimator equals 1 − (1 − p)^k, while the naive one is lower.
- **The execution harness** reports pass, assertion failure, syntax error and timeout (an infinite loop is killed).
- **Stop sequences** cut the sample correctly.
- **Nucleus sampling** keeps the smallest top set.
- **Whitespace tokens** round-trip exactly and shrink indented code.
- **BLEU can prefer a wrong program:** `[x - 1 for x in l]` has higher BLEU than a correct loop rewrite.
- **The 13 Appendix C building blocks** compose into synthetic problems whose reference solutions pass.

**Demo:**

1. **The bias of the naive estimator** is largest when k is a large fraction of n: for p = 0.1, n = 20, k = 10, it gives 0.568 vs the true 0.651.
2. **BLEU vs execution** on the paper's `x_or_y` problem (return x if n is prime, else y):
   - the off-by-one √n loop has **BLEU 0.81** but **fails** (it calls 4 a prime);
   - a correct one-line rewrite has **BLEU 0.10** but **passes**.
3. **Temperature vs k:**
   - **Setup:** 40 synthetic problems × ~6 **real** candidate programs, 243 in all, each actually executed; 73 pass, including some "mistakes" that turn out to be correct, such as swapping two operations that commute. The model's preferences over the candidates are **simulated**.
   - **pass@1 is best at T = 0.2**, while pass@10 and pass@100 keep improving with T.
   - **Honest caveat:** pass@100 here climbs to 1.0 at T = 2.5 only because each toy problem has just ~6 candidates, so high T never produces garbage. A real model's samples degrade at high T, which is why the paper's optimum for pass@100 is 0.8.
   - The pass@1 trend is weak in our toy: 0.347 at T = 0.2 vs 0.311 at T = 2.5.
4. **One-sample selection with simulated token log-probs** (correct samples slightly more confident per token, −0.45 vs −0.55; wrong samples often shorter):
   - random 0.313;
   - **mean 0.519**;
   - sum 0.361;
   - oracle 0.976.
   - The sum is dragged toward short samples. In our simulation it still beats random; in the paper it is slightly worse. All of these numbers follow from our assumptions; they illustrate the mechanism, not a measurement.
5. **Whitespace tokens on 200 real Python files from this repository:** 17% fewer tokens with our word-level split. That is not comparable to the paper's 30%, which was measured on BPE tokens.

**`experiments.py` (written, not run on this laptop)** trains GPT-2-style models (paper 049) on the local Python standard library, with the paper's file filters:
- **E1:** the loss-vs-size power law;
- **E2:** text-pretrained vs scratch initialisation;
- **E3:** a text BPE on code with and without whitespace-run tokens;
- **E4:** Codex-S fine-tuning on synthetic building-block problems, measuring:
  - pass@k vs temperature;
  - mean/sum/random/oracle selection;
  - BLEU of passing vs failing samples;
  - the pass rate vs chain length (Figure 11);
- **E5:** HumanEval pass@k. Models this small should score about 0%; the paper's smallest, Codex-12M, gets 2.0%.

---

## 11. Check yourself

1. Why is BLEU a poor metric for code?
<details><summary>Answer</summary>BLEU rewards textual overlap with one reference. Correct programs can be written in many different ways (low BLEU), and a one-character change can break a program while barely changing BLEU. Running unit tests measures what we actually care about: does it work?</details>

2. With n = 10 samples of which c = 3 pass, compute pass@1 and pass@5.
<details><summary>Answer</summary>pass@1 = c/n = 0.3. pass@5 = 1 − C(7,5)/C(10,5) = 1 − 21/252 = 1 − 0.0833 = 0.917. (Product form: 1 − (1−5/8)(1−5/9)(1−5/10) = 1 − 0.375·0.444·0.5 = 1 − 0.0833.)</details>

3. For the same counts, what does the naive 1 − (1 − c/n)^5 give, and why is it lower?
<details><summary>Answer</summary>1 − 0.7⁵ = 1 − 0.168 = 0.832 < 0.917. It treats the 5 picks as drawn with replacement, so the same failed sample can be picked repeatedly. Sampling without replacement from n can't repeat a failure, so "all 5 fail" is less likely than the naive formula says.</details>

4. When n − c < k, why is pass@k exactly 1?
<details><summary>Answer</summary>There are fewer than k failing samples, so any set of k samples must include at least one passing sample.</details>

5. Why should you use a low temperature for pass@1 and a higher one for pass@100?
<details><summary>Answer</summary>With one try you want the model's single most likely answer. With 100 tries, repeated identical answers are wasted, and diversity increases the chance that some sample is right. But too high a temperature makes samples nonsense, so there is an optimum (0.8 for the 679M model).</details>

6. Why does ranking by the sum of log-probs prefer short answers, and how does the mean fix it?
<details><summary>Answer</summary>Each token adds a negative log-probability, so longer sequences accumulate more negative terms regardless of quality. The mean divides by length, comparing per-token confidence instead.</details>

7. What did Codex-S change, and why did it help?
<details><summary>Answer</summary>It fine-tuned on ~50k standalone, tested functions (competitive programming plus functions traced from CI runs), with loss only on the solution. This matches HumanEval's format (docstring → function) much better than general GitHub code, giving +6.5 points pass@1 and +15.1 points pass@100 on average.</details>

8. What do whitespace tokens buy you?
<details><summary>Answer</summary>Runs of spaces (indentation) become single tokens, so code needs ~30% fewer tokens. More code fits in the context window, training covers more code per token, and generation needs fewer steps.</details>

9. What does Figure 11's 13-building-block experiment show?
<details><summary>Answer</summary>When a docstring chains more simple string operations, Codex's pass rate drops by roughly 2–3× per extra step: it struggles to compose many simple instructions, unlike a human programmer.</details>

10. In what sense is Codex "misaligned" in Section 7.2?
<details><summary>Answer</summary>When the prompt contains subtle bugs, Codex tends to write buggier code, imitating the context, even though it is capable of writing correct code. It "can" do the task but doesn't, and the gap grows with model size.</details>
