# InstructGPT: Training Language Models to Follow Instructions with Human Feedback, explained simply

**Paper:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike & Ryan Lowe (OpenAI), *Training language models to follow instructions with human feedback*, NeurIPS 2022.

**In one sentence:** take GPT-3, fine-tune it on human-written answers to real user prompts, train a reward model on human **rankings** of its answers, and optimise it with PPO plus a little of its original pre-training data. The resulting **InstructGPT** follows instructions so much better that people preferred a **1.3B** InstructGPT over the **175B** GPT-3. This is the recipe behind ChatGPT.

---

## 1. The big idea

### 1.1 GPT-3 is not trying to help you
- **GPT-3's training objective:** predict the next token of internet text.
- **What it does with an instruction** like "Explain the moon landing to a 6-year-old": it may continue with more questions, a forum discussion, or something off-topic, because that is what follows such text on the web.
- **The authors' word is "misaligned":** the model's objective differs from the user's intent ("be helpful, honest and harmless").
- **Few-shot prompts help** but are clumsy, and the model still makes things up, follows harmful requests, and ignores constraints.

### 1.2 Fine-tune with human feedback
- **Reuse the RLHF recipe of paper 066** (summarisation), now for **any** instruction a user might give.
- **Train on the real distribution of prompts** people sent to the OpenAI API: generation, open QA, brainstorming, chat, rewriting, summarisation, classification, …

---

## 2. Data and labelers (Section 3)
- **About 40 contractors** (Upwork, Scale AI), screened for sensitivity to harmful content and agreement with researchers.
- **Agreement rates:**
  - training labelers agree with each other **72.6%** of the time;
  - **held-out** labelers, who never produced training data, agree **77.3%** of the time with each other.
- **Three datasets:**

| Dataset | Prompts | Used for |
|---|---|---|
| SFT | ~13k (API + labeler-written) | demonstrations of the desired output |
| RM | ~33k (API + labeler-written) | rankings of model outputs |
| PPO | ~31k (API only) | RL prompts (no labels needed) |

---

## 3. The three steps (Section 3.5, Figure 2)

### Step 1: supervised fine-tuning (SFT)
- **Training:** fine-tune GPT-3 on the demonstrations for **16 epochs**.
- **Overfitting is fine:** it overfits the validation loss after 1 epoch, but more epochs still improve RM score and human preference.

### Step 2: reward model (RM) from rankings
- **Collection:** for each prompt, labelers **rank K = 4 to 9** outputs. That gives **C(K, 2)** pairwise comparisons at once; K = 9 gives 36.
- **The loss:**
```
loss(θ) = − 1/C(K,2) · E_(x, y_w, y_l) [ log σ( r_θ(x, y_w) − r_θ(x, y_l) ) ]
```
  where y_w is the preferred output of a pair.
- **A crucial detail: all C(K, 2) pairs from one prompt are ONE batch element.**
  - **If you shuffle them** into a dataset of independent pairs, each completion appears in K − 1 updates, and the RM **overfits** after one epoch.
  - **Grouping them** fixes that, and is cheaper: **K forward passes** instead of C(K, 2). Each completion is scored once and the scores are compared in all pairs.
- **Model choice:** a **6B** RM, because 175B RMs were unstable.
- **Normalisation:** the RM is shifted so labeler demonstrations score 0.

**Worked example:** K = 4 outputs ranked A > B > C > D with scores r = (2, 1, 0.5, −1).
- The 6 pairs and their losses −log σ(Δ):
  - A–B: Δ = 1, loss 0.31;
  - A–C: Δ = 1.5, loss 0.20;
  - A–D: Δ = 3, loss 0.05;
  - B–C: Δ = 0.5, loss 0.47;
  - B–D: Δ = 2, loss 0.13;
  - C–D: Δ = 1.5, loss 0.20.
- **Mean loss = 1.36/6 ≈ 0.23.**

### Step 3: PPO, and PPO-ptx
- **The environment is a bandit:** a random API prompt, one response, one reward.
- **The reward:** the RM score, minus a **per-token KL penalty** to the SFT model (as in paper 066). The value function is initialised from the RM.
- **PPO-ptx also mixes in pre-training gradients:**
```
objective(φ) = E_(x,y)∼π_RL [ r_θ(x, y) − β log( π_RL(y|x) / π_SFT(y|x) ) ]  +  γ · E_x∼D_pretrain [ log π_RL(x) ]
```
- **Why:** RLHF alone causes **performance regressions** on public NLP benchmarks (SQuAD, DROP, HellaSwag, translation). This cost of aligning the model is the **alignment tax**. The γ term keeps the model good at its original distribution.
- **Naming:** "InstructGPT" = the PPO-ptx models.

---

## 4. Results (Section 4)

### 4.1 People prefer InstructGPT (Figure 1, Figure 3)
- **175B InstructGPT** is preferred to **175B GPT-3** **85 ± 3%** of the time, and to **few-shot** GPT-3 **71 ± 4%**.
- **1.3B InstructGPT** is preferred to **175B GPT-3**, despite having **100× fewer parameters**.
- **Held-out labelers agree,** so it's not just fitting the training labelers' tastes.
- **On the API distribution,** it beats GPT-3 fine-tuned on the public FLAN and T0 instruction datasets. Those datasets don't look like real user requests.

### 4.2 Truthfulness, toxicity, bias
- **TruthfulQA:** truthful and informative answers about **twice** as often as GPT-3.
- **Hallucination:** on "closed-domain" tasks (summarise or answer using only the given text), InstructGPT makes up information **21%** of the time vs **41%** for GPT-3.
- **Toxicity:** about **25% fewer** toxic outputs when asked to be respectful. Without that instruction, the gap mostly vanishes.
- **Bias** (Winogender, CrowS-Pairs): **no improvement.**

### 4.3 The alignment tax, and its fix
- **PPO without pre-training data** loses some performance on public benchmarks.
- **PPO-ptx recovers most of it** without hurting human preference. Simply increasing the KL coefficient instead does not work as well.

### 4.4 Limitations
- **It still makes simple mistakes,** follows false premises, hedges too much, and can be steered into harmful outputs ("it follows instructions, including bad ones").
- **"Aligned to whom?"** The outcome reflects ~40 labelers, the researchers' instructions and API customers, not humanity at large.

---

## 5. Why it matters
- **SFT → RM → PPO became the standard post-training pipeline:** ChatGPT, Claude, Llama-chat and others.
- **It showed alignment is cheap relative to pre-training:** the RLHF compute was a small fraction of GPT-3's, yet it changed the user experience completely.
- **It made "helpful, honest, harmless"** the working definition of alignment for assistants.

---

## 6. What our code found
A toy InstructGPT, with real tiny networks (paper 056's LLaMA block):
- **"Web text" for pre-training:**
  - 80% periodic letter patterns ("abcabca…"); predicting them is our "public NLP benchmark";
  - 20% instruction-looking text followed by **unrelated** letters, never the right answer.
- **The instructions:** **SORT**, **FIRST2** and **LAST2** of 3–5 letters.
- **The labelers' utility:** +2 × fraction of positions right, +1 for an exact answer, −0.3 per length mismatch.

| Model | Instruction utility | Exact | Pre-training benchmark |
|---|---|---|---|
| Base (pre-trained only) | −0.08 | 0.3% | **99.9%** |
| SFT (200 demonstrations) | 2.13 | 58.7% | 89.3% |
| PPO (β = 0.05, 40 iterations) | 2.11 | 56.3% | 89.5% |
| **PPO-ptx (γ = 0.3)** | **2.14** | 58.0% | **98.3%** |

- **The base model is "misaligned":** asked "SORT d b c a", it continues with "c d c", like the web text it saw. That's despite being nearly perfect on its own benchmark.
- **SFT creates instruction following** but costs **10 points** on the benchmark: the alignment tax.
- **The reward model follows the paper's design:**
  - K-way rankings (K = 4–9) with all pairs of a prompt in one batch element;
  - 1,000 ranked prompts gave 19,749 comparisons from only 6,577 forward passes;
  - **66.4%** held-out pairwise accuracy.
- **PPO-ptx keeps SFT's instruction following** and recovers the benchmark (**89.3% → 98.3%**), the paper's alignment-tax fix.
- **Honest negative result: PPO did not improve instruction following beyond SFT.**
  - **Why:** the RM (66% accurate) is weaker than the SFT policy's own competence (59% exact answers), so it can't point the way to better outputs. Longer PPO slowly made things worse.
  - **Contrast with paper 066's toy,** where the RM's signal exceeded the policy's and RL helped. InstructGPT used a 6B RM trained on 33k ranked prompts.
- **An earlier version of our toy (REVERSE instead of LAST2) was worse:** checking a reversal requires the same mirror alignment as producing it, so the RM reached only 55–58%. That matches paper 061's lesson: learned judges help when **checking is easier than doing**.

**`experiments.py` (written, not run on this laptop):**
- **E1:** how strong the RM must be for RL to help;
- **E2:** per-prompt batching vs shuffled pairs (the paper's overfitting claim);
- **E3:** a γ sweep (alignment tax);
- **E4:** a β sweep;
- **E5:** SFT data size.

---

## 7. Check yourself

1. What does "misaligned" mean for GPT-3 here?
<details><summary>Answer</summary>Its training objective (predict the next token of web text) differs from what users want (follow the instruction helpfully, honestly, harmlessly), so it often continues text instead of doing the task.</details>

2. Labelers rank K = 6 responses. How many comparisons is that, and how many RM forward passes are needed?
<details><summary>Answer</summary>C(6, 2) = 15 comparisons, but only 6 forward passes: each response is scored once and the scores are compared in all 15 pairs.</details>

3. Why treat the C(K, 2) pairs from one prompt as a single batch element?
<details><summary>Answer</summary>The pairs are highly correlated (each response appears in K − 1 of them). Shuffled as independent examples, the RM sees each response many times per epoch and overfits after one pass. Grouping them avoids that and is K-vs-C(K, 2) times cheaper.</details>

4. Compute the mean RM loss for a ranking A > B with r_A = 1 and r_B = 0, plus A > C with r_C = −1, plus B > C.
<details><summary>Answer</summary>Pairs: A–B (Δ = 1): 0.313; A–C (Δ = 2): 0.127; B–C (Δ = 1): 0.313. Mean ≈ 0.25.</details>

5. Write the PPO-ptx objective and explain the γ term.
<details><summary>Answer</summary>E[r(x, y) − β log(π_RL/π_SFT)] + γ E_pretrain[log π_RL(x)]. The γ term keeps maximising the likelihood of pre-training text during RL, which prevents the model from losing general abilities (the alignment tax on public NLP benchmarks).</details>

6. Why can a 1.3B InstructGPT beat a 175B GPT-3 in human preference?
<details><summary>Answer</summary>Preference depends on following the user's intent. The 175B GPT-3 has more knowledge but isn't trying to follow instructions. Fine-tuning with human feedback targets exactly what people judge, which matters more than raw size here.</details>

7. Which problems did InstructGPT improve, and which did it not?
<details><summary>Answer</summary>Improved: instruction following, truthfulness (TruthfulQA ~2×), closed-domain hallucination (21% vs 41%), toxicity when asked to be respectful (~25% fewer). Not improved: bias (Winogender, CrowS-Pairs). It still follows harmful instructions and makes simple mistakes.</details>

8. In our toy, SFT drops the pre-training benchmark from 99.9% to 89.3%. What fixes it, and why?
<details><summary>Answer</summary>PPO-ptx: adding γ × the pre-training log-likelihood to the objective keeps training the model on its original distribution, recovering 98.3% while keeping instruction following.</details>

9. Our PPO didn't beat SFT. Give the reason, and what would be needed.
<details><summary>Answer</summary>The reward model (66% pairwise accuracy) was less informative than the policy already was (59% exact answers), so optimising it couldn't find better outputs. A stronger RM (more and cleaner rankings, a larger model) or tasks where judging is easier than doing would be needed, as in InstructGPT's 6B RM on 33k prompts.</details>

10. What does "aligned to whom?" refer to?
<details><summary>Answer</summary>The model's behaviour reflects the preferences of ~40 contractors, the researchers' labeling instructions, and API customers' prompts, not a broad or democratic notion of human values. The paper discusses this as a key limitation.</details>
