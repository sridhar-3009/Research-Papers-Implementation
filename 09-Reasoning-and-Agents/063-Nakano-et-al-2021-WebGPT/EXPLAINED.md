# WebGPT: Browser-assisted question-answering with human feedback, explained simply

**Paper:** Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess & John Schulman (OpenAI), *WebGPT: Browser-assisted question-answering with human feedback*, 2021.

**In one sentence:** teach GPT-3 to **use a text web browser** (search, click, scroll, quote) to answer open-ended questions **with references**. First copy human demonstrations, then improve it with a **reward model** trained on human comparisons, used to pick the best of many answers. The best model's answers were preferred over **human-written** answers 56% of the time.

---

## 1. The big idea

### 1.1 Long-form answers are hard to trust
- **The task:** questions like ELI5 ("Explain Like I'm Five", from Reddit), e.g. "Why do we get goosebumps?" They need **paragraph-length** answers.
- **Two problems with plain language models:**
  - **they make up facts** (hallucination);
  - **their claims are hard to check**, because nothing says where they came from.

### 1.2 Two ideas
1. **Retrieval through a browser.** Let the model use a **real search engine** (Bing) through a simple text interface, and collect **quotes** as it browses. The answer must be backed by those references.
   - **Why it matters:** this outsources *finding* documents to a strong existing tool. The model learns the higher-level skill of **searching like a person**.
2. **Human feedback.** Ask people which of two answers they prefer, then optimise for that preference. "Good answer" is hard to define by a formula, but people can **compare** answers.

---

## 2. The browsing environment (Section 2, Table 1)

### 2.1 What the model sees
At each step the model gets a **text summary of the state**:
- the question;
- the quotes collected so far;
- the current page's text around the cursor;
- how many actions it has left.

It then writes **one command**:

| Command | Effect |
|---|---|
| `Search <query>` | send the query to Bing and show the results |
| `Clicked on link <id>` | open a result or link |
| `Find in page: <text>` | jump to the next occurrence of the text |
| `Quote: <text>` | save the extract as a reference (with page title and domain) |
| `Scrolled down / up <1, 2, 3>` | scroll |
| `Top`, `Back` | go to the top of the page, or back to the previous page |
| `End: Answer` | stop browsing and write the answer |
| `End: <Nonsense, Controversial>` | stop without answering |

- **No memory beyond the summary:** each step starts from a fresh context, so the model must quote what matters.
- **The end of browsing:** browsing ends on `End`, at the action limit, or at the quote-length limit. The model then writes its answer from the question and quotes, citing them as [1], [2], …

### 2.2 Why references matter
- **For readers:** a human can check claims against the quotes.
- **For training:** it makes **human evaluation** of factual accuracy feasible; labelers judge whether claims are supported, instead of researching each claim themselves.

---

## 3. Training (Section 3)

### 3.1 Data
- **About 6,000 demonstrations:** people used the same browser (via a GUI) to answer questions.
- **About 21,500 comparisons:** two model answers to the same question, each with references, and a person picks the better one, or a tie.
- **The source:** almost all questions come from ELI5. The instructions stressed relevant, coherent, **trustworthy** answers.

### 3.2 Four methods (on GPT-3 760M, 13B and 175B)
1. **Behaviour cloning (BC):** supervised fine-tuning on the demonstrations; learn to imitate human browsing and answering.
2. **Reward model (RM):** start from the BC model, replace the output layer with a single number r, and train it on comparisons.
   - **The Bradley–Terry model:** P(A preferred to B) = σ(r_A − r_B), where σ(x) = 1/(1 + e^(−x)).
   - **The loss** is cross-entropy against the human label (1, 0, or **0.5 for ties**).
   - **r is an Elo score:** a difference of 1 means σ(1) = **73%** preference.
3. **Reinforcement learning (RL):** PPO (paper 066 explains it) against the RM. The episode reward is the RM score at the end, minus a **KL penalty** at every token:
```
R = RM(answer) − β Σ_t [ log π(token_t) − log π_BC(token_t) ]
```
   The penalty keeps the policy close to BC, so it can't drift into strange answers that merely **fool** the RM.
4. **Rejection sampling (best-of-n):** sample n answers (n = 4, 16, 64) from the BC model and return the one the **RM scores highest**. No extra training, just more compute at answer time.

**The final models** are BC plus best-of-n, with an RM of the same size: 760M best-of-4, 13B best-of-16, 175B best-of-64.

**Worked example of the RM loss:**
- Answer A scores r_A = 1.2 and answer B scores r_B = 0.2, so P(A > B) = σ(1.0) = 0.73.
- If people preferred A: loss = −log 0.73 = 0.31.
- If they called it a tie: loss = −0.5·log 0.73 − 0.5·log 0.27 = 0.81.

---

## 4. Results (Section 4)

### 4.1 Human preference
| Comparison | WebGPT (175B best-of-64) preferred |
|---|---|
| vs **human demonstrators** using the same browser | **56%** |
| vs **top-voted Reddit answer** (citations stripped) | **69%** |
| (previous best system vs Reddit answers) | 23% |

- **Beating the demonstrators matters.** Pure imitation should top out around 50% against the people it imitates, so the gain comes from **human feedback** (through the RM).
- **The comparison vs demonstrators is the more meaningful one:** same answer style, references available for fact-checking, and detailed shared criteria.

### 4.2 TruthfulQA
- **Every** WebGPT model beats **every** GPT-3 model on truthful and on truthful-and-informative answers.
- WebGPT's truthful-and-informative score **rises with size**; GPT-3's doesn't.

### 4.3 Rejection sampling beats RL (Section 5.1)
- **Head to head against BC (175B):**
  - **best-of-64** is preferred to BC **68%** of the time;
  - **RL** is preferred only **58%**;
  - **RL plus rejection sampling** is no better than rejection sampling alone.
- **Possible reasons:**
  - many attempts use more inference compute;
  - with hindsight, it can pick the attempt that happened to find good websites;
  - the RM was trained on data from BC and rejection-sampling policies, so RL moves further out of its training distribution;
  - RL needs hyperparameter tuning;
  - RL reduces sample diversity.
- **Tuning the BC baseline** (number of epochs, temperature) closed much of the apparent gap to RL.

### 4.4 Scaling (Figures 6–8)
- **Data:**
  - doubling demonstrations raises the policy's RM score by about **0.13**;
  - doubling comparisons raises RM accuracy by about **1.8%**.
- **Parameters:**
  - doubling policy parameters gives about **+0.09** RM score;
  - doubling RM parameters gives about **+0.4%** accuracy.
- **Best-of-n:** for a fixed inference budget, there is a best combination of model size and n. More samples than that gives diminishing or negative returns.

### 4.5 Risks (Section 6)
- **The model can cherry-pick sources** that support an answer it already "decided".
- **It can quote unreliable sites.** References can give a **false sense of authority**, and people may over-trust cited answers.
- **Labelers find it hard to judge factual accuracy** even with references. Labeler agreement was 73%.
- **Live web access** is a new safety surface: a model that can act online.

---

## 5. Why it matters
- **It was a direct precursor of ChatGPT's browsing and of retrieval-augmented assistants.** Paper 071 (RAG) is the retrieval-model counterpart.
- **It was an early large-scale use of the RLHF toolkit:** comparisons → reward model → best-of-n / PPO with KL. InstructGPT (paper 067) used the same toolkit for general instructions.
- **The finding that best-of-n is a strong, simple baseline** for optimising against a reward model is still widely used.

---

## 6. What our code found
We can't browse Bing with GPT-3, so we built a faithful miniature.

**The browser** (`Browser`, `LocalWeb`) implements Table 1's commands on a local web:
- **search:** TF-IDF ranking, with a higher "authority" weight for reliable domains;
- **links, find, quote** (only text actually on the page), **scroll, top, back, end**;
- the **state summary** the model would see, and **reference formatting**.

**The toy web:** 40 topics, each with a reliable **encyclopedia** page (the true fact) and an unreliable **blog** with a **wrong** fact.
- **The "BC" policy** is stochastic: it searches precisely or vaguely, clicks the top or a random result, uses Find or quotes a random sentence, and adds 0–N filler sentences.
- **The simulated labeler's true utility:**
  - correct +2, +0.5 if supported by a quote;
  - wrong −1;
  - **some detail is liked** (+0.4 per filler sentence up to 2), but **verbosity is not** (−0.6 per sentence beyond 2).
- **The reward model:** linear Bradley–Terry on visible features (cites, supported, encyclopedia domain, filler count), trained on 3,000 noisy comparisons.
  - **Held-out accuracy: 70.3%.** It learned positive weights for support (1.96) and the reliable domain (1.51), plus a small positive weight for **length** (0.12). A linear RM can't express "some detail is good, too much is bad".

**Best-of-n against the RM** (200 questions):

| n | RM score of chosen | TRUE quality | Preferred to BC |
|---|---|---|---|
| 1 | 2.35 | 0.76 | 51.8% |
| 4 | 3.26 | 2.16 | 70.1% |
| 16 | 3.74 | **2.93** | **80.5%** |
| 64 | 3.85 | 2.56 | 75.9% |
| 256 | 3.95 | 2.03 | 70.5% |

- **Reward over-optimisation, in miniature.** The RM score keeps rising with n, but the true quality **peaks at n = 16 and falls after**. With enough samples, the search finds the RM's blind spot (it thinks longer is always better). This is why best-of-n has a best n (Figure 8) and why RL needs a KL penalty.

**RL** (REINFORCE as a stand-in for PPO, on the policy's 4 parameters):
- **β = 0:** the RM score rises 2.32 → 4.36, but true quality **falls** 0.69 → **−0.25**.
  - The policy did learn the good habits (top result 0.99, Find 0.99), **and** padded answers to the filler cap (8 sentences). That is reward hacking.
- **β = 0.3:** the KL penalty keeps the answers short (filler 1.5) and true quality rises to **2.71**.
- **An honest difference:** our RL with KL comes close to the best best-of-n (2.93), while the paper found rejection sampling clearly better than RL (68% vs 58%). Our 4-parameter policy is far easier to optimise than a 175B model with PPO.

**Bugs the tests caught:**
1. **TF-IDF length normalisation ranked the short blog above the encyclopedia.** "Click the top result" then led to wrong facts, so we added a domain-authority weight, as real engines rank reliable sites.
2. **A quote about a different topic counted as "supporting".** Support now requires the quote to mention the asked topic.

**`experiments.py` (written, not run on this laptop):**
- **E1:** a **real** reward model fine-tuned on OpenAI's released `webgpt_comparisons` dataset;
- **E2:** best-of-n with an open model, scored by that reward model;
- **E3:** RM accuracy vs number of comparisons;
- **E4:** the best-of-n peak for RMs trained on more or less data;
- **E5:** a β sweep for RL with KL.

---

## 7. Check yourself

1. Why does WebGPT collect quotes while browsing?
<details><summary>Answer</summary>The quotes become the answer's references: they let readers and labelers check claims, which makes human evaluation of factual accuracy feasible. The model also has no memory between steps except the summary, so quotes are how it keeps information.</details>

2. Write the reward-model probability and loss for a comparison.
<details><summary>Answer</summary>P(A preferred) = σ(r_A − r_B). Loss = −[y log P + (1 − y) log(1 − P)] with y = 1, 0 or 0.5 for a tie.</details>

3. Two answers have RM scores 2.0 and 1.0. What preference does the RM predict?
<details><summary>Answer</summary>σ(1.0) ≈ 73% for the first answer.</details>

4. Why can't behaviour cloning alone beat the demonstrators, and how did WebGPT get to 56%?
<details><summary>Answer</summary>Imitation copies the demonstrators' behaviour, so at best it matches them (~50% preference). The reward model learned from comparisons what people prefer, and best-of-64 selection against it produced answers better than typical demonstrations.</details>

5. What is the KL penalty in RL, and why is it needed?
<details><summary>Answer</summary>R = RM score − β Σ (log π − log π_BC). It penalises moving away from the behaviour-cloned policy. Without it, the policy can find outputs that score highly on the imperfect reward model but are worse in reality (reward hacking), as our β = 0 run did with very long answers.</details>

6. In our best-of-n table, the RM score keeps increasing with n but the true quality peaks at n = 16. Explain.
<details><summary>Answer</summary>The RM is imperfect. Among more samples, the one it rates highest is increasingly likely to exploit its errors (here: preferring longer answers even when verbosity hurts). Selection pressure first finds genuinely better answers, then finds the RM's blind spots: over-optimisation.</details>

7. Give two reasons the paper suggests for rejection sampling beating RL.
<details><summary>Answer</summary>Any two of: it spends more inference compute; it can choose, with hindsight, the attempt that found good sources; the RM was trained on BC and rejection-sampling outputs, so RL drifts further out of its training distribution; RL needs tuning; RL reduces diversity.</details>

8. Why is the comparison against human demonstrators more meaningful than against Reddit answers?
<details><summary>Answer</summary>Same answer style and format (better blinding); references let labelers fact-check; detailed shared criteria make judgements interpretable; Reddit answers aim at originality and simplicity, and many are low-effort.</details>

9. What risks of a browsing, citing model does the paper flag?
<details><summary>Answer</summary>Cherry-picking sources that support a chosen answer; citing unreliable websites; references creating false authority and over-reliance; difficulty judging factual accuracy; and the new surface of a model acting on the live web.</details>

10. In our toy, the RM gave a positive weight to answer length. Why, and why does it matter?
<details><summary>Answer</summary>In the comparison data, a little extra detail was preferred, so on average longer answers won and a linear RM learned "longer is better". It can't represent "good up to 2 sentences, bad beyond". Optimising hard against it (large best-of-n, RL without KL) produces overly long answers that people dislike.</details>
