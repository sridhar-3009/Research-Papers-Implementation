# ReAct: Synergizing Reasoning and Acting in Language Models, explained simply

**Paper:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan & Yuan Cao (Princeton & Google Brain), *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023.

**In one sentence:** let a language model alternate between **thinking** ("I need to find who directed this film first") and **acting** (calling a tool such as Wikipedia search), and feed what the tool returns back into its context. The model's reasoning is then **grounded in real facts**, and its actions are **guided by reasoning**. This loop is the blueprint of today's LLM **agents**.

---

## 1. The big idea

### 1.1 Two abilities that were studied separately
- **Reasoning only** (chain of thought, paper 059): the model thinks step by step, but **only from its memory**. If a fact is missing, wrong or out of date, it **hallucinates**, and an early wrong fact spoils every later step.
- **Acting only:** the model issues actions (search, click, move) and reads the results, but **without explicit reasoning**. It can lose track of its goal, repeat useless actions, or fail to combine what it found into an answer.

### 1.2 Combine them
Humans do both together. Cooking: "the recipe needs salt, I'm out of it, let me check the cupboard (act), there's soy sauce instead, that'll work (reason)."

**ReAct:** enlarge the model's action space with **language**. A **thought** is an "action" that changes nothing in the world, only the model's own context. The trajectory interleaves:
```
Thought 1: I need to search Apple Remote and find the program it was designed to interact with.
Action 1: search[Apple Remote]
Observation 1: The Apple Remote ... originally designed to control the Front Row media center program ...
Thought 2: I need to search Front Row next and find what other device can control it.
Action 2: search[Front Row]
Observation 2: Could not find [Front Row]. Similar: ['Front Row Seat to Earth', ..., 'Front Row (software)']
Thought 3: Front Row is not found. I need to search Front Row (software).
Action 3: search[Front Row (software)]
Observation 3: Front Row is a discontinued media center software ...
Thought 4: Front Row is controlled by an Apple Remote or the keyboard function keys. So the answer is keyboard function keys.
Action 4: finish[keyboard function keys]
```
- **Thought 3 handles an exception** (the search failed) and **reformulates** the query.
- **Observations are written by the environment, not the model.** This is what keeps the trajectory factual.

**What thoughts do:**
- decompose the goal ("first find X, then Y");
- extract the key fact from a long observation;
- track progress ("done with step 1");
- use commonsense ("a desk lamp is probably on a desk");
- handle failures;
- state the final answer.

---

## 2. Knowledge tasks: HotpotQA and FEVER (Section 3)
- **HotpotQA:** multi-hop questions that need facts from **two or more** Wikipedia pages.
- **FEVER:** decide whether a claim is SUPPORTED, REFUTED or there is NOT ENOUGH INFO.
- **The models see only the question**, no supporting paragraphs.
- **The tool is a deliberately simple Wikipedia API:**
  - `search[entity]`: the first 5 sentences of the page, or 5 similar titles if the page doesn't exist;
  - `lookup[string]`: the next sentence on the current page containing the string (like Ctrl+F);
  - `finish[answer]`: end with this answer.
- **Prompts:** **6** (HotpotQA) or **3** (FEVER) hand-written trajectories, given as few-shot examples to **PaLM-540B**.
- **Baselines are built by deleting parts of the same trajectories** (a clean ablation):

| Method | Keeps |
|---|---|
| Standard | question → answer |
| CoT | thoughts only (reasoning, no tools) |
| CoT-SC | CoT sampled 21 times at T = 0.7, majority answer |
| Act | actions + observations (tools, no thoughts) |
| ReAct | thoughts + actions + observations |

### 2.1 Results (Table 1, PaLM-540B)
| Method | HotpotQA (exact match) | FEVER (accuracy) |
|---|---|---|
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC | 33.4 | 60.4 |
| Act | 25.7 | 58.9 |
| ReAct | 27.4 | 60.9 |
| CoT-SC → ReAct | 34.2 | **64.6** |
| ReAct → CoT-SC | **35.1** | 62.0 |
| (supervised state of the art) | 67.5 | 89.5 |

- **ReAct beats Act on both tasks:** reasoning helps decide actions and, especially, combine findings into an answer.
- **ReAct vs CoT is mixed:**
  - **FEVER (60.9 vs 56.3):** retrieval matters, because claims can differ in small factual details;
  - **HotpotQA (27.4 vs 29.4):** ReAct is slightly behind, because its rigid think-act-observe structure reduces flexibility.
- **The best methods combine internal knowledge with retrieval.** This is the **back-off** (switching to the other method when one fails):
  - **ReAct → CoT-SC:** if ReAct has no answer after 7 (HotpotQA) or 5 (FEVER) steps, use CoT-SC;
  - **CoT-SC → ReAct:** if the CoT-SC majority answer got fewer than half the votes (the model is unsure), use ReAct.
  - **Either combination reaches** CoT-SC's 21-sample accuracy with only **3–5 samples**.

### 2.2 Why does each one fail? (Table 2, 50 trajectories each)
| | ReAct | CoT |
|---|---|---|
| Correct with correct facts | 94% | 86% |
| Correct but with **hallucinated** reasoning | 6% | 14% |
| Failure: **hallucination** | **0%** | **56%** |
| Failure: reasoning error (incl. repetitive loops) | 47% | 16% |
| Failure: search returned nothing useful | 23% | – |
| Failure: label ambiguity | 29% | 28% |

- **The core trade-off:**
  - CoT's main failure is **making up facts**;
  - ReAct almost never does (it reads facts), but it gets **stuck**: it repeats the same thought and action, or gets derailed by an unhelpful search.

### 2.3 Fine-tuning (Figure 3)
- **The data:** they generated 3,000 correct ReAct trajectories with the big model and fine-tuned smaller PaLMs (8B, 62B) on them.
- **When only prompted,** ReAct is the **worst** method for small models; they can't learn the format from a few examples.
- **When fine-tuned, ReAct is the best:**
  - **PaLM-8B fine-tuned on ReAct** beats every prompting method on PaLM-62B;
  - **PaLM-62B fine-tuned on ReAct** beats every prompting method on PaLM-540B.
- **Why fine-tuning on Standard or CoT does worse:** it teaches the model to **memorise facts** (possibly hallucinated). Fine-tuning on ReAct or Act teaches a **skill**: how to look things up.

---

## 3. Decision-making tasks (Section 4)
- **ALFWorld:** a text game set in a simulated house, e.g. "examine paper under desklamp". It needs dozens of steps, exploration, and commonsense about where objects usually are.
  - **Prompts:** 2 annotated trajectories per task type, with **sparse** thoughts: plan, track subgoals, and guess where things are.
  - **ReAct** (best of 6 prompts): **71%** success.
  - **Act:** 45%.
  - **BUTLER**, an imitation-learning agent trained on 10⁵ expert trajectories: 37%.
- **WebShop:** buying a product that matches an instruction on a simulated shopping website with 1.18M real products.
  - **ReAct:** success **40.0%**;
  - **Act:** 30.1%;
  - **imitation learning (IL):** 29.1%;
  - **IL + RL:** 28.7%.
- **Two examples beat a model trained on thousands.**

---

## 4. Why it matters
- **ReAct is the template of LLM agents:** tool-using chatbots, coding agents, browsing agents and the "thought → tool call → tool result" loop in modern assistant APIs.
- **It shows a general principle:** language models are better at **deciding what to look up** and **interpreting the result** than at storing every fact.
- **Its failure modes became the field's open problems:** loops, unhelpful tool results, and deciding when to stop.

---

## 5. What our code found
The paper prompts PaLM-540B with live Wikipedia. On a laptop we do two things.

### 5.1 The loop itself
- `react_loop` runs **Thought / Action / Observation** against `WikiEnv`, a local page dictionary with the paper's `search` (first 5 sentences, or similar titles), `lookup` (the next matching sentence) and `finish`.
- The demo drives it with a scripted "model" to show the transcript format.
- The tests check every action and the loop.

### 5.2 A toy encyclopedia
- **The world:** 200 films → 120 directors → 20 cities. Each question asks "where was the director of film F born?", which takes two hops.
- **The models:** four tiny Transformers (paper 056's block), trained by imitation on trajectories for Standard, CoT, Act and ReAct (like the paper's fine-tuning set-up).
  - **During tool use, the environment writes the page into the context.** The model never learns to generate observations.
  - **Training detail:** tool users see random "counterfactual" pages in half their training trajectories, so they must **read** rather than recall. We added this because without it our tool users memorised the facts too and scored only 43–47% on changed facts.

| Method | Facts as trained | Facts **changed** after training | Search tool down |
|---|---|---|---|
| Standard | 100% | 3% | 100% |
| CoT | 100% | 3% | 100% |
| Act | 100% | **100%** | 0% |
| ReAct | 100% | **100%** | 0% |
| ReAct → CoT back-off | | | **95%** |

(Chance is 5%.)

- **The closed-book models confidently state stale facts** when the world changes:
  - **The example:** CoT says the director of F100 is P112, born in C12, which was true during training. The page now says P79, born in C0.
  - **Why it matters:** this is Table 2's **hallucination** failure, at its simplest.
- **The tool users read the current page and stay right.**
- **When the tool fails, ReAct gets stuck.** Backing off to CoT recovers 95% (the paper's motivation for combining internal and external knowledge).
  - **It's not 100% because** in 5 of 100 outage cases ReAct still wrote an answer without any evidence, so the back-off rule never fired. **A back-off only helps when the agent admits it is stuck.**
- **Honest gap: Act = ReAct in our toy.** The task has one fixed plan, so thoughts add nothing. In the paper, thoughts pay off when the model must choose what to search, recover from failed searches, and synthesise answers.

**`experiments.py` (written, not run on this laptop):**
- **E1:** HotpotQA with an open 7B instruction model and the **live** Wikipedia API; Standard, CoT, Act and ReAct built by ablating the paper's Figure 1 trajectory;
- **E2:** flaky search with one retry;
- **E3:** training-set size (the fine-tuning curve);
- **E4:** both back-off rules under stale facts plus outages.

---

## 6. Check yourself

1. What is a "thought" in ReAct, formally?
<details><summary>Answer</summary>An action in the language space L that does not affect the environment. It only adds text to the model's own context. The action space becomes A ∪ L.</details>

2. Why does CoT hallucinate while ReAct rarely does?
<details><summary>Answer</summary>CoT can only use facts stored in its parameters, so missing or wrong facts get invented and propagate. ReAct fetches facts from Wikipedia; observations are written by the environment, so its reasoning is grounded (0% hallucination failures vs 56% for CoT in Table 2).</details>

3. What are ReAct's main failure modes?
<details><summary>Answer</summary>Reasoning errors, especially repetitive loops (repeating the same thought and action), and unhelpful search results that derail it (23% of failures).</details>

4. Explain the two back-off rules.
<details><summary>Answer</summary>ReAct → CoT-SC: if ReAct hasn't finished within its step budget, use the CoT-SC majority answer. CoT-SC → ReAct: if the CoT-SC majority answer got fewer than half the votes (low confidence), run ReAct instead.</details>

5. Why is ReAct the worst method when prompting small models, but the best when fine-tuning them?
<details><summary>Answer</summary>Small models can't learn the complex think-act-observe format from a few in-context examples. Fine-tuned on 3,000 trajectories, they learn the skill of retrieving and using information, which generalises better than memorising answers (Standard and CoT fine-tuning).</details>

6. How were the baselines constructed, and why is that a fair comparison?
<details><summary>Answer</summary>By deleting parts of the same hand-written ReAct trajectories: Standard keeps only question and answer, CoT keeps thoughts, Act keeps actions and observations. Every method sees the same examples; only the presence of reasoning or acting differs.</details>

7. In our toy, why do Standard and CoT score 3% when facts change?
<details><summary>Answer</summary>They answer from memorised training facts. When a film's director changes, they still produce the old director's birthplace, a confident stale answer. That's below chance because the old answer is specifically wrong.</details>

8. Why did we train the tool users with counterfactual pages?
<details><summary>Answer</summary>In the plain set-up every page was consistent with memory, so the model could "copy" or "recall" interchangeably and learned partly to recall (only 43–47% on changed facts). Counterfactual pages make reading the observation the only reliable strategy, which mirrors how real LLMs must rely on retrieved context.</details>

9. Our ReAct → CoT back-off reached 95%, not 100%, during the outage. Why?
<details><summary>Answer</summary>In 5 of 100 cases ReAct produced an answer with no evidence instead of giving up, so the back-off (which only triggers on "no answer") never ran. An agent must recognise and admit failure for a fallback to help.</details>

10. ReAct beat imitation learning on ALFWorld (71% vs 37%) with only 2 examples per task type. What does the model contribute that the imitation agent lacks?
<details><summary>Answer</summary>Pre-trained commonsense and reasoning: the LLM can plan subgoals, track progress, and guess where objects are likely to be (a desk lamp on a desk), which an agent trained only on expert trajectories in that environment must learn from scratch.</details>
