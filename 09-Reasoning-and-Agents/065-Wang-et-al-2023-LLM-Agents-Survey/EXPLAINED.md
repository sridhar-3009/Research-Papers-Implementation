# A Survey on LLM-based Autonomous Agents, explained simply

**Paper:** Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei & Ji-Rong Wen (Renmin University of China), *A Survey on Large Language Model based Autonomous Agents*, arXiv 2023 / Frontiers of Computer Science 2024.

**In one sentence:** this survey organises the fast-growing field of LLM agents into **one framework** (profile, memory, planning, action), explains how agents gain abilities (fine-tuning or not), where they are used, how to evaluate them, and what is still unsolved.

---

## 1. What an "LLM-based autonomous agent" is
- **An autonomous agent:** a system that **perceives an environment, decides and acts** to reach goals over time, on its own. Classic agents were built with reinforcement learning in narrow worlds.
- **The LLM version:** use a large language model as the **brain**. It brings broad knowledge, language understanding and reasoning, so agents can work in open-ended settings: coding, browsing, simulations, science.
- **The problem the survey solves:** papers 059–064 and hundreds of others each invent their own architecture. The survey shows they share a common structure.

---

## 2. The unified framework (Section 2, Figure 2)
Four modules work together. The LLM calls sit inside them.

### 2.1 Profile: who the agent is
- **What it holds:** the role (coder, teacher, shopper), personality, background, written into the prompt.
- **Three ways to create profiles:**
  - **handcrafting:** "You are an outgoing person";
  - **LLM-generation:** write a few seed profiles, let the LLM generate many more;
  - **dataset alignment:** build profiles from real demographic records, e.g. survey participants, to simulate a population.

### 2.2 Memory: what the agent knows from experience
- **Structures:**
  - **Unified memory:** only short-term, i.e. whatever fits in the prompt. Simple, but limited by the context window.
  - **Hybrid memory:** short-term plus **long-term** storage (e.g. a vector database). Retrieval brings old but relevant memories back. Generative agents (paper 064) use this.
- **Formats:** natural language, embeddings, databases (queried with SQL), structured lists.
- **Operations:**
  - **Reading:** pick the useful memories, typically by m* = argmax over memories of α·recency + β·relevance + γ·importance (generative agents' rule).
  - **Writing:** store new information, which raises two problems:
    - **duplicates:** the same event stored many times;
    - **overflow:** the store is full, so what gets deleted?
  - **Reflection:** summarise memories into higher-level insights.

### 2.3 Planning: how the agent decides what to do
- **Without feedback** (plan, then just execute):
  - **single-path reasoning:** one chain of steps (chain of thought, paper 059; zero-shot CoT, paper 060; plan-then-execute);
  - **multi-path reasoning:** explore several chains (self-consistency votes; Tree of Thoughts searches over partial plans and evaluates them);
  - **external planner:** translate the problem into a formal language (e.g. PDDL) and use a classical planner.
- **With feedback** (adjust the plan using signals):
  - **environment:** the world reports success or failure, e.g. ReAct observations (paper 062), Voyager's code-execution errors;
  - **human:** a person corrects or advises;
  - **model:** another model, or the agent itself, critiques (self-refine, Reflexion: learn verbal lessons from failed attempts).

### 2.4 Action: what the agent does
- **Goal:** completing a task, communicating (with people or other agents), or exploring.
- **Production:** actions come from **memory recollection** (reusing what worked) or **plan following**.
- **Action space:**
  - **external tools:** APIs, databases, search, code interpreters, other models;
  - **internal knowledge:** the LLM's own planning, conversation and commonsense.
- **Impact:** actions change the environment, the agent's internal state (memory), or trigger new actions.

---

## 3. Acquiring capabilities (Section 2.2)
- **With fine-tuning** (change the weights), using datasets that are:
  - **human-annotated;**
  - **LLM-generated** (e.g. ToolBench: tool-use traces written by ChatGPT);
  - **real-world** (e.g. web pages and actions).
- **Without fine-tuning:**
  - **prompt engineering:** few-shot examples, chain of thought, role descriptions;
  - **mechanism engineering:**
    - **trial-and-error:** act, get feedback, revise;
    - **crowd-sourcing / debate:** several agents discuss until they agree;
    - **experience accumulation:** store successful skills in memory (Voyager's skill library);
    - **self-driven evolution:** set own goals, learn from feedback.

---

## 4. Applications (Section 3)
- **Social science:** psychology, political science, social simulation (generative agents).
- **Natural science:** documentation, experiment assistants (e.g. chemistry agents with lab tools), education.
- **Engineering:**
  - software development, where multi-agent "companies" (ChatDev, MetaGPT) assign roles such as product manager, coder and tester;
  - robotics and embodied AI;
  - industrial automation.

---

## 5. Evaluation (Section 4)
- **Subjective (human judgement):**
  - **human annotation:** rate believability or quality;
  - **Turing tests:** can people tell agent from human?
  - Needed for qualities like believability (paper 064's TrueSkill study).
- **Objective:**
  - **metrics:** success rate, human-likeness, efficiency (cost, number of steps);
  - **protocols:** real-world simulation, social evaluation, multi-task, software testing;
  - **benchmarks:** ALFWorld, WebShop, Mind2Web, ToolBench, AgentBench and others.
- **Report cost, not only success.** An agent that succeeds 5% more often with 10× the LLM calls may not be better.

---

## 6. Open challenges (Section 6)
1. **Role-playing capability:** LLMs play some roles poorly (rare jobs, specific personalities).
2. **Generalised human alignment:** for simulation you sometimes **want** agents with human flaws, not idealised assistants.
3. **Prompt robustness:** small prompt changes can break complex multi-module agents.
4. **Hallucination:** confident false statements, made worse when agents act on them.
5. **Knowledge boundary:** an agent simulating a 1950s person shouldn't "know" about smartphones, but LLMs know everything at once.
6. **Efficiency:** many sequential LLM calls make agents slow and expensive.

---

## 7. What our code found
A survey has no experiment to reproduce, so `agentkit.py` implements the **framework as composable modules**:
- **profiles:** handcrafted, generated and dataset-aligned;
- **memories:** unified and hybrid, with de-duplication, importance-aware eviction and reflection;
- **five planning strategies.**

They are tested against a **simulated LLM** that gets each step right with probability p, so every design choice can be measured, and checked against closed-form probabilities.

**Planning on a 6-step task with p = 0.8** (judge accuracy 0.8):

| Strategy | Success | LLM calls / task |
|---|---|---|
| Single path (CoT / plan-then-execute) | 25.4% (closed form 0.8⁶ = 26.2%) | 3.7 |
| Multi-path: self-consistency, 5 chains | 38.6% | 18.4 |
| Multi-path: tree search, 3 proposals + self-evaluation | 63.9% | 22.2 |
| **Environment feedback** (ReAct-style), 2 retries | **94.6%** (closed form 95.3%) | 7.3 |
| Model feedback across trials (Reflexion), 3 trials | 70.7% | 7.6 |

**By task length (p = 0.9):**

| Steps | Single path | Self-consistency | Tree search | Env. feedback | Reflexion |
|---|---|---|---|---|---|
| 2 | 79.1% | 99.1% | 94.0% | 99.7% | 100% |
| 10 | 32.8% | 55.7% | 71.3% | 99.1% | 78.1% |
| 20 | 11.6% | 13.9% | 50.2% | 97.4% | 33.4% |

- **Open-loop plans collapse on long tasks** (0.9²⁰ = 12%).
- **Voting over whole chains** helps only while the right answer is still the most common one.
- **Self-evaluation** helps exactly as much as the judge is accurate.
- **Feedback from the world is the strongest signal per call.**
- **Honest caveat:** our environment says exactly **which** step failed and allows free retries. Real feedback is noisier and delayed, so those numbers are an upper bound.

**Memory (key-door task):** read which key opens a door, then experience `gap` unrelated events, then open the door.

| Gap | Unified memory (window 8) | Hybrid memory |
|---|---|---|
| 2–5 | 100% | 100% |
| 10–200 | **0%** | **100%** |

- **A short-term-only memory forgets anything that leaves the window.**
- **A hybrid memory reads the fact back** by relevance and importance.
- **Writing merges duplicates,** and when the store overflows it evicts the **least important** record. The tests check that the key survives while plain FIFO eviction would lose it (E3).

**`experiments.py`:**
- **E1:** a grid of p × L for all strategies, with costs;
- **E2:** tree search vs judge accuracy;
- **E3:** importance-aware vs FIFO eviction with unique distractors;
- **E4:** a **real LLM** on the Game of 24: single chain vs self-consistency vs propose-and-verify with a Python checker as the environment.
- E1–E3 are quick; E4 needs a GPU.

---

## 8. Check yourself

1. Name the four modules of the survey's framework and one design choice in each.
<details><summary>Answer</summary>Profile (handcrafted / LLM-generated / dataset-aligned), memory (unified vs hybrid; formats; reading / writing / reflection), planning (with or without feedback; single- vs multi-path), action (goal, production, space: tools vs internal knowledge, impact).</details>

2. What is the difference between unified and hybrid memory?
<details><summary>Answer</summary>Unified memory is only short-term: what fits in the prompt. Hybrid memory adds a long-term store (e.g. a vector database) from which relevant old memories are retrieved into the prompt.</details>

3. What two problems does memory writing have to handle?
<details><summary>Answer</summary>Duplicates (the same information stored repeatedly) and overflow (deciding what to forget when storage is full).</details>

4. A plan has 10 steps, each right with probability 0.9, and no feedback. What is the chance it succeeds?
<details><summary>Answer</summary>0.9¹⁰ ≈ 35% (our simulation: 32.8%).</details>

5. With environment feedback and 2 retries per step, what is the success probability for the same task?
<details><summary>Answer</summary>Each step succeeds with 1 − 0.1³ = 0.999, so the task succeeds with 0.999¹⁰ ≈ 99.0%.</details>

6. Why can self-consistency fail on long tasks even with many chains?
<details><summary>Answer</summary>It votes on final answers. If each chain is right with only 0.9²⁰ ≈ 12%, most chains are wrong. If the wrong answers coincide, or the right one isn't the plurality, voting doesn't help (13.9% vs 11.6% at 20 steps in our simulation).</details>

7. What limits a tree search that uses the model to evaluate its own steps?
<details><summary>Answer</summary>The evaluator's accuracy: if it accepts wrong steps or rejects right ones, the search is misled. E2 sweeps judge accuracy to show this.</details>

8. Give two ways to acquire capabilities without fine-tuning.
<details><summary>Answer</summary>Prompt engineering (examples, chain of thought, role descriptions) and mechanism engineering (trial-and-error with feedback, multi-agent debate, accumulating skills in memory, self-driven evolution).</details>

9. Why should agent evaluations report efficiency?
<details><summary>Answer</summary>Strategies differ enormously in cost (in our table, 3.7 vs 22 LLM calls per task). A small gain in success may not justify many times the calls, latency and money.</details>

10. What is the "knowledge boundary" challenge?
<details><summary>Answer</summary>When simulating a specific person or time, the agent should only know what that person would know, but LLMs know everything in their training data and leak it into the role.</details>
