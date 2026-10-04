# Generative Agents: Interactive Simulacra of Human Behavior, explained simply

**Paper:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang & Michael S. Bernstein (Stanford & Google), *Generative Agents: Interactive Simulacra of Human Behavior*, UIST 2023.

**In one sentence:** give each of 25 LLM-driven characters in a small virtual town a **memory stream** of everything they experience, a way to **retrieve** the right memories, the ability to **reflect** into higher-level conclusions, and to **plan** their days. They then behave believably over days: they spread news, form friendships and organise a Valentine's Day party, all without being scripted to.

---

## 1. The big idea

### 1.1 What an LLM alone can't do
- **An LLM can play a character for one reply.** But a believable agent must:
  - **remember** what happened yesterday;
  - **notice** what matters;
  - **draw conclusions** ("Klaus is passionate about research");
  - **act consistently over time.**
- **A prompt can't hold a lifetime of experiences,** and the model forgets everything between calls.

### 1.2 The architecture (Figure 5)
1. **Memory stream:** a long list of natural-language records with timestamps, e.g.
   - "Isabella Rodriguez is setting out the pastries";
   - "Maria Lopez is studying for a chemistry test while drinking coffee".
2. **Retrieval:** when deciding what to do or say, pull the few memories most useful **right now**.
3. **Reflection:** periodically write higher-level thoughts about the memories, and store them as memories too.
4. **Planning:** write a day plan, refine it into hourly and then 5–15-minute steps, and react when something unexpected happens.

The LLM (ChatGPT, gpt-3.5-turbo) does all the language work. **The architecture decides which information it sees.**

---

## 2. Retrieval: three scores (Section 4.1)
For a query (the current situation), every memory gets:
```
score = α_recency · recency + α_importance · importance + α_relevance · relevance      (all α = 1)
```
- **Recency:** exponential decay over **game hours since the memory was last accessed**, with factor **0.995** per hour. Retrieving a memory refreshes it.
- **Importance:** asked of the LLM **once**, when the memory is created:
  > "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory."
  - "Cleaning up the room" → **2**; "asking your crush out on a date" → **8**.
- **Relevance:** the **cosine similarity** between the memory's embedding and the query's embedding.
- **Scaling:** each of the three is **min-max scaled** to [0, 1] over all memories before adding.

**Worked example (our demo):** the query is "What is Klaus passionate about? research" at hour 24.

| Memory | Raw recency | Scaled recency | Importance | Relevance | **Total** |
|---|---|---|---|---|---|
| ate breakfast (hour 0, importance 1) | 0.887 | 0.00 | 0.00 | 0.00 | 0.00 |
| talked with Maria about research (hour 10, importance 6) | 0.932 | 0.44 | 1.00 | 1.00 | **2.44** |
| library desk is free (hour 22, importance 2) | 0.990 | 1.00 | 0.20 | 0.51 | 1.71 |

- **Why it matters:** the raw recencies differ only slightly (0.887 to 0.990), because 0.995 per hour decays slowly. **Min-max scaling stretches** them to the full [0, 1].
- **So recency is a strong force:** we found that hundreds of fresh, trivial perceptions can crowd out important but older news, unless importance pulls it back up.

---

## 3. Reflection (Section 4.2)
- **When:** once the **summed importance** of recent events exceeds **150**, which happened about 2–3 times a game day.
- **How:**
  1. Give the LLM the **100 most recent** memories and ask: "Given only the information above, what are **3 most salient high-level questions** we can answer about the subjects in the statements?" For example, "What topic is Klaus Mueller passionate about?"
  2. Use each question as a **retrieval query** to gather evidence, including earlier reflections.
  3. Ask for **5 high-level insights**, each citing its evidence: "*Klaus Mueller is dedicated to his research on gentrification (because of 1, 2, 8, 15)*".
  4. Store each insight as a **reflection** that **points to** its evidence.
- **The result is a reflection tree** (Figure 7):
  - the leaves are observations;
  - higher nodes are increasingly abstract thoughts built on lower ones.
- **Why it matters:** asked "whom would you spend an hour with?", an agent without reflections picks whoever it interacted with most often. One with reflections can pick someone who **shares its passion**.

---

## 4. Planning and reacting (Section 4.3)
- **Top-down recursive planning:**
  1. From the agent's summary description and a summary of yesterday, write a **day plan** of 5–8 chunks ("1) wake up and complete the morning routine at 8:00 am, 2) go to Oak Hill College to take classes starting 10:00 am, …").
  2. **Decompose** it into **hour-long** actions, then into **5–15-minute** actions.
- **Plans go into the memory stream too,** so they can be retrieved and stay consistent.
- **Reacting:** at each step the agent perceives its surroundings and asks, "Should John react to this observation, and if so, how?" If yes, it re-plans from that moment.
- **Dialogue:** when two agents meet, each utterance is generated from retrieved memories about the other person and the conversation so far.

---

## 5. Smallville and what emerged (Sections 3 and 7)
- **The town:** 25 agents in a 2D world with houses, a café, a bar, a college, a park, stores and more. Each agent starts from one paragraph of description (occupation, relationships). A user can talk to agents or change the world.
- **Emergent behaviour over two game days:**

| Measure | Start | End |
|---|---|---|
| Knew about **Sam's mayoral candidacy** | 1 (4%) | **8 (32%)** |
| Knew about **Isabella's Valentine's party** | 1 (4%) | **13 (52%)** |
| **Network density** (share of pairs that know each other) | 0.167 | **0.74** |

- **Accuracy of what they knew:** among 453 answers about who knows whom, only **1.3%** (6) were hallucinated.
- **The party:**
  - Isabella was given only the **intent** to throw a party.
  - She invited people, decorated the café with Maria's help, and **5 of the 12 invited agents showed up**.
  - Of the 7 who didn't: 3 had conflicts, and 4 said they were interested but never **planned** to go.

---

## 6. Does each piece matter? (Section 6)
- **The interview set-up:**
  - **Agents interviewed:** each agent after 2 game days, with questions on self-knowledge, memory, plans, reactions and reflections.
  - **Judges:** **100 human participants** watched replays and **ranked** the answers from 5 conditions by believability.
  - **Scoring:** rankings were converted to **TrueSkill** ratings (a multi-player generalisation of Elo).

| Condition | TrueSkill μ |
|---|---|
| **Full architecture** | **29.89** |
| No reflection | 26.88 |
| No reflection, no planning | 25.64 |
| Human crowdworkers role-playing the agent | 22.95 |
| No memory at all (the previous state of the art) | 21.21 |

- **Each removed component lowers believability.** Full vs no memory is an effect size of **d = 8.16** standard deviations.
- **All pairwise differences are significant** (Kruskal–Wallis, then Dunn post-hoc tests), except crowdworkers vs no memory.
- **Failure modes:**
  - **retrieval misses:** an agent "hadn't been following the election" although it had heard about it;
  - **embellishment:** adding plausible but false details;
  - **odd places:** e.g. eating lunch at the bar once it learned the bar existed;
  - **overly polite, cooperative behaviour**, inherited from the instruction-tuned model.

---

## 7. Why it matters
- **It defined the memory–reflection–planning template** used by many later LLM agents and multi-agent simulations (social science, games, prototyping social systems).
- **It turned "long-term memory for LLMs" into retrieval with a scoring rule.** Recency, importance and relevance became standard ingredients.
- **It raised ethical questions:**
  - people forming parasocial relationships with agents;
  - the risk of deepfake-like misuse;
  - the need to disclose that a character is an AI.

---

## 8. What our code found
`agents.py` implements the paper's memory stream, retrieval score, reflection trigger and tree, and recursive-planning structure. Where the paper calls ChatGPT, it uses **transparent stand-ins**:
- importance from keywords;
- relevance from hashed bag-of-words embeddings;
- salient questions from frequent words;
- insights of the form "X keeps coming back to Y (because of …)".

**Exact checks (tests):**
- **Retrieval:** the score is the sum of the three min-max-scaled terms, and retrieval refreshes recency.
- **Reflection:** the trigger fires above 150; reflections cite evidence; observations have depth 0 and reflections depth 1.
- **Importance:** the stand-in ranks "asking your crush out" above "cleaning the room".

**The toy town:** 25 agents follow hourly plans between 8 places. They **perceive** who is around ("agent3 is at the cafe", importance 1–2). When two meet, each mentions the first **news** item among its top 30 retrieved memories. Each morning, each agent **plans its day from its top 10 memories** for "what should I do today?". Results over 5 seeds:

| Retrieval | Knew of candidacy | Knew of party | Came to party | Density start → end |
|---|---|---|---|---|
| Full (1, 1, 1) | 9.4 / 25 | 11.8 / 25 | 3.6 | 0.294 → 0.406 |
| **No importance** (1, 0, 1) | **1.0** | **1.0** | **0** | 0.294 → 0.403 |
| No recency (0, 1, 1) | 16.8 | 10.2 | 3.2 | 0.294 → 0.407 |
| No relevance (1, 1, 0) | 17.0 | 10.2 | 3.2 | 0.294 → 0.409 |
| Full, no reflection | 9.4 | 11.8 | 3.6 | 0.294 → 0.406 |

- **Importance is what lets news travel.** Without it, days-old news never outranks the stream of fresh, mundane perceptions, so nobody ever mentions it.
- **Dropping recency or relevance makes news spread *more* here:** both favour fresh, situation-specific perceptions, which compete with old news.
- **Knowing is not enough to act:** about 12 agents knew about the party, but only ~3.6 came. The memory must also be **retrieved when planning the day**, much like the paper's "4 expressed interest but did not plan to come".
- **Honest limits:**
  - **Stand-ins:** our "LLM" pieces are simple rules, so the numbers describe this toy, not Smallville.
  - **Reflection changes nothing here,** because our conversations and planner use only observations (in the paper reflections shape dialogue and plans).
  - **No reflection tree:** in our demo, a second round of reflection retrieved only observations, so no depth-2 node formed.
  - **Tuning:** we tuned the conversation set-up (retrieve 30, share the first news item) because sharing only the single top memory spread nothing at all. Fresh perceptions always won.

**`experiments.py` (written, not run on this laptop):**
- **E1:** the same machinery with a **real LLM** for importance, questions and insights;
- **E2:** interview ablations judged pairwise by an LLM (Bradley–Terry instead of human TrueSkill);
- **E3:** sweeps of decay, retrieval size and reflection threshold.

---

## 9. Check yourself

1. Write the retrieval score and say how each term is computed.
<details><summary>Answer</summary>score = recency + importance + relevance (all weights 1), each min-max scaled to [0, 1]. Recency = 0.995^(hours since last access); importance = the LLM's 1–10 poignancy rating at creation time; relevance = cosine similarity of the memory's embedding with the query's embedding.</details>

2. Raw recencies 0.887, 0.932 and 0.990 become what after min-max scaling?
<details><summary>Answer</summary>(0.887 − 0.887)/(0.990 − 0.887) = 0; (0.932 − 0.887)/0.103 ≈ 0.44; 1.00. Small raw differences become the full [0, 1] range.</details>

3. When does an agent reflect, and what does a reflection look like?
<details><summary>Answer</summary>When the summed importance of recent events exceeds 150 (2–3 times a day). It generates 3 salient questions from the 100 most recent memories, retrieves evidence for each, and writes insights that cite that evidence, e.g. "Klaus is dedicated to his research (because of 1, 2, 8, 15)", stored as memories that point to their sources.</details>

4. Why do reflections form a tree?
<details><summary>Answer</summary>Each reflection points to the memories it was built from, and those can include earlier reflections. Observations are the leaves; higher reflections are increasingly abstract.</details>

5. How does planning work?
<details><summary>Answer</summary>Top-down: a day plan in 5–8 broad chunks, recursively broken into hour-long and then 5–15-minute actions. Plans are stored in the memory stream and revised when the agent decides to react to something it observes.</details>

6. What emerged in Smallville without being scripted?
<details><summary>Answer</summary>Information diffusion (the candidacy: 4% → 32% of agents; the party: 4% → 52%), relationship formation (network density 0.167 → 0.74), and coordination (5 of 12 invitees attended Isabella's party, with help decorating).</details>

7. Rank the five interview conditions by TrueSkill and say what the order shows.
<details><summary>Answer</summary>Full 29.89 > no reflection 26.88 > no reflection/planning 25.64 > crowdworkers 22.95 > no memory 21.21. Every component adds believability, and the full architecture even beat humans role-playing the agents.</details>

8. In our toy, why does removing the importance term stop news from spreading?
<details><summary>Answer</summary>The speaker shares news only if it appears among its top retrieved memories. Without importance, recency and relevance dominate, and both favour fresh perceptions like "X is at the cafe". Old news can never compete, so it is never mentioned.</details>

9. In our toy, about 12 agents knew about the party but only about 3.6 came. Why?
<details><summary>Answer</summary>Coming requires the party memory to be retrieved when the agent plans its day. For many agents other memories ranked higher at planning time, so the party never entered their plan, much like the paper's agents who were interested but never planned to come.</details>

10. Name two failure modes the paper reports.
<details><summary>Answer</summary>Any two of: failing to retrieve relevant memories; embellishing memories with plausible but false details; choosing odd locations for actions as their knowledge of places grows; overly formal and cooperative behaviour inherited from the instruction-tuned LLM.</details>
