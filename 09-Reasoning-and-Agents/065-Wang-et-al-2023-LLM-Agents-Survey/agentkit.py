"""A Survey on Large Language Model based Autonomous Agents (Wang, Ma, Feng, Zhang, Yang, Zhang, Chen, Tang, Chen, Lin,
Zhao, Wei & Wen, Frontiers of Computer Science 2024; arXiv 2023).

The survey's unified framework (Figure 2), implemented as small composable modules:
  PROFILE: who the agent is (handcrafted, LLM-generated, or aligned to a real dataset), written into the prompt.
  MEMORY:
    structure: UNIFIED (short-term only: what fits in the context window) vs HYBRID (short-term + a long-term store);
    formats: natural language, embeddings, databases, structured lists;
    operations: READING (retrieve by recency + relevance + importance), WRITING (handle duplicates and overflow),
      REFLECTION (summarise into higher-level memories).
  PLANNING:
    without feedback: SINGLE-PATH (one chain: CoT, zero-shot CoT, plan-then-execute) and MULTI-PATH (many chains:
      self-consistency votes, Tree of Thoughts searches with self-evaluation), or an external planner;
    with feedback: ENVIRONMENT (ReAct, Voyager: the world says whether a step worked), HUMAN, MODEL (self-critique,
      Reflexion: learn from failed attempts in memory).
  ACTION: goal (task, communication, exploration), production (from memory recollection or plan following),
    space (external tools vs the LLM's own knowledge), impact (changes the environment, internal state, new actions).
  Capability acquisition: fine-tuning (human-annotated, LLM-generated, real-world data) or not (prompt engineering,
    mechanism engineering: trial-and-error, crowd-sourcing/debate, experience accumulation, self-driven evolution).
  Evaluation: subjective (human annotation, Turing tests) and objective (metrics, protocols, benchmarks); efficiency.

Because a survey has no single experiment, the modules here run against a SIMULATED LLM whose step accuracy is a
parameter, so every design choice's effect on success AND cost (number of LLM calls) can be measured and checked
against closed-form probabilities.
"""

import math
import random
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------------------------------- profile
def handcrafted_profile(name, role, traits):
    return f"You are {name}, a {role}. You are {', '.join(traits)}."


def generated_profiles(n, rng, roles=("teacher", "coder", "doctor", "farmer"), traits=("curious", "careful", "outgoing", "quiet")):
    """Stand-in for LLM-generated profiles: fill a template from seed attributes (the survey's second method)."""
    return [handcrafted_profile(f"agent{i}", rng.choice(roles), rng.sample(traits, 2)) for i in range(n)]


def dataset_aligned_profiles(rows):
    """Third method: profiles from real demographic records (e.g. survey participants)."""
    return [f"You are a {r['age']}-year-old {r['occupation']} from {r['state']}." for r in rows]


# ---------------------------------------------------------------------------------------------------- memory
class UnifiedMemory:
    """Short-term only: the last `window` records (what fits in the prompt)."""

    def __init__(self, window):
        self.window, self.records = window, []

    def write(self, text, importance=1):
        self.records.append(text)
        self.records = self.records[-self.window:]                                # overflow: the oldest are lost

    def read(self, query, k=3):
        return [r for r in self.records if any(w in r for w in query.split())][:k] or self.records[-k:]


class HybridMemory:
    """Short-term window + a long-term store read by recency + relevance + importance (as in generative agents).
    Writing de-duplicates (the survey's 'memory duplicated' problem) and evicts the least important record when the
    long-term store overflows."""

    def __init__(self, window, capacity=1000):
        self.short, self.long, self.window, self.capacity, self.t = [], [], window, capacity, 0

    def write(self, text, importance=1):
        self.t += 1
        self.short = (self.short + [text])[-self.window:]
        for rec in self.long:                                                     # duplicate: refresh instead of copy
            if rec["text"] == text:
                rec["t"], rec["importance"] = self.t, max(rec["importance"], importance)
                return
        self.long.append({"text": text, "t": self.t, "importance": importance})
        if len(self.long) > self.capacity:                                        # overflow: drop the least important
            self.long.remove(min(self.long, key=lambda r: (r["importance"], r["t"])))

    def read(self, query, k=3):
        q = set(query.split())

        def score(r):
            recency = 0.99 ** (self.t - r["t"])
            relevance = len(q & set(r["text"].split())) / max(1, len(q))
            return recency + relevance + r["importance"] / 10
        return [r["text"] for r in sorted(self.long, key=score, reverse=True)[:k]]


def reflect_memories(texts, min_count=3):
    """Memory reflection stand-in: repeated subjects become a summary ('X happened n times')."""
    c = Counter(t.split(" ")[0] for t in texts)
    return [f"{subj} occurs {n} times" for subj, n in c.items() if n >= min_count]


# ---------------------------------------------------------------------------------------------------- simulated LLM
class SimLLM:
    """A stand-in LLM for planning experiments: asked for the next step of a task, it is right with probability p;
    asked to judge a step, it is right with probability q. Every call is counted (the cost metric)."""

    def __init__(self, p, q=0.8, n_options=4, seed=0):
        self.p, self.q, self.n_options, self.rng, self.calls = p, q, n_options, random.Random(seed), 0

    def propose(self, correct):
        self.calls += 1
        return correct if self.rng.random() < self.p else f"wrong{self.rng.randrange(self.n_options - 1)}"

    def judge(self, step, correct):
        self.calls += 1
        truth = step == correct
        return truth if self.rng.random() < self.q else not truth


# ---------------------------------------------------------------------------------------------------- planning
def single_path(llm, steps):
    """CoT / plan-then-execute: one chain, no feedback; one wrong step ruins it. P(success) = p^L."""
    return all(llm.propose(s) == s for s in steps)


def self_consistency(llm, steps, n_paths=5):
    """Multi-path, vote on the FINAL outcome: each full chain is right with p^L; wrong chains scatter over many
    wrong answers, so the vote needs only a plurality."""
    finals = []
    for _ in range(n_paths):
        ok = all(llm.propose(s) == s for s in steps)
        finals.append("right" if ok else f"wrong{llm.rng.randrange(10)}")
    return Counter(finals).most_common(1)[0][0] == "right"


def tree_of_thoughts(llm, steps, breadth=3):
    """Multi-path with self-evaluation (ToT-style BFS): at each step propose `breadth` candidates, keep the first one
    the model judges correct (or the last if none). Errors happen only if the judge accepts a wrong step."""
    for s in steps:
        cands = [llm.propose(s) for _ in range(breadth)]
        chosen = next((c for c in cands if llm.judge(c, s)), cands[-1])
        if chosen != s:
            return False
    return True


def with_environment_feedback(llm, steps, retries=2):
    """ReAct / Voyager-style: execute each step; the ENVIRONMENT reports failure; retry up to `retries` times.
    P(success) = (1 - (1 - p)^(retries + 1))^L."""
    for s in steps:
        if not any(llm.propose(s) == s for _ in range(retries + 1)):
            return False
    return True


def reflexion(llm, steps, trials=3):
    """Model feedback across episodes (Reflexion-style): after a failed attempt the agent stores a lesson about the
    step it got wrong; on the next trial that step is done correctly from memory (no LLM call)."""
    lessons = set()
    for _ in range(trials):
        failed = None
        for i, s in enumerate(steps):
            if i in lessons:
                continue
            if llm.propose(s) != s:
                failed = i
                break
        if failed is None:
            return True
        lessons.add(failed)
    return False


def analytic(strategy, p, L, **kw):
    """Closed-form success probabilities (independent steps) for the strategies that have one."""
    if strategy == "single_path":
        return p ** L
    if strategy == "environment_feedback":
        return (1 - (1 - p) ** (kw.get("retries", 2) + 1)) ** L
    raise ValueError(strategy)


def evaluate(strategy, p, L, n=2000, q=0.8, seed=0, **kw):
    """Objective evaluation: success rate and mean LLM calls per task (efficiency)."""
    llm = SimLLM(p, q, seed=seed)
    steps = [f"step{i}" for i in range(L)]
    wins = sum(strategy(llm, steps, **kw) for _ in range(n))
    return wins / n, llm.calls / n


# ---------------------------------------------------------------------------------------------------- memory task
def key_door_episode(memory, gap, rng, distractors=("saw a tree", "saw a bench", "heard a bird", "saw a cloud")):
    """The agent reads which key opens the door at the start, then experiences `gap` unrelated observations, then must
    open the door: success iff its memory READ for 'key door' returns the right key."""
    key = f"key{rng.randrange(10)}"
    memory.write(f"note says {key} opens door", importance=8)
    for _ in range(gap):
        memory.write(rng.choice(distractors), importance=1)
    recalled = " ".join(memory.read("key door", k=3))
    return key in recalled


# ---------------------------------------------------------------------------------------------------- survey facts
FRAMEWORK = {
    "profile": ["handcrafting", "LLM-generation", "dataset alignment"],
    "memory structure": ["unified (short-term)", "hybrid (short + long-term)"],
    "memory format": ["natural languages", "embeddings", "databases", "structured lists"],
    "memory operation": ["reading", "writing", "reflection"],
    "planning without feedback": ["single-path reasoning", "multi-path reasoning", "external planner"],
    "planning with feedback": ["environmental", "human", "model"],
    "action": ["goal", "production", "space", "impact"],
    "capability acquisition": ["fine-tuning", "prompt engineering", "mechanism engineering"],
    "evaluation": ["subjective (human annotation, Turing test)", "objective (metrics, protocols, benchmarks)"],
}
CHALLENGES = ["role-playing capability", "generalized human alignment", "prompt robustness", "hallucination",
              "knowledge boundary", "efficiency"]
