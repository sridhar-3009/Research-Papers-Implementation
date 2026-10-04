"""Generative Agents: Interactive Simulacra of Human Behavior (Park, O'Brien, Cai, Morris, Liang & Bernstein,
UIST 2023).

  25 LLM-driven agents live in 'Smallville' (a Sims-like town). Architecture (Section 4), around a MEMORY STREAM of
  natural-language records (observations, reflections, plans), each with creation time, last-access time, importance:
    Retrieval: score = a_rec * recency + a_imp * importance + a_rel * relevance, all a = 1, each term min-max scaled
      to [0, 1]; recency = 0.995^(game hours since last access); importance = an integer 1-10 the LLM assigns when the
      memory is created ('2' cleaning the room, '8' asking your crush out); relevance = cosine similarity between the
      memory's embedding and the query's. The top memories that fit the context are put in the prompt.
    Reflection: when the summed importance of recent events exceeds 150 (2-3 times a day), ask the LLM for the
      3 most salient questions about the 100 most recent records, retrieve for each, and ask for 5 insights
      'insight (because of 1, 5, 3)' -> stored as reflections POINTING to their evidence: a reflection tree.
    Planning: a day plan of 5-8 chunks, recursively decomposed into hour-long, then 5-15 minute actions; plans are
      memories too and can be revised when the agent reacts to observations; dialogue is generated from memories.
  Evaluation: 'interviews' (self-knowledge, memory, plans, reactions, reflections); 100 participants ranked 5
    conditions; TrueSkill mu: full 29.89, no reflection 26.88, no reflection/planning 25.64, human crowdworkers 22.95,
    no memory/reflection/planning 21.21 (Cohen's d = 8.16 full vs none).
  Emergence over 2 game days: Sam's candidacy known by 1 -> 8 agents (4% -> 32%), Isabella's party 1 -> 13 (4% ->
    52%), network density 0.167 -> 0.74, 1.3% of 453 answers hallucinated; 5 of 12 invited agents came to the party.
  Failure modes: retrieval misses, embellishment, odd location choices, overly formal / cooperative behaviour.

Where the paper calls ChatGPT, this file uses small, transparent STAND-INS (keyword importance, hashed bag-of-words
embeddings, templated questions/insights); the memory, retrieval, reflection and planning machinery is the paper's.
"""

import hashlib
import math
import re
from collections import Counter

import numpy as np

DECAY = 0.995
REFLECT_THRESHOLD = 150


# ---------------------------------------------------------------------------------------------------- stand-ins for the LLM
IMPORTANT_WORDS = {"party": 6, "valentine": 6, "election": 6, "mayor": 7, "candidacy": 7, "running": 4, "invite": 5,
                   "crush": 8, "date": 5, "research": 4, "paper": 3, "project": 3, "breakup": 9, "accepted": 8,
                   "exhibition": 5, "decorate": 3, "talking": 2}


def rate_importance(text):
    """Stand-in for 'On the scale of 1 to 10 ... rate the likely poignancy': keyword-based, clipped to 1-10."""
    words = re.findall(r"[a-z]+", text.lower())
    return int(min(10, max(1, 1 + max([IMPORTANT_WORDS.get(w, 0) for w in words] or [0]))))


def embed(text, dim=256):
    """Stand-in for an LLM embedding: hashed bag of (lower-cased, crudely stemmed) words, unit length."""
    v = np.zeros(dim)
    for w in re.findall(r"[a-z]+", text.lower()):
        w = re.sub(r"(ing|ed|s)$", "", w) if len(w) > 4 else w
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


def minmax(x):
    x = np.asarray(x, float)
    return np.zeros_like(x) if x.max() == x.min() else (x - x.min()) / (x.max() - x.min())


# ---------------------------------------------------------------------------------------------------- memory stream
class Memory:
    def __init__(self, text, t, kind="observation", importance=None, evidence=()):
        self.text, self.created, self.last_access, self.kind = text, t, t, kind
        self.importance = importance if importance is not None else rate_importance(text)
        self.embedding, self.evidence = embed(text), list(evidence)


class MemoryStream:
    def __init__(self, weights=(1.0, 1.0, 1.0)):
        self.records, self.weights = [], weights
        self.importance_since_reflection = 0

    def add(self, text, t, kind="observation", importance=None, evidence=()):
        m = Memory(text, t, kind, importance, evidence)
        self.records.append(m)
        if kind == "observation":
            self.importance_since_reflection += m.importance
        return m

    def scores(self, query, t):
        """The three components, each min-max scaled, and the weighted total (Section 4.1)."""
        recency = minmax([DECAY ** (t - m.last_access) for m in self.records])
        importance = minmax([m.importance for m in self.records])
        q = embed(query)
        relevance = minmax([float(m.embedding @ q) for m in self.records])
        a_rec, a_imp, a_rel = self.weights
        return a_rec * recency + a_imp * importance + a_rel * relevance, (recency, importance, relevance)

    def retrieve(self, query, t, k=5):
        if not self.records:
            return []
        total, _ = self.scores(query, t)
        top = [self.records[i] for i in np.argsort(-total)[:k]]
        for m in top:
            m.last_access = t                                                     # retrieval refreshes recency
        return top

    def should_reflect(self):
        return self.importance_since_reflection > REFLECT_THRESHOLD


def salient_questions(records, n=3, exclude=()):
    """Stand-in for 'what are 3 most salient high-level questions?': the most frequent content words of the most
    recent records become questions about them."""
    stop = set("the a an is are was to of in at on and with for about his her their from by it this that be".split())
    stop |= {e.lower() for e in exclude}
    words = Counter(w for m in records for w in re.findall(r"[a-z]+", m.text.lower()) if w not in stop and len(w) > 3)
    return [f"What is important about {w}?" for w, _ in words.most_common(n)]


def reflect(agent_name, stream, t, n_recent=100):
    """Questions from the 100 most recent records -> retrieve evidence for each -> an insight that CITES the evidence
    (stand-in wording 'X keeps coming back to Y'), stored as a reflection pointing at its sources."""
    recent = stream.records[-n_recent:]
    new = []
    for q in salient_questions(recent, exclude=(agent_name,)):
        evidence = stream.retrieve(q, t, k=5)
        topic = q.split("about ")[-1].rstrip("?")
        cites = [stream.records.index(m) + 1 for m in evidence]
        text = f"{agent_name} keeps coming back to {topic} (because of {', '.join(map(str, cites))})"
        new.append(stream.add(text, t, kind="reflection", importance=max(m.importance for m in evidence), evidence=evidence))
    stream.importance_since_reflection = 0
    return new


def reflection_depth(m):
    """Leaves (observations) have depth 0; a reflection is one more than its deepest evidence: the reflection tree."""
    return 0 if not m.evidence else 1 + max(reflection_depth(e) for e in m.evidence)


# ---------------------------------------------------------------------------------------------------- planning
def plan_day(agent, t_day):
    """Top-down recursive planning: a coarse day plan in chunks, decomposed into hourly blocks (the paper further
    splits into 5-15 minute actions). Stand-in: the agent's routine, plus any event it RETRIEVES as relevant when
    planning (e.g. a party it heard about), inserted at its time."""
    blocks = [(h, place) for h0, h1, place in agent.routine for h in range(h0, h1)]
    plan = dict(blocks)
    for m in agent.memory.retrieve(f"what should {agent.name} do today", t_day, k=agent.plan_k):
        if (e := agent.world.events.get(m.text)) and e["day"] == t_day // 24:
            for h in range(e["start"], e["end"]):
                plan[h] = e["place"]
    agent.memory.add(f"{agent.name} plans: " + ", ".join(f"{h}:00 {p}" for h, p in sorted(plan.items())[:6]), t_day, "plan", 2)
    return plan


# ---------------------------------------------------------------------------------------------------- the town
class Agent:
    def __init__(self, name, routine, world, weights=(1.0, 1.0, 1.0), plan_k=5, reflect=True):
        self.name, self.routine, self.world = name, routine, world
        self.memory, self.plan_k, self.can_reflect = MemoryStream(weights), plan_k, reflect
        self.known = set()                                                        # people they have met


class Town:
    """A small Smallville: agents follow plans between places; agents in the same place observe and talk. In a
    conversation each speaker shares the memory it retrieves as most relevant to the other person, so news spreads
    only as far as retrieval carries it."""

    PLACES = ["home", "cafe", "college", "park", "market", "bar", "library", "office"]

    def __init__(self, n_agents=25, seed=0, weights=(1.0, 1.0, 1.0), plan_k=10, reflect=True, chat_prob=0.15, chat_k=30):
        rng = np.random.default_rng(seed)
        self.rng, self.events, self.t, self.chat_prob, self.chat_k = rng, {}, 0, chat_prob, chat_k
        self.agents = []
        for i in range(n_agents):
            work = self.PLACES[1 + i % (len(self.PLACES) - 1)]
            leisure = self.PLACES[1 + rng.integers(len(self.PLACES) - 1)]
            routine = [(0, 8, "home"), (8, 12, work), (12, 13, "cafe"), (13, 17, work), (17, 20, leisure), (20, 24, "home")]
            self.agents.append(Agent(f"agent{i}", routine, self, weights, plan_k, reflect))
        for a in self.agents:                                                     # initial acquaintances
            for b in rng.choice(self.agents, 4, replace=False):
                if b is not a:
                    a.known.add(b.name); b.known.add(a.name)
        self.plans = {}

    def seed_fact(self, agent_idx, text, event=None):
        a = self.agents[agent_idx]
        a.memory.add(text, self.t, importance=None)
        if event:
            self.events[text] = event

    def knows(self, keyword):
        return [a.name for a in self.agents if any(keyword in m.text for m in a.memory.records)]

    def density(self):
        n = len(self.agents)
        return sum(len(a.known) for a in self.agents) / (n * (n - 1))

    def step_hour(self):
        if self.t % 24 == 0:
            self.plans = {a.name: plan_day(a, self.t) for a in self.agents}
        hour = self.t % 24
        where = {}
        for a in self.agents:
            where.setdefault(self.plans[a.name].get(hour, "home"), []).append(a)
        for place, group in where.items():
            for a in group:                                                       # perception: who is around
                a.memory.add(f"{a.name} is at the {place}", self.t, importance=1)
                for b in group[:4]:
                    if b is not a:
                        a.memory.add(f"{b.name} is at the {place}", self.t, importance=2)
            if place == "home" or len(group) < 2:
                continue
            for a in group:
                for b in group:
                    if a is not b and self.rng.random() < self.chat_prob / len(group):
                        self.converse(a, b, place)
        for a in self.agents:
            if a.can_reflect and a.memory.should_reflect():
                reflect(a.name, a.memory, self.t)
        self.t += 1

    def converse(self, a, b, place):
        a.known.add(b.name); b.known.add(a.name)
        for speaker, listener in ((a, b), (b, a)):
            # the speaker sees its top retrieved memories and talks about the first one that is news (in the paper the
            # LLM chooses what to say from the retrieved memories); routine perceptions are not worth mentioning
            for m in speaker.memory.retrieve(f"{speaker.name} is chatting with {listener.name} at the {place}", self.t, k=self.chat_k):
                if m.kind == "observation" and " is at the " not in m.text and "was talking with" not in m.text:
                    if not any(r.text == m.text for r in listener.memory.records):
                        listener.memory.add(m.text, self.t, importance=m.importance)
                    break
        a.memory.add(f"{a.name} was talking with {b.name} at the {place}", self.t, importance=2)
        b.memory.add(f"{b.name} was talking with {a.name} at the {place}", self.t, importance=2)

    def run(self, hours):
        for _ in range(hours):
            self.step_hour()


# ---------------------------------------------------------------------------------------------------- reported
TRUESKILL = {"full architecture": (29.89, 0.72), "no reflection": (26.88, 0.69),
             "no reflection, no planning": (25.64, 0.68), "human crowdworkers": (22.95, 0.69),
             "no observation, no reflection, no planning": (21.21, 0.70)}
EMERGENCE = {"knew of Sam's candidacy": ("1 (4%)", "8 (32%)"), "knew of Isabella's party": ("1 (4%)", "13 (52%)"),
             "network density": (0.167, 0.74), "hallucinated awareness answers": "6 of 453 (1.3%)",
             "invited agents who came to the party": "5 of 12"}


def run_town(seed=0, hours=48, **kw):
    """Seed the paper's two pieces of news (a candidacy, a Valentine's party) in one agent each, simulate, measure."""
    town = Town(seed=seed, **kw)
    town.seed_fact(0, "agent0 is running for mayor in the local election")
    party = "agent5 is throwing a valentine party at the cafe on day 2 from 17 to 19"
    town.seed_fact(5, party, event={"day": 1, "start": 17, "end": 19, "place": "cafe"})
    d0 = town.density()
    town.run(hours)
    usual = lambda a: dict((h, p) for h0, h1, p in a.routine for h in range(h0, h1)).get(17)
    knew = town.knows("valentine")
    attended = [a.name for a in town.agents if a.name in knew and a.name != "agent5"
                and town.plans[a.name].get(17) == "cafe" and usual(a) != "cafe"]
    return {"mayor": len(town.knows("mayor")), "party": len(knew), "density": (d0, town.density()),
            "attended": len(attended), "reflections": sum(m.kind == "reflection" for a in town.agents for m in a.memory.records),
            "town": town}
