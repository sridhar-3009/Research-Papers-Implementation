"""ReAct: Synergizing Reasoning and Acting in Language Models (Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao, ICLR 2023).

  Idea: let the model interleave free-form THOUGHTS (language that changes nothing in the world) with ACTIONS (that
    call an environment) and read the resulting OBSERVATIONS:  Thought 1 / Action 1 / Observation 1 / Thought 2 / ...
    Formally the action space becomes A u L (environment actions plus language). Thoughts decompose the goal, extract
    facts from observations, track progress, handle exceptions and decide the answer; actions fetch real information.
  Knowledge tasks (HotpotQA multi-hop QA, FEVER fact checking), question only, with a tiny Wikipedia API:
    search[entity] -> first 5 sentences of the page (or 5 similar titles), lookup[string] -> next sentence containing
    the string (Ctrl+F), finish[answer]. Few-shot prompts: 6 (HotpotQA) / 3 (FEVER) hand-written trajectories.
    Baselines are ablations of the same trajectories: Standard (no thoughts/actions), CoT (thoughts only), Act (no
    thoughts). Back-off: ReAct -> CoT-SC if no answer within 7 (HotpotQA) / 5 (FEVER) steps; CoT-SC -> ReAct if the
    majority among 21 samples has < n/2 votes.
  Table 1 (PaLM-540B; HotpotQA EM / FEVER acc): Standard 28.7/57.1, CoT 29.4/56.3, CoT-SC 33.4/60.4, Act 25.7/58.9,
    ReAct 27.4/60.9, CoT-SC->ReAct 34.2/64.6, ReAct->CoT-SC 35.1/62.0.
  Table 2 (HotpotQA failure analysis): CoT's main failure is HALLUCINATION (56% of failures, false positives 14%);
    ReAct's are reasoning errors incl. repetitive loops (47%) and unhelpful search results (23%), 0% hallucination.
  Decision making: ALFWorld success 71% (ReAct best of 6) vs 45% (Act) vs 37% (BUTLER imitation learning, 10^5
    trajectories); WebShop success 40.0% vs 30.1% (Act) vs 29.1% (IL). Fine-tuned on 3,000 ReAct trajectories,
    PaLM-8B ReAct beats all PaLM-62B prompting methods.
"""

import difflib
import re

# ---------------------------------------------------------------------------------------------------- environment
class WikiEnv:
    """A local stand-in for the paper's Wikipedia API: pages = {title: text}."""

    def __init__(self, pages):
        self.pages = pages
        self.page, self.lookup_key, self.lookup_hits, self.answer = None, None, [], None

    @staticmethod
    def sentences(text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def search(self, entity):
        if entity in self.pages:
            self.page = entity
            return " ".join(self.sentences(self.pages[entity])[:5])
        similar = difflib.get_close_matches(entity, list(self.pages), n=5, cutoff=0.3)
        return f"Could not find [{entity}]. Similar: {similar}."

    def lookup(self, key):
        if self.page is None:
            return "No page loaded; use search first."
        if key != self.lookup_key:
            self.lookup_key = key
            self.lookup_hits = [s for s in self.sentences(self.pages[self.page]) if key.lower() in s.lower()]
        if not self.lookup_hits:
            return "No more results."
        return f"(Result {len(self.lookup_hits)} left) " + self.lookup_hits.pop(0)

    def step(self, action):
        """Returns (observation, done)."""
        m = re.fullmatch(r"\s*(search|lookup|finish)\[(.*)\]\s*", action)
        if not m:
            return f"Invalid action: {action}", False
        kind, arg = m.groups()
        if kind == "finish":
            self.answer = arg
            return f"Episode finished, answer: {arg}", True
        return (self.search(arg) if kind == "search" else self.lookup(arg)), False


# ---------------------------------------------------------------------------------------------------- prompting
def react_loop(generate, question, env, exemplars="", max_steps=7):
    """The ReAct loop. `generate(prompt, stop)` returns text up to (not including) a stop string. For step i the model
    writes 'Thought i: ...' and 'Action i: ...'; the environment's reply is appended as 'Observation i: ...'.
    Returns (answer or None, transcript)."""
    transcript = exemplars + f"Question: {question}\n"
    for i in range(1, max_steps + 1):
        step = generate(transcript + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        if f"\nAction {i}:" not in step:                                           # the model forgot to act
            step = step.split("\n")[0] + f"\nAction {i}:" + generate(transcript + f"Thought {i}:{step.split(chr(10))[0]}\nAction {i}:", stop=["\n"])
        thought, action = step.split(f"\nAction {i}:", 1)
        obs, done = env.step(action.strip())
        transcript += f"Thought {i}:{thought}\nAction {i}: {action.strip()}\nObservation {i}: {obs}\n"
        if done:
            return env.answer, transcript
    return None, transcript


def ablate(trajectory, keep):
    """Build the baselines from one ReAct trajectory (list of (kind, text)), as the paper does: Standard keeps only the
    final answer, CoT keeps thoughts, Act keeps actions + observations, ReAct keeps everything."""
    allowed = {"standard": set(), "cot": {"thought"}, "act": {"action", "observation"},
               "react": {"thought", "action", "observation"}}[keep]
    return [(k, t) for k, t in trajectory if k in allowed or k == "answer"]


def react_then_cot(react_answer, cot_sc_answers):
    """ReAct -> CoT-SC: if ReAct gave no answer within its step budget, fall back to the CoT-SC majority."""
    if react_answer is not None:
        return react_answer, "react"
    return majority(cot_sc_answers)[0], "cot-sc"


def cot_then_react(cot_sc_answers, react_fn):
    """CoT-SC -> ReAct: if the CoT-SC majority has fewer than n/2 votes (not confident), run ReAct instead."""
    ans, votes = majority(cot_sc_answers)
    if votes >= len(cot_sc_answers) / 2:
        return ans, "cot-sc"
    return react_fn(), "react"


def majority(answers):
    counts = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return None, 0
    best = max(counts, key=counts.get)
    return best, counts[best]


# ---------------------------------------------------------------------------------------------------- toy world
SPECIAL = ["Q", "T1", "T2", "T3", "S", "O", "FIN", "dir", "born", "END"]
N_FILMS, N_PEOPLE, N_CITIES, CTX = 200, 120, 20, 24


class ToyWorld:
    """Films -> directors -> birth cities. Questions: 'Q F17' = 'where was the director of film F17 born?' (2 hops).
    The 'encyclopedia' page of a film says who directed it; a person's page says where they were born."""

    def __init__(self, seed=0):
        import random
        rng = random.Random(seed)
        self.vocab = SPECIAL + [f"F{i}" for i in range(N_FILMS)] + [f"P{i}" for i in range(N_PEOPLE)] + [f"C{i}" for i in range(N_CITIES)]
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        self.director = {f"F{i}": f"P{rng.randrange(N_PEOPLE)}" for i in range(N_FILMS)}
        self.born = {f"P{i}": f"C{rng.randrange(N_CITIES)}" for i in range(N_PEOPLE)}
        self.films = list(self.director)
        self.outage = False                                                       # the search tool is down

    def page(self, entity):
        if self.outage:
            return []
        if entity in self.director:
            return [entity, "dir", self.director[entity]]
        if entity in self.born:
            return [entity, "born", self.born[entity]]
        return []

    def answer(self, film):
        return self.born[self.director[film]]

    def trajectory(self, film, method, rng=None):
        """A training trajectory. For the tool-using methods, half the time the pages are COUNTERFACTUAL (random
        director / city), so the only way to be right is to read the observation instead of recalling."""
        p = self.director[film]; c = self.born[p]
        pages = {film: [film, "dir", p], p: [p, "born", c]}
        if rng is not None and method in ("react", "act") and rng.random() < 0.5:
            p = f"P{rng.randrange(N_PEOPLE)}"; c = f"C{rng.randrange(N_CITIES)}"
            pages = {film: [film, "dir", p], p: [p, "born", c]}
        if method == "react":
            return ["Q", film, "T1", "S", film, "O"] + pages[film] + ["T2", "S", p, "O"] + pages[p] + ["T3", "FIN", c, "END"]
        if method == "act":
            return ["Q", film, "S", film, "O"] + pages[film] + ["S", p, "O"] + pages[p] + ["FIN", c, "END"]
        if method == "cot":
            return ["Q", film, "T1", film, "dir", p, "T2", p, "born", c, "T3", "FIN", c, "END"]
        return ["Q", film, "FIN", c, "END"]                                      # standard


def _llama():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "08-Pretraining-and-Scaling-LLMs" / "056-Touvron-et-al-2023-LLaMA" / "llama.py"
    spec = importlib.util.spec_from_file_location("llama_056", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def train_toy(world, method, steps, d=64, seed=0, lr=3e-3, batch=64):
    """Imitation of trajectories (like the paper's fine-tuning on 3,000 ReAct trajectories). The loss skips the
    observation tokens: the environment writes those, not the model."""
    import random
    import torch
    import torch.nn.functional as F
    L = _llama()
    torch.manual_seed(seed); rng = random.Random(seed + 1)
    m = L.LLaMA(vocab=len(world.vocab), d=d, layers=2, heads=4, ctx=CTX, hidden=L.ffn_hidden(d, 16))
    opt = torch.optim.AdamW(m.parameters(), lr)
    for _ in range(steps):
        X, M = [], []
        for _ in range(batch):
            t = world.trajectory(rng.choice(world.films), method, rng)
            ids = [world.idx[x] for x in t]
            mask = [0, 0] + [0 if "O" in t[max(0, j - 3):j] else 1 for j in range(2, len(t))]
            X.append(ids + [world.idx["END"]] * (CTX - len(ids))); M.append(mask + [0] * (CTX - len(ids)))
        X = torch.tensor(X); M = torch.tensor(M)[:, 1:].float()
        nll = F.cross_entropy(m(X)[:, :-1].reshape(-1, len(world.vocab)), X[:, 1:].reshape(-1), reduction="none").view(M.shape)
        loss = (nll * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval(), loss.item()


def run_toy(world, model, film, method, max_tokens=20):
    """Greedy decoding; after the model writes 'O' (call the tool), the ENVIRONMENT appends the page of the entity
    it searched for. Returns (answer or None, token trajectory)."""
    import torch
    seq = ["Q", film]
    with torch.no_grad():
        for _ in range(max_tokens):
            nxt = world.vocab[int(model(torch.tensor([[world.idx[x] for x in seq[-CTX:]]]))[0, -1].argmax())]
            seq.append(nxt)
            if nxt == "O" and method in ("react", "act"):
                seq += world.page(seq[-2])
            if len(seq) >= 2 and seq[-2] == "FIN":
                return (nxt if nxt.startswith("C") else None), seq
            if nxt == "END":
                return None, seq
    return None, seq


# ---------------------------------------------------------------------------------------------------- reported
TABLE_1 = {  # PaLM-540B: (HotpotQA EM, FEVER accuracy)
    "Standard": (28.7, 57.1), "CoT": (29.4, 56.3), "CoT-SC (21 samples)": (33.4, 60.4), "Act": (25.7, 58.9),
    "ReAct": (27.4, 60.9), "CoT-SC -> ReAct": (34.2, 64.6), "ReAct -> CoT-SC": (35.1, 62.0),
    "Supervised SoTA": (67.5, 89.5)}
TABLE_2 = {  # HotpotQA, 50 sampled trajectories each: (ReAct %, CoT %)
    "success: true positive": (94, 86), "success: false positive (hallucinated)": (6, 14),
    "failure: reasoning error": (47, 16), "failure: search result error": (23, None),
    "failure: hallucination": (0, 56), "failure: label ambiguity": (29, 28)}
DECISION_MAKING = {"ALFWorld success: ReAct (best of 6)": 71, "ALFWorld: ReAct (avg)": 57, "ALFWorld: Act (best of 6)": 45,
                   "ALFWorld: BUTLER (best of 8)": 37, "WebShop success: ReAct": 40.0, "WebShop: Act": 30.1,
                   "WebShop: IL": 29.1, "WebShop: IL+RL": 28.7}
