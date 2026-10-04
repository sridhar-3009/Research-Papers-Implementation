"""WebGPT: Browser-assisted question-answering with human feedback (Nakano, Hilton, Balaji, Wu, Ouyang, Kim, Hesse,
Jain, Kosaraju, Saunders, Jiang, Cobbe, Eloundou, Krueger, Button, Knight, Chess & Schulman, OpenAI 2021).

  Task: long-form answers to ELI5 questions, with REFERENCES, by browsing the web through a text interface.
  Environment (Table 1): the model sees a text summary (question, quotes so far, current page around the cursor, past
    actions) and issues one command: Search <query> (Bing), Clicked on link <id>, Find in page: <text>,
    Quote: <text> (adds a reference: title, domain, extract), Scrolled down/up <1,2,3>, Top, Back, End: Answer,
    End: <Nonsense, Controversial>. Then, given the question and its quotes, it writes the answer.
  Data: ~6,000 human demonstrations of browsing, ~21,500 comparisons of two model answers (with references).
  Methods (GPT-3 760M / 13B / 175B):
    1. behaviour cloning (BC) on demonstrations;
    2. reward model (RM): BC model with a scalar head, trained on comparisons with cross-entropy:
       P(A preferred) = sigmoid(r_A - r_B), ties as soft 0.5 labels; r is an Elo score (difference 1 -> 73%);
    3. RL: PPO against the RM, with a per-token KL penalty from the BC model (against over-optimisation);
    4. rejection sampling (best-of-n): sample n = 4 / 16 / 64 answers, return the one the RM ranks highest.
  Results: 175B best-of-64 answers preferred 56% to human demonstrators, 69% to the top-voted Reddit answers;
    best-of-64 preferred 68% to plain BC, RL only 58%; RL + rejection sampling ~ rejection sampling alone.
    TruthfulQA: all WebGPT models beat all GPT-3 models on truthful and truthful+informative.
    Scaling: doubling demonstrations +0.13 RM score; doubling comparisons +1.8% RM accuracy; compute-optimal
    best-of-n grows with the budget (Figure 8). Risks: cherry-picking references, over-reliance on references.
"""

import math
import re
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------------------------------- the browser
class LocalWeb:
    """A tiny 'internet': pages = list of {title, domain, text, authority?}; search ranks pages by TF-IDF of the
    query times an optional authority weight (like a real engine favouring reliable sites)."""

    def __init__(self, pages):
        self.pages = pages
        docs = [self.tokens(p["title"] + " " + p["text"]) for p in pages]
        df = Counter(w for d in docs for w in set(d))
        self.idf = {w: math.log((1 + len(docs)) / (1 + c)) + 1 for w, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    @staticmethod
    def tokens(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query, k=5):
        q = self.tokens(query)
        scores = [p.get("authority", 1.0) * sum(tf[w] * self.idf.get(w, 0) for w in q) / (1 + sum(tf.values())) ** 0.5
                  for p, tf in zip(self.pages, self.tf)]
        order = sorted(range(len(self.pages)), key=lambda i: -scores[i])
        return [i for i in order[:k] if scores[i] > 0]


class Browser:
    """The text environment of Table 1. step(command) -> observation; quotes collect references."""

    WINDOW = 3                                                                    # sentences shown around the cursor

    def __init__(self, web, question, max_actions=20):
        self.web, self.question, self.max_actions = web, question, max_actions
        self.history, self.results, self.page, self.cursor, self.quotes = [], [], None, 0, []
        self.actions, self.done, self.skip = 0, False, False

    def sentences(self):
        return re.split(r"(?<=[.!?])\s+", self.web.pages[self.page]["text"]) if self.page is not None else []

    def view(self):
        if self.page is None and self.results:
            return "\n".join(f"[{i}] {self.web.pages[p]['title']} ({self.web.pages[p]['domain']})" for i, p in enumerate(self.results))
        s = self.sentences()
        return " ".join(s[self.cursor:self.cursor + self.WINDOW])

    def summary(self):
        """What the model is shown each step (it has no other memory): question, quotes, page view, actions left."""
        quotes = "\n".join(f"  [{i + 1}] {q['title']} ({q['domain']}): {q['extract']}" for i, q in enumerate(self.quotes))
        where = self.web.pages[self.page]["title"] if self.page is not None else "search results"
        return (f"Question: {self.question}\nQuotes:\n{quotes or '  (none)'}\nPast actions: {self.actions} of "
                f"{self.max_actions}\nCurrent: {where}\n{self.view()}\nNext command:")

    def step(self, command):
        if self.done:
            return "Episode is over."
        self.actions += 1
        c = command.strip()
        if m := re.fullmatch(r"Search (.+)", c):
            self.history.append((self.page, self.cursor)); self.results, self.page, self.cursor = self.web.search(m[1]), None, 0
        elif m := re.fullmatch(r"Clicked on link (\d+)", c):
            if int(m[1]) < len(self.results):
                self.history.append((self.page, self.cursor)); self.page, self.cursor = self.results[int(m[1])], 0
        elif m := re.fullmatch(r"Find in page: (.+)", c):
            s = self.sentences()
            hits = [i for i in range(self.cursor + 1, len(s)) if m[1].lower() in s[i].lower()] or \
                   [i for i in range(len(s)) if m[1].lower() in s[i].lower()]
            if hits:
                self.cursor = hits[0]
        elif m := re.fullmatch(r"Quote: (.+)", c):
            if self.page is not None and m[1] in self.web.pages[self.page]["text"]:
                p = self.web.pages[self.page]
                self.quotes.append({"title": p["title"], "domain": p["domain"], "extract": m[1]})
        elif m := re.fullmatch(r"Scrolled (down|up) ([123])", c):
            self.cursor = max(0, min(len(self.sentences()) - 1, self.cursor + (1 if m[1] == "down" else -1) * int(m[2]) * self.WINDOW))
        elif c == "Top":
            self.cursor = 0
        elif c == "Back" and self.history:
            self.page, self.cursor = self.history.pop()
        elif c == "End: Answer":
            self.done = True
        elif c in ("End: Nonsense", "End: Controversial"):
            self.done = self.skip = True
        # anything else: an invalid action, which still counts towards the maximum
        if self.actions >= self.max_actions:
            self.done = True
        return self.summary()


def format_answer(text, quotes):
    """The answering phase: the answer cites references as [1], [2], ... and lists them."""
    refs = "\n".join(f"[{i + 1}] {q['title']} ({q['domain']})" for i, q in enumerate(quotes))
    return f"{text}\n\nReferences:\n{refs}" if quotes else text


# ---------------------------------------------------------------------------------------------------- reward model
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def elo_preference(delta):
    """The RM score is an Elo-like scale: P(A preferred over B) = sigmoid(r_A - r_B). A difference of 1 -> 73%."""
    return sigmoid(delta)


def bradley_terry_loss(r_a, r_b, label):
    """Cross-entropy for a comparison; label = 1 (A better), 0 (B better) or 0.5 (tie, a soft label)."""
    p = sigmoid(r_a - r_b)
    return -(label * np.log(p + 1e-12) + (1 - label) * np.log(1 - p + 1e-12))


def train_reward_model(feats_a, feats_b, labels, l2=1e-3, steps=2000, lr=0.5):
    """A LINEAR reward model r = w . features, fitted to comparisons by gradient descent on the Bradley-Terry loss."""
    X = np.asarray(feats_a, float) - np.asarray(feats_b, float)
    y = np.asarray(labels, float)
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = sigmoid(X @ w)
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
    return w


def best_of_n(candidates, score_fn):
    """Rejection sampling: return the candidate with the highest reward-model score."""
    return max(candidates, key=score_fn)


def kl_penalised_return(rm_score, logp_policy, logp_bc, beta):
    """The RL reward used with PPO: the RM score at the end of the episode minus beta * sum over tokens of
    (log pi(token) - log pi_BC(token)), an estimate of KL(pi || pi_BC) along the sampled trajectory."""
    return rm_score - beta * float(np.sum(np.asarray(logp_policy) - np.asarray(logp_bc)))


# ---------------------------------------------------------------------------------------------------- toy world
FILLER = ["Many people find this topic interesting.", "There is a long history behind it.",
          "Experts have discussed this for years.", "It is often mentioned in textbooks.",
          "Some details vary by source.", "The topic attracts curious readers."]


def make_web(n_topics=40, n_values=12, seed=0):
    """Each topic has a reliable encyclopedia page (true fact among filler) and an unreliable blog with a WRONG fact."""
    rng = np.random.default_rng(seed)
    truth, pages = {}, []
    for t in range(n_topics):
        name, value = f"topic{t}", f"value{rng.integers(n_values)}"
        wrong = f"value{(int(value[5:]) + 1 + rng.integers(n_values - 1)) % n_values}"
        truth[name] = value
        fill = list(rng.choice(FILLER, 5, replace=False))
        pages.append({"title": f"{name} - Encyclopedia", "domain": "encyclopedia.org", "authority": 1.5,
                      "text": " ".join(fill[:2] + [f"The key fact of {name} is {value}."] + fill[2:])})
        pages.append({"title": f"My thoughts on {name}", "domain": "someblog.net",
                      "text": " ".join(fill[:1] + [f"I heard the key fact of {name} is {wrong}."] + fill[1:3])})
    return LocalWeb(pages), truth


def policy_episode(web, topic, theta, rng):
    """A stochastic browsing policy (stand-in for the behaviour-cloned model). theta = probabilities:
      precise  : search the topic name (else a vague query)
      top      : click the first result (else a random one)
      find     : use 'Find in page: key fact' before quoting (else quote a random visible sentence)
      fill     : expected number of filler sentences added to the answer (a style choice)
    Returns (answer text, quotes, log-probability of the choices, the browser); the browser also records
    b.choices = [(name, taken)] and b.n_fill, which give exact policy gradients."""
    b = Browser(web, f"What is the key fact of {topic}?")
    b.choices = []
    logp = 0.0

    def choose(p, name=None):
        nonlocal logp
        x = rng.random() < p
        logp += math.log(p if x else 1 - p)
        b.choices.append((name, x))
        return x
    b.step(f"Search key fact {topic}" if choose(theta["precise"], "precise") else "Search key fact")
    if b.results:
        b.step("Clicked on link 0" if choose(theta["top"], "top") else f"Clicked on link {rng.integers(len(b.results))}")
        if choose(theta["find"], "find"):
            b.step("Find in page: key fact")
            sentence = b.view().split(". ")[0].rstrip(".") + "."
        else:
            vis = re.split(r"(?<=[.!?])\s+", b.view())
            sentence = vis[rng.integers(len(vis))]
        b.step(f"Quote: {sentence}")
    b.step("End: Answer")
    facts = [re.search(r"is (value\d+)", q["extract"]) for q in b.quotes]
    facts = [f[1] for f in facts if f]
    value = facts[0] if facts else f"value{rng.integers(12)}"                      # no evidence: a guess
    n_fill = rng.poisson(theta["fill"])
    b.n_fill = n_fill
    logp += n_fill * math.log(theta["fill"] + 1e-9) - theta["fill"] - math.lgamma(n_fill + 1)
    text = f"The key fact of {topic} is {value}" + (" [1]." if facts else ".") + " " + " ".join(rng.choice(FILLER, n_fill))
    return text.strip(), b.quotes, logp, b


def answer_features(text, quotes, topic):
    """What a reward model can see: cites a reference, the reference supports the stated value, the reference comes
    from the encyclopedia domain, number of filler sentences (length)."""
    m = re.search(r"is (value\d+)", text)
    value = m[1] if m else None
    supported = any(value and value in q["extract"] and f"{topic} " in q["extract"] for q in quotes)
    good_domain = any(q["domain"] == "encyclopedia.org" for q in quotes)
    n_fill = sum(text.count(f) for f in FILLER)
    return np.array([1.0, float(bool(quotes)), float(supported), float(good_domain), float(n_fill)])


def true_quality(text, quotes, topic, truth):
    """The (simulated) human labeler's utility: correctness dominates, a supporting reference helps, and a LITTLE
    extra detail is liked but verbosity beyond 2 filler sentences is disliked (a non-linear taste the linear RM
    cannot represent: over-optimising its length weight hurts)."""
    m = re.search(r"is (value\d+)", text)
    correct = bool(m and m[1] == truth[topic])
    supported = any(m and m[1] in q["extract"] and f"{topic} " in q["extract"] for q in quotes)
    n_fill = sum(text.count(f) for f in FILLER)
    return 2.0 * correct + 0.5 * (correct and supported) - 1.0 * (not correct) + 0.4 * min(n_fill, 2) - 0.6 * max(0, n_fill - 2)


def labeler_compare(qa, qb, rng, noise=1.0):
    """Noisy human comparison: A preferred with probability sigmoid((qa - qb) / noise); 10% ties."""
    if rng.random() < 0.1:
        return 0.5
    return float(rng.random() < sigmoid((qa - qb) / noise))


def policy_logp(choices, n_fill, theta):
    """log-probability of an episode's choices under another policy (e.g. the BC policy) -> KL estimates."""
    lp = sum(math.log(theta[n] if x else 1 - theta[n]) for n, x in choices)
    return lp + n_fill * math.log(theta["fill"]) - theta["fill"] - math.lgamma(n_fill + 1)


def rl_finetune(web, truth, theta_bc, reward_fn, beta, steps=300, batch=64, lr=0.05, seed=0):
    """REINFORCE (a stand-in for PPO) on the policy's parameters: logits for the three choices, log-rate for fill.
    Reward per episode = RM score - beta * (log pi - log pi_BC) (the KL-penalised return). Returns the trajectory of
    (mean RM score, mean true quality) and the final theta."""
    rng = np.random.default_rng(seed)
    z = {k: math.log(v / (1 - v)) for k, v in theta_bc.items() if k != "fill"}
    u = math.log(theta_bc["fill"])
    history = []
    for _ in range(steps):
        theta = {k: 1 / (1 + math.exp(-v)) for k, v in z.items()} | {"fill": math.exp(u)}
        grads, rets, rm_scores, quals = [], [], [], []
        for _ in range(batch):
            topic = f"topic{rng.integers(len(truth))}"
            text, quotes, logp, b = policy_episode(web, topic, theta, rng)
            rm = reward_fn(text, quotes, topic)
            ret = kl_penalised_return(rm, [logp], [policy_logp(b.choices, b.n_fill, theta_bc)], beta)
            g = {k: 0.0 for k in z} | {"fill": 0.0}
            for name, x in b.choices:
                g[name] += float(x) - theta[name]                                  # d log Bernoulli / d logit
            g["fill"] += (b.n_fill - theta["fill"]) / max(theta["fill"], 1.0)         # d log Poisson / d log-rate (scaled)
            grads.append(g); rets.append(ret); rm_scores.append(rm); quals.append(true_quality(text, quotes, topic, truth))
        base = float(np.mean(rets))
        for k in z:
            z[k] += lr * float(np.mean([g[k] * (r - base) for g, r in zip(grads, rets)]))
        u = min(math.log(8.0), u + lr * float(np.mean([g["fill"] * (r - base) for g, r in zip(grads, rets)])))
        history.append((float(np.mean(rm_scores)), float(np.mean(quals))))
    theta = {k: 1 / (1 + math.exp(-v)) for k, v in z.items()} | {"fill": math.exp(u)}
    return history, theta


# ---------------------------------------------------------------------------------------------------- reported
RESULTS = {"175B best-of-64 vs human demonstrators": 0.56, "175B best-of-64 vs ELI5 top answers": 0.69,
           "best-of-64 BC vs BC (175B)": 0.68, "RL vs BC (175B)": 0.58, "Krishna et al. 2021 vs ELI5 answers": 0.23,
           "demonstrations": 6000, "comparisons": 21500, "RM training comparisons": 16000}
SCALING = {"doubling demonstrations: RM score": "+0.13", "doubling comparisons: RM accuracy": "+1.8%",
           "doubling policy parameters: RM score": "+0.09", "doubling RM parameters: accuracy": "+0.4%"}
