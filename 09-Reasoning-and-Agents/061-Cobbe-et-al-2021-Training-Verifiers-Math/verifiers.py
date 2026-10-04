"""Training Verifiers to Solve Math Word Problems (Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek,
Hilton, Nakano, Hesse & Schulman, OpenAI 2021).

  GSM8K: 8.5K grade-school maths word problems (7.5K train / 1K test), 2-8 steps, natural-language solutions with
    calculator annotations <<expr=result>> and the final answer after '####'.
  Baseline (finetuning): finetune GPT-3 (6B / 175B) on the solutions; at test time ONE low-temperature (T = 0) sample.
    Solving directly (no steps) drops 6B from 20.6% to 5.2%. A calculator overrides tokens after '=' inside <<...>>.
  Verification (Figure 4):
    1. finetune a GENERATOR for only 2 epochs (test@100 coverage peaks early, test@1 keeps rising);
    2. sample 100 solutions per TRAINING problem (T = 0.7) and label each correct/incorrect by its final answer only;
    3. train a VERIFIER for 1 epoch: a token-level value head predicting P(correct) after every token, plus the
       language-modelling objective (joint);
    4. at test time sample 100 solutions, return the one with the highest verifier score (or let the top 3-5 vote).
  Findings: 6B + verifier ~ finetuned 175B (a ~30x model-size boost); verifiers need enough data (worse than the
    baseline on small training sets) and then scale better with data; token-level > solution-level; joint LM
    objective helps; a bigger generator matters more than a bigger verifier; searching over too many samples
    (> ~400 for 6B) finds adversarial solutions that fool the verifier; 20% residual dropout helps both methods.
"""

import random
import re
from collections import Counter

# ---------------------------------------------------------------------------------------------------- GSM8K format
CALC = re.compile(r"<<([^=<>]+)=([^<>]*)>>")


def final_answer(solution):
    """GSM8K solutions end with '#### 72'."""
    m = re.search(r"####\s*(-?[\d,\.]+)", solution)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def calculator_fix(text):
    """Replace every <<expr=anything>> with the true value of expr (what the test-time calculator enforces)."""
    def ev(m):
        expr = m.group(1)
        if not re.fullmatch(r"[\d\.\s\+\-\*/\(\)]+", expr):
            return m.group(0)
        v = eval(expr)
        v = int(v) if float(v).is_integer() else round(v, 6)
        return f"<<{expr}={v}>>"
    return CALC.sub(ev, text)


def calculator_step(partial):
    """During sampling: if the text ends with '<<expr=', return the calculator result to insert (else None)."""
    m = re.search(r"<<([\d\.\s\+\-\*/\(\)]+)=$", partial)
    if not m:
        return None
    v = eval(m.group(1))
    return str(int(v) if float(v).is_integer() else round(v, 6))


def strip_annotations(text):
    return CALC.sub("", text)


# ---------------------------------------------------------------------------------------------------- selection
def coverage_at_n(correct_lists, n):
    """test@N: the share of problems where at least one of the first N samples is correct (coverage)."""
    return sum(any(c[:n]) for c in correct_lists) / len(correct_lists)


def select_by_verifier(answers, scores, top_k=1):
    """Rank samples by verifier score; top_k = 1 returns the best one's answer, top_k > 1 lets the top_k vote."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    if top_k == 1:
        return answers[order[0]]
    votes = Counter(answers[i] for i in order[:top_k] if answers[i] is not None)
    return votes.most_common(1)[0][0] if votes else None


def majority_vote(answers):
    votes = Counter(a for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None


# ---------------------------------------------------------------------------------------------------- toy task
VOCAB = sorted(set("Q0123456789=;+t"))
STOI = {c: i for i, c in enumerate(VOCAB)}
N_DIGITS, CTX = 6, 20


def toy_problem(rng):
    """'Find two of these digits that add up to t': generation is a SEARCH over pairs, checking a proposed pair is a
    local sum plus a membership test, the asymmetry that makes verifiers useful. At most 2 valid pairs."""
    while True:
        ds = [rng.randint(0, 9) for _ in range(N_DIGITS)]
        i, j = rng.sample(range(N_DIGITS), 2)
        t = ds[i] + ds[j]
        pairs = [(a, b) for a in range(N_DIGITS) for b in range(N_DIGITS) if a != b and ds[a] + ds[b] == t]
        if len(pairs) <= 2:
            return {"q": "Q" + "".join(map(str, ds)) + f"t{t}=", "gold": f"{ds[i]}+{ds[j]};", "digits": ds, "target": t}


def toy_correct(p, solution):
    try:
        x, y = (int(v) for v in solution.rstrip(";").split("+"))
    except ValueError:
        return False
    left = Counter(p["digits"]); left[x] -= 1; left[y] -= 1
    return x + y == p["target"] and left[x] >= 0 and left[y] >= 0


def _llama():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "08-Pretraining-and-Scaling-LLMs" / "056-Touvron-et-al-2023-LLaMA" / "llama.py"
    spec = importlib.util.spec_from_file_location("llama_056", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _encode(q, a):
    t = [STOI[c] for c in q + a][:CTX]
    return t + [STOI[";"]] * (CTX - len(t)), len(q), len(t)


def make_lm(d=64, layers=2):
    L = _llama()
    return L.LLaMA(vocab=len(VOCAB), d=d, layers=layers, heads=max(1, d // 16), ctx=CTX, hidden=L.ffn_hidden(d, 16))


def train_generator(steps, seed=0, lr=3e-3, batch=64):
    """Supervised finetuning on gold solutions (loss on the solution only). Deliberately short, like the paper's
    2-epoch generator: good coverage, mediocre single-sample accuracy."""
    import torch
    import torch.nn.functional as F
    rng = random.Random(seed); torch.manual_seed(seed)
    m = make_lm()
    opt = torch.optim.AdamW(m.parameters(), lr)
    for _ in range(steps):
        X, M = [], []
        for _ in range(batch):
            p = toy_problem(rng)
            x, lq, lt = _encode(p["q"], p["gold"])
            X.append(x); M.append([0] * lq + [1] * (lt - lq) + [0] * (CTX - lt))
        X = torch.tensor(X); M = torch.tensor(M)[:, 1:].float()
        nll = F.cross_entropy(m(X)[:, :-1].reshape(-1, len(VOCAB)), X[:, 1:].reshape(-1), reduction="none").view(M.shape)
        loss = (nll * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def sample_solutions(gen, questions, n, temperature=1.0, max_new=6, seed=0):
    """n samples per question, batched over questions of equal length. temperature 0 = greedy."""
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    out, groups = [None] * len(questions), {}
    for i, q in enumerate(questions):
        groups.setdefault(len(q), []).append(i)
    with torch.no_grad():
        for ln, idx in groups.items():
            x = torch.tensor([[STOI[c] for c in questions[i]] for i in idx for _ in range(n)])
            done = torch.zeros(len(x), dtype=torch.bool)
            for _ in range(max_new):
                lg = gen(x)[:, -1]
                nx = lg.argmax(-1) if temperature == 0 else torch.multinomial(F.softmax(lg / temperature, -1), 1, generator=g).squeeze(1)
                nx = torch.where(done, torch.full_like(nx, STOI[";"]), nx)
                x = torch.cat([x, nx[:, None]], 1); done |= nx == STOI[";"]
                if done.all():
                    break
            rows = ["".join(VOCAB[t] for t in r[ln:].tolist()).split(";")[0] + ";" for r in x]
            for j, i in enumerate(idx):
                out[i] = rows[j * n:(j + 1) * n]
    return out


class Verifier:
    """Initialised from the generator (the paper initialises both from the same pretrained GPT-3); a scalar head on
    every position predicts P(solution correct) (token-level value function); trained jointly with LM loss."""

    def __init__(self, generator, token_level=True, joint=True):
        import torch
        import torch.nn as nn
        self.lm = make_lm(); self.lm.load_state_dict(generator.state_dict())
        self.head = nn.Linear(self.lm.tok.embedding_dim, 1)
        self.token_level, self.joint = token_level, joint
        self.params = list(self.lm.parameters()) + list(self.head.parameters())
        self.torch = torch

    def forward(self, X):
        h = self.lm.tok(X)
        for b in self.lm.blocks:
            h = b(h, self.lm.cos, self.lm.sin)
        h = self.lm.norm(h)
        return self.lm.out(h), self.head(h).squeeze(-1)

    def train(self, data, steps, lr=2e-3, batch=64, seed=0):
        """data: (question, solution, correct?) triples."""
        import torch.nn.functional as F
        torch = self.torch
        opt = torch.optim.AdamW(self.params, lr)
        for s in range(steps):
            b = random.Random(seed + s).sample(data, min(batch, len(data)))
            X, M, Y, last = [], [], [], []
            for q, a, ok in b:
                x, lq, lt = _encode(q, a)
                X.append(x); M.append([0] * lq + [1] * (lt - lq) + [0] * (CTX - lt)); Y.append(float(ok)); last.append(lt - 1)
            X, M, Y = torch.tensor(X), torch.tensor(M).float(), torch.tensor(Y)
            lg, v = self.forward(X)
            if self.token_level:                                                   # P(correct) after EVERY token
                vl = (F.binary_cross_entropy_with_logits(v, Y[:, None].expand_as(v), reduction="none") * M).sum() / M.sum()
            else:                                                                  # only after the final token
                vl = F.binary_cross_entropy_with_logits(v[torch.arange(len(X)), torch.tensor(last)], Y)
            loss = vl
            if self.joint:
                Mn = M[:, 1:]
                lm = F.cross_entropy(lg[:, :-1].reshape(-1, len(VOCAB)), X[:, 1:].reshape(-1), reduction="none").view(Mn.shape)
                loss = loss + (lm * Mn).sum() / Mn.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def score(self, q, solutions):
        torch = self.torch
        X, last = [], []
        for a in solutions:
            x, _, lt = _encode(q, a); X.append(x); last.append(lt - 1)
        with torch.no_grad():
            _, v = self.forward(torch.tensor(X))
        return v[torch.arange(len(X)), torch.tensor(last)].tolist()


# ---------------------------------------------------------------------------------------------------- reported
FINDINGS = {"6B no intermediate steps (direct answer)": (20.6, 5.2),
            "verification ~ model-size boost": "about 30x (6B verifier ~ finetuned 175B)",
            "best number of samples to rank (6B)": "~400; more finds adversarial solutions",
            "top-k voting with 100 samples": "top 3-5", "dropout": "20% residual dropout helps both"}
