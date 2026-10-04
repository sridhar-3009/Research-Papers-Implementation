"""Generative-agents experiments (Park et al. 2023): the architecture with a real LLM, plus toy sweeps.

  E1  LLM-backed memory: an open instruction model (Hugging Face `transformers`) rates importance with the paper's
      prompt, writes the 3 salient questions and the 5 insights 'insight (because of 1, 5, 3)', and generates
      dialogue from retrieved memories, for a 5-agent town over one game day; report how the party news spreads and
      sample reflections.
  E2  Interview ablations (Section 6): for each agent, answer the paper-style interview questions with the full
      memory, with observations only, and with no memory; an LLM judge ranks the answers; Bradley-Terry scores per
      condition (the paper used 100 humans and TrueSkill).
  E3  Toy sweeps: recency decay (0.9 ... 0.999), conversation retrieval size k (5 ... 60), reflection threshold
      (50 ... 400): news spread and party attendance, 10 seeds each.

!! HEAVY. Not run on the author's laptop (E1-E2 need a GPU and a model download).
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

import agents as A

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMPORTANCE_PROMPT = ("On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is "
                     "extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the "
                     "following piece of memory.\nMemory: {m}\nRating:")
INTERVIEW = ["Give an introduction of yourself.", "Who is agent5?", "What are you planning to do today?",
             "Do you know about any upcoming party?", "What do you care about most these days?"]


class LLM:
    def __init__(self, name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()

    def __call__(self, prompt, max_new=80):
        ids = self.tok(prompt, return_tensors="pt").to(DEV)
        with torch.no_grad():
            g = self.m.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def install_llm(llm):
    """Replace the stand-ins in agents.py with LLM calls (importance rating, salient questions, insights)."""
    def rate(text):
        m = re.search(r"\d+", llm(IMPORTANCE_PROMPT.format(m=text), 4))
        return int(min(10, max(1, int(m.group())))) if m else 3
    A.rate_importance = rate

    def questions(records, n=3, exclude=()):
        statements = "\n".join(m.text for m in records[-100:])
        out = llm(f"{statements}\nGiven only the information above, what are {n} most salient high-level questions we "
                  f"can answer about the subjects in the statements?\n1.", 120)
        return [q.strip(" .") + "?" for q in re.split(r"\n\d\.|\?", "1." + out) if len(q.strip()) > 8][:n]
    A.salient_questions = questions


def e1(a):
    llm = LLM(a.model)
    install_llm(llm)
    r = A.run_town(seed=0, hours=a.hours, n_agents=a.n_agents)
    town = r.pop("town")
    refl = [m.text for ag in town.agents for m in ag.memory.records if m.kind == "reflection"][:10]
    return r | {"sample reflections": refl}


def e2(a):
    llm = LLM(a.model)
    town = A.run_town(seed=0, hours=a.hours, n_agents=a.n_agents)["town"]
    conds = {"full memory": lambda ag, q: ag.memory.retrieve(q, town.t, k=10),
             "observations only": lambda ag, q: [m for m in ag.memory.retrieve(q, town.t, k=20) if m.kind == "observation"][:10],
             "no memory": lambda ag, q: []}
    wins = {c: 0.0 for c in conds}; games = {c: 0 for c in conds}
    for ag in town.agents:
        for q in INTERVIEW:
            answers = {c: llm("You are " + ag.name + ". Relevant memories:\n" + "\n".join(m.text for m in f(ag, q)) +
                              f"\nInterviewer: {q}\n{ag.name}:", 60) for c, f in conds.items()}
            names = list(answers)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    v = llm(f"Which answer is more believable for {ag.name}, given that agent's life?\nQuestion: {q}\n"
                            f"A: {answers[names[i]]}\nB: {answers[names[j]]}\nAnswer A or B:", 2)
                    w = names[i] if "A" in v[:2] else names[j]
                    wins[w] += 1; games[names[i]] += 1; games[names[j]] += 1
    return {c: wins[c] / max(games[c], 1) for c in conds} | {"paper TrueSkill mu": A.TRUESKILL}


def e3(a):
    out = {}
    for name, values, apply in (("decay", a.decays, lambda v: setattr(A, "DECAY", v)),
                                ("chat_k", a.ks, None), ("reflect threshold", a.thresholds, lambda v: setattr(A, "REFLECT_THRESHOLD", v))):
        row = {}
        for v in values:
            old = (A.DECAY, A.REFLECT_THRESHOLD)
            if apply:
                apply(v)
            rs = [A.run_town(seed=s, **({"chat_k": v} if name == "chat_k" else {})) for s in range(a.seeds)]
            A.DECAY, A.REFLECT_THRESHOLD = old
            row[v] = {k: float(np.mean([r[k] for r in rs])) for k in ("mayor", "party", "attended", "reflections")}
        out[name] = row
        print("  E3", name, row, flush=True)
    return out


def report(R, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 4)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.model, a.hours, a.n_agents = "Qwen/Qwen2.5-7B-Instruct", 24, 6
    a.decays, a.ks, a.thresholds, a.seeds = (0.9, 0.99, 0.995, 0.999), (5, 10, 20, 30, 60), (50, 150, 400), 10
    if a.quick:
        a.model, a.hours, a.n_agents = "sshleifer/tiny-gpt2", 2, 6
        a.decays, a.ks, a.thresholds, a.seeds = (0.995,), (30,), (150,), 1
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3)):
            if a.only in (None, name):
                try:
                    R[name] = fn(a)
                except ImportError as e:
                    R[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(R, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
