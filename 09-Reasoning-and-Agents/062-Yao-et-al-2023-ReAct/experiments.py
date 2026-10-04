"""ReAct experiments (Yao et al. 2023): a real HotpotQA run with an open model and the live Wikipedia API, plus toy
studies of the paper's claims.

  E1  Table 1 (real): HotpotQA dev (Hugging Face `datasets`), question only, the search/lookup/finish API on live
      Wikipedia (`requests`), an open instruction model (`transformers`); Standard, CoT, Act, ReAct prompts built by
      ablating the same exemplar trajectory; exact match; max 7 steps.
  E2  Toy: the search tool fails for 30% of questions; when the agent gives no answer, the harness retries once
      with the tool back up. Act vs ReAct accuracy (a first step; training explicit retry thoughts is left open).
  E3  Toy (Figure 3, fine-tuning): number of training trajectories (100 ... 3000) vs accuracy for Standard, CoT, Act,
      ReAct on a changed world.
  E4  Toy back-off rules: ReAct -> CoT and CoT-SC -> ReAct under a mix of stale facts and tool outages.

!! HEAVY. Not run on the author's laptop (E1 needs a GPU, internet and model downloads).
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from react import ToyWorld, ablate, react_then_cot, cot_then_react, react_loop, run_toy, train_toy

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"

EXEMPLAR = [  # the paper's Figure 1 HotpotQA example, written as one trajectory
    ("question", "Aside from the Apple Remote, what other device can control the program Apple Remote was originally "
                 "designed to interact with?"),
    ("thought", "I need to search Apple Remote and find the program it was originally designed to interact with."),
    ("action", "search[Apple Remote]"),
    ("observation", "The Apple Remote is a remote control introduced in October 2005 by Apple ... originally designed "
                    "to control the Front Row media center program ..."),
    ("thought", "Apple Remote was originally designed to control the Front Row media center program. I need to search "
                "Front Row next and find what other device can control it."),
    ("action", "search[Front Row]"),
    ("observation", "Could not find [Front Row]. Similar: ['Front Row Seat to Earth', 'Front Row Motorsports', "
                    "'Front Row (software)', ...]"),
    ("thought", "Front Row is not found. I need to search Front Row (software)."),
    ("action", "search[Front Row (software)]"),
    ("observation", "Front Row is a discontinued media center software ..."),
    ("thought", "Front Row (software) is controlled by an Apple Remote or the keyboard function keys. So the answer is "
                "keyboard function keys."),
    ("action", "finish[keyboard function keys]"),
    ("answer", "keyboard function keys")]


class LiveWiki:
    """search/lookup/finish against the real Wikipedia API (plain-text extracts)."""

    def __init__(self):
        import requests
        self.r, self.page_text, self.hits, self.key, self.answer = requests, None, [], None, None

    def _get(self, title):
        p = {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json", "redirects": 1}
        pages = self.r.get("https://en.wikipedia.org/w/api.php", params=p, timeout=10).json()["query"]["pages"]
        return next(iter(pages.values())).get("extract")

    def step(self, action):
        m = re.fullmatch(r"\s*(search|lookup|finish)\[(.*)\]\s*", action)
        if not m:
            return f"Invalid action: {action}", False
        kind, arg = m.groups()
        if kind == "finish":
            self.answer = arg
            return "done", True
        if kind == "search":
            text = self._get(arg)
            if not text:
                s = self.r.get("https://en.wikipedia.org/w/api.php", params={"action": "opensearch", "search": arg,
                               "limit": 5, "format": "json"}, timeout=10).json()
                return f"Could not find [{arg}]. Similar: {s[1]}.", False
            self.page_text, self.key = text, None
            return " ".join(re.split(r"(?<=[.!?])\s+", text)[:5]), False
        if self.page_text is None:
            return "No page loaded.", False
        if arg != self.key:
            self.key, self.hits = arg, [s for s in re.split(r"(?<=[.!?])\s+", self.page_text) if arg.lower() in s.lower()]
        return (self.hits.pop(0) if self.hits else "No more results."), False


def render(traj, method):
    lines, i = [], 0
    for kind, text in ablate(traj, method) if method != "react" else traj:
        if kind == "question":
            lines.append(f"Question: {text}")
        elif kind == "thought":
            i += 1; lines.append(f"Thought {i}: {text}")
        elif kind == "action":
            if method == "act":                                                  # no thoughts: actions advance the step
                i += 1
            lines.append(f"Action {i}: {text}")
        elif kind == "observation":
            lines.append(f"Observation {i}: {text}")
        elif kind == "answer" and method in ("standard", "cot"):
            lines.append(f"Answer: {text}")
    q = [t for k, t in traj if k == "question"][0]
    return "\n".join([f"Question: {q}"] + [l for l in lines if not l.startswith("Question")]) + "\n\n"


def normalize(s):
    s = re.sub(r"\b(a|an|the)\b", " ", (s or "").lower())
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


def e1(a):
    import datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()

    def generate(prompt, stop):
        ids = tok(prompt, return_tensors="pt").to(DEV)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        out = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        for s in stop:
            out = out.split(s)[0]
        return out

    rows = datasets.load_dataset("hotpot_qa", "distractor", split="validation").select(range(a.n))
    res = {m: [] for m in ("standard", "cot", "act", "react")}
    for r in rows:
        q, truth = r["question"], normalize(r["answer"])
        for m in ("standard", "cot"):
            out = generate(render(EXEMPLAR, m) + f"Question: {q}\n" + ("Answer:" if m == "standard" else "Thought 1:"),
                           stop=["\nQuestion:"])
            ans = out.split("Answer:")[-1].split("answer is")[-1].strip().split("\n")[0]
            res[m].append(normalize(ans) == truth)
        ans, _ = react_loop(generate, q, LiveWiki(), render(EXEMPLAR, "react"), max_steps=7)
        res["react"].append(normalize(ans) == truth)
        prompt = render(EXEMPLAR, "act") + f"Question: {q}\n"
        env, ans = LiveWiki(), None
        for i in range(1, 8):                                                     # Act: actions only
            act = generate(prompt + f"Action {i}:", stop=["\n"]).strip()
            obs, done = env.step(act)
            prompt += f"Action {i}: {act}\nObservation {i}: {obs}\n"
            if done:
                ans = env.answer; break
        res["act"].append(normalize(ans) == truth)
    return {m: float(np.mean(v)) for m, v in res.items()} | {"paper (PaLM-540B EM)": {"standard": 28.7, "cot": 29.4, "act": 25.7, "react": 27.4}}


# ---------------------------------------------------------------------------------------------------- toys
def changed_world_accuracy(world, model, method, rng):
    old = dict(world.director)
    films = world.films[100:]
    for f in films:
        world.director[f] = f"P{rng.randrange(120)}"
    acc = float(np.mean([run_toy(world, model, f, method)[0] == world.answer(f) for f in films]))
    world.director.update(old)
    return acc


def e2(a):
    """Flaky search: the first attempt for a film returns empty pages 30% of the time; if the agent ends without an
    answer, retry once with the tool back up."""
    out = {}
    for method in ("act", "react"):
        world = ToyWorld()
        m, _ = train_toy(world, method, a.steps)
        rng, ok = random.Random(0), 0
        for f in world.films:
            world.outage = rng.random() < 0.3
            ans, seq = run_toy(world, m, f, method)
            world.outage = False
            if ans is None:                                                       # one retry with the tool back
                ans, seq = run_toy(world, m, f, method)
            ok += ans == world.answer(f)
        out[method] = ok / len(world.films)
    out["note"] = "retry is done by the harness here; a fuller version would train retry thoughts into the trajectories"
    return out


def e3(a):
    out = {}
    for n in a.sizes:
        for method in ("standard", "cot", "act", "react"):
            world = ToyWorld()
            world.films = world.films[:n] if n < len(world.films) else world.films
            m, _ = train_toy(world, method, a.steps)
            world.films = ToyWorld().films
            out.setdefault(n, {})[method] = changed_world_accuracy(world, m, method, random.Random(1))
        print("  E3", n, out[n], flush=True)
    return out


def e4(a):
    world = ToyWorld()
    react_m, _ = train_toy(world, "react", a.steps)
    cot_m, _ = train_toy(world, "cot", a.steps)
    rng = random.Random(2)
    res = {"react": [], "cot": [], "react->cot": [], "cot-sc->react": []}
    for f in world.films[:a.n_eval]:
        stale = rng.random() < 0.3
        outage = rng.random() < 0.3
        old = world.director[f]
        if stale:
            world.director[f] = f"P{rng.randrange(120)}"
        truth = world.answer(f)
        world.outage = outage
        r_ans = run_toy(world, react_m, f, "react")[0]
        world.outage = False
        c_ans = run_toy(world, cot_m, f, "cot")[0]
        res["react"].append(r_ans == truth); res["cot"].append(c_ans == truth)
        res["react->cot"].append(react_then_cot(r_ans, [c_ans])[0] == truth)
        res["cot-sc->react"].append(cot_then_react([c_ans], lambda: r_ans)[0] == truth)
        world.director[f] = old
    return {k: float(np.mean(v)) for k, v in res.items()}


def report(R, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 5)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.model, a.n, a.max_new, a.steps, a.sizes, a.n_eval = "Qwen/Qwen2.5-7B-Instruct", 300, 128, 800, (25, 50, 100, 200), 200
    if a.quick:
        a.model, a.n, a.max_new, a.steps, a.sizes, a.n_eval = "sshleifer/tiny-gpt2", 1, 8, 3, (25,), 5
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
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
