"""Chain-of-thought experiments (Wei et al. 2022): the real prompting setting with open models, plus controlled toys.

  E1  The paper's main experiment on real models: GSM8K test (Hugging Face `datasets`), the 8 exemplars of Table 20,
      standard vs chain-of-thought prompting, greedy decoding, answer = number after the last 'The answer is', for a
      ladder of open base models of increasing size (Hugging Face `transformers`; e.g. Qwen2.5 0.5B/1.5B/3B/7B).
      Does the CoT gain grow with scale (Figure 4)?
  E2  Ablations on the same models (Figure 5): equation only, variable compute only (dots as long as the equation),
      chain of thought after the answer, built from the same 8 exemplars.
  E3  Robustness (Figure 6): chains written by annotator B / C (the paper's Tables 29-30 style) and shuffled
      exemplar orders, 3 seeds.
  E4  Toy emergence in the PROMPTING setting: models of several sizes pre-trained on documents that each contain 5
      problems in ONE format; at test time two exemplars choose the format (standard vs chain of thought). Is there
      a size below which chain-of-thought exemplars do not help?
  E5  Toy difficulty sweep: gain of chain of thought vs number of steps k (one-step problems should gain nothing).

!! HEAVY. Not run on the author's laptop (E1-E3 need a GPU and model downloads).
       python3 experiments.py --quick
       python3 experiments.py --only e4
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

from cot import (FORMATS, GSM8K_EXEMPLARS, STOI, VOCAB, _llama, cot_prompt, evaluate_toy, extract_answer,
                 standard_prompt, toy_answer, toy_item, train_toy)

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------- real models
def gsm8k(n):
    import datasets
    ds = datasets.load_dataset("gsm8k", "main", split="test")
    rows = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds]
    return rows[:n]


def generate(model, tok, prompts, max_new):
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(text.split("\nQ:")[0])                                      # stop at the next question
    return outs


def ablation_exemplars(kind):
    out = []
    for e in GSM8K_EXEMPLARS:
        eqs = re.findall(r"\d+ [-+*x] \d+ (?:=|is) \d+", e["chain"])
        eq = "; ".join(eqs) if eqs else e["answer"]
        if kind == "equation only":
            chain = eq + "."
        elif kind == "variable compute only":
            chain = "." * len(eq)
        elif kind == "chain after answer":
            chain = None
        out.append({**e, "chain": chain} if chain is not None else e)
    return out


def after_answer_prompt(exemplars, q):
    shots = "".join(f"Q: {e['question']}\nA: The answer is {e['answer']}. {e['chain']}\n\n" for e in exemplars)
    return shots + f"Q: {q}\nA:"


def run_model(name, rows, a, variants):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
    res = {}
    for vname, make in variants.items():
        prompts = [make(q) for q, _ in rows]
        outs = generate(model, tok, prompts, a.max_new)
        if vname == "chain after answer":
            preds = [(re.findall(r"answer is\s*\$?(-?[\d,]*\.?\d+)", o) or [None])[0] for o in outs]
            preds = [p.replace(",", "") if p else None for p in preds]
        else:
            preds = [extract_answer(o) for o in outs]
        res[vname] = float(np.mean([p == t for p, (_, t) in zip(preds, rows)]))
        print(f"  {name} {vname}: {res[vname]:.3f}", flush=True)
    del model
    return res


def e1(a):
    rows = gsm8k(a.n_gsm)
    variants = {"standard": lambda q: standard_prompt(GSM8K_EXEMPLARS, q),
                "chain of thought": lambda q: cot_prompt(GSM8K_EXEMPLARS, q)}
    return {m: run_model(m, rows, a, variants) for m in a.models}


def e2(a):
    rows = gsm8k(a.n_gsm)
    variants = {"equation only": lambda q: cot_prompt(ablation_exemplars("equation only"), q),
                "variable compute only": lambda q: cot_prompt(ablation_exemplars("variable compute only"), q),
                "chain after answer": lambda q: after_answer_prompt(GSM8K_EXEMPLARS, q)}
    return {m: run_model(m, rows, a, variants) for m in a.models[-2:]}


def e3(a):
    rows = gsm8k(a.n_gsm)
    terse = [{**e, "chain": re.sub(r"[A-Z][^.]*?(?=\d+ [-+*x] \d+)", "", e["chain"]).strip()} for e in GSM8K_EXEMPLARS]
    variants = {}
    for seed in range(3):
        order = list(range(8)); random.Random(seed).shuffle(order)
        variants[f"annotator A, order {seed}"] = (lambda o: lambda q: cot_prompt([GSM8K_EXEMPLARS[i] for i in o], q))(order)
    variants["terse chains (other style)"] = lambda q: cot_prompt(terse, q)
    return {m: run_model(m, rows, a, variants) for m in a.models[-1:]}


# ---------------------------------------------------------------------------------------------------- toys
def mixed_doc(rng, n=4):
    fmt = rng.choice(["standard", "chain of thought"])
    return "".join(q + a for q, a, _, _ in (toy_item(rng, fmt, rng.randint(2, 6)) for _ in range(n)))


def e4(a):
    import torch.nn.functional as F
    L = _llama()
    out = {}
    for d, layers in a.sizes:
        rng = random.Random(0); torch.manual_seed(0)
        m = L.LLaMA(vocab=len(VOCAB), d=d, layers=layers, heads=max(1, d // 16), ctx=a.ctx, hidden=L.ffn_hidden(d, 16)).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), 3e-3)
        for _ in range(a.steps4):
            batch = []
            for _ in range(a.batch):
                doc = mixed_doc(rng)
                while len(doc) < a.ctx:
                    doc += mixed_doc(rng)
                batch.append([STOI[c] for c in doc[:a.ctx]])
            loss = m.loss(torch.tensor(batch, device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        row = {}
        for fmt in ("standard", "chain of thought"):
            trng, ok = random.Random(5), 0
            for _ in range(a.n_eval):
                shots = "".join(q + ans for q, ans, _, _ in (toy_item(trng, fmt, trng.randint(2, 6)) for _ in range(2)))
                q, _, truth, _ = toy_item(trng, fmt, 5)
                p = [STOI[c] for c in shots + q]
                with torch.no_grad():
                    for _ in range(14):
                        n = int(m(torch.tensor([p[-a.ctx:]], device=DEV))[0, -1].argmax()); p.append(n)
                        if VOCAB[n] == ";":
                            break
                completion = "".join(VOCAB[i] for i in p)[len(shots + q):]
                ok += toy_answer(fmt, completion) == str(truth)
            row[fmt] = ok / a.n_eval
        out[f"d={d}, layers={layers}"] = row
        print("  E4", d, layers, row, flush=True)
    return out


def e5(a):
    out = {}
    for fmt in ("standard", "chain of thought"):
        m, _ = train_toy(fmt, a.steps5, k_range=(1, 6))
        out[fmt] = {k: evaluate_toy(m, fmt, k, n=a.n_eval)[0] for k in range(1, 7)}
        print("  E5", fmt, out[fmt], flush=True)
    out["gain"] = {k: out["chain of thought"][k] - out["standard"][k] for k in range(1, 7)}
    return out


def report(R, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
    a.n_gsm, a.max_new = 500, 256
    a.sizes, a.ctx, a.batch, a.steps4, a.n_eval, a.steps5 = [(16, 1), (32, 2), (64, 2), (128, 4)], 96, 64, 4000, 300, 1500
    if a.quick:
        a.models, a.n_gsm, a.max_new = ["sshleifer/tiny-gpt2"], 2, 8
        a.sizes, a.ctx, a.batch, a.steps4, a.n_eval, a.steps5 = [(16, 1)], 64, 4, 3, 3, 3
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
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
