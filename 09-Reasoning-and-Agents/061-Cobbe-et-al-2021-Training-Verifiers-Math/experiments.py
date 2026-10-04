"""Verifier experiments (Cobbe et al. 2021): toy ablations that mirror the paper's figures, and the real GSM8K
pipeline with a small open model.

  E1  Figure 5 (toy): finetuning baseline (greedy) vs verification (best-of-N) for training sets of 50 ... 4000
      problems; 3 seeds.
  E2  Figure 6a/6b (toy): token-level vs solution-level verifier; joint (value + LM) vs verification-only.
  E3  Figure 6c (toy): generator strength vs verifier strength (generator trained 300 / 600 steps; verifier from a
      weak or strong initialisation).
  E4  Figure 7 (toy): number of samples ranked (1 ... 64) and top-k voting (k = 1 ... 16).
  E5  Real GSM8K (Hugging Face `transformers` + `datasets`): finetune a small causal LM on GSM8K train for 2 epochs
      (generator, calculator-annotated text), sample N solutions per training problem (T = 0.7), train a verifier
      (the same LM + scalar head, token-level, joint) for 1 epoch, compare test@1 (T = 0), majority vote and verifier
      best-of-N on the test set.

!! HEAVY. Not run on the author's laptop (E5 needs a GPU).
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from verifiers import (Verifier, calculator_step, final_answer, majority_vote, sample_solutions, select_by_verifier,
                       toy_correct, toy_problem, train_generator)

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------- toy helpers
def toy_setup(a, gen_steps=None, seed=0):
    gen = train_generator(gen_steps or a.gen_steps, seed=seed)
    rng = random.Random(100 + seed)
    train = [toy_problem(rng) for _ in range(max(a.sizes))]
    test = [toy_problem(random.Random(50_000 + i)) for i in range(a.n_test)]
    tr_s = sample_solutions(gen, [p["q"] for p in train], a.n_samples, seed=seed + 1)
    te_s = sample_solutions(gen, [p["q"] for p in test], max(a.ns), seed=seed + 2)
    greedy = np.mean([toy_correct(p, g[0]) for p, g in zip(test, sample_solutions(gen, [p["q"] for p in test], 1, 0))])
    return gen, train, test, tr_s, te_s, greedy


def labelled(train, tr_s, n):
    return [(p["q"], s, toy_correct(p, s)) for p, ss in zip(train[:n], tr_s[:n]) for s in ss]


def best_of(ver, test, te_s, n=None, top_k=1):
    hits = []
    for p, ss in zip(test, te_s):
        ss = ss[:n] if n else ss
        sc = ver.score(p["q"], ss)
        hits.append(toy_correct(p, select_by_verifier(ss, sc, top_k=top_k) or ""))
    return float(np.mean(hits))


def e1(a):
    out = {}
    for seed in range(a.seeds):
        gen, train, test, tr_s, te_s, greedy = toy_setup(a, seed=seed)
        for n in a.sizes:
            v = Verifier(gen).train(labelled(train, tr_s, n), a.ver_steps, seed=seed)
            out.setdefault(n, {"greedy": [], "verifier best-of-N": []})
            out[n]["greedy"].append(greedy); out[n]["verifier best-of-N"].append(best_of(v, test, te_s, a.n_samples))
            print("  E1", seed, n, out[n]["verifier best-of-N"][-1], greedy, flush=True)
    return {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in out.items()}


def e2(a):
    gen, train, test, tr_s, te_s, greedy = toy_setup(a)
    data = labelled(train, tr_s, max(a.sizes))
    out = {"greedy": greedy}
    for name, kw in (("token-level, joint", {}), ("solution-level, joint", {"token_level": False}),
                     ("token-level, verification-only", {"joint": False}),
                     ("solution-level, verification-only", {"token_level": False, "joint": False})):
        out[name] = best_of(Verifier(gen, **kw).train(data, a.ver_steps), test, te_s, a.n_samples)
        print("  E2", name, out[name], flush=True)
    return out


def e3(a):
    out = {}
    for g_steps in a.gen_ladder:
        gen, train, test, tr_s, te_s, greedy = toy_setup(a, gen_steps=g_steps)
        data = labelled(train, tr_s, max(a.sizes))
        for v_init in a.gen_ladder:
            init = gen if v_init == g_steps else train_generator(v_init)
            ver = Verifier(init).train(data, a.ver_steps)
            out[f"generator {g_steps} steps, verifier init {v_init} steps"] = {"greedy": greedy, "verifier": best_of(ver, test, te_s, a.n_samples)}
            print("  E3", g_steps, v_init, out[f"generator {g_steps} steps, verifier init {v_init} steps"], flush=True)
    return out


def e4(a):
    gen, train, test, tr_s, te_s, greedy = toy_setup(a)
    ver = Verifier(gen).train(labelled(train, tr_s, max(a.sizes)), a.ver_steps)
    return {"greedy": greedy,
            "best-of-N": {n: best_of(ver, test, te_s, n) for n in a.ns},
            f"top-k vote of {max(a.ns)}": {k: best_of(ver, test, te_s, None, top_k=k) for k in a.topk},
            "majority vote (no verifier)": float(np.mean([toy_correct(p, majority_vote(s) or "") for p, s in zip(test, te_s)]))}


# ---------------------------------------------------------------------------------------------------- real GSM8K
def e5(a):
    import datasets
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tr = datasets.load_dataset("gsm8k", "main", split="train").select(range(a.gsm_train))
    te = datasets.load_dataset("gsm8k", "main", split="test").select(range(a.gsm_test))
    tok = AutoTokenizer.from_pretrained(a.model); tok.pad_token = tok.pad_token or tok.eos_token
    fmt = lambda q: f"Question: {q}\nAnswer:"

    def finetune(model, texts, prompts, epochs, extra_head=None, labels=None):
        params = list(model.parameters()) + (list(extra_head.parameters()) if extra_head else [])
        opt = torch.optim.AdamW(params, a.lr)
        for _ in range(epochs):
            idx = list(range(len(texts))); random.shuffle(idx)
            for s in range(0, len(idx), a.bs):
                b = idx[s:s + a.bs]
                enc = tok([texts[i] for i in b], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEV)
                plen = [len(tok(prompts[i])["input_ids"]) for i in b]
                mask = enc["attention_mask"].clone()
                for j, pl in enumerate(plen):
                    mask[j, :pl] = 0
                out = model(**enc, output_hidden_states=extra_head is not None)
                lm = F.cross_entropy(out.logits[:, :-1].transpose(1, 2), enc["input_ids"][:, 1:], reduction="none")
                loss = (lm * mask[:, 1:]).sum() / mask[:, 1:].sum()
                if extra_head is not None:                                         # token-level value, joint with LM
                    v = extra_head(out.hidden_states[-1]).squeeze(-1)
                    y = torch.tensor([labels[i] for i in b], device=DEV, dtype=v.dtype)[:, None].expand_as(v)
                    loss = loss + (F.binary_cross_entropy_with_logits(v, y, reduction="none") * mask).sum() / mask.sum()
                opt.zero_grad(); loss.backward(); opt.step()

    def sample(model, q, n, T):
        enc = tok([fmt(q)] * n, return_tensors="pt").to(DEV)
        kw = dict(do_sample=True, temperature=T) if T > 0 else dict(do_sample=False)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id, **kw)
        return [tok.decode(r[enc["input_ids"].shape[1]:], skip_special_tokens=True).split("Question:")[0] for r in g]

    gen = AutoModelForCausalLM.from_pretrained(a.model).to(DEV)
    texts = [fmt(r["question"]) + " " + r["answer"] + tok.eos_token for r in tr]
    finetune(gen, texts, [fmt(r["question"]) for r in tr], epochs=2)
    gen.eval()
    vtexts, vprompts, vlabels = [], [], []
    for r in tr:
        for s in sample(gen, r["question"], a.n_gsm_samples, 0.7):
            vtexts.append(fmt(r["question"]) + s); vprompts.append(fmt(r["question"]))
            vlabels.append(float(final_answer(s) == final_answer(r["answer"])))
    ver = AutoModelForCausalLM.from_pretrained(a.model).to(DEV)
    head = nn.Linear(ver.config.hidden_size, 1).to(DEV)
    finetune(ver, vtexts, vprompts, epochs=1, extra_head=head, labels=vlabels)
    ver.eval()
    res = {"test@1 (T=0)": [], "majority vote": [], "verifier best-of-N": [], "coverage": []}
    for r in te:
        truth = final_answer(r["answer"])
        res["test@1 (T=0)"].append(final_answer(sample(gen, r["question"], 1, 0)[0]) == truth)
        sols = sample(gen, r["question"], a.n_gsm_samples, 0.7)
        answers = [final_answer(s) for s in sols]
        res["majority vote"].append(majority_vote(answers) == truth)
        res["coverage"].append(truth in answers)
        enc = tok([fmt(r["question"]) + s for s in sols], return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            h = ver(**enc, output_hidden_states=True).hidden_states[-1]
        last = enc["attention_mask"].sum(1) - 1
        scores = head(h[torch.arange(len(sols)), last]).squeeze(-1).tolist()
        res["verifier best-of-N"].append(select_by_verifier(answers, scores) == truth)
    return {k: float(np.mean(v)) for k, v in res.items()} | {"note": "calculator_step can be hooked into sampling"}


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
    a.gen_steps, a.sizes, a.n_test, a.n_samples, a.ns, a.topk = 400, (50, 100, 250, 500, 1000, 2000, 4000), 1000, 16, (1, 2, 4, 8, 16, 32, 64), (1, 3, 5, 8, 16)
    a.ver_steps, a.seeds, a.gen_ladder = 2000, 3, (300, 600)
    a.model, a.gsm_train, a.gsm_test, a.n_gsm_samples, a.lr, a.bs, a.max_new = "Qwen/Qwen2.5-0.5B", 7473, 1319, 20, 1e-5, 8, 256
    if a.quick:
        a.gen_steps, a.sizes, a.n_test, a.n_samples, a.ns, a.topk = 3, (4,), 3, 2, (1, 2), (1, 2)
        a.ver_steps, a.seeds, a.gen_ladder = 2, 1, (3,)
        a.model, a.gsm_train, a.gsm_test, a.n_gsm_samples, a.bs, a.max_new = "sshleifer/tiny-gpt2", 2, 2, 2, 2, 8
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
