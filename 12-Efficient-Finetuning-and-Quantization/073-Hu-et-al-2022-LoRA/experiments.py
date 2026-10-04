"""LoRA experiments (Hu et al. 2022): toy sweeps of the paper's tables, and LoRA on a real pre-trained model.

  E1  Table 6: rank r in {1, 2, 4, 8, 16, 64} x weight types {Wq}, {Wq, Wv}, {Wq, Wk, Wv, Wo}; 3 seeds; plus full
      fine-tuning; learning rate tuned per method over {1e-3, 3e-3, 1e-2, 3e-2}.
  E2  Table 5: fixed parameter budgets (1k, 2k, 4k) spread over different weight types; 3 seeds.
  E3  Figure 3/4: subspace similarity between r = 8 and r = 64 adapters trained from DIFFERENT seeds, and between two
      r = 64 runs with different seeds (the paper's check that the top directions are not noise).
  E4  Table 7 for every layer and Wq / Wv: amplification factor vs r.
  E5  Small-data regime: 32 ... 2048 downstream examples, full fine-tuning vs LoRA r = 8 (over-fitting).
  E6  Alpha / learning-rate coupling: with the alpha / r scaling, does the best learning rate stay put as r changes?
  E7  Real LoRA: GPT-2 (124M, Hugging Face `transformers`) with LoRA r = 8 on c_attn, fine-tuned on SST-2 sentences as
      a language-modelling task ("review: ... sentiment: positive"), vs full fine-tuning: accuracy, trainable params,
      checkpoint size, peak memory, step time.

!! HEAVY for E7 (GPU recommended, downloads GPT-2 + SST-2); E1-E6 take minutes to an hour on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import lora as L

HERE = Path(__file__).parent


def base_model(a, seed=0):
    return L.pretrain(L.new_model(seed), np.random.default_rng(seed), steps=a.pre_steps)


def best_over_lr(make, a, seed):
    best = 0.0
    for lr in a.lrs:
        m = L.finetune(make(), "desc", np.random.default_rng(seed + 1), steps=a.ft_steps, lr=lr)
        best = max(best, L.accuracy(m, "desc"))
    return best


def e1(a):
    out = []
    for seed in range(a.seeds):
        base = base_model(a, seed)
        out.append({"seed": seed, "method": "full", "params": L.trainable(base),
                    "acc": best_over_lr(lambda: copy.deepcopy(base), a, seed)})
        for targets in (("wq",), ("wq", "wv"), ("wq", "wk", "wv", "wo")):
            for r in a.ranks:
                mk = lambda: L.add_lora(copy.deepcopy(base), targets, r)
                out.append({"seed": seed, "method": "+".join(targets), "r": r, "params": L.trainable(mk()),
                            "acc": best_over_lr(mk, a, seed)})
    return out


def e2(a):
    out = []
    for seed in range(a.seeds):
        base = base_model(a, seed)
        for budget_r in a.budget_ranks:                       # r for a single matrix; halves for two, quarters for four
            for targets in (("wq",), ("wk",), ("wv",), ("wo",), ("wq", "wk"), ("wq", "wv"), ("wq", "wk", "wv", "wo")):
                r = max(1, budget_r // len(targets))
                mk = lambda: L.add_lora(copy.deepcopy(base), targets, r)
                out.append({"seed": seed, "budget (single-matrix r)": budget_r, "targets": "+".join(targets), "r": r,
                            "acc": best_over_lr(mk, a, seed)})
    return out


def e3(a):
    base = base_model(a)
    runs = {}
    for name, r, seed in (("r8 seed0", 8, 0), ("r64 seed1", 64, 1), ("r64 seed2", 64, 2)):
        torch.manual_seed(seed)
        m = L.add_lora(copy.deepcopy(base), ("wq", "wv"), r)
        for layer in L.lora_layers(m):                       # re-draw A with this run's seed
            with torch.no_grad():
                layer.A.normal_(0, 1 / np.sqrt(layer.A.shape[1]))
        runs[name] = L.finetune(m, "desc", np.random.default_rng(seed), steps=a.ft_steps, lr=1e-2)
    out = {}
    for pair in (("r8 seed0", "r64 seed1"), ("r64 seed1", "r64 seed2")):
        A1 = [l.A.detach() for l in L.lora_layers(runs[pair[0]])][-1]
        A2 = [l.A.detach() for l in L.lora_layers(runs[pair[1]])][-1]
        out[" vs ".join(pair)] = {f"i={i},j={j}": L.subspace_similarity(A1, A2, i, j) for i in (1, 2, 4, 8) for j in (1, 2, 4, 8, 16, 32)}
    return out


def e4(a):
    base = base_model(a)
    out = {}
    for r in (1, 4, 16, 64):
        m = L.finetune(L.add_lora(copy.deepcopy(base), ("wq", "wv"), r), "desc", np.random.default_rng(0),
                       steps=a.ft_steps, lr=1e-2)
        for li, layer in enumerate(L.lora_layers(m)):
            out[f"r={r} adapter {li}"] = L.amplification(layer.base.weight.detach(), layer.delta().detach(), min(r, 8))
    return out


def e5(a):
    base = base_model(a)
    out = {}
    for n in a.sizes:
        data = L.make_batch("desc", n, np.random.default_rng(7))
        for name, mk, lr in (("full", lambda: copy.deepcopy(base), 3e-3),
                             ("LoRA r=8", lambda: L.add_lora(copy.deepcopy(base), ("wq", "wv"), 8), 1e-2)):
            m = L.finetune(mk(), "desc", np.random.default_rng(0), steps=a.ft_steps, lr=lr, train_data=data)
            out[f"{n} examples, {name}"] = L.accuracy(m, "desc")
    return out


def e6(a):
    base = base_model(a)
    out = {}
    for r in (2, 8, 32):
        for lr in a.lrs:
            m = L.finetune(L.add_lora(copy.deepcopy(base), ("wq", "wv"), r, alpha=8), "desc",
                           np.random.default_rng(0), steps=a.ft_steps, lr=lr)
            out[f"r={r}, alpha=8, lr={lr}"] = L.accuracy(m, "desc")
    return out


def e7(a):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.hf_model)
    tok.pad_token = tok.eos_token
    sst = load_dataset("glue", "sst2")
    train, val = sst["train"].shuffle(seed=0).select(range(a.n_train)), sst["validation"]
    label_word = {0: " negative", 1: " positive"}
    word_ids = [tok(label_word[i]).input_ids[0] for i in (0, 1)]

    def batch(rows):
        texts = [f"review: {s.strip()} sentiment:{label_word[l]}" for s, l in zip(rows["sentence"], rows["label"])]
        x = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=96).to(dev)
        labels = x.input_ids.clone()
        labels[x.attention_mask == 0] = -100
        return x, labels

    out = {}
    for method in ("full", "lora"):
        m = AutoModelForCausalLM.from_pretrained(a.hf_model).to(dev)
        if method == "lora":
            for p in m.parameters():
                p.requires_grad_(False)
            for blk in m.transformer.h:                       # GPT-2's fused q,k,v projection is a Conv1D (in x out)
                conv = blk.attn.c_attn
                lin = torch.nn.Linear(conv.weight.shape[0], conv.weight.shape[1]).to(dev)
                with torch.no_grad():
                    lin.weight.copy_(conv.weight.T)
                    lin.bias.copy_(conv.bias)
                blk.attn.c_attn = L.LoRALinear(lin, a.r, alpha=2 * a.r).to(dev)
                blk.attn.c_attn.base.bias.requires_grad_(False)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.hf_lr[method])
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        for s in range(0, len(train), a.bs):
            x, labels = batch(train[s:s + a.bs])
            loss = m(**x, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        step_time = (time.time() - t) / max(1, len(train) // a.bs)
        m.eval()
        hits = []
        with torch.no_grad():
            for s in range(0, len(val), 64):
                rows = val[s:s + 64]
                x = tok([f"review: {t.strip()} sentiment:" for t in rows["sentence"]], return_tensors="pt", padding=True).to(dev)
                logits = m(**x).logits
                last = x.attention_mask.sum(1) - 1
                pred = logits[torch.arange(len(last)), last][:, word_ids].argmax(-1).cpu()
                hits += (pred == torch.tensor(rows["label"])).tolist()
        n_train = sum(p.numel() for p in params)
        out[method] = {"val accuracy": float(np.mean(hits)), "trainable params": n_train,
                       "checkpoint MB (fp16)": n_train * 2 / 1e6, "seconds per step": step_time,
                       "peak GPU memory GB": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None}
    return out


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 8)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.seeds, a.pre_steps, a.ft_steps, a.lrs = 3, 800, 300, (1e-3, 3e-3, 1e-2, 3e-2)
    a.ranks, a.budget_ranks, a.sizes = (1, 2, 4, 8, 16, 64), (4, 8, 16), (32, 128, 512, 2048)
    a.hf_model, a.n_train, a.bs, a.r, a.hf_lr = "gpt2", 8000, 16, 8, {"full": 5e-5, "lora": 5e-4}
    if a.quick:
        a.seeds, a.pre_steps, a.ft_steps, a.lrs = 1, 2, 2, (1e-2,)
        a.ranks, a.budget_ranks, a.sizes = (1,), (4,), (32,)
        a.hf_model, a.n_train, a.bs = "sshleifer/tiny-gpt2", 8, 4
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6), ("e7", e7)):
            if a.only in (None, name):
                try:
                    Rs[name] = fn(a)
                except ImportError as e:
                    Rs[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(Rs, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(Rs, a)


if __name__ == "__main__":
    main()
