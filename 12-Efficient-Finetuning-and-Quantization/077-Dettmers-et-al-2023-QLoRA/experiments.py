"""QLoRA experiments (Dettmers et al. 2023): toy sweeps of the data-type and adapter studies, and a real QLoRA run.

  E1  Data types x block size {16, 32, 64, 128, 256} on Gaussian, Laplace (heavy-tailed) and real model weights:
      relative MSE and code entropy (does NF4's advantage depend on the weights being normal?).
  E2  Whole-model post-training quantization: {Int, FP, NF} x {3, 4} bits x block sizes, with and without double
      quantization, 3 seeds of the tiny LLaMA: loss and task accuracy.
  E3  QLoRA fine-tuning: base data type {16-bit, NF4, FP4 E2M1, FP4 E3M0, Int4, NF3, Int3} x LoRA placement {q,v;
      attention; all linear} x r {2, 8, 32}; 3 seeds.
  E4  "Lost performance is recovered by adapters": quantize to 3 bits, then train LoRA on the ORIGINAL tasks only -
      how much of the quantization loss comes back?
  E5  Real QLoRA: a small causal LM (e.g. `facebook/opt-350m`) loaded in 4-bit NF4 with double quantization via
      `bitsandbytes` (BitsAndBytesConfig), LoRA r=16 on all linear layers via `peft`, fine-tuned on an instruction
      dataset slice (`tatsu-lab/alpaca`); compare with 16-bit LoRA: loss, memory, step time.

!! HEAVY for E5 (needs a CUDA GPU with bitsandbytes, model + data downloads); E1-E4 take minutes on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import qlora as Q

HERE = Path(__file__).parent
L = Q.L
TYPES = dict(Q.DATA_TYPES, **Q.DATA_TYPES_3BIT)


def base_model(a, seed=0):
    return L.pretrain(L.new_model(seed), np.random.default_rng(seed), steps=a.pre_steps)


def entropy(codes, n):
    p = torch.bincount(codes.flatten(), minlength=n).float()
    p = p / p.sum()
    return float(-(p[p > 0] * p[p > 0].log2()).sum())


def e1(a):
    torch.manual_seed(0)
    base = base_model(a)
    real = torch.cat([p.detach().flatten() for n, p in base.named_parameters() if "blocks" in n and p.dim() == 2])
    real = real[: (len(real) // 256) * 256]
    weights = {"gaussian": torch.randn(len(real)), "laplace": torch.distributions.Laplace(0, 1).sample((len(real),)),
               "tiny LLaMA weights": real}
    out = {}
    for wname, w in weights.items():
        for bs in a.blocks:
            W = w.reshape(-1, bs)
            for k, v in TYPES.items():
                codes, _ = Q.quantize_blockwise(W, v, bs)
                out[f"{wname} | block {bs} | {k}"] = {"rel mse": float((Q.fake_quant(W, v, bs) - W).pow(2).mean() / W.var()),
                                                      "entropy": entropy(codes, len(v))}
    return out


def e2(a):
    out = []
    for seed in range(a.seeds):
        base = base_model(a, seed)
        seq = torch.cat([L.make_batch(t, 200, np.random.default_rng(9)) for t in ("copy", "reverse", "sort")])
        for k, v in TYPES.items():
            for bs in a.blocks:
                for dq in (False, True):
                    m = Q.quantize_model(base, v, bs, dq)
                    with torch.no_grad():
                        out.append({"seed": seed, "type": k, "block": bs, "dq": dq, "loss": float(L.seq_loss(m, seq)),
                                    "acc": float(np.mean([L.accuracy(m, t) for t in ("copy", "reverse", "sort")]))})
    return out


def e3(a):
    out = []
    for seed in range(a.seeds):
        base = base_model(a, seed)
        for tname in ("16-bit", "NF4", "FP4 (E2M1)", "FP4 (E3M0)", "Int4", "NF3", "Int3"):
            qb = base if tname == "16-bit" else Q.quantize_model(base, TYPES[tname], 16 if "3" in tname[-1] else 64)
            for place in ("qv", "attention", "all"):
                for r in a.ranks:
                    m = copy.deepcopy(qb)
                    if place == "qv":
                        L.add_lora(m, ("wq", "wv"), r)
                    elif place == "attention":
                        L.add_lora(m, ("wq", "wk", "wv", "wo"), r)
                    else:
                        Q.add_lora_everywhere(m, r)
                    L.finetune(m, "desc", np.random.default_rng(seed + 1), steps=a.ft_steps, lr=1e-2)
                    out.append({"seed": seed, "base": tname, "lora": place, "r": r, "acc": L.accuracy(m, "desc")})
    return out


def e4(a):
    base = base_model(a)
    tasks = ("copy", "reverse", "sort")
    out = {"fp": float(np.mean([L.accuracy(base, t) for t in tasks]))}
    for k in ("Int3", "NF3"):
        qb = Q.quantize_model(base, TYPES[k], 64)
        out[f"{k} before adapters"] = float(np.mean([L.accuracy(qb, t) for t in tasks]))
        m = Q.add_lora_everywhere(copy.deepcopy(qb), 8)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-3)
        rng = np.random.default_rng(3)
        for _ in range(a.ft_steps):
            loss = sum(L.seq_loss(m, L.make_batch(t, 32, rng)) for t in tasks) / 3
            opt.zero_grad()
            loss.backward()
            opt.step()
        out[f"{k} after LoRA on the original tasks"] = float(np.mean([L.accuracy(m, t) for t in tasks]))
    return out


def e5(a):
    import peft
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    assert torch.cuda.is_available(), "QLoRA's 4-bit kernels need a CUDA GPU"
    tok = AutoTokenizer.from_pretrained(a.hf_model)
    data = load_dataset("tatsu-lab/alpaca", split=f"train[:{a.n_train}]")
    texts = [f"### Instruction:\n{x['instruction']}\n{x['input']}\n### Response:\n{x['output']}" for x in data]
    out = {}
    for name in ("16-bit LoRA", "QLoRA NF4+DQ"):
        kw = {"torch_dtype": torch.bfloat16}
        if name.startswith("QLoRA"):
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(a.hf_model, device_map="cuda", **kw)
        if name.startswith("QLoRA"):
            model = peft.prepare_model_for_kbit_training(model)
        cfg = peft.LoraConfig(r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.05, task_type="CAUSAL_LM")
        model = peft.get_peft_model(model, cfg)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        torch.cuda.reset_peak_memory_stats()
        t, losses = time.time(), []
        for s in range(0, len(texts), a.bs):
            x = tok(texts[s:s + a.bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
            loss = model(**x, labels=x.input_ids).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out[name] = {"final loss (mean of last 20 steps)": float(np.mean(losses[-20:])),
                     "peak memory GB": torch.cuda.max_memory_allocated() / 1e9,
                     "seconds per step": (time.time() - t) / len(losses)}
    return out


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.seeds, a.pre_steps, a.ft_steps, a.blocks, a.ranks = 3, 800, 300, (16, 32, 64), (2, 8, 32)
    a.hf_model, a.n_train, a.bs = "facebook/opt-350m", 2000, 8
    if a.quick:
        a.seeds, a.pre_steps, a.ft_steps, a.blocks, a.ranks = 1, 2, 2, (64,), (2,)
        a.hf_model, a.n_train, a.bs = "facebook/opt-125m", 8, 4
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                try:
                    Rs[name] = fn(a)
                except (ImportError, AssertionError) as e:
                    Rs[name] = {"skipped": str(e)}
                path.write_text(json.dumps(Rs, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(Rs, a)


if __name__ == "__main__":
    main()
