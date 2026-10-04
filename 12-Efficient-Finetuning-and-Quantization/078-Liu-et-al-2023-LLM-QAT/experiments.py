"""LLM-QAT experiments (Liu et al. 2023): toy versions of the paper's tables, and data-free QAT of a real small LM.

  E1  Table 1 analogue: W-A-KV in {8-8-8, 8-8-4, 4-8-8, 4-8-4, 4-6-16, 3-8-3, 2-8-8}: PTQ vs QAT (generated data,
      logits distillation); 3 seeds.
  E2  Table 3 analogue: training data = real / generated top-1 / sampled / hybrid (k_det = 1, 2, 3, 5) / a DIFFERENT
      real distribution (only 'copy' sequences, like fine-tuning on WikiText); evaluated per task.
  E3  Table 5 analogue: hard labels vs logits distillation vs logits + hidden-state MSE; sampling temperature 0.7 / 1 / 1.3.
  E4  Table 4 analogue: MinMax vs clipping (clip each tensor at its 99th / 99.9th percentile) for weights and
      activations, symmetric vs asymmetric.
  E5  Amount of generated data: 300 / 1k / 3k / 10k sequences; training steps 100 / 300 / 1000.
  E6  Real model: data-free QAT of OPT-125M (Hugging Face `transformers`): generate 2k sequences of 128 tokens with
      hybrid sampling, fake-quantize all decoder linears at W4-A8 (+ 4-bit K/V projections), logits-distil for a few
      hundred steps (AdamW 2e-5, cosine), WikiText-2 perplexity vs RTN PTQ.

!! HEAVY for E6 (downloads OPT-125M + WikiText; GPU recommended); E1-E5 take minutes on a CPU.
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
import torch.nn.functional as F

import llmqat as Q

HERE = Path(__file__).parent


def teacher(a, seed=0):
    return Q.pretrain(Q.new_model(seed), np.random.default_rng(seed), steps=a.pre_steps)


def e1(a):
    out = []
    for seed in range(a.seeds):
        T = teacher(a, seed)
        gen = Q.generate(T, a.n_gen, "sample", seed=seed + 1)
        for cfg in a.configs:
            out.append({"seed": seed, "W-A-KV": "-".join(map(str, cfg)), "ptq": Q.accuracy(Q.quantize(T, *cfg)),
                        "qat": Q.accuracy(Q.train_qat(Q.quantize(T, *cfg), T, gen, steps=a.steps, seed=seed))})
    return out


def e2(a):
    T = teacher(a)
    sources = {"real (all tasks)": Q.real_batch(a.n_gen, np.random.default_rng(5)),
               "real, copy only": Q.make_seq("copy", a.n_gen, np.random.default_rng(5)),
               "generated top-1": Q.generate(T, a.n_gen, "top1"), "generated sampled": Q.generate(T, a.n_gen, "sample", seed=1)}
    for k in a.k_dets:
        sources[f"generated hybrid k={k}"] = Q.generate(T, a.n_gen, "hybrid", k_det=k, seed=1)
    out = {}
    for name, data in sources.items():
        s = Q.train_qat(Q.quantize(T, 3, 8, 3), T, data, steps=a.steps)
        out[name] = {t: Q.accuracy(s, t) for t in Q.TASK} | {"diversity / valid": Q.diversity(data)}
    return out


def train_hidden(student, teacher_m, data, steps, w_hidden=1.0):
    """Logits distillation plus MSE between the students' and teacher's final hidden states."""
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.0)
    feats = {}

    def hook(name):
        def f(mod, inp, out):
            feats[name] = out
        return f
    hs = student.norm.register_forward_hook(hook("s"))
    ht = teacher_m.norm.register_forward_hook(hook("t"))
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        seq = data[torch.randint(len(data), (64,), generator=g)]
        sl = student(seq[:, :-1])
        with torch.no_grad():
            p = torch.softmax(teacher_m(seq[:, :-1]), -1)
        loss = -(p * torch.log_softmax(sl, -1)).sum(-1).mean() + w_hidden * F.mse_loss(feats["s"], feats["t"].detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    hs.remove()
    ht.remove()
    return student


def e3(a):
    T = teacher(a)
    out = {}
    for temp in (0.7, 1.0, 1.3):
        gen = Q.generate(T, a.n_gen, "sample", temperature=temp, seed=1)
        for loss in ("hard", "logits"):
            out[f"T={temp}, {loss}"] = Q.accuracy(Q.train_qat(Q.quantize(T, 3, 8, 3), T, gen, steps=a.steps, loss=loss))
    gen = Q.generate(T, a.n_gen, "sample", seed=1)
    out["T=1.0, logits + hidden MSE"] = Q.accuracy(train_hidden(Q.quantize(T, 3, 8, 3), T, gen, a.steps))
    return out


def e4(a):
    T = teacher(a)
    gen = Q.generate(T, a.n_gen, "sample", seed=1)
    original = Q.fake_quant
    out = {}
    for name in ("minmax", "clip 99.9%", "clip 99%"):
        if name != "minmax":
            pct = 0.999 if "99.9" in name else 0.99

            def clipped(x, bits, dim, pct=pct):
                if bits is None or bits >= 16:
                    return x
                qmax = 2 ** (bits - 1) - 1
                alpha = torch.quantile(x.detach().abs().float(), pct, dim=dim, keepdim=True).clamp(min=1e-8) / qmax
                q = torch.clamp(torch.round(x / alpha), -qmax, qmax) * alpha
                return x + (q - x).detach()
            Q.fake_quant = clipped
        out[name] = {"ptq": Q.accuracy(Q.quantize(T, 4, 6, 4)),
                     "qat": Q.accuracy(Q.train_qat(Q.quantize(T, 4, 6, 4), T, gen, steps=a.steps))}
        Q.fake_quant = original
    return out


def e5(a):
    T = teacher(a)
    out = {}
    for n in a.data_sizes:
        gen = Q.generate(T, n, "sample", seed=1)
        for steps in a.step_list:
            out[f"{n} sequences, {steps} steps"] = Q.accuracy(Q.train_qat(Q.quantize(T, 2, 8, 8), T, gen, steps=steps))
    return out


def e6(a):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.hf_model)
    T = AutoModelForCausalLM.from_pretrained(a.hf_model).to(dev).eval()
    test = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(test, return_tensors="pt").input_ids[:, :a.eval_tokens].to(dev)

    def ppl(m):
        nll, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, ids.shape[1] - 1, 512):
                c = ids[:, s:s + 512]
                if c.shape[1] < 2:
                    break
                nll += m(c, labels=c).loss.item() * (c.shape[1] - 1)
                n += c.shape[1] - 1
        return float(np.exp(nll / n))

    with torch.no_grad():                                                   # hybrid generation from BOS
        seqs = []
        for _ in range(a.n_hf_gen // 16):
            x = torch.full((16, 1), tok.bos_token_id, device=dev)
            for i in range(a.gen_len):
                logits = T(x).logits[:, -1]
                nxt = logits.argmax(-1) if i < 3 else torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
                x = torch.cat([x, nxt[:, None]], 1)
            seqs.append(x)
        data = torch.cat(seqs)

    def fq(m):
        s = copy.deepcopy(m)
        for layer in s.model.decoder.layers:
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"):
                parent, child = name.rsplit(".", 1) if "." in name else ("", name)
                mod = layer.get_submodule(parent) if parent else layer
                lin = getattr(mod, child)
                kv = 4 if child in ("k_proj", "v_proj") else None

                class QL(torch.nn.Module):
                    def __init__(self, lin, kv):
                        super().__init__()
                        self.lin, self.kv = lin, kv

                    def forward(self, x):
                        y = F.linear(Q.fake_quant(x, 8, -1), Q.fake_quant(self.lin.weight, 4, 1), self.lin.bias)
                        return Q.fake_quant(y, self.kv, -1) if self.kv else y
                setattr(mod, child, QL(lin, kv))
        return s

    S = fq(T).train()
    out = {"fp": ppl(T), "ptq W4-A8-KV4": ppl(fq(T))}
    opt = torch.optim.AdamW(S.parameters(), lr=2e-5, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.hf_steps)
    for step in range(a.hf_steps):
        x = data[torch.randint(len(data), (8,))]
        with torch.no_grad():
            p = torch.softmax(T(x).logits, -1)
        loss = -(p * torch.log_softmax(S(x).logits, -1)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    out["data-free QAT W4-A8-KV4"] = ppl(S.eval())
    return out


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs):
        Ls += [f"## {k.upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 7)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.seeds, a.pre_steps, a.steps, a.n_gen = 3, 900, 300, 3000
    a.configs = ((8, 8, 8), (8, 8, 4), (4, 8, 8), (4, 8, 4), (4, 6, 16), (3, 8, 3), (2, 8, 8))
    a.k_dets, a.data_sizes, a.step_list = (1, 2, 3, 5), (300, 1000, 3000, 10000), (100, 300, 1000)
    a.hf_model, a.eval_tokens, a.n_hf_gen, a.gen_len, a.hf_steps = "facebook/opt-125m", 40000, 2048, 128, 500
    if a.quick:
        a.seeds, a.pre_steps, a.steps, a.n_gen = 1, 2, 2, 30
        a.configs, a.k_dets, a.data_sizes, a.step_list = ((4, 8, 4),), (3,), (30,), (2,)
        a.eval_tokens, a.n_hf_gen, a.gen_len, a.hf_steps = 600, 16, 8, 2
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6)):
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
