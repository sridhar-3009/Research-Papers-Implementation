"""AWQ experiments (Lin et al. 2023): toy sweeps of the paper's tables and a real AWQ run on OPT.

  E1  Table 1: keep {1, 2, 3, 5, 10}% of input channels in FP chosen by activation / weight norm / random, INT3 and
      INT4 with groups of 16 and 32; 3 seeds of the outlier model.
  E2  Table 2: fixed scale s in {1, 1.25, 1.5, 2, 3, 4, 8, 16} for the salient channels; share of changed Delta.
  E3  Calibration robustness (Figure 8): calibrate AWQ and GPTQ on COPY-only sequences vs on all three tasks, then
      evaluate on all three; also 4 vs 16 vs 128 calibration sequences.
  E4  AWQ + GPTQ (scale first, then GPTQ on the scaled weights) and GPTQ with act-order, INT2 / INT3 / INT4.
  E5  Search grid size {5, 10, 20, 50} and the distribution of chosen alphas per layer type.
  E6  Real model: OPT-1.3B (Hugging Face `transformers`), INT3 and INT4 with g128, RTN vs AWQ (scale search folded
      into the LayerNorms / previous linear layers, as in awq.py), WikiText-2 perplexity.

!! HEAVY for E6 (downloads OPT-1.3B + WikiText-2; GPU recommended); E1-E5 take minutes on a CPU.
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

import awq as A

HERE = Path(__file__).parent
G = A.G


def outlier_model(a, seed=0):
    base = A.L.pretrain(A.L.new_model(seed), np.random.default_rng(seed), steps=a.pre_steps)
    return A.SQ.inject_outliers(base, np.random.default_rng(seed + 2), steps=a.re_steps)


def e1(a):
    out = []
    for seed in range(a.seeds):
        m = outlier_model(a, seed)
        calib = A.task_batch(("copy", "reverse", "sort"), seed=seed + 1)
        for bits in (3, 4):
            for group in a.group_sizes:
                out.append({"seed": seed, "bits": bits, "group": group, "method": "rtn",
                            "acc": G.task_accuracy(A.rtn_model(m, bits, group))})
                for frac in a.fracs:
                    for how in ("act", "weight", "random"):
                        q = A.quantize_keep_fp(m, calib, bits, group, frac, how)
                        out.append({"seed": seed, "bits": bits, "group": group, "frac": frac, "how": how,
                                    "acc": G.task_accuracy(q)})
    return out


def e2(a):
    m = outlier_model(a)
    calib = A.task_batch(("copy", "reverse", "sort"))
    out = {}
    for s in a.scales:
        q, ch = A.scale_salient(m, calib, 4, 16, s, frac=0.05)
        out[s] = {"changed Delta": ch, "acc": G.task_accuracy(q), "loss": G.model_loss(q)}
    return out


def e3(a):
    out = {}
    for name, model in (("outlier model", outlier_model(a)),
                        ("plain model", A.L.pretrain(A.L.new_model(), np.random.default_rng(0), steps=a.pre_steps))):
        for calib_name, tasks, n in (("all tasks, 126", ("copy", "reverse", "sort"), 126), ("copy only, 126", ("copy",), 126),
                                     ("all tasks, 12", ("copy", "reverse", "sort"), 12), ("all tasks, 3", ("copy", "reverse", "sort"), 3)):
            calib = A.task_batch(tasks, n=n)
            for bits in (2, 3):
                aw = A.awq_model(model, calib, bits, 16)[0]
                gp = G.quantize_model(model, calib, bits, "gptq", 16)
                out[f"{name} | calib {calib_name} | INT{bits}"] = {"awq": G.task_accuracy(aw), "gptq": G.task_accuracy(gp)}
    return out


@torch.no_grad()
def awq_then_gptq(model, calib, bits, group):
    """Fold AWQ's scales, then quantize the scaled weights with GPTQ instead of round-to-nearest."""
    m = copy.deepcopy(model)
    for b in range(len(m.blocks)):
        blk = m.blocks[b]
        for (sub, names), fold in A.GROUPS:
            X = A.group_input_stats(m, calib, b, sub, names[0])
            lins = [getattr(getattr(blk, sub), n) for n in names]
            s, _, _ = A.awq_scale([l.weight.data for l in lins], X, bits, group)
            if fold[0] == "norm":
                getattr(blk, fold[1]).weight.div_(s)
            else:
                getattr(getattr(blk, fold[1]), fold[2]).weight.div_(s[:, None])
            H = G.hessian((X / s).T.double())
            for l in lins:
                l.weight.data = G.gptq((l.weight.data * s).double(), H, bits, 128, group)
    return m


@torch.no_grad()
def gptq_act_order(model, calib, bits, group):
    m = copy.deepcopy(model)
    for b in range(len(m.blocks)):
        for sub, names in G.GROUPS:
            X = G.layer_inputs(m, calib, b, sub, names[0])
            H = G.hessian(X)
            perm = torch.argsort(torch.diag(H), descending=True)
            for n in names:
                lin = getattr(getattr(m.blocks[b], sub), n)
                W = lin.weight.data.double()[:, perm]
                Q = G.gptq(W, H[perm][:, perm], bits, 128, group)
                lin.weight.data = Q[:, torch.argsort(perm)].float()
    return m


def e4(a):
    m = outlier_model(a)
    calib = A.task_batch(("copy", "reverse", "sort"))
    out = {}
    for bits in (2, 3, 4):
        out[f"INT{bits}"] = {"rtn": G.task_accuracy(A.rtn_model(m, bits, 16)),
                             "awq": G.task_accuracy(A.awq_model(m, calib, bits, 16)[0]),
                             "gptq": G.task_accuracy(G.quantize_model(m, calib, bits, "gptq", 16)),
                             "gptq act-order": G.task_accuracy(gptq_act_order(m, calib, bits, 16)),
                             "awq + gptq": G.task_accuracy(awq_then_gptq(m, calib, bits, 16))}
    return out


def e5(a):
    m = outlier_model(a)
    calib = A.task_batch(("copy", "reverse", "sort"))
    out = {}
    for grid in a.grids:
        q, alphas = A.awq_model(m, calib, 3, 16, grid=grid)
        out[f"grid {grid}"] = {"acc": G.task_accuracy(q), "alphas (qkv, o, w1w3, w2 per block)": alphas}
    return out


def e6(a):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.hf_model)
    base = AutoModelForCausalLM.from_pretrained(a.hf_model, torch_dtype=torch.float32).to(dev).eval()
    train = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
    test = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    calib = tok(train, return_tensors="pt").input_ids[:, :a.calib_tokens].to(dev).view(-1, 512)
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

    def inputs_of(m, lin):
        store = []
        h = lin.register_forward_hook(lambda mod, i, o: store.append(i[0].reshape(-1, i[0].shape[-1])))
        with torch.no_grad():
            m(calib)
        h.remove()
        return torch.cat(store).float()

    def quantize(method, bits):
        m = copy.deepcopy(base)
        for layer in m.model.decoder.layers:
            groups = (((layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj), ("ln", layer.self_attn_layer_norm)),
                      ((layer.self_attn.out_proj,), ("rows", layer.self_attn.v_proj)),
                      ((layer.fc1,), ("ln", layer.final_layer_norm)),
                      ((layer.fc2,), (None, None)))                       # ReLU in between: no exact fold for fc2
            for lins, (kind, prev) in groups:
                if method == "awq" and kind is not None:
                    X = inputs_of(m, lins[0])
                    s, _, _ = A.awq_scale([l.weight.data.float() for l in lins], X, bits, 128)
                    with torch.no_grad():
                        if kind == "ln":
                            prev.weight.div_(s)
                            prev.bias.div_(s)
                        else:
                            prev.weight.div_(s[:, None])
                            prev.bias.div_(s)
                        for l in lins:
                            l.weight.mul_(s)
                for l in lins:
                    l.weight.data = A.wq(l.weight.data.float(), bits, 128).to(l.weight.dtype)
        return m

    out = {"fp32": ppl(base)}
    for bits in (4, 3):
        for method in ("rtn", "awq"):
            out[f"{method} INT{bits}-g128"] = ppl(quantize(method, bits))
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
    a.seeds, a.pre_steps, a.re_steps, a.group_sizes = 3, 800, 300, (16, 32)
    a.fracs, a.scales, a.grids = (0.01, 0.02, 0.05, 0.1), (1, 1.25, 1.5, 2, 3, 4, 8, 16), (5, 10, 20, 50)
    a.hf_model, a.calib_tokens, a.eval_tokens = "facebook/opt-1.3b", 512 * 64, 60000
    if a.quick:
        a.seeds, a.pre_steps, a.re_steps, a.group_sizes = 1, 2, 2, (16,)
        a.fracs, a.scales, a.grids = (0.05,), (2,), (5,)
        a.hf_model, a.calib_tokens, a.eval_tokens = "facebook/opt-125m", 1024, 1024
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
