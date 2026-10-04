"""GPTQ experiments (Frantar et al. 2023): toy sweeps of the paper's analyses, and real GPTQ on OPT.

  E1  Layer level: RTN vs OBQ (greedy) vs GPTQ (fixed order) vs GPTQ with 'act-order' (columns sorted by diag(H)),
      bits {2, 3, 4} x input correlation {none, medium, strong} x layer width {32, 64, 128}; 5 seeds.
  E2  Dampening lambda in {0, 0.1%, 1%, 10%} and calibration size {8, 32, 128, 512} sequences.
  E3  Model level (Table 3 analogue): tiny LLaMAs of width {32, 64, 128} quantized to {4, 3, 2} bits, per row and
      with groups {8, 16, 32}, RTN vs GPTQ.
  E4  Runtime scaling: OBQ vs GPTQ on square layers 32 ... 512 (log-log slopes vs the predicted d^4 and d^3).
  E5  Real GPTQ: OPT-125M / OPT-350M (Hugging Face `transformers`) quantized layer by layer with 128 C4 segments of
      2048 tokens (here: WikiText-2 train text), 4 / 3 bits, per row and g128; WikiText-2 perplexity vs RTN.

!! HEAVY for E5 (downloads OPT + data; GPU recommended); E1-E4 take minutes on a CPU.
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

import gptq as G

HERE = Path(__file__).parent


def layer(d_row, d_col, corr, seed, n=512):
    g = torch.Generator().manual_seed(seed)
    mix = torch.eye(d_col) + corr * torch.randn(d_col, d_col, generator=g)
    X = (mix @ torch.randn(d_col, n, generator=g)).double()
    return torch.randn(d_row, d_col, generator=g).double(), X


def act_order(W, H, bits):
    perm = torch.argsort(torch.diag(H), descending=True)
    Q = G.gptq(W[:, perm], H[perm][:, perm], bits)
    return Q[:, torch.argsort(perm)]


def e1(a):
    out = []
    for width in a.widths:
        for corr in (0.0, 0.5, 2.0):
            for bits in (2, 3, 4):
                for seed in range(a.seeds):
                    W, X = layer(width, width, corr, seed)
                    H = G.hessian(X)
                    row = {"width": width, "corr": corr, "bits": bits, "seed": seed,
                           "rtn": G.layer_error(W, G.rtn(W, bits).double(), X),
                           "gptq": G.layer_error(W, G.gptq(W, H, bits).double(), X),
                           "gptq act-order": G.layer_error(W, act_order(W, H, bits).double(), X)}
                    if width <= a.obq_max:
                        row["obq"] = G.layer_error(W, G.obq(W, H, bits).double(), X)
                    out.append(row)
    return out


def e2(a):
    out = {}
    W, X = layer(64, 64, 2.0, 0, n=2048)
    for damp in (0.0, 0.001, 0.01, 0.1):
        try:
            out[f"damp {damp}"] = G.layer_error(W, G.gptq(W, G.hessian(X, damp), 3).double(), X)
        except RuntimeError as e:                                       # Cholesky fails without dampening
            out[f"damp {damp}"] = f"failed: {str(e)[:60]}"
    base = G.L.pretrain(G.L.new_model(), np.random.default_rng(0), steps=a.pre_steps)
    for n in a.calib_sizes:
        q = G.quantize_model(base, G.calibration_batch(n), 2, "gptq")
        out[f"{n} calibration sequences, 2-bit"] = {"acc": G.task_accuracy(q), "loss": G.model_loss(q)}
    return out


def e3(a):
    out = []
    for width in a.model_widths:
        torch.manual_seed(0)
        base = G.L.pretrain(G.L.new_model(d=width), np.random.default_rng(0), steps=a.pre_steps)
        calib = G.calibration_batch()
        out.append({"width": width, "bits": 16, "acc": G.task_accuracy(base), "loss": G.model_loss(base)})
        for bits in (4, 3, 2):
            for gs in (None,) + tuple(g for g in a.groups if g < width):
                for method in ("rtn", "gptq"):
                    q = G.quantize_model(base, calib, bits, method, gs)
                    out.append({"width": width, "bits": bits, "group": gs, "method": method,
                                "acc": G.task_accuracy(q), "loss": G.model_loss(q)})
    return out


def e4(a):
    out = {}
    for d in a.sizes:
        W, X = layer(d, d, 0.5, 0, n=2 * d)
        H = G.hessian(X)
        t = time.time()
        G.gptq(W, H, 4)
        tg = time.time() - t
        to = None
        if d <= a.obq_max:
            t = time.time()
            G.obq(W, H, 4)
            to = time.time() - t
        out[d] = {"gptq s": tg, "obq s": to}
    return out


def e5(a):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for name in a.hf_models:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
        train = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"])
        test = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
        tr_ids = tok(train, return_tensors="pt").input_ids
        te_ids = tok(test, return_tensors="pt").input_ids[:, :a.eval_tokens].to(dev)
        g = torch.Generator().manual_seed(0)
        starts = torch.randint(0, tr_ids.shape[1] - a.seq, (a.n_calib,), generator=g)
        calib = torch.stack([tr_ids[0, s:s + a.seq] for s in starts]).to(dev)

        def ppl(m):
            nll, n = 0.0, 0
            with torch.no_grad():
                for s in range(0, te_ids.shape[1] - 1, a.seq):
                    c = te_ids[:, s:s + a.seq]
                    if c.shape[1] < 2:
                        break
                    nll += m(c, labels=c).loss.item() * (c.shape[1] - 1)
                    n += c.shape[1] - 1
            return float(np.exp(nll / n))

        def quantize(method, bits, gs):
            m = copy.deepcopy(model)
            for layer_mod in m.model.decoder.layers:                   # forward order inside each decoder layer
                for lin_name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "fc1", "fc2"):
                    lin = layer_mod.get_submodule(lin_name)
                    store = {}

                    def hook(mod, inp, outp):
                        store["x"] = inp[0].reshape(-1, inp[0].shape[-1])
                    h = lin.register_forward_hook(hook)
                    with torch.no_grad():
                        for i in range(0, len(calib), 8):
                            m(calib[i:i + 8])
                            x = store["x"].double().T
                            store["H"] = store.get("H", 0) + 2 * x @ x.T
                    h.remove()
                    H = store["H"] + 0.01 * torch.diag(store["H"]).mean() * torch.eye(store["H"].shape[0], device=dev, dtype=torch.float64)
                    W = lin.weight.data.double().cpu()
                    Q = G.rtn(W, bits, gs) if method == "rtn" else G.gptq(W, H.cpu(), bits, 128, gs)
                    lin.weight.data = Q.float().to(dev)
            return m

        res = {"fp32": ppl(model)}
        for bits in (4, 3):
            for gs in (None, 128):
                for method in ("rtn", "gptq"):
                    res[f"{method} {bits}-bit{'' if gs is None else ' g128'}"] = ppl(quantize(method, bits, gs))
        out[name] = res
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
    a.widths, a.seeds, a.obq_max, a.pre_steps = (32, 64, 128), 5, 64, 800
    a.calib_sizes, a.model_widths, a.groups, a.sizes = (8, 32, 128, 512), (32, 64, 128), (8, 16, 32), (32, 64, 128, 256, 512)
    a.hf_models, a.n_calib, a.seq, a.eval_tokens = ("facebook/opt-125m", "facebook/opt-350m"), 128, 2048, 100000
    if a.quick:
        a.widths, a.seeds, a.obq_max, a.pre_steps = (16,), 1, 16, 2
        a.calib_sizes, a.model_widths, a.groups, a.sizes = (6,), (32,), (8,), (16,)
        a.hf_models, a.n_calib, a.seq, a.eval_tokens = ("facebook/opt-125m",), 2, 64, 256
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
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
