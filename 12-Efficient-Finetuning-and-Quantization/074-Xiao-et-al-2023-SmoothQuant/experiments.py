"""SmoothQuant experiments (Xiao et al. 2023): toy sweeps of the paper's ablations, and fake-quantized W8A8 on a real
OPT model.

  E1  Granularity x bits (Table 1 analogue): activations per-tensor / per-token / per-channel / static, weights
      per-tensor / per-channel, W8A8 / W6A6 / W4A8; with and without SmoothQuant; 3 seeds of outlier injection.
  E2  Migration strength alpha in {0, 0.1, ..., 1} x bits {8, 6, 5} x outlier strength {30, 150, 500} (Figure 10:
      does the best alpha move toward 1 when activations get harder, as for GLM-130B?).
  E3  Calibration set size: 4 / 16 / 64 / 512 sequences for the smoothing factors and the static scales.
  E4  Where to smooth: only attention inputs, only FFN inputs, both.
  E5  Real model: OPT-1.3B (Hugging Face `transformers`), fake-quantized W8A8 per-tensor static (O3-like) with and
      without SmoothQuant (alpha = 0.5, 512 calibration sentences), WikiText-2 perplexity and the activation outlier
      statistics of every layer.

!! HEAVY for E5 (downloads OPT-1.3B + WikiText; GPU recommended); E1-E4 take minutes on a CPU.
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

import smoothquant as S

HERE = Path(__file__).parent


def models(a, seed=0, factor=150.0):
    base = S.L.pretrain(S.L.new_model(seed), np.random.default_rng(seed), steps=a.pre_steps)
    return S.inject_outliers(base, np.random.default_rng(seed + 2), factor=factor, steps=a.re_steps)


def e1(a):
    out = []
    for seed in range(a.seeds):
        m = models(a, seed)
        calib = S.calibration_batch(a.calib, seed + 1)
        sm = S.smooth(m, calib, 0.5)
        for wb, ab in a.bit_pairs:
            for w_gran in ("tensor", "channel"):
                for a_gran in ("tensor", "token", "channel", "static"):
                    for name, mm in (("plain", m), ("smoothquant", sm)):
                        acc = S.task_accuracy(S.quantized_copy(mm, calib, w_bits=wb, a_bits=ab, w_gran=w_gran, a_gran=a_gran))
                        out.append({"seed": seed, "bits": f"W{wb}A{ab}", "w": w_gran, "a": a_gran, "model": name, "acc": acc})
    return out


def e2(a):
    out = {}
    for factor in a.factors:
        m = models(a, 0, factor)
        calib = S.calibration_batch(a.calib)
        for bits in a.bits:
            out[f"factor {factor}, W{bits}A{bits}"] = {
                al: S.task_accuracy(S.quantized_copy(S.smooth(m, calib, al), calib, a_gran="static", w_bits=bits, a_bits=bits))
                for al in a.alphas}
    return out


def e3(a):
    m = models(a)
    test_calib = S.calibration_batch(a.calib)
    out = {}
    for n in a.calib_sizes:
        calib = S.calibration_batch(n, seed=5)
        sm = S.smooth(m, calib, 0.5)
        out[n] = {"W8A8 static": S.task_accuracy(S.quantized_copy(sm, calib, a_gran="static")),
                  "W6A6 static": S.task_accuracy(S.quantized_copy(sm, calib, a_gran="static", w_bits=6, a_bits=6))}
    return out


def e4(a):
    m = models(a)
    calib = S.calibration_batch(a.calib)
    out = {}
    for name, groups in (("attention inputs only", S.NORM_GROUPS[:1]), ("FFN inputs only", S.NORM_GROUPS[1:]),
                         ("both", S.NORM_GROUPS)):
        saved = S.NORM_GROUPS
        S.NORM_GROUPS = groups
        sm = S.smooth(m, calib, 0.5)
        S.NORM_GROUPS = saved
        out[name] = {f"W{b}A{b}": S.task_accuracy(S.quantized_copy(sm, calib, a_gran="static", w_bits=b, a_bits=b)) for b in (8, 6)}
    return out


def e5(a):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.hf_model)
    base = AutoModelForCausalLM.from_pretrained(a.hf_model, torch_dtype=torch.float32).to(dev).eval()
    test = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    calib_text = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
    calib = [t for t in calib_text if len(t) > 200][:a.hf_calib]
    ids = tok(test, return_tensors="pt").input_ids[:, :a.eval_tokens].to(dev)

    def ppl(model):
        nll, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, ids.shape[1] - 1, 512):
                chunk = ids[:, s:s + 513]
                out = model(chunk[:, :-1], labels=chunk[:, :-1])
                nll += out.loss.item() * (chunk.shape[1] - 2)
                n += chunk.shape[1] - 2
        return float(np.exp(nll / n))

    def absmax_inputs(model):
        stats, hooks = {}, []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "decoder.layers" in name:
                hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: stats.__setitem__(
                    n, torch.maximum(stats.get(n, torch.zeros(i[0].shape[-1], device=dev)), i[0].reshape(-1, i[0].shape[-1]).abs().amax(0)))))
        with torch.no_grad():
            for t in calib:
                model(tok(t, return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev))
        for h in hooks:
            h.remove()
        return stats

    def smooth_opt(model, alpha):
        m = copy.deepcopy(model)
        stats = absmax_inputs(m)
        for li, layer in enumerate(m.model.decoder.layers):
            for ln, lins in ((layer.self_attn_layer_norm, (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)),
                             (layer.final_layer_norm, (layer.fc1,))):
                key = f"model.decoder.layers.{li}." + ("self_attn.q_proj" if len(lins) == 3 else "fc1")
                ax = stats[key].clamp(min=1e-5)
                aw = torch.stack([l.weight.abs().amax(0) for l in lins]).amax(0).clamp(min=1e-5)
                s = ax ** alpha / aw ** (1 - alpha)
                with torch.no_grad():
                    ln.weight.div_(s)
                    ln.bias.div_(s)
                    for l in lins:
                        l.weight.mul_(s)
        return m

    def quantize_opt(model):
        m = copy.deepcopy(model)
        stats = absmax_inputs(m)
        for name, mod in list(m.named_modules()):
            for cname, child in list(mod.named_children()):
                full = f"{name}.{cname}" if name else cname
                if isinstance(child, torch.nn.Linear) and "decoder.layers" in full:
                    q = S.QuantLinear(child, static_scale=stats[full].max() / 127)
                    bias = child.bias

                    class WithBias(torch.nn.Module):
                        def __init__(self, q, b):
                            super().__init__()
                            self.q, self.b = q, b

                        def forward(self, x):
                            return self.q(x) + self.b
                    setattr(mod, cname, WithBias(q, bias))
        return m

    stats = absmax_inputs(base)
    ratio = {k: float(v.max() / v.median()) for k, v in stats.items() if "q_proj" in k or "fc1" in k}
    return {"FP32 perplexity": ppl(base), "W8A8 per-tensor static perplexity": ppl(quantize_opt(base)),
            "SmoothQuant + W8A8 per-tensor static perplexity": ppl(quantize_opt(smooth_opt(base, 0.5))),
            "max/median input |x| per layer (q_proj, fc1)": ratio}


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
    a.seeds, a.pre_steps, a.re_steps, a.calib = 3, 800, 300, 128
    a.bit_pairs, a.factors, a.bits = ((8, 8), (6, 6), (4, 8)), (30.0, 150.0, 500.0), (8, 6, 5)
    a.alphas, a.calib_sizes = tuple(round(x, 1) for x in np.arange(0, 1.01, 0.1)), (4, 16, 64, 512)
    a.hf_model, a.hf_calib, a.eval_tokens = "facebook/opt-1.3b", 512, 40000
    if a.quick:
        a.seeds, a.pre_steps, a.re_steps, a.calib = 1, 2, 2, 12
        a.bit_pairs, a.factors, a.bits, a.alphas, a.calib_sizes = ((8, 8),), (150.0,), (8,), (0.5,), (6,)
        a.hf_model, a.hf_calib, a.eval_tokens = "facebook/opt-125m", 2, 600
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
