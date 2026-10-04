"""Efficiently Scaling Transformer Inference experiments (Pope et al. 2022): sweeps of the analytical model and the
chip simulation, plus a real measurement of prefill vs decode on a small model.

  E1  Table 1 generalised: max context for multihead / baseline multiquery / optimized multiquery over batch
      {1 ... 1024}, chips {16, 64, 256} and KV budget {10%, 30%, 50%}.
  E2  Figure 3 by simulation: bytes sent per chip for 1D WS, 2D WS (all X x YZ splits) and weight-gathered, over
      chips {16, 64, 256} and tokens per batch {16 ... 65536}; which layout wins where; fit of the 2D slope in chips.
  E3  Figure 1: latency vs cost Pareto frontiers (chips x batch x int8/bf16) for PaLM 8B / 62B / 540B, prefill and
      decode, keeping only configurations whose weights + KV cache fit in HBM.
  E4  Real measurement (Hugging Face `transformers`, GPU recommended): GPT-2 (multihead) and a multiquery model
      (`bigcode/tiny_starcoder_py`) -- prefill time for 512 tokens and decode time per token vs batch size {1 ... 64},
      KV-cache bytes per token, and the resulting MFU estimate.

!! HEAVY for E4 (model downloads; GPU recommended); E1-E3 run in seconds.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import inference as I

HERE = Path(__file__).parent


def e1(a):
    out = {}
    for chips in a.chips:
        for frac in (0.1, 0.3, 0.5):
            for v in ("multihead", "baseline multiquery", "optimized multiquery"):
                out[f"{chips} chips, {frac:.0%}, {v}"] = {b: I.max_context(v, b, chips, frac) for b in a.batches}
    return out


def e2(a):
    rng = np.random.default_rng(0)
    E, F = 512, 2048
    Wi, Wo = rng.standard_normal((E, F)) / 22, rng.standard_normal((F, E)) / 45
    out = []
    for n in a.chips:
        splits = [(X, n // X) for X in (1, 2, 4, 8, 16) if n % X == 0 and n // X <= F and X <= E]
        for tokens in a.tokens:
            x = rng.standard_normal((tokens, E))
            row = {"chips": n, "tokens": tokens}
            m = I.Mesh(n)
            I.ffn_1d_ws(np.array_split(x, n, axis=-1), Wi, Wo, m)
            row["1D WS"] = m.sent.mean()
            for X, YZ in splits:
                m = I.Mesh(n)
                I.ffn_2d_ws(x, Wi, Wo, m, X, YZ)
                row[f"2D WS X={X}"] = m.sent.mean()
            m = I.Mesh(n)
            I.ffn_weight_gathered(x, Wi, Wo, m)
            row["WG"] = m.sent.mean()
            row["best"] = min((k for k in row if k not in ("chips", "tokens")), key=lambda k: row[k])
            out.append(row)
    return out


def e3(a):
    out = {}
    for model in ("8B", "62B", "540B"):
        n, L, E, F, H, dh = I.PALM[model]
        for phase in ("prefill", "decode"):
            pts = []
            for chips in a.pareto_chips:
                for b in a.batches:
                    for wb in (1, 2):
                        for layout in ("2D WS", "WG"):
                            mem = n * wb + I.kv_bytes_per_token(L, H, dh, 1) * b * 2048
                            if mem > 0.9 * chips * I.TPU_V4["hbm_bytes"]:
                                continue
                            r = I.step_time(model, chips, b, 2048, phase, layout, wb)
                            lat = r["total"] * (64 if phase == "decode" else 1)
                            pts.append({"chips": chips, "batch": b, "bytes": wb, "layout": layout, "latency s": lat,
                                        "chip-ms per token": r["total"] * 1e3 * chips / (b if phase == "decode" else b * 2048),
                                        "MFU": r["MFU"]})
            pts.sort(key=lambda p: p["latency s"])
            frontier, best = [], float("inf")
            for p in pts:
                if p["chip-ms per token"] < best:
                    frontier.append(p)
                    best = p["chip-ms per token"]
            out[f"{model} {phase}"] = frontier
    return out


def e4(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for name in a.hf_models:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).to(dev).eval()
        n_params = sum(p.numel() for p in model.parameters())
        res = {"params": n_params}
        for b in a.hf_batches:
            ids = torch.randint(100, 1000, (b, a.prompt), device=dev)
            with torch.no_grad():
                t = time.time()
                o = model(ids, use_cache=True)
                if dev == "cuda":
                    torch.cuda.synchronize()
                pre = time.time() - t
                past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                t = time.time()
                for _ in range(a.gen):
                    o = model(nxt, past_key_values=past, use_cache=True)
                    past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                if dev == "cuda":
                    torch.cuda.synchronize()
                dec = (time.time() - t) / a.gen
            kv = sum(t_.numel() * t_.element_size() for layer in past for t_ in layer) if isinstance(past, (tuple, list)) else None
            res[f"batch {b}"] = {"prefill s": pre, "decode s/token": dec, "KV bytes per token": kv / (b * (a.prompt + a.gen)) if kv else None,
                                 "decode tokens/s": b / dec}
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
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 5)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.chips, a.batches, a.tokens = (16, 64, 256), (1, 4, 16, 64, 128, 256, 512, 1024), (16, 256, 4096, 65536)
    a.pareto_chips = (8, 16, 32, 64, 128, 256)
    a.hf_models, a.hf_batches, a.prompt, a.gen = ("gpt2", "bigcode/tiny_starcoder_py"), (1, 4, 16, 64), 512, 32
    if a.quick:
        a.chips, a.batches, a.tokens, a.pareto_chips = (16,), (1, 64), (16,), (64,)
        a.hf_models, a.hf_batches, a.prompt, a.gen = ("sshleifer/tiny-gpt2",), (1,), 8, 2
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)):
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
