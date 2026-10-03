"""Codex's experiments (Chen et al. 2021) at small scale, with real Transformer code models.

Corpus: every .py file of the local Python standard library (plus site-packages if --site), filtered as in Section 3.1
(files > 1 MB, average line length > 100, max line length > 1000, alphanumeric fraction < 0.25, exact duplicates);
the GPT-2 model and byte-level BPE from paper 049.

  E1  Figure 4: test loss on held-out code vs model size; fit L = (N / N_c)^-alpha (paper: alpha 0.13, N_c 5.92e7).
  E2  Section 3.2: initialise from a model pre-trained on English text (Gutenberg) vs from scratch; the paper found
      no final gain but faster convergence.
  E3  Section 3.2: a BPE learned on English text applied to code, with and without extra tokens for whitespace runs
      (paper: ~30% fewer tokens).
  E4  Codex-S on the Appendix C task: fine-tune on synthetic 'string_manipulation' problems (loss on the solution
      only, prompts left-padded), then on held-out problems measure
        - pass@k for k = 1, 10, 100 at T = 0.2 ... 1.2 (Figure 5: best T grows with k),
        - one-sample selection by mean / sum log-prob / random vs the oracle (Figure 7),
        - BLEU of passing vs failing samples (Figure 8),
        - pass rate vs number of chained building blocks 1..7 (Figure 11: ~2-3x drop per extra block).
  E5  HumanEval (downloaded from github.com/openai/human-eval): pass@1/10/100 of the E1 models. Expect ~0% for models
      this small; the paper's smallest Codex (12M) scores 2.0% pass@1 after 100B tokens.

!! HEAVY. Not run on the author's laptop. Generated code is executed in subprocesses WITHOUT a real sandbox.
       python3 experiments.py --quick
       python3 experiments.py --only e4
       python3 experiments.py --report-only
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import random
import re
import sysconfig
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from codex import (STOP, bleu, check_correctness, pass_at_k, synthetic_problem, truncate)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
spec = importlib.util.spec_from_file_location("gpt2_049", HERE.parent / "049-Radford-et-al-2019-GPT-2" / "gpt2.py")
gpt2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpt2)
SHAPES = [(2, 64), (2, 128), (4, 192), (4, 256), (6, 384), (8, 512)]
BOOKS = [1342, 84, 11, 2701, 1661, 98, 1952, 345]
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


# ---------------------------------------------------------------------------------------------------- data
def keep_file(text):
    lines = text.splitlines() or [""]
    if len(text.encode()) > 1_000_000: return False
    if sum(map(len, lines)) / len(lines) > 100 or max(map(len, lines)) > 1000: return False
    return sum(c.isalnum() for c in text) / max(len(text), 1) >= 0.25


def code_corpus(a):
    roots = [Path(sysconfig.get_paths()["stdlib"])] + ([Path(sysconfig.get_paths()["purelib"])] if a.site else [])
    seen, files = set(), []
    for root in roots:
        for f in sorted(root.rglob("*.py")):
            try:
                t = f.read_text(encoding="utf-8")
            except Exception:
                continue
            h = hashlib.md5(t.encode()).hexdigest()
            if h not in seen and keep_file(t):
                seen.add(h); files.append(t)
    random.Random(0).shuffle(files)
    return files[:a.max_files]


def text_corpus(a):
    d = DATA / "gutenberg"; d.mkdir(parents=True, exist_ok=True)
    parts = []
    for b in BOOKS[:a.n_books]:
        f = d / f"{b}.txt"
        if not f.exists():
            urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt", f)
        parts.append(f.read_text(encoding="utf-8", errors="ignore").split("*** START")[-1].split("*** END")[0])
    return "\n\n".join(parts)


_cache = {}


def setup(a):
    if "code" not in _cache:
        files = code_corpus(a)
        text = "\n\n".join(files)
        bpe = gpt2.ByteBPE(text[:a.bpe_chars], a.merges)
        ids = torch.tensor(bpe.encode(text[:a.max_chars]))
        cut = int(0.98 * len(ids))
        _cache["code"] = (ids[:cut], ids[cut:], bpe)
        print(f"  corpus: {len(files)} files, {len(ids):,} tokens, vocab {len(bpe)}", flush=True)
    return _cache["code"]


# ---------------------------------------------------------------------------------------------------- training
def make(L, d, V, a):
    torch.manual_seed(0)
    m = gpt2.GPT2(vocab=V, n_ctx=a.n_ctx, d=d, layers=L, heads=max(1, d // 64)).to(DEV)
    non_emb = sum(p.numel() for n, p in m.named_parameters() if "wte" not in n and "wpe" not in n)
    return m, non_emb


def batches(ids, a):
    while True:
        st = torch.randint(0, len(ids) - a.n_ctx - 1, (a.batch,))
        yield torch.stack([ids[i:i + a.n_ctx] for i in st]).to(DEV)


@torch.no_grad()
def val_loss(m, val, a, n=50):
    m.eval()
    g = torch.Generator().manual_seed(0)
    tot = sum(m.loss(val[s:s + a.n_ctx][None].to(DEV)).item()
              for s in torch.randint(0, len(val) - a.n_ctx - 1, (n,), generator=g))
    m.train()
    return tot / n


def train_lm(m, data, a, steps, lr=None, record=0, val=None):
    """Adam(0.9, 0.95, 1e-8), weight decay 0.1, 175-step-style linear warm-up (scaled), cosine decay."""
    opt = torch.optim.AdamW(m.parameters(), lr or a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    warm = max(1, min(175, steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    it, curve = batches(data, a), []
    for s in range(1, steps + 1):
        loss = m.loss(next(it))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if record and val is not None and s % max(1, steps // record) == 0:
            curve.append((s, val_loss(m, val, a, 20)))
    return curve


# ---------------------------------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample_completions(m, bpe, prompt, n, T, a, top_p=0.95):
    """n samples (batched) with nucleus sampling; stop at Codex's stop sequences; returns [(text, token log-probs)]."""
    m.eval()
    ids = bpe.encode(prompt)[-(a.n_ctx - a.max_new):]
    x = torch.tensor([ids] * n, device=DEV)
    lps = [[] for _ in range(n)]
    done = [False] * n
    out = [[] for _ in range(n)]
    for _ in range(a.max_new):
        lg = m(x[:, -a.n_ctx:])[:, -1]
        logp = F.log_softmax(lg, -1)
        if T == 0:
            nxt = lg.argmax(-1)
        else:
            p = F.softmax(lg / T, -1)
            sp, si = p.sort(-1, descending=True)
            cut = sp.cumsum(-1) - sp > top_p
            sp[cut] = 0
            nxt = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)).squeeze(1)
        x = torch.cat([x, nxt[:, None]], 1)
        for i in range(n):
            if not done[i]:
                out[i].append(int(nxt[i])); lps[i].append(float(logp[i, nxt[i]]))
                if any(s in bpe.decode(out[i][-8:]) for s in STOP):
                    done[i] = True
        if all(done):
            break
    m.train()
    res = []
    for o, lp in zip(out, lps):
        text = truncate(bpe.decode(o))
        k = len(bpe.encode(text)) if text else 0
        res.append((text, lp[:max(k, 1)]))
    return res


def evaluate(m, bpe, problems, a, T, n):
    """For each problem: n samples, their pass/fail, and their log-probs."""
    out = []
    for pr in problems:
        samples = sample_completions(m, bpe, pr["prompt"], n, T, a)
        ok = [check_correctness(pr["prompt"], s, pr["test"], pr["entry_point"], timeout=a.timeout) == "passed"
              for s, _ in samples]
        out.append({"samples": samples, "ok": ok, "canonical": pr["canonical"]})
    return out


# ---------------------------------------------------------------------------------------------------- experiments
def e1(a):
    tr, va, bpe = setup(a)
    rows = []
    for L, d in SHAPES[:a.n_shapes]:
        m, N = make(L, d, len(bpe), a)
        train_lm(m, tr, a, a.steps)
        rows.append((N, val_loss(m, va, a)))
        torch.save(m.state_dict(), HERE / f"code_{L}x{d}.pt")
        print(f"  E1 N={N:.2e} loss={rows[-1][1]:.3f}", flush=True)
    x, y = np.log([r[0] for r in rows]), np.log([r[1] for r in rows])
    slope, icpt = np.polyfit(x, y, 1)
    return {"(non-embedding N, loss)": rows, "alpha": -slope, "N_c": math.exp(icpt / -slope) if slope else None,
            "paper": {"alpha": 0.13, "N_c": 5.92e7}}


def e2(a):
    tr, va, bpe = setup(a)
    L, d = SHAPES[min(3, a.n_shapes - 1)]
    text_ids = torch.tensor(bpe.encode(text_corpus(a)[:a.max_chars]))
    out = {}
    m, _ = make(L, d, len(bpe), a)
    train_lm(m, text_ids, a, a.steps // 2)                                      # 'GPT' pre-training on English
    out["from text-pretrained"] = train_lm(m, tr, a, a.steps, record=10, val=va)
    m, _ = make(L, d, len(bpe), a)
    out["from scratch"] = train_lm(m, tr, a, a.steps, record=10, val=va)
    out["paper"] = "no final improvement from GPT-3 initialisation, but faster convergence"
    return out


def e3(a):
    files = code_corpus(a)
    code = "\n\n".join(files)[:a.max_chars // 10]
    text_bpe = gpt2.ByteBPE(text_corpus(a)[:a.bpe_chars], a.merges)
    plain = len(text_bpe.encode(code))
    ws_tokens = 0
    for chunk in re.split(r"( {2,})", code):
        if not chunk:
            continue
        ws_tokens += math.ceil(len(chunk) / 24) if chunk.startswith("  ") and set(chunk) == {" "} else len(text_bpe.encode(chunk))
    return {"text BPE on code": plain, "text BPE + whitespace-run tokens": ws_tokens,
            "saving": 1 - ws_tokens / plain, "paper": "~30% fewer tokens"}


def synthetic_set(n, blocks, seed):
    r = random.Random(seed)
    return [synthetic_problem(r.choice(blocks) if isinstance(blocks, list) else blocks, r) for _ in range(n)]


def codex_s(a, bpe, base_state, L, d):
    """Supervised fine-tuning on synthetic problems: loss only on the solution tokens, prompts LEFT-padded so the
    first solution tokens line up (Section 4.4); LR 1/10 of pre-training."""
    m, _ = make(L, d, len(bpe), a)
    if base_state is not None:
        m.load_state_dict(base_state)
    train = synthetic_set(a.n_sft, [1, 2, 3], seed=1)
    opt = torch.optim.AdamW(m.parameters(), a.lr / 10, betas=(0.9, 0.95), weight_decay=0.1)
    for s in range(a.sft_steps):
        batch = random.Random(s).sample(train, a.batch)
        P = [bpe.encode(p["prompt"]) for p in batch]; S = [bpe.encode(p["canonical"]) for p in batch]
        lp = max(map(len, P)); ls = max(map(len, S))
        ids = torch.tensor([[0] * (lp - len(p)) + p + s_ + [0] * (ls - len(s_)) for p, s_ in zip(P, S)], device=DEV)
        mask = torch.tensor([[0] * lp + [1] * len(s_) + [0] * (ls - len(s_)) for s_ in S], device=DEV)[:, 1:].float()
        ids, mask = ids[:, -a.n_ctx:], mask[:, -(a.n_ctx - 1):]
        lg = m(ids)[:, :-1]
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="none").view(mask.shape)
        loss = (nll * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def e4(a):
    tr, va, bpe = setup(a)
    L, d = SHAPES[a.n_shapes - 1]
    f = HERE / f"code_{L}x{d}.pt"
    base = torch.load(f, map_location=DEV) if f.exists() else None
    m = codex_s(a, bpe, base, L, d)
    test = synthetic_set(a.n_eval, [1, 2, 3], seed=99)
    out = {"base model loaded": base is not None, "temperature sweep": {}}
    for T in a.temps:
        res = evaluate(m, bpe, test, a, T, a.n_samples)
        out["temperature sweep"][T] = {f"pass@{k}": float(np.mean([pass_at_k(len(r["ok"]), sum(r["ok"]), k) for r in res]))
                                       for k in (1, 10, 100) if k <= a.n_samples}
        print("  E4 T", T, out["temperature sweep"][T], flush=True)
        if T == 0.8:
            sel = {"random": [], "mean log-prob": [], "sum log-prob": [], "oracle": []}
            b_ok, b_bad = [], []
            for r in res:
                pairs = list(zip(r["samples"], r["ok"]))
                sel["random"].append(np.mean(r["ok"]))
                sel["mean log-prob"].append(max(pairs, key=lambda p: np.mean(p[0][1]))[1])
                sel["sum log-prob"].append(max(pairs, key=lambda p: np.sum(p[0][1]))[1])
                sel["oracle"].append(any(r["ok"]))
                for (txt, _), ok in pairs:
                    (b_ok if ok else b_bad).append(bleu(txt, r["canonical"]))
            out["one-sample selection (T=0.8)"] = {k: float(np.mean(v)) for k, v in sel.items()}
            out["BLEU (T=0.8)"] = {"passing mean": float(np.mean(b_ok)) if b_ok else None,
                                   "failing mean": float(np.mean(b_bad)) if b_bad else None,
                                   "fraction of failing samples with BLEU above the passing median":
                                       float(np.mean(np.array(b_bad) > np.median(b_ok))) if b_ok and b_bad else None}
    chain = {}
    for nb in range(1, 8):
        res = evaluate(m, bpe, synthetic_set(a.n_eval, nb, seed=200 + nb), a, a.chain_T, a.n_samples)
        chain[nb] = float(np.mean([np.mean(r["ok"]) for r in res]))
        print("  E4 blocks", nb, chain[nb], flush=True)
    out["pass rate vs chained blocks"] = chain
    out["paper"] = "Figure 11: pass rate falls ~2-3x per extra block; Codex-S best T: 0 for pass@1, 1 for pass@100"
    return out


def e5(a):
    tr, va, bpe = setup(a)
    path = DATA / "HumanEval.jsonl.gz"
    if not path.exists():
        DATA.mkdir(exist_ok=True); urllib.request.urlretrieve(HUMANEVAL_URL, path)
    probs = [json.loads(l) for l in gzip.open(path, "rt")][:a.n_humaneval]
    for p in probs:
        p["canonical"] = p["canonical_solution"]
    out = {}
    for L, d in SHAPES[:a.n_shapes]:
        f = HERE / f"code_{L}x{d}.pt"
        if not f.exists():
            continue
        m, N = make(L, d, len(bpe), a); m.load_state_dict(torch.load(f, map_location=DEV))
        row = {}
        for T, ks in ((0.2, (1,)), (0.8, (10, 100))):
            res = evaluate(m, bpe, probs, a, T, a.n_samples)
            for k in ks:
                if k <= a.n_samples:
                    row[f"pass@{k} (T={T})"] = float(np.mean([pass_at_k(len(r["ok"]), sum(r["ok"]), k) for r in res]))
        out[f"N={N:.1e}"] = row
        print("  E5", N, row, flush=True)
    out["paper"] = "Codex-12M: 2.0 / 3.6 / 8.6 %; Codex-12B: 28.8 / 46.8 / 72.3 %"
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        L += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--site", action="store_true", help="also use site-packages code")
    a = ap.parse_args()
    a.max_files, a.bpe_chars, a.merges, a.max_chars, a.n_books = 100000, 3_000_000, 4000, 500_000_000, 8
    a.n_ctx, a.batch, a.lr, a.steps, a.n_shapes, a.max_new, a.timeout = 512, 32, 6e-4, 20000, 6, 128, 3.0
    a.n_sft, a.sft_steps, a.n_eval, a.n_samples, a.n_humaneval = 20000, 3000, 100, 100, 164
    a.temps, a.chain_T = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2), 0.2
    if a.quick:
        a.max_files, a.bpe_chars, a.merges, a.max_chars, a.n_books = 50, 30_000, 100, 300_000, 1
        a.n_ctx, a.batch, a.steps, a.n_shapes, a.max_new = 128, 4, 20, 2, 24
        a.n_sft, a.sft_steps, a.n_eval, a.n_samples, a.n_humaneval, a.temps = 50, 5, 2, 3, 2, (0.2, 0.8)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
