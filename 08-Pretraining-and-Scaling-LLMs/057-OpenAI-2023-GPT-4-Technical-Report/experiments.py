"""The GPT-4 report's methods, run for real at small scale (nothing here reproduces GPT-4: its details are withheld).

Models: the LLaMA-style decoder of paper 056, byte-level, trained on Gutenberg books (downloaded) or this repository's
markdown as a fallback.

  E1  Section 3.1: train a family over ~1000x of compute; fit L(C) = a C^b + c on all but the largest, WRITE the
      prediction to disk (pre-registration), then train the largest and compare.
  E2  Section 3.2: capability on addition problems: per problem, sample n answers per model and estimate its pass
      rate; keep problems every model solves at least once; fit -mean log pass rate = alpha C^-k on the smaller
      models and predict the largest; report per difficulty bucket (number of digits).
  E3  Figure 3: Hindsight Neglect items (expected value vs outcome) scored few-shot by every model in E1 via the
      likelihood of ' Y' vs ' N': does accuracy go down, up, or U-shaped with size?
  E4  Figure 8: calibration of a base model on a 4-choice synthetic task (ECE of its answer probabilities), then a
      REINFORCE fine-tune that rewards correct answers (an RLHF stand-in); ECE and accuracy before vs after.
  E5  Appendix C: plant eval items in the training text (verbatim, re-formatted, paraphrased); detection rate and
      false-positive rate of the 3 x 50-character substring check vs substring length and number of samples.

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import importlib.util
import json
import math
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gpt4 import (eligible_problems, expected_calibration_error, fit_capability, fit_power_law_offset,
                  hindsight_neglect_item, is_contaminated, mean_log_pass_rate, predict)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
spec = importlib.util.spec_from_file_location("llama_056", HERE.parent / "056-Touvron-et-al-2023-LLaMA" / "llama.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
BOOKS = [1342, 84, 11, 2701, 1661, 98, 1952, 345, 2600, 4300]


def corpus(a):
    try:
        d = DATA / "gutenberg"; d.mkdir(parents=True, exist_ok=True)
        parts = []
        for b in BOOKS[:a.n_books]:
            f = d / f"{b}.txt"
            if not f.exists():
                urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt", f)
            parts.append(f.read_text(encoding="utf-8", errors="ignore"))
        return "\n\n".join(parts)
    except Exception:
        return "\n".join(p.read_text(errors="ignore") for p in sorted(HERE.parents[1].rglob("*.md")))


def make(d, layers, a):
    torch.manual_seed(0)
    m = L.LLaMA(vocab=256, d=d, layers=layers, heads=max(1, d // 32), ctx=a.ctx, hidden=L.ffn_hidden(d, 32)).to(DEV)
    N = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(("tok", "out")))
    return m, N


def train(m, ids, a, steps, lr):
    opt = torch.optim.AdamW(m.parameters(), lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: L.lr_schedule(s, 1.0, warmup=max(1, steps // 20), total=steps))
    for _ in range(steps):
        i = torch.randint(0, len(ids) - a.ctx - 1, (a.batch,))
        loss = m.loss(torch.stack([ids[j:j + a.ctx] for j in i]).to(DEV))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    return m


@torch.no_grad()
def val_loss(m, ids, a, n=100):
    m.eval()
    g = torch.Generator().manual_seed(0)
    v = sum(m.loss(ids[s:s + a.ctx][None].to(DEV)).item() for s in torch.randint(0, len(ids) - a.ctx - 1, (n,), generator=g)) / n
    m.train()
    return v


def family(a):
    """(d, layers, steps): compute grows ~4x per member; tokens per parameter kept roughly constant."""
    return [(d, l, int(a.tokens_per_param * (12 * l * d * d) / (a.batch * a.ctx))) for d, l in a.shapes]


# ---------------------------------------------------------------------------------------------------- E1
def e1(a):
    ids = torch.tensor(list(corpus(a)[:a.max_chars].encode()))
    cut = int(0.98 * len(ids)); tr, va = ids[:cut], ids[cut:]
    rows = []
    fam = family(a)
    for i, (d, l, steps) in enumerate(fam):
        if i == len(fam) - 1:                                                   # pre-register before the big run
            C = np.array([r[0] for r in rows]); Lv = np.array([r[1] for r in rows])
            params = fit_power_law_offset(C / C.max(), Lv)
            m, N = make(d, l, a)
            C_big = 6 * N * steps * a.batch * a.ctx
            registered = {"fit (a, b, c) on normalised compute": params, "C_big / C_max_small": C_big / C.max(),
                          "predicted loss": float(predict(params, C_big / C.max()))}
            (HERE / "e1_preregistered.json").write_text(json.dumps(registered, default=float))
            print("  E1 registered", registered, flush=True)
        m, N = make(d, l, a)
        train(m, tr, a, steps, a.lr * (64 / d) ** 0.5)                         # mild LR scaling with width
        rows.append((6 * N * steps * a.batch * a.ctx, val_loss(m, va, a), N))
        torch.save(m.state_dict(), HERE / f"e1_{d}x{l}.pt")
        print("  E1", d, l, steps, rows[-1], flush=True)
    reg = json.loads((HERE / "e1_preregistered.json").read_text())
    return {"runs (C, loss, N)": rows, "preregistered": reg, "actual largest loss": rows[-1][1],
            "relative error": reg["predicted loss"] / rows[-1][1] - 1}


# ---------------------------------------------------------------------------------------------------- E2
def addition_text(n, rng, max_digits):
    out = []
    for _ in range(n):
        k = rng.randint(1, max_digits)
        x, y = rng.randint(0, 10 ** k - 1), rng.randint(0, 10 ** k - 1)
        out.append(f"{x}+{y}={x + y}\n")
    return "".join(out)


@torch.no_grad()
def pass_counts(m, problems, a):
    m.eval()
    counts = []
    for x, y in problems:
        prompt = torch.tensor([list(f"{x}+{y}=".encode())] * a.n_samples, device=DEV)
        out = prompt
        for _ in range(len(str(x + y)) + 1):
            lg = m(out[:, -a.ctx:])[:, -1] / a.temperature
            out = torch.cat([out, torch.multinomial(F.softmax(lg, -1), 1)], 1)
        ans = [bytes(r[prompt.shape[1]:].tolist()).decode(errors="ignore").split("\n")[0] for r in out.cpu()]
        counts.append(sum(s == str(x + y) for s in ans))
    m.train()
    return counts


def e2(a):
    rng = random.Random(0)
    ids = torch.tensor(list(addition_text(a.n_add, rng, a.max_digits).encode()))
    problems = [(rng.randint(10 ** (k - 1), 10 ** k - 1), rng.randint(10 ** (k - 1), 10 ** k - 1))
                for k in range(1, a.max_digits + 1) for _ in range(a.per_digit)]
    Cs, all_counts = [], []
    for d, l, steps in family(a):
        m, N = make(d, l, a)
        train(m, ids, a, steps, a.lr * (64 / d) ** 0.5)
        Cs.append(6 * N * steps * a.batch * a.ctx); all_counts.append(pass_counts(m, problems, a))
        print("  E2", d, l, np.mean(all_counts[-1]) / a.n_samples, flush=True)
    counts = np.array(all_counts)
    keep = eligible_problems(counts)
    if len(keep) == 0:
        return {"note": "no problem solved by every model; increase n_samples or training"}
    metric = np.array([mean_log_pass_rate(c[keep] / a.n_samples) for c in counts])
    Cn = np.array(Cs) / Cs[-1]
    alpha, k = fit_capability(Cn[:-1], metric[:-1])
    by_digits = {}
    for nd in range(1, a.max_digits + 1):
        sel = [i for i in keep if len(str(problems[i][0])) == nd]
        if sel:
            mm = np.array([mean_log_pass_rate(c[sel] / a.n_samples) for c in counts])
            try:
                al, kk = fit_capability(Cn[:-1], mm[:-1])
                by_digits[nd] = {"predicted": al, "actual": float(mm[-1])}
            except Exception:
                pass
    return {"eligible problems": len(keep), "metric per model": metric.tolist(), "alpha, k": [alpha, k],
            "predicted for largest": alpha, "actual for largest": float(metric[-1]), "by digits (buckets)": by_digits}


# ---------------------------------------------------------------------------------------------------- E3
@torch.no_grad()
def e3(a):
    rng = random.Random(1)
    shots = []
    for _ in range(a.shots):
        t, lab, _ = hindsight_neglect_item(rng)
        shots.append(f"{t} Answer: {lab}\n")
    items = [hindsight_neglect_item(rng) for _ in range(a.n_items)]
    out = {}
    for d, l, _ in family(a):
        f = HERE / f"e1_{d}x{l}.pt"
        if not f.exists():
            continue
        m, N = make(d, l, a); m.load_state_dict(torch.load(f, map_location=DEV)); m.eval()
        correct = 0
        for t, lab, _ in items:
            prefix = list(("".join(shots) + f"{t} Answer: ").encode())[-(a.ctx - 1):]
            lp = F.log_softmax(m(torch.tensor([prefix], device=DEV))[0, -1], -1)
            pred = "Y" if lp[ord("Y")] > lp[ord("N")] else "N"
            correct += pred == lab
        out[f"N={N:.1e}"] = correct / len(items)
        print("  E3", N, out[f"N={N:.1e}"], flush=True)
    out["note"] = "byte-level scoring of the next character after 'Answer: '; chance = 0.5"
    return out


# ---------------------------------------------------------------------------------------------------- E4
def mc_items(n, rng):
    """4-choice questions 'x+y? a b c d' with exactly one correct sum."""
    items = []
    for _ in range(n):
        x, y = rng.randint(10, 99), rng.randint(10, 99)
        opts = list({x + y, x + y + rng.choice([-10, 10]), x + y + rng.choice([-1, 1]), x + y + rng.choice([-2, 2, 9, -9])})
        while len(opts) < 4:
            opts.append(opts[-1] + 3)
        rng.shuffle(opts)
        items.append((f"{x}+{y}? A {opts[0]} B {opts[1]} C {opts[2]} D {opts[3]} Answer: ", "ABCD"[opts.index(x + y)]))
    return items


def mc_probs(m, items):
    P = []
    for q, _ in items:
        lg = m(torch.tensor([list(q.encode())], device=DEV))[0, -1]
        P.append(F.softmax(lg[[ord(c) for c in "ABCD"]], -1))
    return torch.stack(P)


def e4(a):
    rng = random.Random(2)
    train_items, test_items = mc_items(a.n_mc, rng), mc_items(a.n_mc_test, rng)
    text = "".join(q + ans + "\n" for q, ans in train_items)
    ids = torch.tensor(list(text.encode()))
    d, l, _ = family(a)[-2]
    m, _ = make(d, l, a)
    train(m, ids, a, a.mc_steps, a.lr)
    y = torch.tensor(["ABCD".index(ans) for _, ans in test_items], device=DEV)

    def measure():
        with torch.no_grad():
            P = mc_probs(m, test_items)
        conf, pred = P.max(-1)
        return {"accuracy": float((pred == y).float().mean()),
                "ECE": expected_calibration_error(conf.cpu().numpy(), (pred == y).cpu().numpy())}
    before = measure()
    opt = torch.optim.Adam(m.parameters(), a.lr / 10)
    for s in range(a.rl_steps):                                                  # REINFORCE on 'is the answer correct'
        batch = random.Random(s).sample(train_items, a.batch)
        P = mc_probs(m, batch)
        choice = torch.multinomial(P, 1).squeeze(1)
        reward = (choice == torch.tensor(["ABCD".index(ans) for _, ans in batch], device=DEV)).float()
        loss = -((reward - reward.mean()) * torch.log(P.gather(1, choice[:, None]).squeeze(1) + 1e-9)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = measure()
    return {"base model": before, "after reward fine-tuning": after, "report": {"pre-trained": 0.007, "PPO": 0.074}}


# ---------------------------------------------------------------------------------------------------- E5
def e5(a):
    rng = random.Random(3)
    text = corpus(a)[:a.max_chars // 10]
    sents = [s.strip() for s in text.replace("\n", " ").split(".") if 120 < len(s.strip()) < 400]
    rng.shuffle(sents)
    evals, rest = sents[:a.n_eval], sents[a.n_eval:]
    planted = []
    kinds = {}
    for i, e in enumerate(evals):
        kind = ["verbatim", "reformatted", "paraphrase-ish", "absent"][i % 4]
        kinds[i] = kind
        if kind == "verbatim":
            planted.append(e)
        elif kind == "reformatted":
            planted.append(e.upper().replace(" ", "  ").replace(",", " ;"))
        elif kind == "paraphrase-ish":
            planted.append(" ".join("the" if j % 7 == 0 else w for j, w in enumerate(e.split())))  # every 7th word changed
    train_texts = [" ".join(rest[i:i + 50]) for i in range(0, len(rest), 50)] + planted
    out = {}
    for length in (20, 50, 100):
        for k in (1, 3, 10):
            hits = {kd: [] for kd in ("verbatim", "reformatted", "paraphrase-ish", "absent")}
            for i, e in enumerate(evals):
                hits[kinds[i]].append(is_contaminated(e, train_texts, random.Random(i), k=k, length=length))
            out[f"length {length}, k {k}"] = {kd: float(np.mean(v)) for kd, v in hits.items()}
            print("  E5", length, k, out[f"length {length}, k {k}"], flush=True)
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
    a.n_books, a.max_chars, a.ctx, a.batch, a.lr, a.tokens_per_param = 10, 30_000_000, 256, 32, 3e-3, 40
    a.shapes = [(32, 2), (48, 2), (64, 3), (96, 4), (128, 4), (192, 6), (256, 8)]
    a.n_add, a.max_digits, a.per_digit, a.n_samples, a.temperature = 300_000, 5, 40, 200, 1.0
    a.shots, a.n_items, a.n_mc, a.n_mc_test, a.mc_steps, a.rl_steps, a.n_eval = 4, 400, 20000, 1000, 3000, 500, 400
    if a.quick:
        a.n_books, a.max_chars, a.ctx, a.batch, a.tokens_per_param = 1, 300_000, 64, 4, 0.5
        a.shapes = [(16, 1), (24, 1), (32, 1)]
        a.n_add, a.max_digits, a.per_digit, a.n_samples = 500, 2, 2, 4
        a.shots, a.n_items, a.n_mc, a.n_mc_test, a.mc_steps, a.rl_steps, a.n_eval = 1, 4, 50, 8, 3, 2, 8
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
