"""The GPT-4 report's methods in a few seconds: predict a bigger model's loss from smaller real models with
L(C) = a C^b + c; the capability metric -E[log pass_rate] (simulated under the report's own hypothesis); a trend that
reverses (inverse scaling); calibration (ECE); the substring contamination check; a toy rule-based reward model; and
the reported numbers. Nothing about GPT-4 itself can be reproduced: the report withholds size, data and compute."""

import importlib.util
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from gpt4 import (CALIBRATION_ECE, SAFETY, TABLE_1, TABLE_2, difficulty_buckets, eligible_problems,
                  expected_calibration_error, fit_capability, fit_power_law_offset, hindsight_neglect_item,
                  is_contaminated, mean_log_pass_rate, predict, rbrm_reward, reliability_table, toy_rbrm_classify)

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------------------------------- 1
section("1. Predictable scaling of the loss (Section 3.1): fit small runs, predict the big one BEFORE seeing it")
spec = importlib.util.spec_from_file_location("llama_056", HERE.parent / "056-Touvron-et-al-2023-LLaMA" / "llama.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
text = "\n".join(p.read_text(errors="ignore") for p in sorted(HERE.parents[1].rglob("EXPLAINED.md")))[:600_000]
data = torch.tensor(list(text.encode()))
cut = int(0.95 * len(data)); tr, va = data[:cut], data[cut:]
CTX, B = 64, 16
rows = []
for d, steps in [(8, 60), (12, 90), (16, 130), (24, 190), (32, 280), (48, 420)]:
    torch.manual_seed(0)
    m = L.LLaMA(vocab=256, d=d, layers=2, heads=2, ctx=CTX, hidden=L.ffn_hidden(d, 8))
    N = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(("tok", "out")))
    opt = torch.optim.AdamW(m.parameters(), 1e-2, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: L.lr_schedule(s, 1.0, warmup=10, total=steps))
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        i = torch.randint(0, len(tr) - CTX - 1, (B,), generator=g)
        loss = m.loss(torch.stack([tr[j:j + CTX] for j in i]))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    gv = torch.Generator().manual_seed(1)
    with torch.no_grad():
        v = sum(m.loss(torch.stack([va[j:j + CTX] for j in torch.randint(0, len(va) - CTX - 1, (B,), generator=gv)])).item()
                for _ in range(8)) / 8
    rows.append((6 * N * steps * B * CTX, v, N))
C = np.array([r[0] for r in rows]); Lv = np.array([r[1] for r in rows])
Cn = C / C[-1]                                                               # normalised so the 'big' run is 1
params = fit_power_law_offset(Cn[:-1], Lv[:-1])
for (c, v, N), cn in zip(rows, Cn):
    print(f"  N = {N:6,} non-embedding params, compute {cn:8.4f} (normalised): val loss {v:.3f} nats/byte")
pred = float(predict(params, 1.0))
print(f"  fit on the 5 smaller runs: L(C) = {params[0]:.3f} C^{params[1]:.3f} + {params[2]:.3f}")
print(f"  predicted loss of the largest run: {pred:.3f}; actual {Lv[-1]:.3f} (error {pred - Lv[-1]:+.3f}, {abs(pred / Lv[-1] - 1):.1%})")
print(f"  -> a {C[-1] / C[-2]:.1f}x extrapolation from real tiny runs lands within ~{abs(pred / Lv[-1] - 1):.0%}; the report extrapolates")
print("     10,000x and is much more accurate, because they built a training stack with predictable behaviour at every")
print("     scale. Our tiny runs use one hand-picked learning rate and very few steps, so they are less 'properly trained'.")

# ---------------------------------------------------------------------------------------------------- 2
section("2. Predicting a capability (Section 3.2): -E[log pass_rate] = alpha C^-k (simulated)")
rng = np.random.default_rng(0)
n_prob, n_samples = 60, 1000
alpha_j = rng.lognormal(0.0, 1.0, n_prob)                                    # each problem's own difficulty
Cs = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1.0])                                  # the last one plays 'GPT-4'
true_p = np.exp(-np.outer(Cs ** -0.25, alpha_j) * 0.05)                       # -log p_j = 0.05 alpha_j C^-0.25
counts = rng.binomial(n_samples, true_p)                                     # c correct out of n samples
small = counts[:-1]
keep = eligible_problems(small)
est = small[:, keep] / n_samples
metric = np.array([mean_log_pass_rate(p) for p in est])
alpha, k = fit_capability(Cs[:-1], metric)
gpt4_actual = mean_log_pass_rate(np.maximum(counts[-1, keep], 1) / n_samples)
print(f"  {len(keep)} of {n_prob} problems are solved at least once by every small model (the others are dropped)")
for c_, m_, e_ in zip(Cs[:-1], metric, est):
    print(f"    compute {c_:7.0e}: -mean log pass rate {m_:.3f}, mean pass@1 {e_.mean():.3f}")
print(f"  fit: alpha = {alpha:.3f}, k = {k:.3f} (true k = 0.25); prediction at C = 1: {alpha:.3f} vs measured {gpt4_actual:.3f}")
print(f"  -> -mean log pass rate is a straight line in log-log, so it extrapolates 1000x; the mean pass@1 instead bends")
print(f"     toward 1 (measured {(counts[-1, keep] / n_samples).mean():.3f} at C = 1), which is much harder to extrapolate.")
print("     (Simulated under the report's own power-law hypothesis, so success here only shows the METHOD works.)")
b = difficulty_buckets(est[0], n_buckets=6, exclude_hardest=min(15, len(keep) // 4))
print(f"  difficulty buckets by the smallest model: sizes {[len(x) for x in b]} (easiest first)")

section("3. ...but some trends reverse (Section 3.2, Figure 3: Hindsight Neglect)")
item, label, ev = hindsight_neglect_item(random.Random(3))
print(f"  {item}\n  expected value {ev:+.1f} -> correct answer '{label}' (judge the decision, not the outcome)")
print("  small models answer by the outcome, and get WORSE with scale (inverse scaling); GPT-4 reverses the trend.")
print("  An extrapolation from ada -> babbage -> curie would have predicted the opposite of what GPT-4 does.")

# ---------------------------------------------------------------------------------------------------- 4
section("4. Calibration (Figure 8): pre-trained ECE 0.007 vs RLHF ECE 0.074 (simulated)")
conf = rng.uniform(0.25, 1.0, 5000)                                         # 4-choice MMLU: chance confidence 0.25
correct = rng.random(5000) < conf                                           # a perfectly calibrated model
over = np.clip(0.25 + 1.4 * (conf - 0.25), 0.25, 1.0)                      # an overconfident model, same answers
print(f"  calibrated model:    ECE {expected_calibration_error(conf, correct):.3f}")
print(f"  overconfident model: ECE {expected_calibration_error(over, correct):.3f}   (same answers, inflated confidence)")
print("  reliability (overconfident): " + "; ".join(f"conf {c:.2f} -> acc {a:.2f}" for _, _, _, c, a in reliability_table(over, correct)[-4:]))
print(f"  reported: {CALIBRATION_ECE}")

# ---------------------------------------------------------------------------------------------------- 5
section("5. Contamination check (Appendix C): 3 random 50-character substrings after stripping spaces and symbols")
question = ("A 30-year-old man presents with fever, a stiff neck, and photophobia for two days. Which test should be "
            "performed first to confirm the most likely diagnosis?")
docs = {"copied with different formatting": "forum: a 30 year old man presents with FEVER; a stiff-neck & photophobia "
                                            "for two days. Which test should be performed first, to confirm the most "
                                            "likely diagnosis?? answers below",
        "paraphrased": "A man aged thirty has had a fever, neck stiffness and light sensitivity for 2 days; what test first?",
        "unrelated": "How to cook pasta: boil salted water, add the pasta, stir."}
for name, doc in docs.items():
    print(f"  training doc {name:34s} -> contaminated: {is_contaminated(question, [doc], random.Random(0))}")
print("  -> formatting can't hide a copy, but a paraphrase slips through (a false negative the report acknowledges)")

# ---------------------------------------------------------------------------------------------------- 6
section("6. Rule-based reward model (Section 6): classify (prompt, response) with a rubric, reward accordingly")
cases = [(True, "I can't help with making weapons, but if you're worried about safety I can point you to resources."),
         (True, "As an AI model developed by OpenAI, I cannot. It is important to note this is illegal and harmful."),
         (True, "Sure. Step 1: [redacted-instructions]"),
         (False, "Cheaper options include buying in bulk or duty-free shops. Smoking is harmful; quitting helps."),
         (False, "I cannot help with that. It is important to note that smoking is harmful to your health.")]
for harmful, resp in cases:
    print(f"  {'harmful' if harmful else 'allowed'} prompt, response class {toy_rbrm_classify(resp)} -> reward "
          f"{rbrm_reward(harmful, resp):+.1f}: {resp[:60]}...")
print("  (the report's RBRMs are zero-shot GPT-4 classifiers reading a rubric; ours is a keyword stand-in)")

# ---------------------------------------------------------------------------------------------------- 7
section("7. Reported results")
for k_, (a, b_) in list(TABLE_1.items())[:6]:
    print(f"  {k_:22s} GPT-4 {a:20s} GPT-3.5 {b_}")
for k_, (g4, g35, lm, sota) in TABLE_2.items():
    print(f"  {k_:20s} GPT-4 {g4:5.1f}  GPT-3.5 {g35:5.1f}  best LM {lm:5.1f}  SOTA {sota:5.1f}")
for k_, v in SAFETY.items():
    print(f"  {k_}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
