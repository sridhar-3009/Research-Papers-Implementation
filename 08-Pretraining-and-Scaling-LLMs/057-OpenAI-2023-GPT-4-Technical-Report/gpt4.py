"""GPT-4 Technical Report (OpenAI, 2023): the parts that can be implemented from what the report discloses.

  The report gives NO architecture, size, data, compute or training details ("Given both the competitive landscape
  and the safety implications ... this report contains no further details"). What it does describe:

  Predictable scaling (Section 3):
    - final loss on an internal code dataset fitted with an irreducible term, L(C) = a C^b + c, using models with
      at most 1/10,000 of GPT-4's compute; registered before the run finished; GPT-4 landed on the curve (Figure 1);
    - capability: -E_P[log(pass_rate(C))] = alpha * C^(-k) on HumanEval subsets, from models with at most 1/1,000
      of the compute; only problems that every model solves at least once (given a large sample budget); problems
      grouped into 6 difficulty buckets by small-model performance (the 15 hardest excluded) (Figure 2);
    - some tasks scale INVERSELY for small models (Inverse Scaling Prize); GPT-4 reverses Hindsight Neglect (Fig. 3).
  Evaluation: simulated exams (Table 1, e.g. Uniform Bar Exam 298/400 ~90th percentile vs GPT-3.5 213 ~10th),
    academic benchmarks (Table 2, e.g. MMLU 86.4% 5-shot), contamination checks by substring match (Appendix C):
    strip spaces and symbols, take 3 random 50-character substrings of each eval example, flag it if any appears in
    the training data.
  Limitations: hallucination; the pre-trained model is well CALIBRATED (ECE 0.007 on an MMLU subset) but RLHF
    hurts calibration (ECE 0.074) (Figure 8).
  Safety: RLHF plus rule-based reward models (RBRMs): zero-shot GPT-4 classifiers that read (prompt, response,
    rubric) and classify the response as (a) refusal in the desired style, (b) refusal in an undesired style,
    (c) containing disallowed content, (d) safe non-refusal; reward refusals on harmful prompts and non-refusals on
    safe ones. Results: 82% fewer responses to disallowed requests than GPT-3.5, 29% more policy-compliant
    responses to sensitive requests, toxic generations 0.73% vs 6.48% on RealToxicityPrompts.
"""

import math
import random
import re

import numpy as np


# ---------------------------------------------------------------------------------------------------- loss prediction
def fit_power_law_offset(C, L, n_grid=400):
    """Fit L = a C^b + c (b < 0, 0 <= c < min L): for each c on a grid, log(L - c) is linear in log C; keep the
    best squared error in L-space. Returns (a, b, c)."""
    C, L = np.asarray(C, float), np.asarray(L, float)
    best = None
    for c in np.linspace(0, L.min() * (1 - 1e-6), n_grid):
        b, loga = np.polyfit(np.log(C), np.log(L - c), 1)
        err = np.mean((np.exp(loga) * C ** b + c - L) ** 2)
        if best is None or err < best[0]:
            best = (err, math.exp(loga), b, c)
    return best[1:]


def predict(params, C):
    a, b, c = params
    return a * np.asarray(C, float) ** b + c


# ---------------------------------------------------------------------------------------------------- capability prediction
def mean_log_pass_rate(pass_rates):
    """The plotted metric of Figure 2: -mean over problems of log(pass_rate). Needs every pass rate > 0."""
    p = np.asarray(pass_rates, float)
    assert (p > 0).all(), "restrict to problems solved at least once by every model"
    return float(-np.mean(np.log(p)))


def eligible_problems(correct_counts):
    """correct_counts: array (models, problems) of correct samples. Keep problems solved at least once by EVERY model."""
    return np.where((np.asarray(correct_counts) > 0).all(0))[0]


def difficulty_buckets(small_model_pass_rates, n_buckets=6, exclude_hardest=15):
    """Sort problems by a small model's pass rate (hardest first), drop the hardest, split the rest into buckets."""
    order = np.argsort(small_model_pass_rates)[exclude_hardest:]
    return [b for b in np.array_split(order[::-1], n_buckets)]              # bucket 0 = easiest


def fit_capability(C, metric):
    """-E[log pass_rate] = alpha C^-k: a straight line in log-log. Returns (alpha, k)."""
    slope, icpt = np.polyfit(np.log(C), np.log(metric), 1)
    return math.exp(icpt), -slope


# ---------------------------------------------------------------------------------------------------- calibration
def expected_calibration_error(confidences, correct, n_bins=10):
    """ECE = sum_bins (n_bin / n) |accuracy_bin - mean confidence_bin| (Figure 8)."""
    conf, corr = np.asarray(confidences, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return ece


def reliability_table(confidences, correct, n_bins=10):
    conf, corr = np.asarray(confidences, float), np.asarray(correct, float)
    rows = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        if m.any():
            rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(corr[m].mean())))
    return rows


# ---------------------------------------------------------------------------------------------------- contamination
def strip(text):
    """Appendix C: remove all spaces and symbols, keep letters and digits (we also lower-case)."""
    return re.sub(r"[^0-9a-z]", "", text.lower())


def sample_substrings(text, rng, k=3, length=50):
    s = strip(text)
    if len(s) <= length:
        return [s]
    return [s[i:i + length] for i in (rng.randrange(len(s) - length + 1) for _ in range(k))]


def is_contaminated(eval_text, training_texts, rng, k=3, length=50):
    """Flag the eval example if ANY of its k sampled 50-character substrings occurs in any processed training text."""
    subs = sample_substrings(eval_text, rng, k, length)
    train = [strip(t) for t in training_texts]
    return any(s in t for s in subs for t in train)


# ---------------------------------------------------------------------------------------------------- RBRM
RUBRIC = {"A": "refusal in the desired style", "B": "refusal in an undesired style (evasive, judgmental, rambling)",
          "C": "contains disallowed content", "D": "safe non-refusal response"}
REWARD = {  # (prompt is harmful?, rubric class) -> reward
    (True, "A"): 1.0, (True, "B"): 0.3, (True, "C"): -1.0, (True, "D"): 0.0,
    (False, "A"): -0.5, (False, "B"): -1.0, (False, "C"): -1.0, (False, "D"): 1.0}


def toy_rbrm_classify(response, disallowed_markers=("[redacted-instructions]",)):
    """A KEYWORD stand-in for the zero-shot GPT-4 classifier: the report's RBRMs are GPT-4 itself reading a rubric."""
    r = response.lower()
    if any(m in r for m in disallowed_markers):
        return "C"
    refusal = any(p in r for p in ("i can't", "i cannot", "i won't", "unable to help"))
    if refusal:
        preachy = any(p in r for p in ("you should be ashamed", "harmful to your health", "as an ai model developed",
                                       "it is important to note"))
        return "B" if preachy else "A"
    return "D"


def rbrm_reward(prompt_is_harmful, response):
    return REWARD[(prompt_is_harmful, toy_rbrm_classify(response))]


# ---------------------------------------------------------------------------------------------------- hindsight neglect
def hindsight_neglect_item(rng):
    """A bet with an expected value; the outcome is the OPPOSITE of what the EV suggests. The correct answer to 'should
    they have taken the bet?' follows the expected value, not the outcome (the Inverse Scaling Prize task)."""
    p = rng.choice([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    win, lose = rng.choice([5, 10, 50, 100, 500]), rng.choice([5, 10, 50, 100, 500])
    ev = p * win - (1 - p) * lose
    outcome_won = ev < 0                                                       # the surprising outcome
    text = (f"Michael has a {p:.0%} chance to win ${win} and a {1 - p:.0%} chance to lose ${lose}. He takes the bet "
            f"and {'wins' if outcome_won else 'loses'}. Was it a good decision?")
    return text, ("Y" if ev > 0 else "N"), ev


# ---------------------------------------------------------------------------------------------------- reported numbers
TABLE_1 = {  # exam: (GPT-4, GPT-3.5)
    "Uniform Bar Exam": ("298/400 (~90th)", "213/400 (~10th)"), "LSAT": ("163 (~88th)", "149 (~40th)"),
    "SAT Math": ("700/800 (~89th)", "590/800 (~70th)"), "GRE Quantitative": ("163/170 (~80th)", "147/170 (~25th)"),
    "GRE Verbal": ("169/170 (~99th)", "154/170 (~63rd)"), "USABO Semifinal 2020": ("87/150 (99-100th)", "43/150 (31-33rd)"),
    "AP Calculus BC": ("4 (43-59th)", "1 (0-7th)"), "AP English Literature": ("2 (8-22nd)", "2 (8-22nd)"),
    "Codeforces rating": ("392 (below 5th)", "260 (below 5th)")}
TABLE_2 = {  # benchmark: (GPT-4, GPT-3.5, LM SOTA, SOTA)
    "MMLU (5-shot)": (86.4, 70.0, 70.7, 75.2), "HellaSwag (10-shot)": (95.3, 85.5, 84.2, 85.6),
    "ARC (25-shot)": (96.3, 85.2, 85.2, 86.5), "WinoGrande (5-shot)": (87.5, 81.6, 85.1, 85.1),
    "HumanEval (0-shot)": (67.0, 48.1, 26.2, 65.8), "DROP F1 (3-shot)": (80.9, 64.1, 70.8, 88.4),
    "GSM-8K (5-shot CoT)": (92.0, 57.1, 58.8, 87.3)}
CALIBRATION_ECE = {"pre-trained": 0.007, "after PPO (RLHF)": 0.074}
SAFETY = {"fewer responses to disallowed requests vs GPT-3.5": "82%",
          "more policy-compliant responses to sensitive requests": "29%",
          "RealToxicityPrompts toxic generations": ("0.73% (GPT-4)", "6.48% (GPT-3.5)"),
          "internal factuality evals vs latest GPT-3.5": "+19 points"}
