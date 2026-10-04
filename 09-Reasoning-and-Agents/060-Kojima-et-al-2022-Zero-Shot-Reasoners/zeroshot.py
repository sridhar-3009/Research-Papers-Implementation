"""Large Language Models are Zero-Shot Reasoners (Kojima, Gu, Reid, Matsuo & Iwasawa, NeurIPS 2022).

  Zero-shot-CoT: no exemplars at all; add the trigger "Let's think step by step." after "A:" and let the model write
  its reasoning, then ask again for the answer. TWO prompts (Figure 2):
    1st, reasoning extraction:  x' = "Q: [X]\\nA: Let's think step by step."  ->  the model writes z
    2nd, answer extraction:     "[x'] [z] Therefore, the answer (arabic numerals) is"  ->  the model writes y
  The answer trigger depends on the format (numbers, A-E multiple choice, Yes/No, ...). Answer cleansing takes the
  FIRST piece of the output that fits the format ('probably 375 and 376' -> 375).
  Results (text-davinci-002, Table 1): MultiArith 17.7 -> 78.7, GSM8K 10.4 -> 40.7, Last Letter 0.2 -> 57.6,
    Coin Flip 12.8 -> 91.4, Date 49.3 -> 67.5; no gain on easy one-step sets (SingleEq, AddSub) or commonsense
    (CommonsenseQA 68.8 -> 64.6). PaLM 540B: GSM8K 12.5 -> 43.0, 70.1 with self-consistency (Table 2).
  Scaling (Figure 3): like few-shot CoT, the gain appears only for large models.
  Template robustness (Table 4, MultiArith): instructive triggers help (78.7 for "Let's think step by step.",
    45.7-77.3 for others), misleading triggers hurt or do nothing (9.3-18.8), irrelevant ones do nothing (13.1-17.5).
  Zero-shot-CoT < few-shot-CoT (78.7 vs 93.0 on MultiArith) but >> standard few-shot (33.8).
"""

import re
from collections import Counter

TRIGGER = "Let's think step by step."
ANSWER_TRIGGERS = {
    "number": "Therefore, the answer (arabic numerals) is",
    "multiple choice": "Therefore, among A through E, the answer is",
    "yes/no": "Therefore, the answer (Yes or No) is",
    "letters": "Therefore, the answer is",
}

TABLE_4 = [  # (category, template, MultiArith accuracy with text-davinci-002)
    ("instructive", "Let's think step by step.", 78.7), ("instructive", "First,", 77.3),
    ("instructive", "Let's think about this logically.", 74.5),
    ("instructive", "Let's solve this problem by splitting it into steps.", 72.2),
    ("instructive", "Let's be realistic and think step by step.", 70.8),
    ("instructive", "Let's think like a detective step by step.", 70.3), ("instructive", "Let's think", 57.5),
    ("instructive", "Before we dive into the answer,", 55.7), ("instructive", "The answer is after the proof.", 45.7),
    ("misleading", "Don't think. Just feel.", 18.8),
    ("misleading", "Let's think step by step but reach an incorrect answer.", 18.7),
    ("misleading", 'Let\'s count the number of "a" in the question.', 16.7),
    ("misleading", "By using the fact that the earth is round,", 9.3),
    ("irrelevant", "By the way, I found a good restaurant nearby.", 17.5), ("irrelevant", "Abrakadabra!", 15.5),
    ("irrelevant", "It's a beautiful day.", 13.1), ("(zero-shot baseline)", "", 17.7)]
TABLE_1 = {  # zero-shot -> zero-shot-CoT, text-davinci-002, with format-specific answer triggers
    "SingleEq": (74.6, 78.0), "AddSub": (72.2, 69.6), "MultiArith": (17.7, 78.7), "GSM8K": (10.4, 40.7),
    "AQUA": (22.4, 33.5), "SVAMP": (58.8, 62.1), "CommonsenseQA": (68.8, 64.6), "StrategyQA": (12.7, 54.8),
    "Date Understanding": (49.3, 67.5), "Shuffled Objects": (31.3, 52.4), "Last Letter (4 words)": (0.2, 57.6),
    "Coin Flip (4 times)": (12.8, 91.4)}
TABLE_2_MULTIARITH = {"Zero-Shot": 17.7, "Few-Shot (8)": 33.8, "Zero-Shot-CoT": 78.7, "Few-Shot-CoT (8)": 93.0,
                      "Zero-Plus-Few-Shot-CoT (8)": 92.8}
TABLE_2_PALM_GSM8K = {"Zero-Shot": 12.5, "Zero-Shot-CoT": 43.0, "Zero-Shot-CoT + self-consistency": 70.1,
                      "Few-Shot": 17.9, "Few-Shot-CoT": 56.9, "Few-Shot-CoT + self-consistency": 74.4}


# ---------------------------------------------------------------------------------------------------- prompting
def reasoning_prompt(question, trigger=TRIGGER):
    return f"Q: {question}\nA: {trigger}"


def answer_prompt(first_prompt, reasoning, fmt="number"):
    return f"{first_prompt} {reasoning.strip()}\n{ANSWER_TRIGGERS[fmt]}"


def zero_shot_prompt(question, fmt="number"):
    """The zero-shot BASELINE also uses an answer trigger, just without any reasoning first."""
    return f"Q: {question}\nA: {ANSWER_TRIGGERS[fmt].replace('Therefore, t', 'T')}"


def cleanse(text, fmt="number"):
    """Appendix A.6: take the FIRST piece of the output that fits the answer format."""
    if fmt == "number":
        m = re.search(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
        if not m:
            return None
        x = m.group().replace(",", "").rstrip(".")
        return x[:-2] if x.endswith(".0") else x
    if fmt == "multiple choice":
        m = re.search(r"\b[A-E]\b", text)
        return m.group() if m else None
    if fmt == "yes/no":
        m = re.search(r"\b(yes|no)\b", text.lower())
        return m.group() if m else None
    return text.strip().split()[0].strip(".") if text.strip() else None


def zero_shot_cot(generate, question, fmt="number", trigger=TRIGGER):
    """The full two-stage pipeline with any `generate(prompt) -> text` function."""
    x1 = reasoning_prompt(question, trigger)
    z = generate(x1)
    y = generate(answer_prompt(x1, z, fmt))
    return cleanse(y, fmt), z, y


def self_consistency(answers):
    """Wang et al. 2022: sample several reasoning paths, return the most common final answer (None ignored)."""
    votes = Counter(a for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None


# ---------------------------------------------------------------------------------------------------- toy
VOCAB = sorted(set("Q0123456789=;>.TXR"))
STOI = {c: i for i, c in enumerate(VOCAB)}
TOY_TRIGGERS = {"T": "instructive ('let's think step by step')", "R": "irrelevant", "X": "misleading"}


def toy_problem(rng, k):
    digits = [rng.randint(0, 9) for _ in range(k)]
    run, s = [], 0
    for d in digits:
        s = (s + d) % 10
        run.append(s)
    return "Q" + "".join(map(str, digits)) + "=", run


def toy_document(rng, mix=(0.3, 0.6, 0.05, 0.05), k_range=(2, 6)):
    """The 'pre-training corpus': after a question the text continues in one of four ways, each announced by a
    trigger: '>' answer directly; 'T' running sums, then '>' answer (reasoning text); 'R' filler, then the answer
    (irrelevant); 'X' random digits and a random answer (misleading)."""
    q, run = toy_problem(rng, rng.randint(*k_range))
    u = rng.random()
    if u < mix[0]:
        a = f">{run[-1]};"
    elif u < mix[0] + mix[1]:
        a = "T" + "".join(map(str, run[:-1])) + f">{run[-1]};"
    elif u < mix[0] + mix[1] + mix[2]:
        a = "R" + "." * (len(run) - 1) + f">{run[-1]};"
    else:
        a = "X" + "".join(str(rng.randint(0, 9)) for _ in run[:-1]) + f">{rng.randint(0, 9)};"
    return q, a


def _llama():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "08-Pretraining-and-Scaling-LLMs" / "056-Touvron-et-al-2023-LLaMA" / "llama.py"
    spec = importlib.util.spec_from_file_location("llama_056", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def train_toy(steps, mix=(0.3, 0.6, 0.05, 0.05), d=64, layers=2, batch=64, ctx=24, lr=3e-3, seed=0):
    import random
    import torch
    import torch.nn.functional as F
    L = _llama()
    rng = random.Random(seed); torch.manual_seed(seed)
    m = L.LLaMA(vocab=len(VOCAB), d=d, layers=layers, heads=max(1, d // 16), ctx=ctx, hidden=L.ffn_hidden(d, 16))
    opt = torch.optim.AdamW(m.parameters(), lr)
    for _ in range(steps):
        X, M = [], []
        for _ in range(batch):
            q, a = toy_document(rng, mix)
            t = [STOI[c] for c in q + a]
            X.append(t + [STOI[";"]] * (ctx - len(t)))
            M.append([0] * len(q) + [1] * len(a) + [0] * (ctx - len(t)))
        X = torch.tensor(X); M = torch.tensor(M)[:, 1:].float()
        nll = F.cross_entropy(m(X)[:, :-1].reshape(-1, len(VOCAB)), X[:, 1:].reshape(-1), reduction="none").view(M.shape)
        loss = (nll * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval(), loss.item()


def toy_generate(m, prefix, max_new=12, temperature=0.0, gen=None, ctx=24, stop=">;"):
    import torch
    import torch.nn.functional as F
    p = [STOI[c] for c in prefix]
    with torch.no_grad():
        for _ in range(max_new):
            lg = m(torch.tensor([p[-ctx:]]))[0, -1]
            n = int(lg.argmax()) if temperature == 0 else int(torch.multinomial(F.softmax(lg / temperature, -1), 1, generator=gen))
            p.append(n)
            if VOCAB[n] in stop:
                break
    return "".join(VOCAB[i] for i in p)[len(prefix):]


def toy_answer(m, question, trigger=None, temperature=0.0, gen=None):
    """trigger None: zero-shot baseline (the answer trigger '>' right away). Otherwise two stages: reason after the
    trigger until the model writes '>', then append the answer trigger '>' and read one digit."""
    if trigger is None:
        return toy_generate(m, question + ">", 1)[:1], ""
    z = toy_generate(m, question + trigger, temperature=temperature, gen=gen).rstrip(">;")
    return toy_generate(m, question + trigger + z + ">", 1)[:1], z
