"""Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei, Wang, Schuurmans, Bosma, Ichter, Xia,
Chi, Le & Zhou, NeurIPS 2022).

  Method: few-shot prompting where each exemplar is <input, CHAIN OF THOUGHT, output>: a series of natural-language
    intermediate steps before the answer. No fine-tuning; 8 hand-written exemplars for the math benchmarks.
  Findings:
    - chain of thought is an EMERGENT ability of scale: it hurts or does nothing below ~100B parameters (small models
      write fluent but illogical chains) and helps a lot above;
    - gains are largest on the hardest, multi-step problems (GSM8K more than doubles for the largest GPT and PaLM)
      and tiny or negative on one-step problems (MAWPS SingleOp);
    - PaLM 540B + 8 CoT exemplars: GSM8K 56.9% (standard 17.9%), beating finetuned GPT-3 + verifier (55%);
    - ablations (Figure 5): 'equation only' helps only when the equation is easy to read off; 'variable compute
      only' (dots as long as the equation) ~ baseline; 'chain of thought AFTER the answer' ~ baseline -> the
      sequential reasoning before the answer matters, not extra tokens or activated knowledge;
    - robust to different annotators, exemplar sets and orders (all far above standard prompting);
    - also commonsense (CSQA, StrategyQA, date/sports understanding, SayCan) and symbolic tasks (last-letter
      concatenation, coin flip), including length generalisation beyond the exemplars for large models.
  Error analysis (LaMDA 137B, GSM8K): of 50 correct answers, 48 had fully correct chains; of 50 wrong answers, 46%
    had minor errors (calculator, symbol mapping, one missing step), 54% major semantic/coherence errors.
"""

import re

# ---------------------------------------------------------------------------------------------------- prompting
EXEMPLARS = [  # Figure 1's exemplar and the first exemplar of Appendix Table 20
    {"question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How "
                 "many tennis balls does he have now?",
     "chain": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11.",
     "answer": "11"},
    {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are "
                 "done, there will be 21 trees. How many trees did the grove workers plant today?",
     "chain": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must "
              "have been 21 - 15 = 6.",
     "answer": "6"},
]


GSM8K_EXEMPLARS = [  # Appendix Table 20 (annotator A): used for every math word problem benchmark except AQuA
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there "
     "will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been "
     "21 - 15 = 6.", "6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.", "5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they "
     "had 74 - 35 = 39.", "39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason "
     "give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.", "8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.", "9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to "
     "thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were "
     "added. 9 + 20 is 29.", "29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls "
     "did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had "
     "35 - 2 = 33 golf balls.", "33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. "
     "23 - 15 is 8.", "8"),
]
GSM8K_EXEMPLARS = [{"question": q, "chain": c, "answer": a} for q, c, a in GSM8K_EXEMPLARS]


def standard_prompt(exemplars, question):
    shots = "".join(f"Q: {e['question']}\nA: The answer is {e['answer']}.\n\n" for e in exemplars)
    return shots + f"Q: {question}\nA:"


def cot_prompt(exemplars, question):
    shots = "".join(f"Q: {e['question']}\nA: {e['chain']} The answer is {e['answer']}.\n\n" for e in exemplars)
    return shots + f"Q: {question}\nA:"


def extract_answer(completion):
    """Take the number after the LAST 'The answer is' (or the last number if absent), commas removed."""
    m = re.findall(r"answer is\s*\$?(-?[\d,]*\.?\d+)", completion)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", completion)
    if not m:
        return None
    x = m[-1].replace(",", "")
    return x[:-2] if x.endswith(".0") else x


# ---------------------------------------------------------------------------------------------------- toy task
FORMATS = ("standard", "chain of thought", "variable compute (dots)", "reasoning after answer")
VOCAB = sorted(set("Q0123456789=;>."))
STOI = {c: i for i, c in enumerate(VOCAB)}


def toy_item(rng, fmt, k):
    """Sum of k digits mod 10. The chain of thought writes the running sums; the ablations keep the same length
    (dots) or put the same steps after the answer."""
    digits = [rng.randint(0, 9) for _ in range(k)]
    run, s = [], 0
    for d in digits:
        s = (s + d) % 10
        run.append(s)
    q = "Q" + "".join(map(str, digits)) + "="
    steps = "".join(map(str, run[:-1]))
    a = {"standard": f"{run[-1]};", "chain of thought": f"{steps}>{run[-1]};",
         "variable compute (dots)": "." * (k - 1) + f">{run[-1]};", "reasoning after answer": f"{run[-1]}>{steps};"}[fmt]
    return q, a, run[-1], run


def toy_answer(fmt, completion):
    if fmt in ("chain of thought", "variable compute (dots)"):
        return completion.split(">")[-1].rstrip(";")[:1] if ">" in completion else None
    return completion[:1]


def chain_is_correct(completion, run):
    """For chain-of-thought outputs: every intermediate running sum is right."""
    steps = completion.split(">")[0]
    return steps == "".join(map(str, run[:-1]))


# ---------------------------------------------------------------------------------------------------- reported numbers
TABLE_2_GSM8K = {  # standard -> chain of thought, solve rate %
    "LaMDA 137B": (6.5, 14.3), "GPT-3 175B (text-davinci-002)": (15.6, 46.9), "Codex (code-davinci-002)": (19.7, 63.1),
    "PaLM 540B": (17.9, 56.9)}
PRIOR_BEST_GSM8K = 55.0                                                       # finetuned GPT-3 175B + verifier
ERROR_ANALYSIS = {"correct answers with correct chains": "48 / 50",
                  "wrong answers: minor errors": 0.46, "wrong answers: major errors": 0.54}


# ---------------------------------------------------------------------------------------------------- toy training
def _llama():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "08-Pretraining-and-Scaling-LLMs" / "056-Touvron-et-al-2023-LLaMA" / "llama.py"
    spec = importlib.util.spec_from_file_location("llama_056", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def train_toy(fmt, steps, k_range=(2, 6), d=64, layers=2, batch=64, ctx=24, seed=0, lr=3e-3):
    """A tiny LLaMA-style model (paper 056) trained on ONE format, loss only on the answer part."""
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
            q, a, _, _ = toy_item(rng, fmt, rng.randint(*k_range))
            t = [STOI[c] for c in q + a]
            X.append(t + [STOI[";"]] * (ctx - len(t)))
            M.append([0] * len(q) + [1] * len(a) + [0] * (ctx - len(t)))
        X = torch.tensor(X); M = torch.tensor(M)[:, 1:].float()
        nll = F.cross_entropy(m(X)[:, :-1].reshape(-1, len(VOCAB)), X[:, 1:].reshape(-1), reduction="none").view(M.shape)
        loss = (nll * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval(), loss.item()


def evaluate_toy(m, fmt, k, n=200, seed=9, ctx=24):
    """Greedy decoding; returns (answer accuracy, fraction of correct answers whose chain is also correct, samples)."""
    import random
    import torch
    rng = random.Random(seed)
    correct = chain_ok = 0
    samples = []
    for _ in range(n):
        q, _, ans, run = toy_item(rng, fmt, k)
        p = [STOI[c] for c in q]
        with torch.no_grad():
            for _ in range(2 * k + 3):
                nxt = int(m(torch.tensor([p[-ctx:]]))[0, -1].argmax()); p.append(nxt)
                if VOCAB[nxt] == ";":
                    break
        out = "".join(VOCAB[i] for i in p)[len(q):]
        ok = toy_answer(fmt, out) == str(ans)
        correct += ok
        if ok and fmt == "chain of thought":
            chain_ok += chain_is_correct(out, run)
        if len(samples) < 2:
            samples.append((q, out, ans))
    return correct / n, (chain_ok / correct if correct and fmt == "chain of thought" else None), samples
