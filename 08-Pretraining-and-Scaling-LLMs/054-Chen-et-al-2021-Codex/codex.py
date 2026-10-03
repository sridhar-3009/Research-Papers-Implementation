"""Evaluating Large Language Models Trained on Code (Codex; Chen, Tworek, Jun, Yuan, Pinto, Kaplan, Edwards, et al.,
OpenAI 2021).

  Data: 54M public GitHub repositories (May 2020), 179 GB of unique Python files < 1 MB, filtered (auto-generated,
    average line length > 100, max line length > 1000, few alphanumerics) to 159 GB.
  Model: GPT-3 family fine-tuned on that code (12M ... 12B); same LR as the GPT model, 175-step warm-up, cosine decay,
    100B tokens, Adam(0.9, 0.95, 1e-8), weight decay 0.1. Starting from GPT-3 did not improve the final result but
    converged faster. The GPT-3 tokenizer plus extra tokens for RUNS OF WHITESPACE: ~30% fewer tokens for code.
  Test loss: L(N) = (N / 5.92e7)^-0.13 (N = non-embedding parameters).
  HumanEval: 164 hand-written problems (signature + docstring + body, avg 7.7 unit tests), judged by EXECUTION in a
    sandbox (gVisor + eBPF firewall), not by text match (BLEU fails: wrong programs often get higher BLEU).
  pass@k: generate n >= k samples (n = 200), count the c that pass, and estimate
      pass@k = E[ 1 - C(n-c, k) / C(n, k) ]            (Eq. 1, unbiased; Figure 3's stable product form)
    whereas 1 - (1 - c/n)^k is biased low (Appendix A).
  Sampling: nucleus top-p 0.95; stop at '\\nclass', '\\ndef', '\\n#', '\\nif', '\\nprint'. Best temperature grows
    with k (679M model: T* = 0.2 for pass@1, 0.8 for pass@100).
  Picking ONE sample without tests: highest MEAN token log-prob beats random; SUM log-prob is slightly worse than random.
  Codex-12B: 28.8% pass@1, 46.8% pass@10, 72.3% pass@100. Codex-S (fine-tuned on ~10k competitive-programming and
    ~40k CI-traced standalone functions, loss only on the solution): 37.7% pass@1; 77.5% with 100 samples, and 44.5%
    when one sample is chosen by mean log-prob.
  Limits: pass rate drops ~2-3x per extra chained operation in a docstring (13 building blocks, Appendix C).
"""

import math
import multiprocessing as mp
import random
import re
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------------------------------- pass@k
def pass_at_k(n, c, k):
    """Figure 3 verbatim: 1 - prod_{i=n-c+1}^{n} (1 - k/i) = 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def pass_at_k_comb(n, c, k):
    """The same quantity with exact binomial coefficients (big integers; for checking)."""
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_naive(n, c, k):
    """1 - (1 - p_hat)^k: drawing k WITH replacement from the n samples; biased low (Appendix A)."""
    return 1.0 - (1.0 - c / n) ** k


# ---------------------------------------------------------------------------------------------------- execution
def _run(program, q):
    try:
        exec(program, {"__name__": "__candidate__"})
        q.put("passed")
    except AssertionError:
        q.put("failed: assertion")
    except BaseException as e:                                                   # noqa: BLE001
        q.put(f"failed: {type(e).__name__}")


def check_correctness(prompt, completion, test, entry_point, timeout=2.0):
    """Run prompt + completion + tests in a separate process with a timeout (the human-eval harness pattern).
    NOT a security sandbox: the paper used gVisor; here a separate process only protects against hangs/crashes."""
    program = prompt + completion + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    q = ctx.Queue()
    p = ctx.Process(target=_run, args=(program, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.kill(); p.join()
        return "timed out"
    return q.get() if not q.empty() else "failed: crashed"


STOP = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]


def truncate(completion, stops=STOP):
    """Cut the sample at the first stop sequence (otherwise the model keeps writing more functions)."""
    cut = min([completion.find(s) for s in stops if s in completion] or [len(completion)])
    return completion[:cut]


# ---------------------------------------------------------------------------------------------------- sampling
def softmax(logits, T):
    z = np.asarray(logits, float) / max(T, 1e-8)
    z = np.exp(z - z.max())
    return z / z.sum()


def nucleus(probs, top_p=0.95):
    """Keep the smallest set of most-likely options whose total probability reaches top_p, renormalise."""
    order = np.argsort(-probs)
    keep = order[: int(np.searchsorted(np.cumsum(probs[order]), top_p) + 1)]
    out = np.zeros_like(probs); out[keep] = probs[keep]
    return out / out.sum()


def sample(logits, T, top_p=0.95, rng=np.random):
    if T == 0:
        return int(np.argmax(logits))
    p = nucleus(softmax(logits, T), top_p)
    return int(rng.choice(len(p), p=p))


def rank(samples, by):
    """Choose one sample without tests. samples: list of (token log-probs, passed). by: 'mean', 'sum' or 'random'."""
    if by == "random":
        return random.choice(samples)
    key = (lambda s: np.mean(s[0])) if by == "mean" else (lambda s: np.sum(s[0]))
    return max(samples, key=key)


# ---------------------------------------------------------------------------------------------------- tokens
def whitespace_tokens(code, max_run=24):
    """Codex's lexer change: a run of k spaces becomes ONE token ' '*k (k up to max_run), instead of k tokens."""
    out = []
    for piece in re.findall(r" +|\n|\t|[A-Za-z_]+|[0-9]+|[^\sA-Za-z_0-9]", code):
        if piece.startswith(" "):
            out += [" " * max_run] * (len(piece) // max_run) + ([" " * (len(piece) % max_run)] if len(piece) % max_run else [])
        else:
            out.append(piece)
    return out


def plain_tokens(code):
    """The same split, but every space is its own token (what a text tokenizer roughly does to indentation)."""
    return [t for piece in re.findall(r" +|\n|\t|[A-Za-z_]+|[0-9]+|[^\sA-Za-z_0-9]", code)
            for t in (list(piece) if piece.startswith(" ") else [piece])]


# ---------------------------------------------------------------------------------------------------- BLEU
def bleu(candidate, reference, n_max=4):
    """Sentence BLEU-4 on whitespace tokens (geometric mean of clipped n-gram precisions x brevity penalty), with
    add-one smoothing so short programs don't get 0."""
    c, r = candidate.split(), reference.split()
    logs = []
    for n in range(1, n_max + 1):
        cn = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
        rn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        match = sum(min(v, rn[g]) for g, v in cn.items())
        logs.append(math.log((match + 1) / (max(sum(cn.values()), 0) + 1)))
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / max(len(c), 1))
    return bp * math.exp(sum(logs) / n_max)


# ---------------------------------------------------------------------------------------------------- scaling
def code_loss(N):
    """Figure 4: test loss after code fine-tuning, (N / 5.92e7)^-0.13 nats/token."""
    return (N / 5.92e7) ** -0.13


TABLE_1 = {  # HumanEval pass@1 / @10 / @100 (%)
    "GPT-Neo 125M": (0.75, 1.88, 2.97), "GPT-Neo 1.3B": (4.79, 7.47, 16.30), "GPT-Neo 2.7B": (6.41, 11.27, 21.37),
    "GPT-J 6B": (11.62, 15.74, 27.74), "TabNine": (2.58, 4.35, 7.59),
    "Codex-12M": (2.00, 3.62, 8.58), "Codex-25M": (3.21, 7.1, 12.89), "Codex-42M": (5.06, 8.8, 15.55),
    "Codex-85M": (8.22, 12.81, 22.4), "Codex-300M": (13.17, 20.37, 36.27), "Codex-679M": (16.22, 25.7, 40.95),
    "Codex-2.5B": (21.36, 35.42, 59.5), "Codex-12B": (28.81, 46.81, 72.31),
}


# ---------------------------------------------------------------------------------------------------- Appendix C
BUILDING_BLOCKS = [
    ("remove all instances of the letter e from the string", 's = s.replace("e", "")'),
    ("replace all spaces with exclamation points in the string", 's = s.replace(" ", "!")'),
    ("convert the string s to lowercase", "s = s.lower()"),
    ("remove the first and last two characters of the string", "s = s[2:-2]"),
    ("removes all vowels from the string", 's = "".join(char for char in s if char not in "aeiouAEIOU")'),
    ("remove every third character from the string", 's = "".join(char for i, char in enumerate(s) if i % 3 != 0)'),
    ("drop the last half of the string, as computed by characters", "s = s[: len(s) // 2]"),
    ("replace spaces with triple spaces", 's = s.replace(" ", "   ")'),
    ("reverse the order of words in the string", 's = " ".join(s.split()[::-1])'),
    ("drop the first half of the string, as computed by number of words", 's = " ".join(s.split()[len(s.split()) // 2 :])'),
    ("add the word apples after every word in the string", 's = " ".join(word + " apples" for word in s.split())'),
    ("make every other character in the string uppercase",
     's = "".join(char.upper() if i % 2 == 0 else char for i, char in enumerate(s))'),
    ("delete all exclamation points, question marks, and periods from the string", 's = "".join([x for x in s if x not in ".!?"])'),
]


def synthetic_problem(n_blocks, rng):
    """Chain n building blocks: prompt (signature + docstring), reference body, and a unit test on random strings."""
    blocks = rng.sample(BUILDING_BLOCKS, n_blocks)
    doc = "\n".join(f"    -{d}" for d, _ in blocks)
    prompt = ("def string_manipulation(s: str):\n    \"\"\"\n    This function takes a string as input, then returns the "
              f"result of performing\n    the following sequence of manipulations on that string:\n{doc}\n    \"\"\"\n")
    body = "".join(f"    {c}\n" for _, c in blocks) + "    return s\n"
    ns = {}; exec(prompt + body, ns)
    words = ["Hello there", "the quick brown fox. jumps!", "Codex writes code?", "a e i o u", "Apples and pears"]
    cases = [(w, ns["string_manipulation"](w)) for w in words]
    test = "def check(f):\n" + "".join(f"    assert f({a!r}) == {b!r}\n" for a, b in cases)
    return {"prompt": prompt, "canonical": body, "test": test, "entry_point": "string_manipulation", "blocks": blocks}


# ---------------------------------------------------------------------------------------------------- mini HumanEval
MINI_HUMANEVAL = [
    {"prompt": 'def has_close_elements(numbers, threshold):\n    """Return True if any two numbers are closer than threshold."""\n',
     "canonical": "    for i, a in enumerate(numbers):\n        for b in numbers[i + 1:]:\n            if abs(a - b) < threshold:\n                return True\n    return False\n",
     "test": "def check(f):\n    assert f([1.0, 2.0, 3.0], 0.5) is False\n    assert f([1.0, 2.8, 3.0], 0.3) is True\n    assert f([], 1.0) is False\n",
     "entry_point": "has_close_elements"},
    {"prompt": 'def incr_list(l):\n    """Return the list with every element incremented by 1."""\n',
     "canonical": "    return [x + 1 for x in l]\n",
     "test": "def check(f):\n    assert f([1, 2, 3]) == [2, 3, 4]\n    assert f([]) == []\n",
     "entry_point": "incr_list"},
    {"prompt": 'def is_palindrome(text):\n    """Check whether the string reads the same backwards."""\n',
     "canonical": "    return text == text[::-1]\n",
     "test": "def check(f):\n    assert f('aba')\n    assert not f('ab')\n    assert f('')\n",
     "entry_point": "is_palindrome"},
    {"prompt": 'def x_or_y(n, x, y):\n    """Return x if n is a prime number, otherwise return y."""\n',
     "canonical": "    if n < 2:\n        return y\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return y\n    return x\n",
     "test": "def check(f):\n    assert f(7, 34, 12) == 34\n    assert f(15, 8, 5) == 5\n    assert f(1, 2, 0) == 0\n    assert f(2, 2, 0) == 2\n",
     "entry_point": "x_or_y"},
]
