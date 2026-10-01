"""The problems of Jozefowicz et al. (2015), Section 3.5, as character streams.

  memorization      the filter: read 5 symbols (26 possibilities), then reproduce them
  arithmetic        '3e36d9-h1h39f94eeh43keg3c=-13991064.': two numbers (up to 8 digits, either sign) with
                    random distractor letters between characters; after '=' predict the answer digit by digit
  xml               synthetic nested XML tags of 2-10 lowercase letters
  VOCAB             40 symbols: 26 lowercase letters, 10 digits, '+', '-', '=', '.' (plus '<', '>', '/' for XML)
"""

import random

LETTERS = "abcdefghijklmnopqrstuvwxyz"
VOCAB = LETTERS + "0123456789+-=.<>/ "
IDX = {c: i for i, c in enumerate(VOCAB)}


def encode(s):
    return [IDX[c] for c in s]


def memorization(rng=random, n=5):
    """'5 symbols in sequence (with 26 possibilities) are to be read in sequence and then reproduced'.
    Returns (string, index of the first answer character): the model is judged on the answer part."""
    syms = "".join(rng.choice(LETTERS) for _ in range(n))
    s = syms + "=" + syms + "."
    return s, n + 1


def arithmetic(rng=random, max_digits=8, distractor_p=0.3):
    """'The RNN is required to compute the digits of the sum or difference of two numbers ... we introduced
    a random number of distractor symbols between successive input characters.'"""
    a = rng.randint(-(10 ** max_digits - 1), 10 ** max_digits - 1)
    b = rng.randint(0, 10 ** max_digits - 1)
    op = rng.choice("+-")
    question = f"{a}{op}{b}"
    noisy = ""
    for ch in question:
        noisy += ch
        while rng.random() < distractor_p:
            noisy += rng.choice(LETTERS)
    answer = str(a + b if op == "+" else a - b)
    s = noisy + "=" + answer + "."
    return s, len(noisy) + 1


def xml(rng=random, max_steps=50):
    """'With 50% probability or after 50 iterations, we either close the most recently opened tag, or stop if no
    open tags remain. Otherwise, open a new random tag' (tags of 2-10 lowercase characters)."""
    out, stack = [], []
    for it in range(1000):
        if rng.random() < 0.5 or it >= max_steps:
            if not stack:
                break
            out.append(f"</{stack.pop()}>")
        else:
            tag = "".join(rng.choice(LETTERS) for _ in range(rng.randint(2, 10)))
            stack.append(tag)
            out.append(f"<{tag}>")
    return "".join(out), 0


def xml_is_well_formed(s):
    stack, i = [], 0
    while i < len(s):
        j = s.index(">", i)
        tag = s[i + 1:j]
        if tag.startswith("/"):
            if not stack or stack.pop() != tag[1:]:
                return False
        else:
            stack.append(tag)
        i = j + 1
    return not stack


TASKS = {"memorization": memorization, "arithmetic": arithmetic, "xml": xml}
