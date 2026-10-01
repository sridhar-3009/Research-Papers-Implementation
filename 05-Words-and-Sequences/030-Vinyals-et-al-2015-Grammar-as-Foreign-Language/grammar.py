"""A toy probabilistic grammar for fast tests and the demo: trees in Penn Treebank style, with POS tags."""

import random

LEX = {"DT": ["the", "a", "every", "some"], "NN": ["dog", "cat", "man", "park", "telescope", "bird", "house"],
       "JJ": ["big", "old", "red", "small"], "VBZ": ["sees", "likes", "chases", "finds"],
       "VBD": ["slept", "ran", "laughed"], "IN": ["in", "with", "near", "under"], ".": ["."]}


def gen(rng, label="S", depth=0):
    """S -> NP VP . | NP -> DT NN | DT JJ NN | NP PP | VP -> VBZ NP | VBZ NP PP | VBD | PP -> IN NP"""
    if label in LEX:
        return (label, [rng.choice(LEX[label])])
    deep = depth > 3
    if label == "S":
        kids = ["NP", "VP", "."]
    elif label == "NP":
        r = rng.random()
        kids = ["DT", "NN"] if r < 0.5 or deep else ["DT", "JJ", "NN"] if r < 0.75 else ["NP", "PP"]
    elif label == "VP":
        r = rng.random()
        kids = ["VBZ", "NP"] if r < 0.5 or deep else ["VBZ", "NP", "PP"] if r < 0.8 else ["VBD"]
    else:                                                  # PP
        kids = ["IN", "NP"]
    return (label, [gen(rng, k, depth + 1) for k in kids])


def corpus(n, seed=0, max_words=None):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        t = gen(rng)
        from parser import words
        if max_words is None or len(words(t)) <= max_words:
            out.append(t)
    return out
