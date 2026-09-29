"""The two tasks of the paper: mirror symmetry (Figure 1) and family trees (Figures 2-4)."""

from itertools import product

import numpy as np


# ---------------------------------------------------------------------------
# Figure 1: is a 6-bit input vector symmetrical about its centre?
# ---------------------------------------------------------------------------

def symmetry_data(n=6):
    """All 2^6 = 64 input vectors, and target 1 if the vector reads the same
    backwards (e.g. 101101), else 0. Only 8 of the 64 are symmetrical."""
    X = np.array(list(product([0, 1], repeat=n)), dtype=float)
    d = np.array([[float(np.array_equal(x, x[::-1]))] for x in X])
    return X, d


# ---------------------------------------------------------------------------
# Figures 2-4: two isomorphic family trees
# ---------------------------------------------------------------------------

ENGLISH = ["Christopher", "Penelope", "Andrew", "Christine", "Margaret", "Arthur",
           "Victoria", "James", "Jennifer", "Charles", "Colin", "Charlotte"]
ITALIAN = ["Roberto", "Maria", "Pierro", "Francesca", "Gina", "Emilio",
           "Lucia", "Marco", "Angela", "Tomaso", "Alfonso", "Sophia"]
PEOPLE = ENGLISH + ITALIAN                     # 24 people; Italian i matches English i
RELATIONS = ["father", "mother", "husband", "wife", "son", "daughter",
             "uncle", "aunt", "brother", "sister", "nephew", "niece"]

# One tree, written with the English names (Figure 2, top). The Italian tree has
# exactly the same shape: swap each name for the Italian name at the same index.
_MALE = {"Christopher", "Andrew", "Arthur", "James", "Charles", "Colin"}
_COUPLES = [("Christopher", "Penelope"), ("Andrew", "Christine"), ("Margaret", "Arthur"),
            ("Victoria", "James"), ("Jennifer", "Charles")]
_CHILDREN = {("Christopher", "Penelope"): ["Arthur", "Victoria"],
             ("Andrew", "Christine"): ["James", "Jennifer"],
             ("Victoria", "James"): ["Colin", "Charlotte"]}


def _english_facts():
    """Every (person1, relation, person2) true in the English tree."""
    male = lambda p: p in _MALE
    spouse, parents = {}, {}
    for a, b in _COUPLES:
        spouse[a], spouse[b] = b, a
    for couple, kids in _CHILDREN.items():
        for k in kids:
            parents[k] = couple
    siblings = lambda p: [k for k in _CHILDREN.get(parents.get(p), []) if k != p] if p in parents else []

    facts = set()
    for p in ENGLISH:
        if p in spouse:
            facts.add((p, "husband" if male(spouse[p]) else "wife", spouse[p]))
        if p in parents:
            for par in parents[p]:
                facts.add((p, "father" if male(par) else "mother", par))
        for couple, kids in _CHILDREN.items():
            if p in couple:
                for k in kids:
                    facts.add((p, "son" if male(k) else "daughter", k))
        for s in siblings(p):
            facts.add((p, "brother" if male(s) else "sister", s))
        # uncles/aunts: the parents' siblings and those siblings' spouses
        if p in parents:
            for par in parents[p]:
                for s in siblings(par):
                    for relative in [s] + ([spouse[s]] if s in spouse else []):
                        facts.add((p, "uncle" if male(relative) else "aunt", relative))
    # nephews/nieces are the reverse of uncles/aunts
    for p, rel, q in list(facts):
        if rel in ("uncle", "aunt"):
            facts.add((q, "nephew" if male(p) else "niece", p))
    return facts


def family_facts():
    """All true triples in BOTH trees (the Italian tree mirrors the English one)."""
    to_italian = dict(zip(ENGLISH, ITALIAN))
    eng = _english_facts()
    ita = {(to_italian[a], r, to_italian[b]) for a, r, b in eng}
    return sorted(eng | ita)


def family_cases():
    """Group the triples into training CASES: one per (person1, relation) that has
    an answer. The target switches on EVERY correct person2 (Colin has two aunts,
    so that case has two output units on, as in Figure 3).

    Returns (person1 one-hot (n, 24), relation one-hot (n, 12), targets (n, 24),
    list of (person1, relation))."""
    answers = {}
    for a, r, b in family_facts():
        answers.setdefault((a, r), []).append(b)
    keys = sorted(answers)
    P = np.zeros((len(keys), 24)); R = np.zeros((len(keys), 12)); D = np.zeros((len(keys), 24))
    for n, (a, r) in enumerate(keys):
        P[n, PEOPLE.index(a)] = 1
        R[n, RELATIONS.index(r)] = 1
        for b in answers[(a, r)]:
            D[n, PEOPLE.index(b)] = 1
    return P, R, D, keys


# The five-layer network of Figure 3: 24 people -> 6, 12 relations -> 6,
# both -> 12 central units -> 6 -> 24 output people.
FAMILY_NET = [("person", 24, []), ("relation", 12, []),
              ("person_code", 6, ["person"]), ("relation_code", 6, ["relation"]),
              ("central", 12, ["person_code", "relation_code"]),
              ("penultimate", 6, ["central"]),
              ("out", 24, ["penultimate"])]
