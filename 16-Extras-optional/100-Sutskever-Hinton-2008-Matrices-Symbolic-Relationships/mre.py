"""Using Matrices to Model Symbolic Relationships (Sutskever & Hinton, NIPS 2008) -- Matrix Relational Embedding (MRE).

  LRE (Paccanaro & Hinton)  objects are VECTORS A, relations are MATRICES R; R A should be close to every B with
                            (A, B) in R. Cost (Eq. 1), a softmax over all objects C:
                                C = - sum_R sum_{(A,B) in R} log [ exp(-||R A - B||^2) / sum_C exp(-||R A - C||^2) ]
  MRE                       objects AND relations are N x N MATRICES (Frobenius distances), so relations can themselves
                            be arguments: (3, +3) in plus, (+3, +9) in inverse, (has_father, has_mother) in higher_oppsex
  higher-order cost (Eq. 2) the discriminative Eq. 1 only needs R~A to be CLOSER to B than to the alternatives, so the
                            product '3 x plus' need not equal '+3' and does not work as a relation; Eq. 2 asks for
                            equality:  C = sum ||R~ A - B||^2
  learning                  conjugate gradient (scipy) on all cases, unit-Gaussian init, weight decay 0.01 sum w^2

Tasks: modular arithmetic (base 12, relations +0..+11 and x0..x11, 288 propositions); the two isomorphic family trees
(24 people, 12 relations, 112 cases); the higher-order versions where ALL basic examples of one relation are held out
and it must be understood from higher-order propositions alone; and incremental learning of a new relation with
everything else frozen.
"""

import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------------------------------- the model

class MRE:
    """Entities (objects, then relations) are N x N matrices stored in one array E of shape (n_entities, N, N)."""

    def __init__(self, n_objects, n_relations, N=4, seed=0, wd=0.01):
        self.n_obj, self.n_rel, self.N, self.wd = n_objects, n_relations, N, wd
        self.E = np.random.default_rng(seed).standard_normal((n_objects + n_relations, N, N))

    def rel(self, r):
        return self.n_obj + r

    # cost and gradient -------------------------------------------------------------------------
    def cost_grad(self, E, basic, higher, free=None):
        """basic: array of (relation entity, A entity, B object index) -- discriminative Eq. 1 over all objects;
        higher: array of (relation entity, A entity, B entity) -- squared error Eq. 2. Returns cost and dE.
        `free`: if given, only these entities are regularised (the others are frozen by the caller)."""
        G = np.zeros_like(E)
        cost = 0.0
        objs = E[:self.n_obj]
        if len(basic):
            R, A, B = E[basic[:, 0]], E[basic[:, 1]], basic[:, 2]
            M = R @ A                                                   # (n, N, N)
            D2 = ((M[:, None] - objs[None]) ** 2).sum((2, 3))           # ||M - C||^2 for every object C
            logits = -D2
            lse = np.logaddexp.reduce(logits, axis=1)
            cost += float((lse - logits[np.arange(len(B)), B]).sum())
            P = np.exp(logits - lse[:, None])                           # softmax over objects
            onehot = np.eye(self.n_obj)[B]
            # dL/dM = 2 (sum_c p_c C - B);  dL/dC = 2 (p_c - [c = B]) (M - C)
            dM = 2 * (np.einsum("nc,cij->nij", P, objs) - objs[B])
            dC = 2 * np.einsum("nc,ncij->cij", P - onehot, M[:, None] - objs[None])
            G[:self.n_obj] += dC
            np.add.at(G, basic[:, 0], dM @ A.transpose(0, 2, 1))
            np.add.at(G, basic[:, 1], R.transpose(0, 2, 1) @ dM)
        if len(higher):
            R, A, B = E[higher[:, 0]], E[higher[:, 1]], E[higher[:, 2]]
            Dm = R @ A - B
            cost += float((Dm ** 2).sum())
            np.add.at(G, higher[:, 0], 2 * Dm @ A.transpose(0, 2, 1))
            np.add.at(G, higher[:, 1], 2 * R.transpose(0, 2, 1) @ Dm)
            np.add.at(G, higher[:, 2], -2 * Dm)
        reg = E if free is None else E[free]
        cost += self.wd * float((reg ** 2).sum())
        if free is None:
            G += 2 * self.wd * E
        else:
            G[free] += 2 * self.wd * E[free]
        return cost, G

    def fit(self, basic, higher=(), maxiter=2000, free=None):
        """Conjugate gradient on all cases. `free` = list of entity indices to learn (others frozen)."""
        basic = np.asarray(basic, int).reshape(-1, 3)
        higher = np.asarray(higher, int).reshape(-1, 3)
        shape = self.E.shape
        if free is None:
            def f(x):
                c, g = self.cost_grad(x.reshape(shape), basic, higher)
                return c, g.ravel()
            res = minimize(f, self.E.ravel(), jac=True, method="CG", options={"maxiter": maxiter})
            self.E = res.x.reshape(shape)
        else:
            free = np.asarray(free)
            base = self.E.copy()
            def f(x):
                E = base.copy()
                E[free] = x.reshape(len(free), self.N, self.N)
                c, g = self.cost_grad(E, basic, higher, free)
                return c, g[free].ravel()
            res = minimize(f, self.E[free].ravel(), jac=True, method="CG", options={"maxiter": maxiter})
            self.E[free] = res.x.reshape(len(free), self.N, self.N)
        return res

    # queries -------------------------------------------------------------------------------------
    def answer_ranking(self, r_entity, a_entity):
        M = self.E[r_entity] @ self.E[a_entity]
        return np.argsort(((M[None] - self.E[:self.n_obj]) ** 2).sum((1, 2)))

    def correct(self, r_entity, a_entity, b, all_answers):
        """Correct if B is among the k closest objects, k = number of correct answers to (A, ?) in R."""
        return b in self.answer_ranking(r_entity, a_entity)[:len(all_answers)]

# ----------------------------------------------------------------------------------------------- modular arithmetic

def arithmetic_task(base=12, with_mult=True):
    """Objects 0..base-1; relations +0..+(base-1) (and x0..x(base-1)). Returns (relation names, cases) where each case
    is (relation index, a, b)."""
    names = [f"+{k}" for k in range(base)] + ([f"x{k}" for k in range(base)] if with_mult else [])
    cases = [(k, a, (a + k) % base) for k in range(base) for a in range(base)]
    if with_mult:
        cases += [(base + k, a, (a * k) % base) for k in range(base) for a in range(base)]
    return names, cases


def evaluate(model, cases, train_cases=()):
    """Errors on `cases` (relation index, a, b), with multi-answer queries handled as in the paper."""
    answers = {}
    for r, a, b in list(cases) + list(train_cases):
        answers.setdefault((r, a), set()).add(b)
    return sum(not model.correct(model.rel(r), a, b, answers[(r, a)]) for r, a, b in cases)


def basic_array(model, cases):
    return np.array([(model.rel(r), a, b) for r, a, b in cases], int)


def run_arithmetic(n_test, seed=0, N=4, maxiter=2000):
    names, cases = arithmetic_task()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cases))
    test, train = [cases[i] for i in idx[:n_test]], [cases[i] for i in idx[n_test:]]
    m = MRE(12, len(names), N=N, seed=seed)
    m.fit(basic_array(m, train), maxiter=maxiter)
    return evaluate(m, test, train), evaluate(m, train, test)

# ----------------------------------------------------------------------------------------------- family trees

ENGLISH = {"male": ["Christopher", "Andrew", "Arthur", "James", "Charles", "Colin"],
           "female": ["Penelope", "Christine", "Margaret", "Victoria", "Jennifer", "Charlotte"],
           "couples": [("Christopher", "Penelope"), ("Andrew", "Christine"), ("Arthur", "Margaret"),
                       ("James", "Victoria"), ("Charles", "Jennifer")],
           "children": {("Christopher", "Penelope"): ["Arthur", "Victoria"],
                        ("Andrew", "Christine"): ["James", "Jennifer"],
                        ("James", "Victoria"): ["Colin", "Charlotte"]}}
ITALIAN_NAMES = {"Christopher": "Aurelio", "Penelope": "Maria", "Andrew": "Pierino", "Christine": "Grazia",
                 "Arthur": "Bortolo", "Margaret": "Emma", "James": "Pietro", "Victoria": "Giannina",
                 "Charles": "Marcello", "Jennifer": "Doralice", "Colin": "Alberto", "Charlotte": "Mariemma"}
RELATIONS = ["has_husband", "has_wife", "has_son", "has_daughter", "has_father", "has_mother", "has_brother",
             "has_sister", "has_nephew", "has_niece", "has_uncle", "has_aunt"]
OPPOSITE = {"has_husband": "has_wife", "has_son": "has_daughter", "has_father": "has_mother",
            "has_brother": "has_sister", "has_nephew": "has_niece", "has_uncle": "has_aunt"}


def family_facts(tree, rename=None):
    """All (person, relation, answer) facts of one family tree (aunts / uncles include those by marriage)."""
    rn = (lambda x: rename[x]) if rename else (lambda x: x)
    male = set(tree["male"])
    spouse = {}
    for h, w in tree["couples"]:
        spouse[h], spouse[w] = w, h
    parents = {c: list(p) for p, cs in tree["children"].items() for c in cs}
    children = {p: cs for (h, w), cs in tree["children"].items() for p in (h, w)}
    people = tree["male"] + tree["female"]
    sibs = {x: [s for s in children.get(parents[x][0], []) if s != x] if x in parents else [] for x in people}
    facts = []
    for x in people:
        g = lambda ys, m, f: [(y, m if y in male else f) for y in ys]
        rels = []
        if x in spouse:
            rels += g([spouse[x]], "has_husband", "has_wife")
        rels += g(children.get(x, []), "has_son", "has_daughter")
        rels += g(parents.get(x, []), "has_father", "has_mother")
        rels += g(sibs[x], "has_brother", "has_sister")
        nn = [c for s in sibs[x] for c in children.get(s, [])]
        if x in spouse:
            nn += [c for s in sibs[spouse[x]] for c in children.get(s, [])]
        rels += g(nn, "has_nephew", "has_niece")
        ua = []
        for p in parents.get(x, []):
            for s in sibs[p]:
                ua += [s] + ([spouse[s]] if s in spouse else [])
        rels += g(ua, "has_uncle", "has_aunt")
        facts += [(rn(x), r, rn(y)) for y, r in rels]
    return facts


def family_task():
    """Both trees; returns people, cases (relation index, person index, answer index)."""
    facts = family_facts(ENGLISH) + family_facts(ENGLISH, ITALIAN_NAMES)
    people = ENGLISH["male"] + ENGLISH["female"] + [ITALIAN_NAMES[p] for p in ENGLISH["male"] + ENGLISH["female"]]
    pid = {p: i for i, p in enumerate(people)}
    return people, [(RELATIONS.index(r), pid[x], pid[y]) for x, r, y in facts]


def run_family(n_test, seed=0, N=4, maxiter=2000):
    people, cases = family_task()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cases))
    test, train = [cases[i] for i in idx[:n_test]], [cases[i] for i in idx[n_test:]]
    m = MRE(len(people), len(RELATIONS), N=N, seed=seed)
    m.fit(basic_array(m, train), maxiter=maxiter)
    return evaluate(m, test, train), evaluate(m, train, test)

# ----------------------------------------------------------------------------------------------- higher-order tasks

def higher_order_arithmetic(held_out, seed=0, N=4, maxiter=3000, incremental=False, discriminative_higher=False):
    """Basic relations +0..+11 only; higher-order relations plus (n, +n), minus (n, +(-n)), inverse (+a, +(-a)).
    ALL basic examples of +held_out are removed; MRE must answer (x, ?) in +held_out from higher-order facts alone.
    incremental=True: first learn everything except +held_out (with its higher-order facts removed too), freeze, then
    learn only the matrix for +held_out from its higher-order facts. discriminative_higher=True uses Eq. 1-style
    training for higher-order facts instead of Eq. 2 (the version the paper says failed)."""
    base = 12
    names, cases = arithmetic_task(base, with_mult=False)
    hnames = ["plus", "minus", "inverse"]
    m = MRE(base, base + len(hnames), N=N, seed=seed)
    R = lambda k: m.rel(k)
    H = lambda name: m.rel(base + hnames.index(name))
    higher = ([(H("plus"), n, R(n)) for n in range(base)] + [(H("minus"), n, R((-n) % base)) for n in range(base)]
              + [(H("inverse"), R(a), R((-a) % base)) for a in range(base)])
    train = [c for c in cases if c[0] != held_out]
    test = [c for c in cases if c[0] == held_out]
    involves = lambda h: R(held_out) in (h[1], h[2])
    if discriminative_higher:
        # treat higher-order facts as softmax cases over the RELATION matrices (answers are relations)
        m2 = _DiscHigher(m, base)
        m2.fit(basic_array(m, train), np.array(higher), maxiter)
    elif incremental:
        m.fit(basic_array(m, train), [h for h in higher if not involves(h)], maxiter=maxiter)
        m.fit(np.zeros((0, 3), int), [h for h in higher if involves(h)], maxiter=maxiter, free=[R(held_out)])
    else:
        m.fit(basic_array(m, train), higher, maxiter=maxiter)
    return evaluate(m, test, train), len(test)


class _DiscHigher:
    """Ablation: train higher-order facts with a discriminative softmax over all relation matrices (like Eq. 1)."""

    def __init__(self, m, base):
        self.m, self.base = m, base

    def fit(self, basic, higher, maxiter):
        m = self.m
        rel_ents = np.arange(m.n_obj, m.n_obj + self.base)
        shape = m.E.shape

        def f(x):
            E = x.reshape(shape)
            c, g = m.cost_grad(E, basic, np.zeros((0, 3), int))
            R, A, B = E[higher[:, 0]], E[higher[:, 1]], higher[:, 2]
            M = R @ A
            cands = E[rel_ents]
            logits = -((M[:, None] - cands[None]) ** 2).sum((2, 3))
            lse = np.logaddexp.reduce(logits, axis=1)
            bi = B - m.n_obj
            c += float((lse - logits[np.arange(len(bi)), bi]).sum())
            P = np.exp(logits - lse[:, None])
            dM = 2 * (np.einsum("nc,cij->nij", P, cands) - cands[bi])
            dC = 2 * np.einsum("nc,ncij->cij", P - np.eye(self.base)[bi], M[:, None] - cands[None])
            g[rel_ents] += dC
            np.add.at(g, higher[:, 0], dM @ A.transpose(0, 2, 1))
            np.add.at(g, higher[:, 1], R.transpose(0, 2, 1) @ dM)
            return c, g.ravel()
        res = minimize(f, m.E.ravel(), jac=True, method="CG", options={"maxiter": maxiter})
        m.E = res.x.reshape(shape)


def higher_order_family(held_out, seed=0, N=4, maxiter=3000, incremental=False):
    """Basic family relations plus the higher-order relation higher_oppsex (12 facts, e.g. (has_father, has_mother));
    ALL basic facts of `held_out` are removed."""
    people, cases = family_task()
    m = MRE(len(people), len(RELATIONS) + 1, N=N, seed=seed)
    HO = m.rel(len(RELATIONS))
    higher = []
    for a, b in OPPOSITE.items():
        higher += [(HO, m.rel(RELATIONS.index(a)), m.rel(RELATIONS.index(b))),
                   (HO, m.rel(RELATIONS.index(b)), m.rel(RELATIONS.index(a)))]
    k = RELATIONS.index(held_out)
    train = [c for c in cases if c[0] != k]
    test = [c for c in cases if c[0] == k]
    involves = lambda h: m.rel(k) in (h[1], h[2])
    if incremental:
        m.fit(basic_array(m, train), [h for h in higher if not involves(h)], maxiter=maxiter)
        m.fit(np.zeros((0, 3), int), [h for h in higher if involves(h)], maxiter=maxiter, free=[m.rel(k)])
    else:
        m.fit(basic_array(m, train), higher, maxiter=maxiter)
    return evaluate(m, test, train), len(test)


REPORTED = {
    "Table 1 (mod-12 arithmetic, +0..+11 and x0..x11, 4x4 matrices)": "mean held-out errors over 5 runs: 30 held out "
        "-> 0.0; 60 -> 6.8 (runs 29, 4, 0, 1, 0); 90 -> 24.0; no training errors",
    "Table 2 (family trees, 112 cases)": "10 held out -> 0.4; 20 -> 1.2; 30 -> 2.0 mean errors (24 possible answers)",
    "Table 3 (higher-order arithmetic, all basic +k held out, 12 test cases)": "mean errors +1: 1.0, +4: 2.6, +6: 2.8, "
        "+10: 3.6; the discriminative cost for higher-order facts first gave only slightly-better-than-chance results",
    "Table 4 (higher-order family, via higher_oppsex)": "mean errors has_father (12): 2.4, has_aunt (8): 4.0, "
                                                        "has_sister (6): 0.4, has_nephew (8): 1.6",
    "Table 5 (incremental: learn the held-out relation with everything else frozen)": "arithmetic +1: 1.2, +4: 5.8, "
        "+6: 2.6, +10: 4.4; family has_father 2.0, has_aunt 1.6, has_sister 0.0, has_nephew 0.0",
    "setup": "scipy conjugate gradient (default parameters), unit-Gaussian initialisation, weight decay 0.01 sum w^2, "
             "4x4 matrices best for arithmetic",
}
