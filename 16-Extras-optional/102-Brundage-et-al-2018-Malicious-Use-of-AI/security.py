"""The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation (Brundage, Avin, Clark, Toner,
Eckersley, Garfinkel, Dafoe, Scharre, Zeitzoff, Filar, Anderson, Roff, Allen, Steinhardt, Flynn, O hEigeartaigh,
Beard, Belfield, Farquhar, Lyle, Crootof, Evans, Page, Bryson, Yampolskiy, Amodei; 2018).

The report is a policy analysis, not an algorithm. This file implements, at toy scale and from the DEFENDER's side, the
two parts of its argument that can be made quantitative:

  1  'expansion of existing threats': AI makes labour-intensive tasks efficient and scalable, which 'alleviates the
     tradeoff between the scale and efficacy of attacks'. An abstract cost-benefit model (no operational content):
     attackers choose, per target, between a cheap low-success generic attempt and an expensive high-success tailored
     one; automation lowers the tailoring cost. We measure how many targets become worth a tailored attempt, the
     expected harm, how many (budget-limited) actors become capable -- and how much a defence must lower success
     rates to undo the change.
  2  'today's AI systems suffer from novel unresolved vulnerabilities': data poisoning and adversarial examples, shown on
     a small digit classifier together with standard defences (loss-based data sanitisation, adversarial training) and
     wrapped into a 'red-team report' -- the report's recommendation to import practices such as red teaming from
     computer security.
"""

import numpy as np

# ----------------------------------------------------------------------------------------------- 1. attack economics

def attack_economics(tailoring_cost, n_targets=10000, generic_cost=0.01, p_generic=0.002, p_tailored=0.05,
                     seed=0, defence_factor=1.0):
    """Targets have values v ~ lognormal. For each target the (rational, abstract) attacker takes the better of
      generic:  p_generic * v - generic_cost        tailored:  p_tailored * v - tailoring_cost      or nothing.
    `defence_factor` multiplies both success probabilities (e.g. 0.5 = defences halve success).
    Returns the share of targets attacked at all, the share given a tailored attempt, total expected harm, and the
    attacker's total cost."""
    v = np.random.default_rng(seed).lognormal(mean=2.0, sigma=1.5, size=n_targets)
    pg, pt = p_generic * defence_factor, p_tailored * defence_factor
    g = pg * v - generic_cost
    t = pt * v - tailoring_cost
    choice = np.where((t > g) & (t > 0), 2, np.where(g > 0, 1, 0))
    harm = np.where(choice == 2, pt * v, np.where(choice == 1, pg * v, 0)).sum()
    cost = np.where(choice == 2, tailoring_cost, np.where(choice == 1, generic_cost, 0)).sum()
    return {"attacked": float((choice > 0).mean()), "tailored": float((choice == 2).mean()),
            "expected harm": float(harm), "attacker cost": float(cost)}


def capable_actors(tailoring_cost, budgets, min_tailored_profit=0.0, **kw):
    """Share of actors (with the given budgets) who can afford at least one profitable tailored attempt: the report's
    'expand the set of actors who can carry out particular attacks'."""
    return float(np.mean(np.asarray(budgets) >= tailoring_cost))


def defence_needed(tailoring_cost_new, tailoring_cost_old, **kw):
    """The success-rate multiplier a defence must achieve so that expected harm with cheap tailoring falls back to the
    level it had with expensive tailoring (bisection)."""
    target = attack_economics(tailoring_cost_old, **kw)["expected harm"]
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if attack_economics(tailoring_cost_new, defence_factor=mid, **kw)["expected harm"] > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

# ----------------------------------------------------------------------------------------------- 2. AI vulnerabilities

def load_digits():
    from sklearn.datasets import load_digits as ld
    d = ld()
    X, y = d.data / 16.0, d.target
    idx = np.random.default_rng(0).permutation(len(y))
    return X[idx[:1300]], y[idx[:1300]], X[idx[1300:]], y[idx[1300:]]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train_softmax(X, y, epochs=200, lr=0.5, l2=1e-3, adv_eps=0.0, seed=0):
    """Multinomial logistic regression by gradient descent; with adv_eps > 0, each step also trains on FGSM versions of
    the batch made against the current model (adversarial training)."""
    W, b = np.zeros((X.shape[1], 10)), np.zeros(10)
    Y = np.eye(10)[y]
    for _ in range(epochs):
        Xs, Ys = X, Y
        if adv_eps > 0:
            Xa = fgsm(W, b, X, y, adv_eps)
            Xs, Ys = np.vstack([X, Xa]), np.vstack([Y, Y])
        P = softmax(Xs @ W + b)
        W -= lr * (Xs.T @ (P - Ys) / len(Xs) + l2 * W)
        b -= lr * (P - Ys).mean(0)
    return W, b


def accuracy(W, b, X, y):
    return float(((X @ W + b).argmax(1) == y).mean())


def fgsm(W, b, X, y, eps):
    """Fast gradient sign method: move each pixel by eps in the direction that increases the loss (clipped to [0, 1])."""
    P = softmax(X @ W + b)
    grad = (P - np.eye(10)[y]) @ W.T                                   # d loss / d input for a linear model
    return np.clip(X + eps * np.sign(grad), 0, 1)


def poison_labels(y, frac, rng, targeted=False):
    """Label flipping: `frac` of the training labels changed -- at random, or (targeted) all relabelled 7 -> 1 first."""
    y = y.copy()
    n = int(frac * len(y))
    if targeted:
        cand = np.where(y == 7)[0]
        idx = rng.choice(cand, min(n, len(cand)), replace=False)
        y[idx] = 1
    else:
        idx = rng.choice(len(y), n, replace=False)
        y[idx] = (y[idx] + rng.integers(1, 10, n)) % 10
    return y


def sanitize(X, y, drop_frac):
    """A standard defence: train, drop the `drop_frac` training points with the highest loss (likely mislabelled),
    retrain on the rest."""
    W, b = train_softmax(X, y)
    loss = -np.log(softmax(X @ W + b)[np.arange(len(y)), y] + 1e-12)
    keep = np.argsort(loss)[: int((1 - drop_frac) * len(y))]
    return train_softmax(X[keep], y[keep]), keep


def red_team_report(Xtr, ytr, Xte, yte, eps=(0.05, 0.1, 0.2), poison=(0.1, 0.3), seed=0):
    """Run the vulnerability tests the report names against a model trained normally and one hardened by adversarial
    training, and return a small table -- the analogue of red teaming an ML system before deployment."""
    rng = np.random.default_rng(seed)
    out = {}
    base = train_softmax(Xtr, ytr)
    hard = train_softmax(Xtr, ytr, adv_eps=0.1)
    out["clean accuracy"] = {"standard": accuracy(*base, Xte, yte), "adversarially trained": accuracy(*hard, Xte, yte)}
    for e in eps:
        out[f"FGSM eps={e}"] = {"standard": accuracy(*base, fgsm(*base, Xte, yte, e), yte),
                                "adversarially trained": accuracy(*hard, fgsm(*hard, Xte, yte, e), yte)}
    for f in poison:
        yp = poison_labels(ytr, f, rng)
        Wp = train_softmax(Xtr, yp)
        (Ws, keep) = sanitize(Xtr, yp, f)
        out[f"random label flips {int(f * 100)}%"] = {"no defence": accuracy(*Wp, Xte, yte),
                                                      "loss-based sanitisation": accuracy(*Ws, Xte, yte)}
    yt = poison_labels(ytr, 1.0, rng, targeted=True)                   # every training 7 relabelled as 1
    Wt = train_softmax(Xtr, yt)
    sevens = yte == 7
    (Wts, _) = sanitize(Xtr, yt, 0.1)
    out["targeted: all training 7s labelled 1"] = {
        "overall accuracy": accuracy(*Wt, Xte, yte),
        "test 7s classified as 1": float(((Xte[sevens] @ Wt[0] + Wt[1]).argmax(1) == 1).mean()),
        "... after loss-based sanitisation": float(((Xte[sevens] @ Wts[0] + Wts[1]).argmax(1) == 1).mean())}
    return out


REPORTED = {
    "authors": "26 authors from Future of Humanity Institute, Centre for the Study of Existential Risk, OpenAI, "
               "Electronic Frontier Foundation, Center for a New American Security, and others (February 2018)",
    "three changes to the threat landscape": "expansion of existing threats (lower attack costs -> more actors, higher "
                                             "rate, more targets); introduction of new threats (tasks impractical for "
                                             "humans; exploiting vulnerabilities of defenders' AI systems); change to the "
                                             "typical character of threats (more effective, finely targeted, difficult "
                                             "to attribute, exploiting AI vulnerabilities)",
    "three domains": "digital security (e.g. automating labour-intensive cyberattacks such as spear phishing, speech "
                     "synthesis for impersonation, automated hacking, adversarial examples and data poisoning); physical "
                     "security (e.g. drones and autonomous weapons, subverting cyber-physical systems, swarms); political "
                     "security (surveillance, persuasion, deception -- e.g. manipulated videos)",
    "properties of AI": "dual-use; efficient and scalable; can exceed human capabilities; can increase anonymity and "
                        "psychological distance; rapid diffusion; novel unresolved vulnerabilities (data poisoning, "
                        "adversarial examples, flawed goal design)",
    "four high-level recommendations": "(1) policymakers collaborate closely with technical researchers; (2) "
                                       "researchers take the dual-use nature of their work seriously; (3) import best "
                                       "practices from fields such as computer security; (4) expand the range of "
                                       "stakeholders and domain experts involved",
    "four priority research areas": "learning from and with the cybersecurity community (red teaming, formal "
                                    "verification, responsible disclosure of AI vulnerabilities, security tools, secure "
                                    "hardware); exploring different openness models (pre-publication risk assessment, "
                                    "central access licensing, sharing regimes); promoting a culture of responsibility; "
                                    "developing technological and policy solutions",
}
