"""Rosenblatt's photoperceptron (1958), as described in the paper.

    S-units (retina)  -->  A-units (association cells)  -->  R-units (responses)
     on/off points        random wiring, fixed threshold      compete; winner responds

- Each A-unit has x excitatory and y inhibitory connections to RANDOM retina
  points and fires if (lit excitatory) - (lit inhibitory) >= theta.
  This wiring never changes.
- Each A-unit belongs to the "source-set" of one response (the sets don't
  overlap). What learning changes is each A-unit's VALUE V: how strongly its
  impulses count.
- A response's strength is the sum (Sigma-system) or the mean (mu-system) of the
  values of its ACTIVE source-set units. The stronger response wins; the loser
  is inhibited (the paper's feedback rule b).

Reinforcement ("value dynamics", Table 1):
  alpha: every active unit of the reinforced source-set gains +1 and keeps it.
  gamma: active units gain +1, the INACTIVE units of the same source-set lose
         the same total, so a source-set's total value never changes.
Bivalent (trial and error): the machine answers; the source-set of the correct
  response gains on its active units, the wrong response's active units lose.
"""

import numpy as np


class Photoperceptron:
    def __init__(self, n_points, n_assoc, n_resp=2, x=5, y=5, theta=1, rng=None):
        """n_points: retina size (number of S-points). n_assoc: N_A, number of A-units.
        n_resp: N_R, number of responses. x, y, theta: A-unit wiring and threshold."""
        rng = np.random.default_rng(rng)
        self.exc = rng.integers(0, n_points, (n_assoc, x))    # excitatory origin points
        self.inh = rng.integers(0, n_points, (n_assoc, y))    # inhibitory origin points
        self.theta = theta
        # Disjoint source-sets: each A-unit feeds exactly one response.
        self.source = rng.integers(0, n_resp, n_assoc)
        self.n_resp = n_resp
        self.V = np.zeros(n_assoc)                            # all units start equal
        self.alive = np.ones(n_assoc, bool)                   # for "remove part of the brain"
        self.rng = rng

    # ---- The predominant phase: which A-units fire ----

    def activate(self, S):
        """S: (n, n_points) 0/1 stimuli. Returns (n, N_A) 0/1 A-unit activity."""
        S = np.asarray(S, dtype=np.int32)
        net = S[:, self.exc].sum(-1) - S[:, self.inh].sum(-1)
        return (net >= self.theta) & self.alive

    # ---- The postdominant phase: which response wins ----

    def strengths(self, A, mode="mu"):
        """Strength of every response for every stimulus, shape (n, N_R).

        mode="sigma": sum of the values of the active units in the source-set.
        mode="mu":    their mean (less affected by how many units happen to fire).
        """
        out = np.zeros((len(A), self.n_resp))
        for r in range(self.n_resp):
            mask = A & (self.source == r)
            total = mask @ self.V
            if mode == "mu":
                count = mask.sum(1)
                total = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
            out[:, r] = total
        return out

    def respond(self, S, mode="mu"):
        """The winning response for each stimulus (ties broken at random)."""
        st = self.strengths(self.activate(S), mode)
        noise = self.rng.random(st.shape) * 1e-9               # random tie-break
        return (st + noise).argmax(1)

    # ---- Learning ----

    def reinforce(self, a, r, system="gamma", amount=1.0):
        """Reinforce response r for one stimulus whose A-activity is a (N_A,)."""
        src = self.source == r
        active = a & src
        if system == "alpha":
            self.V[active] += amount
        elif system == "gamma":
            n_on, n_all = active.sum(), src.sum()
            if 0 < n_on < n_all:
                self.V[active] += amount
                self.V[src & ~a] -= amount * n_on / (n_all - n_on)   # pay for it
        else:
            raise ValueError(system)

    def train_forced(self, S, labels, system="gamma"):
        """'Forced' learning series: for each stimulus, the experimenter makes the
        correct response occur, and its active source-set units are reinforced."""
        A = self.activate(S)
        for a, r in zip(A, labels):
            self.reinforce(a, r, system)

    def train_bivalent(self, S, labels, epochs=1, mode="mu"):
        """Trial-and-error learning with reward and punishment (bivalent system).
        Positive reinforcement goes to the correct response's active units,
        negative to the other responses' active units. Returns errors per epoch."""
        A = self.activate(S)
        errors = []
        for _ in range(epochs):
            wrong = 0
            for k in self.rng.permutation(len(S)):
                a, r = A[k], labels[k]
                guess = (self.strengths(a[None], mode)[0] + self.rng.random(self.n_resp) * 1e-9).argmax()
                wrong += guess != r
                self.V[a & (self.source == r)] += 1
                self.V[a & (self.source != r)] -= 1
            errors.append(int(wrong))
        return errors

    # ---- Testing ----

    def p_correct(self, S, labels, mode="mu"):
        """Probability of the correct response: P_r if S are the training stimuli,
        P_g if they are new stimuli from the same classes. Ties count as half."""
        st = self.strengths(self.activate(S), mode)
        idx = np.arange(len(S))
        correct = st[idx, labels]
        st_other = st.copy()
        st_other[idx, labels] = -np.inf
        best_other = st_other.max(1)
        return float(np.mean((correct > best_other) + 0.5 * (correct == best_other)))
