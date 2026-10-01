"""Chapter 3 of Sutskever's thesis (2013): the Recurrent Temporal Restricted Boltzmann Machine (RTRBM),
from Sutskever, Hinton & Taylor (2009). Binary units, NumPy, gradients by hand.

At every time step there is an RBM over the visible frame v_t whose HIDDEN BIAS depends on the past:
    b_t = b_h + W' r_{t-1}
and the past is summarized by the deterministic mean-field state
    r_t = sigmoid(W v_t + b_h + W' r_{t-1})                      (Eq. 3.20, inference = Algorithm 4)
So log P(v_1..v_T) = sum_t log P_RBM(v_t | hidden bias b_t)       (Eq. 3.22)

  infer             Algorithm 4: compute r_1..r_T (exact posterior, a point mass)
  sample            Algorithm 3: v_t ~ RBM(. | b_t) with Gibbs sampling, then r_t <- mean-field
  gradient          BPTT through r (Eqs. 3.23-3.25); each RBM term's gradient by CD-k, or EXACTLY
                    (by enumerating all states) for tiny models
  rbm_log_prob_exact, rbm_grad_exact   enumeration helpers (tiny RBMs only)
"""

import itertools

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class RTRBM:
    def __init__(self, n_visible, n_hidden, init_std=0.005, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.H = n_visible, n_hidden
        self.W = rng.normal(0, 0.01, (n_hidden, n_visible))           # visible-to-hidden (RBM weights)
        self.Wp = rng.normal(0, init_std, (n_hidden, n_hidden))       # W': hidden-to-hidden
        self.bv = rng.normal(0, init_std, n_visible)
        self.bh = rng.normal(0, init_std, n_hidden)
        self.r0 = rng.normal(0, init_std, n_hidden)                   # the initial state h_0
        self.rng = rng

    def params(self):
        return [self.W, self.Wp, self.bv, self.bh, self.r0]

    # ---- Algorithm 4: inference ----
    def infer(self, vs):
        """r_t = sigmoid(W v_t + b_h + W' r_{t-1}); returns r_0..r_T, shape (T+1, H)."""
        r = [self.r0]
        for v in vs:
            r.append(sigmoid(self.W @ v + self.bh + self.Wp @ r[-1]))
        return np.array(r)

    def hidden_biases(self, rs):
        """The dynamic hidden bias of the RBM at each step: b_t = b_h + W' r_{t-1}, t = 1..T."""
        return self.bh[None, :] + rs[:-1] @ self.Wp.T

    # ---- one RBM's gradient ----
    def cd_grad(self, v, b_hid, k):
        """CD-k for an RBM with weights W, visible bias bv, hidden bias b_hid, at data v.
        Returns (dW, dbv, db_hid): estimates of d log P(v) / d(.)."""
        ph0 = sigmoid(self.W @ v + b_hid)
        h = (self.rng.random(self.H) < ph0).astype(float)
        vk = v
        for _ in range(k):
            pv = sigmoid(self.W.T @ h + self.bv)
            vk = (self.rng.random(self.V) < pv).astype(float)
            phk = sigmoid(self.W @ vk + b_hid)
            h = (self.rng.random(self.H) < phk).astype(float)
        return np.outer(ph0, v) - np.outer(phk, vk), v - vk, ph0 - phk

    def exact_grad(self, v, b_hid):
        """The exact d log P(v) / d(W, bv, b_hid) of a tiny RBM, by enumerating all visible states."""
        return rbm_grad_exact(self.W, self.bv, b_hid, v)

    # ---- Eqs. 3.23-3.25: the full gradient ----
    def gradient(self, vs, k=10, exact=False):
        """d log P(v_1..v_T) / d params. Each time step contributes an RBM gradient (CD-k or exact)
        w.r.t. W, bv and its hidden bias b_t; b_t = b_h + W' r_{t-1} then passes the bias gradient
        into W', b_h and back through r_{t-1} (BPTT, Eq. 3.23)."""
        T = len(vs)
        rs = self.infer(vs)                                            # r_0..r_T
        bs = self.hidden_biases(rs)
        gW, gWp, gbv, gbh, gr0 = (np.zeros_like(p) for p in self.params())
        g_bt = []
        for t in range(T):
            dW, dbv, db = self.exact_grad(vs[t], bs[t]) if exact else self.cd_grad(vs[t], bs[t], k)
            gW += dW; gbv += dbv
            g_bt.append(db)                                            # d log P(v_t | r_{t-1}) / d b_t
        # backward pass over r: dL/dr_t (t = T..0)
        dr_next = np.zeros(self.H)                                     # dL/dr_{t+1} from later steps
        for t in range(T, -1, -1):
            # r_t feeds (a) the RBM bias b_{t+1} = b_h + W' r_t, and (b) r_{t+1} = sigmoid(... + W' r_t)
            dr = np.zeros(self.H)
            if t < T:
                dr += self.Wp.T @ g_bt[t]
                gWp += np.outer(g_bt[t], rs[t]); gbh += g_bt[t]
                pre = rs[t + 1] * (1 - rs[t + 1]) * dr_next             # through r_{t+1}'s sigmoid
                dr += self.Wp.T @ pre
                gWp += np.outer(pre, rs[t]); gbh += pre
                gW += np.outer(pre, vs[t])                             # r_{t+1} also uses W v_{t+1}
            if t == 0:
                gr0 += dr
            dr_next = dr
        return [gW, gWp, gbv, gbh, gr0]

    def log_prob_exact(self, vs):
        """log P(v_1..v_T) = sum_t log P_RBM(v_t | b_t), by enumeration (tiny models only)."""
        rs = self.infer(vs)
        return sum(rbm_log_prob_exact(self.W, self.bv, b, v) for v, b in zip(vs, self.hidden_biases(rs)))

    # ---- Algorithm 3: sampling ----
    def sample(self, T, gibbs=25):
        r = self.r0
        out = []
        for _ in range(T):
            b = self.bh + self.Wp @ r
            v = (self.rng.random(self.V) < 0.5).astype(float)
            for _ in range(gibbs):                                     # sample v_t ~ P(v_t | r_{t-1})
                h = (self.rng.random(self.H) < sigmoid(self.W @ v + b)).astype(float)
                pv = sigmoid(self.W.T @ h + self.bv)
                v = (self.rng.random(self.V) < pv).astype(float)
            out.append(pv)                                             # show probabilities, like the thesis' videos
            r = sigmoid(self.W @ v + b)                                # r_t <- mean-field (not a sample!)
        return np.array(out)

    def predict_next(self, vs, gibbs=25):
        """Mean-field-ish prediction of each next frame (for the mean squared prediction error)."""
        rs = self.infer(vs)
        preds = []
        for t in range(1, len(vs)):
            b = self.bh + self.Wp @ rs[t]
            pv = sigmoid(self.bv)
            for _ in range(gibbs):
                ph = sigmoid(self.W @ pv + b)
                pv = sigmoid(self.W.T @ ph + self.bv)
            preds.append(pv)
        return np.array(preds)


# ---------------------------------------------------------------------------
# Exact RBM quantities for tiny models (enumerate all 2^V visible states)
# ---------------------------------------------------------------------------

def free_energy(W, bv, bh, v):
    """F(v) = -bv.v - sum_j log(1 + exp(bh_j + W_j.v)); P(v) = exp(-F(v)) / Z."""
    return -bv @ v - np.logaddexp(0, bh + W @ v).sum()


def rbm_log_prob_exact(W, bv, bh, v):
    states = np.array(list(itertools.product([0.0, 1.0], repeat=len(bv))))
    neg_F = np.array([-free_energy(W, bv, bh, s) for s in states])
    logZ = np.logaddexp.reduce(neg_F)
    return -free_energy(W, bv, bh, v) - logZ


def rbm_grad_exact(W, bv, bh, v):
    """d log P(v)/d theta = <E-gradient>_data - <E-gradient>_model, both computed exactly."""
    states = np.array(list(itertools.product([0.0, 1.0], repeat=len(bv))))
    neg_F = np.array([-free_energy(W, bv, bh, s) for s in states])
    p = np.exp(neg_F - np.logaddexp.reduce(neg_F))                    # P(v) for every state
    ph_data = sigmoid(W @ v + bh)
    ph_states = sigmoid(states @ W.T + bh)                             # (2^V, H)
    dW = np.outer(ph_data, v) - (p[:, None, None] * ph_states[:, :, None] * states[:, None, :]).sum(0)
    dbv = v - p @ states
    dbh = ph_data - p @ ph_states
    return dW, dbv, dbh
