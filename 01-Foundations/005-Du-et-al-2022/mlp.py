"""Multilayer perceptron with backpropagation, Section 4 of Du et al. (2022).

Notation follows the paper: layer m has J_m nodes; W[m] maps layer m to m+1.
    net^(m) = o^(m-1) W^(m-1) + theta^(m)        Eq. (24)
    o^(m)   = phi^(m)(net^(m))                   Eq. (25)

The objective is the paper's MSE, Eq. (26), plus optional weight decay, Eq. (34):
    E(w) = 1/(2N) * sum_p ||y_p - o_p||^2  +  decay * sum_i w_i^2

All weights and biases can be read and written as ONE flat vector. That's what
the second-order optimizers (Newton, BFGS, CG) in optimizers.py work on.
"""

import numpy as np

# ---- Activation functions, Eqs. (3)-(6) and (64) ----
# Each maps net -> output. The derivative is written in terms of the OUTPUT o,
# which is what backpropagation has at hand.
ACTIVATIONS = {
    # 1/(1+e^-a) written as 0.5*(1+tanh(a/2)): same function, no overflow for big |a|
    "logistic": (lambda a: 0.5 * (1 + np.tanh(a / 2)), lambda o: o * (1 - o)),  # Eq. (4), beta = 1
    "tanh":     (np.tanh,                        lambda o: 1 - o ** 2),       # Eq. (5)
    "linear":   (lambda a: a,                    lambda o: np.ones_like(o)),  # Eq. (6)
    "relu":     (lambda a: np.maximum(0, a),     lambda o: (o > 0) * 1.0),    # Eq. (64)
}


class MLP:
    def __init__(self, sizes, hidden="logistic", output="linear", rng=None):
        """sizes: nodes per layer, e.g. [4, 4, 3] = 4 inputs, 4 hidden, 3 outputs.

        Weights start uniform in [-3/sqrt(n), 3/sqrt(n)], n = fan-in of the node,
        the heuristic from Section 7.3.1. Random values break the symmetry
        between hidden nodes; scaling by fan-in keeps them out of saturation.
        """
        rng = np.random.default_rng(rng)
        self.sizes = list(sizes)
        self.acts = [hidden] * (len(sizes) - 2) + [output]
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            r = 3 / np.sqrt(fan_in)
            self.W.append(rng.uniform(-r, r, (fan_in, fan_out)))
            self.b.append(rng.uniform(-r, r, fan_out))
        # Open-node faults (Section 10.1): mask[m][j] = 0 means hidden node j of
        # hidden layer m is broken and its output is stuck at zero.
        self.masks = [np.ones(n) for n in sizes[1:-1]]

    # ---- Flat parameter vector ----

    @property
    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def get_params(self):
        return np.concatenate([np.concatenate([W.ravel(), b]) for W, b in zip(self.W, self.b)])

    def set_params(self, w):
        i = 0
        for m, (W, b) in enumerate(zip(self.W, self.b)):
            self.W[m] = w[i:i + W.size].reshape(W.shape); i += W.size
            self.b[m] = w[i:i + b.size].copy();          i += b.size

    # ---- Forward pass ----

    def forward(self, X):
        """Return the outputs of every layer: [o^(1) = X, o^(2), ..., o^(M)]."""
        outs = [X]
        for m, (W, b) in enumerate(zip(self.W, self.b)):
            phi = ACTIVATIONS[self.acts[m]][0]
            o = phi(outs[-1] @ W + b)                       # Eqs. (24)-(25)
            if m < len(self.masks):
                o = o * self.masks[m]                       # faulty nodes output 0
            outs.append(o)
        return outs

    def predict(self, X):
        return self.forward(X)[-1]

    # ---- Objective and gradient (backpropagation) ----

    def loss(self, X, Y, decay=0.0):
        """E(w) from Eq. (26), plus the weight-decay penalty of Eq. (34)."""
        E = np.sum((Y - self.predict(X)) ** 2) / (2 * len(X))
        return E + decay * np.sum(self.get_params() ** 2)

    def gradient(self, X, Y, decay=0.0):
        """dE/dw for every weight and bias, by backpropagation (Section 4.3).

        1. Forward pass, keeping every layer's output.
        2. Output layer: delta = dE/dnet = -(y - o) * phi'(net) / N.
           (This is the paper's generalized error term delta, Eq. (30), with the
           sign and 1/N of Eq. (26) folded in.)
        3. For each layer, going backwards:
              dE/dW = o_prev^T delta,  dE/dtheta = sum of delta        Eq. (31)
              delta_prev = (delta W^T) * phi'(o_prev)                  chain rule
        """
        outs = self.forward(X)
        N = len(X)
        delta = -(Y - outs[-1]) * ACTIVATIONS[self.acts[-1]][1](outs[-1]) / N
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for m in reversed(range(len(self.W))):
            gW[m] = outs[m].T @ delta
            gb[m] = delta.sum(axis=0)
            if m > 0:
                delta = (delta @ self.W[m].T) * ACTIVATIONS[self.acts[m - 1]][1](outs[m])
                delta = delta * self.masks[m - 1]           # no error flows into dead nodes
        g = np.concatenate([np.concatenate([gW[m].ravel(), gb[m]]) for m in range(len(self.W))])
        return g + 2 * decay * self.get_params()

    def jacobian(self, X, Y):
        """Residuals e = o - y (flattened, N*K) and their Jacobian J = de/dw.

        Needed by Gauss-Newton and Levenberg-Marquardt, Eqs. (45)-(46):
            gradient = J^T e,  Hessian ~ J^T J.
        Same backward pass as gradient(), but run once per output unit k and
        without summing over the samples, so every (sample, output) pair keeps
        its own row.
        """
        outs = self.forward(X)
        N, K = outs[-1].shape
        e = (outs[-1] - Y).ravel()                      # row index = p*K + k
        J = np.zeros((N, K, self.n_params))
        dphi_out = ACTIVATIONS[self.acts[-1]][1](outs[-1])
        for k in range(K):
            delta = np.zeros((N, K))
            delta[:, k] = dphi_out[:, k]                # d o_pk / d net_pk
            cols = []
            for m in reversed(range(len(self.W))):
                # per-sample outer product instead of a sum over samples
                cols.append((outs[m][:, :, None] * delta[:, None, :]).reshape(N, -1))
                cols[-1] = np.concatenate([cols[-1], delta], axis=1)
                if m > 0:
                    delta = (delta @ self.W[m].T) * ACTIVATIONS[self.acts[m - 1]][1](outs[m])
                    delta = delta * self.masks[m - 1]
            J[:, k, :] = np.concatenate(cols[::-1], axis=1)
        return e, J.reshape(N * K, -1)
