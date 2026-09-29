"""Back-propagation exactly as written in Rumelhart, Hinton & Williams (1986).

A LAYERED network: input units at the bottom, any number of layers above, output
units at the top. Connections only go UP, and may skip layers (page 533). Each
layer lists the earlier layers it receives from, so both a plain stack and the
family-tree network (two input groups, each with its own hidden group) fit.

Forward pass, Eqs. (1)-(2):
    x_j = sum_i y_i w_ji + b_j          total input  (the bias b_j is a weight
    y_j = 1 / (1 + exp(-x_j))           from an extra unit that is always 1)

Error, Eq. (3):
    E = 1/2 sum_c sum_j (y_jc - d_jc)^2    over cases c and output units j

Backward pass, Eqs. (4)-(7), for one case:
    dE/dy_j = y_j - d_j                  output units              (4)
    dE/dx_j = dE/dy_j * y_j (1 - y_j)    slope of the logistic     (5)
    dE/dw_ji = dE/dx_j * y_i             the weight's gradient     (6)
    dE/dy_i = sum_j dE/dx_j * w_ji       error sent down to unit i (7)

Weight change, Eqs. (8)-(9), after a full sweep through all cases:
    dw(t) = -eps * dE/dw(t) + alpha * dw(t-1)
"""

import numpy as np


def logistic(x):
    return 0.5 * (1 + np.tanh(x / 2))           # = 1 / (1 + e^-x), without overflow


class LayeredNet:
    def __init__(self, spec, init_range=0.3, rng=None):
        """spec: list of (name, size, sources) from bottom to top. Layers with no
        sources are input layers; the last layer is the output layer.

        Example, the symmetry net of Figure 1:
            [("in", 6, []), ("hidden", 2, ["in"]), ("out", 1, ["hidden"])]
        Weights start "small and random", uniform in [-init_range, init_range],
        "to break symmetry" (page 535); Figure 1 used 0.3.
        """
        rng = np.random.default_rng(rng)
        self.spec = spec
        self.sizes = {name: size for name, size, _ in spec}
        self.sources = {name: list(src) for name, _, src in spec}
        self.inputs = [name for name, _, src in spec if not src]
        self.output = spec[-1][0]
        # W[(src, dst)] has shape (size of src, size of dst): w_ji for i in src, j in dst
        self.W = {(s, name): rng.uniform(-init_range, init_range, (self.sizes[s], size))
                  for name, size, src in spec for s in src}
        self.b = {name: rng.uniform(-init_range, init_range, size) for name, size, src in spec if src}

    # ---- parameters as a dict, so training code can loop over them ----

    def params(self):
        return {**{("W",) + k: v for k, v in self.W.items()}, **{("b", k): v for k, v in self.b.items()}}

    # ---- Forward pass: Eqs. (1) and (2), layer by layer, bottom to top ----

    def forward(self, inputs):
        """inputs: {input layer name: (cases, size) array}. Returns every layer's y."""
        y = dict(inputs)
        for name, size, src in self.spec:
            if not src:
                continue
            x = self.b[name] + sum(y[s] @ self.W[(s, name)] for s in src)      # Eq. (1)
            y[name] = logistic(x)                                            # Eq. (2)
        return y

    # ---- Error: Eq. (3), with the optional "close enough" margin of Figure 4 ----

    @staticmethod
    def output_error(y_out, d, margin=None):
        """dE/dy for the output units, Eq. (4): y - d.

        margin=(0.2, 0.8) is the family-tree rule (Figure 4 caption): no error for
        a unit that should be on and is above 0.8, or should be off and is below 0.2.
        """
        err = y_out - d
        if margin is not None:
            lo, hi = margin
            err = np.where((d == 1) & (y_out > hi), 0.0, err)
            err = np.where((d == 0) & (y_out < lo), 0.0, err)
        return err

    def error(self, inputs, d, margin=None):
        """E, Eq. (3): half the sum of squared output errors over all cases."""
        return 0.5 * float(np.sum(self.output_error(self.forward(inputs)[self.output], d, margin) ** 2))

    # ---- Backward pass: Eqs. (4)-(7), top to bottom ----

    def gradients(self, inputs, d, margin=None):
        """dE/dw for every weight and bias, summed over all cases (the paper
        accumulates the gradient over the whole set before changing weights)."""
        y = self.forward(inputs)
        dE_dy = {name: np.zeros_like(y[name]) for name in y}
        dE_dy[self.output] = self.output_error(y[self.output], d, margin)        # Eq. (4)
        grads = {}
        for name, size, src in reversed(self.spec):
            if not src:
                continue
            dE_dx = dE_dy[name] * y[name] * (1 - y[name])                        # Eq. (5)
            grads[("b", name)] = dE_dx.sum(0)                                   # bias: y_i = 1
            for s in src:
                grads[("W", s, name)] = y[s].T @ dE_dx                          # Eq. (6)
                dE_dy[s] = dE_dy[s] + dE_dx @ self.W[(s, name)].T               # Eq. (7)
        return grads


def train(net, inputs, d, sweeps=1000, eps=0.1, alpha=0.9, margin=None, decay=0.0,
          schedule=None, stop=None):
    """Batch gradient descent with momentum, Eqs. (8)-(9).

    One sweep = one pass through ALL cases, then one weight change.
    decay:    shrink every weight by this fraction after each change (the family
              tree used 0.2% = 0.002, Figure 4 caption).
    schedule: optional function sweep -> (eps, alpha), e.g. the family tree's
              "eps=0.005, alpha=0.5 for the first 20 sweeps, then 0.01 and 0.9".
    stop:     optional function net -> bool, checked after each sweep.
    Returns the error E after each sweep (and stops early if stop(net) is True).
    """
    velocity = {k: np.zeros_like(v) for k, v in net.params().items()}
    history = []
    for t in range(sweeps):
        if schedule is not None:
            eps, alpha = schedule(t)
        g = net.gradients(inputs, d, margin)
        for k, p in net.params().items():
            velocity[k] = -eps * g[k] + alpha * velocity[k]                     # Eq. (9)
            p += velocity[k]                                                    # (in place)
            if decay:
                p *= (1 - decay)
        history.append(net.error(inputs, d, margin))
        if stop is not None and stop(net):
            break
    return history


# ---------------------------------------------------------------------------
# Figure 5: a recurrent net run for several steps = a layered net with tied weights
# ---------------------------------------------------------------------------

class IterativeNet:
    """A synchronous recurrent net: every unit's state at step t+1 comes from all
    units' states at step t (plus the external input), through ONE weight matrix.

    Figure 5: running it for T steps is the same as a T-layer layered net whose
    layers all share the same weights. To train it, back-propagate through the
    layers and add up dE/dw over all copies of each weight (the paper averages;
    averaging only rescales the step size)."""

    def __init__(self, n_units, n_inputs, init_range=0.5, rng=None):
        rng = np.random.default_rng(rng)
        self.W = rng.uniform(-init_range, init_range, (n_units, n_units))   # unit -> unit
        self.U = rng.uniform(-init_range, init_range, (n_inputs, n_units))  # input -> unit
        self.b = rng.uniform(-init_range, init_range, n_units)

    def run(self, h0, inputs):
        """inputs: list of (cases, n_inputs) arrays, one per step. Returns all states."""
        states = [h0]
        for u in inputs:
            states.append(logistic(states[-1] @ self.W + u @ self.U + self.b))
        return states

    def gradients(self, h0, inputs, targets):
        """Back-propagation through time. targets: (cases, n_units) desired final
        state. E = 1/2 sum (final state - target)^2."""
        states = self.run(h0, inputs)
        gW, gU, gb = np.zeros_like(self.W), np.zeros_like(self.U), np.zeros_like(self.b)
        dE_dy = states[-1] - targets                                   # Eq. (4) at the top layer
        for t in range(len(inputs), 0, -1):                            # layer t = step t
            dE_dx = dE_dy * states[t] * (1 - states[t])                # Eq. (5)
            gW += states[t - 1].T @ dE_dx                              # Eq. (6), summed over the
            gU += inputs[t - 1].T @ dE_dx                              #   copies of each weight
            gb += dE_dx.sum(0)
            dE_dy = dE_dx @ self.W.T                                   # Eq. (7): down one layer
        return gW, gU, gb
