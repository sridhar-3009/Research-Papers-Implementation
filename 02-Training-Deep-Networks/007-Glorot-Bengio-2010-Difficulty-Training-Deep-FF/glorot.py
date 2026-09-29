"""Glorot & Bengio (2010): deep nets, three activations, two initializations.

  Section 2.1   Shapeset-3x2: a generator for the paper's synthetic image dataset
  Section 2.3   deep MLPs with 1-5 hidden layers, softmax output, -log P(y|x),
                SGD on mini-batches of 10, biases 0
  Eq. (1)       "standard" init:   W ~ U[-1/sqrt(n), 1/sqrt(n)]      n = fan-in
  Eq. (16)      "normalized" init: W ~ U[-sqrt(6)/sqrt(n_in + n_out), +...]   (Xavier)
  Section 3     activations: sigmoid, tanh, softsign x / (1 + |x|)
  Sections 3-4  monitoring tools: activation statistics per layer, back-propagated
                gradients per layer, weight gradients, Jacobian singular values

The network is PyTorch (for speed); everything the paper studies - the
initialization, the activations, and every measurement - is written out here.
"""

import numpy as np
import torch

ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "softsign": lambda s: s / (1 + s.abs()),
}


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def standard_init(n_in, n_out, gen):
    """Eq. (1): U[-1/sqrt(n), 1/sqrt(n)], n = size of the previous layer.
    Variance 1/(3n), so n Var[W] = 1/3 (Eq. 15)."""
    a = 1 / np.sqrt(n_in)
    return (torch.rand(n_in, n_out, generator=gen) * 2 - 1) * a


def normalized_init(n_in, n_out, gen):
    """Eq. (16): U[-sqrt(6/(n_in+n_out)), +sqrt(6/(n_in+n_out))].
    Variance 2/(n_in + n_out), the compromise of Eqs. (10)-(12) between keeping
    activation variance (n_in Var[W] = 1) and gradient variance (n_out Var[W] = 1)."""
    a = np.sqrt(6 / (n_in + n_out))
    return (torch.rand(n_in, n_out, generator=gen) * 2 - 1) * a


INITS = {"standard": standard_init, "normalized": normalized_init}


# ---------------------------------------------------------------------------
# The deep network
# ---------------------------------------------------------------------------

class DeepNet(torch.nn.Module):
    """n_in -> [hidden] * depth -> n_out, one activation for all hidden layers,
    softmax output. forward(x, keep=True) also returns every layer's
    pre-activation s^i and activation z^i (Section 4 notation), with gradients
    retained so backward() fills in dCost/ds^i."""

    def __init__(self, n_in, n_out, hidden=1000, depth=5, act="tanh", init="normalized", seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        sizes = [n_in] + [hidden] * depth + [n_out]
        self.act_name, self.act = act, ACTIVATIONS[act]
        self.W = torch.nn.ParameterList(
            [torch.nn.Parameter(INITS[init](a, b, gen)) for a, b in zip(sizes[:-1], sizes[1:])])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(b)) for b in sizes[1:]])   # biases 0

    def forward(self, x, keep=False):
        zs, ss = [x], []
        z = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            s = z @ W + b                              # s^i = z^i W^i + b^i
            if keep:
                if s.requires_grad:
                    s.retain_grad()
                ss.append(s)
            if i < len(self.W) - 1:
                z = self.act(s)                        # z^{i+1} = f(s^i)
                if keep:
                    zs.append(z)
            else:
                z = s                                  # logits; softmax is inside the loss
        return (z, ss, zs) if keep else z


def nll(logits, y):
    """-log P(y|x) with a softmax output (Section 2.3)."""
    return torch.nn.functional.cross_entropy(logits, y)


def quadratic(logits, y):
    """The traditional quadratic cost on softmax outputs, for comparison (Section 4.1)."""
    p = torch.softmax(logits, 1)
    return 0.5 * ((p - torch.nn.functional.one_hot(y, p.shape[1]).float()) ** 2).sum(1).mean()


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

@torch.no_grad()
def activation_stats(net, X):
    """For each hidden layer: mean, std, and 98th percentile of |activation|
    (the quantities plotted in Figures 2, 3 and 10), on a fixed set of inputs."""
    _, _, zs = net(X, keep=True)
    return [(float(z.mean()), float(z.std()), float(torch.quantile(z.abs().flatten()[:200000], 0.98)))
            for z in zs[1:]]


def gradient_stats(net, X, y):
    """Back-propagate once. Returns, per layer: std of the activations z^i,
    std of the back-propagated gradients dCost/ds^i (Figure 7), and std of the
    weight gradients dCost/dW^i (Figure 8)."""
    net.zero_grad()
    logits, ss, zs = net(X, keep=True)
    nll(logits, y).backward()
    act = [float(z.detach().std()) for z in zs[1:]]
    back = [float(s.grad.std()) for s in ss]
    wgrad = [float(W.grad.std()) for W in net.W]
    return act, back, wgrad


@torch.no_grad()
def jacobian_singular_values(net, X, n_examples=20):
    """Average singular value of J_i = dz^{i+1}/dz^i = W^i diag(f'(s^i)), Eq. (17),
    for each hidden-to-hidden layer, averaged over some examples. The paper: about
    0.8 with normalized init, about 0.5 with standard init."""
    _, ss, _ = net(X[:n_examples], keep=True)
    out = []
    for i in range(1, len(net.W) - 1):                 # hidden layer i -> hidden layer i+1
        s = ss[i]
        if net.act_name == "tanh":
            fp = 1 - torch.tanh(s) ** 2
        elif net.act_name == "sigmoid":
            fp = torch.sigmoid(s) * (1 - torch.sigmoid(s))
        else:
            fp = 1 / (1 + s.abs()) ** 2
        vals = [torch.linalg.svdvals(net.W[i] * fp[k]).mean() for k in range(len(s))]
        out.append(float(torch.stack(vals).mean()))
    return out


# ---------------------------------------------------------------------------
# Shapeset-3x2 (Section 2.1)
# ---------------------------------------------------------------------------

SHAPES = ("triangle", "parallelogram", "ellipse")
PAIRS = [(a, b) for a in range(3) for b in range(a, 3)]      # 6 pairs, order ignored
CLASSES = [(a,) for a in range(3)] + PAIRS                   # 3 singles + 6 pairs = 9 classes


def _shape_mask(kind, rng, size, yy, xx):
    """A random shape: random proportions, scale, rotation and position."""
    cx, cy = rng.uniform(0.25, 0.75, 2) * size
    scale = rng.uniform(0.15, 0.32) * size
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    u = (xx - cx) * c + (yy - cy) * s                         # coordinates in the shape's frame
    v = -(xx - cx) * s + (yy - cy) * c
    if kind == 0:                                             # triangle: random corners in the frame
        pts = np.array([[-1, -0.8], [1, -0.8], [rng.uniform(-0.8, 0.8), rng.uniform(0.6, 1.0)]]) * scale
    elif kind == 1:                                           # parallelogram: random slant and aspect
        w, h, sk = 1.0, rng.uniform(0.4, 0.9), rng.uniform(-0.6, 0.6)
        pts = np.array([[-w, -h], [w, -h], [w + sk, h], [-w + sk, h]]) * scale
    else:                                                     # ellipse: random aspect
        a, b = scale, scale * rng.uniform(0.4, 0.9)
        return (u / a) ** 2 + (v / b) ** 2 <= 1
    inside = np.ones_like(u, dtype=bool)                      # convex polygon: inside all edges
    for (x1, y1), (x2, y2) in zip(pts, np.roll(pts, -1, axis=0)):
        inside &= (x2 - x1) * (v - y1) - (y2 - y1) * (u - x1) >= 0
    return inside


def shapeset(n, rng=None, size=32):
    """n images (n, size*size) in [0, 1] and labels 0-8. Each image has 1 or 2
    objects with random grey levels; the second may cover at most 50% of the
    first (as in the paper). The label says WHICH shapes are present."""
    rng = np.random.default_rng(rng)
    yy, xx = np.mgrid[:size, :size] + 0.5
    X = np.zeros((n, size, size), np.float32)
    y = np.zeros(n, np.int64)
    for k in range(n):
        label = rng.integers(len(CLASSES))
        kinds = CLASSES[label]
        if len(kinds) == 2 and rng.random() < 0.5:
            kinds = kinds[::-1]                               # either shape can be in front
        img = np.zeros((size, size), np.float32)
        first = None
        for j, kind in enumerate(kinds):
            for _ in range(50):
                m = _shape_mask(kind, rng, size, yy, xx)
                if m.sum() < 12:
                    continue
                if first is None or (m & first).sum() <= 0.5 * first.sum():
                    break
            img[m] = rng.uniform(0.3, 1.0)
            first = m if first is None else first
        X[k], y[k] = img, label
    return X.reshape(n, -1), y
