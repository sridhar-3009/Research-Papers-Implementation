"""Supervised Contrastive Learning (Khosla, Teterwak, Wang, Sarna, Tian, Isola, Maschinot, Liu, Krishnan; NeurIPS 2020).

  set-up          a batch of N images, each augmented twice -> 2N 'views'; encoder r = Enc(x), normalised; projection
                  z = Proj(r), normalised to the unit sphere; the loss is computed on z, the projection is thrown away
                  and a linear classifier is trained on the frozen r
  self-supervised (Eq. 1, SimCLR)   L_i = -log exp(z_i.z_j(i)/tau) / sum_{a != i} exp(z_i.z_a/tau)      one positive
  SupCon out      (Eq. 2)  L_i = -1/|P(i)| sum_{p in P(i)} log exp(z_i.z_p/tau) / sum_{a != i} exp(z_i.z_a/tau)
                  P(i) = every OTHER view in the batch with the same label
  SupCon in       (Eq. 3)  L_i = -log ( 1/|P(i)| sum_{p in P(i)} exp(z_i.z_p/tau) / sum_{a != i} exp(z_i.z_a/tau) )
                  by Jensen, L_in <= L_out; the paper finds L_out much better (78.7% vs 67.4% on ImageNet)
  gradient (Eq. 4) dL_i/dz_i = (1/tau) [ sum_p z_p (P_ip - X_ip) + sum_n z_n P_in ]: implicit hard positive / negative
                  mining once the normalisation is backpropagated

Everything here is numpy: the three losses with vectorised gradients (checked by finite differences), a small MLP
encoder + projection trained with each loss on augmented 8x8 digits (scikit-learn), a cross-entropy baseline with the same
encoder, linear probes, robustness to corruptions, and the hard-positive gradient analysis.
"""

import numpy as np
from scipy import ndimage

# ----------------------------------------------------------------------------------------------- the losses

def contrastive_loss(Z, labels, tau=0.1, kind="out"):
    """Mean over anchors of the SupCon loss on L2-normalised rows of Z (2N x d). labels: (2N,) ints.
    kind = 'out' (Eq. 2), 'in' (Eq. 3) or 'self' (Eq. 1: the positive is only the other view of the same image --
    pass labels = image ids). Returns the loss and dL/dZ."""
    n = len(Z)
    S = Z @ Z.T / tau
    off = ~np.eye(n, dtype=bool)
    S_m = np.where(off, S, -np.inf)
    lse = np.logaddexp.reduce(S_m, axis=1)                            # log sum_{a != i} exp(S_ia)
    soft = np.exp(S_m - lse[:, None])                                 # P_ia
    pos = (labels[:, None] == labels[None, :]) & off
    npos = pos.sum(1)
    valid = npos > 0
    if kind in ("out", "self"):
        L = -(np.where(pos, S - lse[:, None], 0).sum(1) / np.maximum(npos, 1))
        G = soft - pos / np.maximum(npos, 1)[:, None]                 # dL_i / dS_ia
    elif kind == "in":
        S_p = np.where(pos, S, -np.inf)
        lse_p = np.logaddexp.reduce(S_p, axis=1)
        L = -(lse_p - np.log(np.maximum(npos, 1)) - lse)
        G = soft - np.where(pos, np.exp(S_p - lse_p[:, None]), 0)     # X_ip = softmax over the positives
    else:
        raise ValueError(kind)
    G = np.where(valid[:, None] & off, G, 0) / max(valid.sum(), 1)
    loss = float(L[valid].mean())
    dZ = (G + G.T) @ Z / tau
    return loss, dZ


def normalize(U):
    nrm = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    return U / nrm, nrm


def normalize_backward(Z, nrm, dZ):
    """z = u / |u|  ->  du = (I - z z^T) dz / |u|."""
    return (dZ - Z * (Z * dZ).sum(1, keepdims=True)) / nrm


def hard_positive_gradient(cosines, d=16, tau=0.1, n_neg=20, seed=0):
    """Eq. 4 analysis: one anchor, one positive at cosine similarity c, random negatives. Gradient norm of the loss
    w.r.t. the UN-normalised positive vector w (z_p = w / |w|) as a function of c: large for hard positives
    (c ~ 0), vanishing for easy ones (c -> 1)."""
    rng = np.random.default_rng(seed)
    zi = np.zeros(d); zi[0] = 1.0
    negs = normalize(rng.standard_normal((n_neg, d)))[0]
    out = {}
    for c in cosines:
        w = np.zeros(d); w[0] = c; w[1] = np.sqrt(1 - c ** 2)
        Z = np.vstack([zi, w, negs])
        labels = np.r_[0, 0, np.arange(1, n_neg + 1)]
        _, dZ = contrastive_loss(Z, labels, tau, "out")
        zp, nrm = normalize(w[None])
        out[c] = float(np.linalg.norm(normalize_backward(zp, nrm, dZ[1:2])))
    return out

# ----------------------------------------------------------------------------------------------- data and augmentation

def load_digits_split(seed=0):
    from sklearn.datasets import load_digits
    d = load_digits()
    X, y = d.data / 16.0, d.target
    idx = np.random.default_rng(seed).permutation(len(y))
    tr, te = idx[:1300], idx[1300:]
    return X[tr], y[tr], X[te], y[te]


def augment(X, rng, shift=1.0, noise=0.1, scale=0.2):
    """Random sub-pixel shift, intensity scaling and pixel noise of 8x8 images."""
    out = np.empty_like(X)
    for i, x in enumerate(X):
        img = ndimage.shift(x.reshape(8, 8), rng.uniform(-shift, shift, 2), order=1, mode="constant")
        out[i] = img.ravel()
    out *= rng.uniform(1 - scale, 1 + scale, (len(X), 1))
    return np.clip(out + noise * rng.standard_normal(X.shape), 0, 1)


def corrupt(X, kind, level, seed=0):
    """Test-time corruptions (an ImageNet-C analogue): gaussian noise, blur, shift."""
    rng = np.random.default_rng(seed)
    if kind == "noise":
        return np.clip(X + level * rng.standard_normal(X.shape), 0, 1)
    if kind == "blur":
        return np.array([ndimage.gaussian_filter(x.reshape(8, 8), level).ravel() for x in X])
    if kind == "shift":
        return np.array([ndimage.shift(x.reshape(8, 8), (level, level), order=1, mode="constant").ravel() for x in X])
    raise ValueError(kind)

# ----------------------------------------------------------------------------------------------- networks

class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            self.p[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)


def init_params(d_in=64, hidden=128, d_rep=64, d_proj=32, classes=10, seed=0):
    r = np.random.default_rng(seed)
    he = lambda a, b: r.normal(0, np.sqrt(2 / a), (a, b))
    return {"W1": he(d_in, hidden), "b1": np.zeros(hidden), "W2": he(hidden, d_rep), "b2": np.zeros(d_rep),
            "P1": he(d_rep, d_rep), "c1": np.zeros(d_rep), "P2": he(d_rep, d_proj), "c2": np.zeros(d_proj),
            "Wc": r.normal(0, 0.01, (d_rep, classes)), "bc": np.zeros(classes)}


def encode(p, X):
    """Enc: two-layer ReLU MLP, output normalised to the unit sphere (as in the paper)."""
    h = np.maximum(0, X @ p["W1"] + p["b1"])
    u = h @ p["W2"] + p["b2"]
    r, nrm = normalize(u)
    return h, u, r, nrm


def project(p, r):
    g = np.maximum(0, r @ p["P1"] + p["c1"])
    v = g @ p["P2"] + p["c2"]
    z, nrm = normalize(v)
    return g, v, z, nrm


def train_contrastive(X, y, kind="out", tau=0.1, epochs=60, batch=128, lr=1e-3, seed=0):
    """Stage 1: encoder + projection trained with Eq. 1, 2 or 3 on two augmented views per image."""
    rng = np.random.default_rng(seed)
    p = init_params(seed=seed)
    opt = Adam(p, lr)
    hist = []
    for ep in range(epochs):
        for idx in np.array_split(rng.permutation(len(y)), max(1, len(y) // batch)):
            Xb = np.vstack([augment(X[idx], rng), augment(X[idx], rng)])
            lab = np.r_[y[idx], y[idx]] if kind != "self" else np.r_[np.arange(len(idx)), np.arange(len(idx))]
            h, u, r, nr = encode(p, Xb)
            g, v, z, nz = project(p, r)
            loss, dz = contrastive_loss(z, lab, tau, "out" if kind == "self" else kind)
            dv = normalize_backward(z, nz, dz)
            dg = (dv @ p["P2"].T) * (g > 0)
            dr = dg @ p["P1"].T
            du = normalize_backward(r, nr, dr)
            dh = (du @ p["W2"].T) * (h > 0)
            opt.step({"P2": g.T @ dv, "c2": dv.sum(0), "P1": r.T @ dg, "c1": dg.sum(0),
                      "W2": h.T @ du, "b2": du.sum(0), "W1": Xb.T @ dh, "b1": dh.sum(0)})
            hist.append(loss)
    return p, hist


def linear_probe(p, X, y, epochs=300, lr=0.5):
    """Stage 2: a softmax classifier on the FROZEN normalised representation r."""
    r = encode(p, X)[2]
    W = np.zeros((r.shape[1], 10)); b = np.zeros(10)
    Y = np.eye(10)[y]
    for _ in range(epochs):
        z = r @ W + b
        z -= z.max(1, keepdims=True)
        pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
        W -= lr * r.T @ (pr - Y) / len(y) * 10
        b -= lr * (pr - Y).mean(0) * 10
    return W, b


def probe_accuracy(p, W, b, X, y):
    return float(((encode(p, X)[2] @ W + b).argmax(1) == y).mean())


def train_cross_entropy(X, y, epochs=60, batch=128, lr=1e-3, seed=0):
    """The baseline: the same encoder (including the normalised r) + a linear head, trained end to end with
    cross-entropy on augmented images."""
    rng = np.random.default_rng(seed)
    p = init_params(seed=seed)
    opt = Adam(p, lr)
    for ep in range(epochs):
        for idx in np.array_split(rng.permutation(len(y)), max(1, len(y) // batch)):
            Xb = augment(X[idx], rng)
            h, u, r, nr = encode(p, Xb)
            z = (r @ p["Wc"] + p["bc"]) * 10                          # fixed scale: r is unit-norm
            z -= z.max(1, keepdims=True)
            pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
            dz = (pr - np.eye(10)[y[idx]]) / len(idx) * 10
            dr = dz @ p["Wc"].T
            du = normalize_backward(r, nr, dr)
            dh = (du @ p["W2"].T) * (h > 0)
            opt.step({"Wc": r.T @ dz, "bc": dz.sum(0), "W2": h.T @ du, "b2": du.sum(0), "W1": Xb.T @ dh,
                      "b1": dh.sum(0)})
    return p


def ce_accuracy(p, X, y):
    return float(((encode(p, X)[2] @ p["Wc"] + p["bc"]).argmax(1) == y).mean())


REPORTED = {
    "Table 1": "ImageNet top-1, ResNet-50, batch 6144: L_out 78.7% vs L_in 67.4%",
    "Table 2 (ResNet-50)": "CIFAR-10 / CIFAR-100 / ImageNet top-1: SimCLR 93.6 / 70.7 / 70.2; cross-entropy 95.0 / 75.3 / "
                           "78.2; max-margin 92.4 / 70.5 / 78.0; SupCon 96.0 / 76.5 / 78.7",
    "Table 3": "ResNet-200 with Stacked RandAugment: cross-entropy 80.9% (our impl.) vs SupCon 81.4% (0.8% above the best "
               "reported number for this architecture, 80.6%); ResNet-50 AutoAugment SupCon 78.7%",
    "robustness (ImageNet-C)": "mCE lower for SupCon: ResNet-50 68.6 -> 67.2, ResNet-200 52.4 -> 50.6 (relative mCE "
                               "96.2 -> 94.6 and 69.1 -> 66.5)",
    "other": "memory bank of 8192 with batch 256 gives 79.1% on ResNet-50; cross-entropy with batch 12,288 only 77.5%; "
             "N-pairs loss 57.4%; less sensitive to hyperparameters than cross-entropy; temperature 0.1 used",
}
