"""The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization (Coates & Ng, ICML 2011).

  the question   a feature learner = TRAINING (learn a dictionary D of normalised basis vectors) + ENCODING (map an
                 input x to features f given D). Sparse coding beats vector quantization -- is it the dictionary or
                 the encoder? Decouple them and mix and match.
  training       R (random Gaussian), RP (randomly sampled patches), OMP-1 (gain-shape VQ, ~ spherical K-means),
                 OMP-k (k non-zeros), SC (L1 sparse coding:  min ||D s - x||^2 + lambda ||s||_1,  ||D_j|| = 1)
  encoding       SC (solve the lasso with D fixed), OMP-k, soft threshold T:  f = [max(0, D^T x - alpha),
                 max(0, -D^T x - alpha)]  -- a single feed-forward layer; every encoder splits the polarities
  findings       (1) with a good encoder, almost ANY dictionary works -- random patches or VQ are as good as SC;
                 (2) the plain soft threshold is competitive with sparse coding as the encoder

This file runs the paper's pipeline (patches -> per-patch normalisation -> ZCA whitening -> dictionary -> encoder ->
quadrant average pooling -> linear classifier) on scikit-learn's 8x8 digits with few labels, with every training
algorithm and every encoder implemented in numpy (FISTA for the lasso, batched OMP).
"""

import numpy as np

# ----------------------------------------------------------------------------------------------- data and patches

def load_digits_images():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.images / 16.0, d.target


def extract_patches(images, p=4):
    """All p x p patches with stride 1: (n_images, n_positions, p*p) and the grid size."""
    n, H, W = images.shape
    g = H - p + 1
    out = np.empty((n, g * g, p * p))
    k = 0
    for r in range(g):
        for c in range(g):
            out[:, k] = images[:, r:r + p, c:c + p].reshape(n, -1)
            k += 1
    return out, g


def normalize_patches(P, eps=0.1):
    """Per-patch brightness / contrast normalisation (subtract the mean, divide by the standard deviation)."""
    P = P - P.mean(-1, keepdims=True)
    return P / np.sqrt(P.var(-1, keepdims=True) + eps)


def fit_zca(X, eps=0.01):
    """ZCA whitening (retaining full variance): W = U diag(1 / sqrt(lambda + eps)) U^T."""
    mu = X.mean(0)
    C = np.cov(X - mu, rowvar=False)
    lam, U = np.linalg.eigh(C)
    return mu, U @ np.diag(1 / np.sqrt(lam + eps)) @ U.T


def unit_columns(D):
    return D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)

# ----------------------------------------------------------------------------------------------- encoders

def lasso_fista(X, D, lam, iters=100):
    """Sparse codes for every row of X: argmin_s ||D s - x||^2 + lam ||s||_1, by FISTA (batched over rows)."""
    L = 2 * np.linalg.norm(D, 2) ** 2                                  # Lipschitz constant of the smooth part
    DtD, DtX = D.T @ D, X @ D
    S = np.zeros((len(X), D.shape[1]))
    Y, t = S.copy(), 1.0
    for _ in range(iters):
        G = 2 * (Y @ DtD - DtX)
        S_new = Y - G / L
        S_new = np.sign(S_new) * np.maximum(np.abs(S_new) - lam / L, 0)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Y = S_new + (t - 1) / t_new * (S_new - S)
        S, t = S_new, t_new
    return S


def omp(X, D, k):
    """Orthogonal matching pursuit with at most k non-zeros, batched: greedily add the atom most correlated with the
    residual, then refit all selected coefficients by least squares."""
    n, d = len(X), D.shape[1]
    S = np.zeros((n, d))
    support = np.zeros((n, 0), int)
    R = X.copy()
    rows = np.arange(n)
    for _ in range(k):
        corr = np.abs(R @ D)
        corr[rows[:, None], support] = -1                              # never pick an atom twice
        support = np.hstack([support, corr.argmax(1)[:, None]])
        Ds = D.T[support]                                              # (n, j, dim)
        G = Ds @ Ds.transpose(0, 2, 1) + 1e-8 * np.eye(support.shape[1])
        coef = np.linalg.solve(G, (Ds @ X[:, :, None]))[:, :, 0]
        S = np.zeros((n, d))
        S[rows[:, None], support] = coef
        R = X - S @ D.T
    return S


def split(S):
    return np.hstack([np.maximum(0, S), np.maximum(0, -S)])


def encode(X, D, method, param):
    """method: 'SC' (param = lambda), 'OMP' (param = k), 'T' (param = alpha)."""
    if method == "SC":
        return split(lasso_fista(X, D, param))
    if method == "OMP":
        return split(omp(X, D, int(param)))
    if method == "T":
        Z = X @ D
        return np.hstack([np.maximum(0, Z - param), np.maximum(0, -Z - param)])
    raise ValueError(method)

# ----------------------------------------------------------------------------------------------- training algorithms

def train_dictionary(X, d, method, rng, iters=10, lam=1.0, k=1):
    """Return a dictionary (dim x d) with unit-norm columns.
      R    random Gaussian columns          RP   random training patches
      OMP  alternate OMP-k codes and the least-squares dictionary update (k = 1: gain-shape VQ)
      SC   alternate lasso codes (FISTA) and the least-squares dictionary update (L1 sparse coding)"""
    dim = X.shape[1]
    if method == "R":
        return unit_columns(rng.standard_normal((dim, d)))
    D = unit_columns(X[rng.choice(len(X), d, replace=False)].T.copy())
    if method == "RP":
        return D
    for _ in range(iters):
        S = omp(X, D, k) if method == "OMP" else lasso_fista(X, D, lam, iters=60)
        D_new = X.T @ S @ np.linalg.pinv(S.T @ S + 1e-6 * np.eye(d))    # least squares for D given the codes
        dead = np.linalg.norm(D_new, axis=0) < 1e-8                    # re-seed unused atoms with random patches
        D_new[:, dead] = X[rng.choice(len(X), dead.sum(), replace=False)].T
        D = unit_columns(D_new)
    return D

# ----------------------------------------------------------------------------------------------- the pipeline

class Pipeline:
    """Patches -> normalise -> whiten (fitted on a sample) -> encode with D -> average-pool over the 4 (overlapping)
    quadrants of the patch grid -> standardise -> multinomial logistic regression."""

    def __init__(self, images, p=4, n_sample=20000, seed=0):
        self.rng = np.random.default_rng(seed)
        P, self.g = extract_patches(images, p)
        P = normalize_patches(P)
        flat = P.reshape(-1, P.shape[-1])
        sample = flat[self.rng.choice(len(flat), min(n_sample, len(flat)), replace=False)]
        self.mu, self.W = fit_zca(sample)
        self.patches = (P - self.mu) @ self.W                          # (n_img, positions, dim)
        self.sample = (sample - self.mu) @ self.W

    def features(self, D, method, param):
        n, npos, dim = self.patches.shape
        F = encode(self.patches.reshape(-1, dim), D, method, param).reshape(n, npos, -1)
        g = self.g
        h = (g + 1) // 2
        grid = np.arange(npos).reshape(g, g)
        quads = [grid[:h, :h], grid[:h, g - h:], grid[g - h:, :h], grid[g - h:, g - h:]]
        return np.hstack([F[:, q.ravel()].mean(1) for q in quads])


def logistic_regression(F, y, Ftest, l2=1e-3, iters=300, lr=0.5, classes=10):
    mu, sd = F.mean(0), F.std(0) + 1e-6
    F, Ftest = (F - mu) / sd, (Ftest - mu) / sd
    W = np.zeros((F.shape[1], classes)); b = np.zeros(classes)
    Y = np.eye(classes)[y]
    for _ in range(iters):
        z = F @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        W -= lr * (F.T @ (p - Y) / len(y) + l2 * W)
        b -= lr * (p - Y).mean(0)
    return (Ftest @ W + b).argmax(1)


def few_label_accuracy(F, y, per_class=20, splits=3, seed=0):
    """Train on `per_class` labelled images per class, test on all the rest; mean over `splits` random draws."""
    accs = []
    for s in range(splits):
        r = np.random.default_rng(seed + s)
        tr = np.concatenate([r.choice(np.where(y == c)[0], per_class, replace=False) for c in range(10)])
        te = np.setdiff1d(np.arange(len(y)), tr)
        accs.append(float((logistic_regression(F[tr], y[tr], F[te]) == y[te]).mean()))
    return float(np.mean(accs))


TRAINERS = {"R": dict(method="R"), "RP": dict(method="RP"), "OMP-1": dict(method="OMP", k=1),
            "OMP-5": dict(method="OMP", k=5), "SC": dict(method="SC", lam=1.0)}
NATURAL = {"R": ("T", 0.0), "RP": ("T", 0.0), "OMP-1": ("OMP", 1), "OMP-5": ("OMP", 5), "SC": ("SC", 1.0)}
ENCODERS = {"SC": ("SC", 1.0), "OMP-1": ("OMP", 1), "OMP-5": ("OMP", 5), "T": ("T", 0.25)}

REPORTED = {
    "Table 1 (CIFAR-10, 5-fold CV %, rows = training, columns = Natural / SC / OMP-1 / OMP-10 / T encoder)":
        "R 70.5 / 74.0 / 65.8 / 68.6 / 73.2; RP 76.0 / 76.6 / 70.1 / 71.6 / 78.1; RBM 74.1 / 76.7 / 69.5 / 72.9 / 78.3; "
        "SAE 74.8 / 76.5 / 68.8 / 71.5 / 76.7; SC 77.9 / 78.5 / 70.8 / 75.3 / 78.5; OMP-1 71.4 / 78.7 / 71.4 / 76.0 / "
        "78.9; OMP-2 73.8 / 78.5 / 71.0 / 75.8 / 79.0; OMP-5 75.4 / 78.8 / 71.0 / 76.1 / 79.1; OMP-10 75.3 / 79.0 / "
        "70.7 / 75.3 / 79.4",
    "Table 2 (CIFAR-10 test)": "random patches + soft threshold 79.1% (no training beyond alpha); SC/SC 78.8%; best "
                               "OMP-10/T 80.1%; OMP-1/T with d = 6000 basis vectors 81.5% (best known result)",
    "Table 3 (NORB)": "random patches + SC encoder 95.0% (better than published results; conv net 94.4%); soft "
                      "threshold best 93.6% in 1 hour vs sparse coding's 7 hours on 40 cores",
    "Caltech 101 (SC encoder, 30 train images / class)": "R 67.2%, RP 72.6%, SC 72.6%, OMP-1 71.9%",
    "setup": "1600-element dictionaries on whitened 6x6 colour patches (108-dim); stride 1; average pooling over 4 "
             "quadrants (4 x 2 x 1600 = 12,800 features); L2-SVM",
}
