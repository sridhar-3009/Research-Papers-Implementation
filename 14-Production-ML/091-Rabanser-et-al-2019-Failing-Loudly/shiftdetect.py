"""Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift (Rabanser, Guennemann, Lipton, NeurIPS
2019) -- the full pipeline on a laptop-sized image dataset (scikit-learn's 8x8 handwritten digits, 1,797 images).

  pipeline       source data x ~ p (validation set) and target data x' ~ q (incoming, possibly shifted)
                   -> dimensionality reduction (DR) to K dimensions
                   -> two-sample test(s), H0: p(x) = q(x')  ->  "shift" if p-value < alpha = 0.05
  DR methods     NoRed (raw pixels), PCA, SRP (sparse random projection, eq. 1), UAE (untrained autoencoder / random
                 encoder), TAE (trained autoencoder), BBSDs (softmax of the label classifier), BBSDh (its hard
                 prediction), Classif (a domain classifier trained to tell source from target)
  tests          multiple univariate Kolmogorov-Smirnov tests + Bonferroni (reject if min p < alpha / K); multivariate
                 MMD with an RBF kernel and a permutation test; chi-squared on predicted-class counts (BBSDh);
                 binomial test of the domain classifier's held-out accuracy against 0.5 (Classif)
  shifts         Gaussian noise (s/m/l), image transforms (rotation + translation + zoom, s/m/l), knock-out of class 0,
                 adversarial (FGSM), m img + ko, only-zero + m img -- each applied to a fraction delta of the target
  beyond         most anomalous samples = target samples the domain classifier is most sure are "target";
  detection      malignancy = the label classifier's accuracy on those samples (with their true labels)
"""

import numpy as np
from scipy import ndimage, stats

# ----------------------------------------------------------------------------------------------- data

def load_data(seed=0):
    """Digits scaled to [0, 1], split into train (fit the DR methods and the label classifier), validation (source
    samples p) and test (target pool q, which gets shifted)."""
    from sklearn.datasets import load_digits
    d = load_digits()
    X, y = d.data / 16.0, d.target
    idx = np.random.default_rng(seed).permutation(len(y))
    tr, va, te = idx[:900], idx[900:1350], idx[1350:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])

# ----------------------------------------------------------------------------------------------- a small numpy MLP

def _adam(params, grads, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] = state.get("t", 0) + 1
    for k in params:
        m = state.setdefault("m" + k, np.zeros_like(params[k]))
        v = state.setdefault("v" + k, np.zeros_like(params[k]))
        m[:] = b1 * m + (1 - b1) * grads[k]
        v[:] = b2 * v + (1 - b2) * grads[k] ** 2
        params[k] -= lr * (m / (1 - b1 ** state["t"])) / (np.sqrt(v / (1 - b2 ** state["t"])) + eps)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MLP:
    """64 -> hidden (ReLU) -> out. Used as the label classifier (softmax output) and as the domain classifier."""

    def __init__(self, d_in, hidden, d_out, seed=0):
        r = np.random.default_rng(seed)
        self.p = {"W1": r.normal(0, np.sqrt(2 / d_in), (d_in, hidden)), "b1": np.zeros(hidden),
                  "W2": r.normal(0, np.sqrt(1 / hidden), (hidden, d_out)), "b2": np.zeros(d_out)}

    def forward(self, X):
        h = np.maximum(0, X @ self.p["W1"] + self.p["b1"])
        return h, h @ self.p["W2"] + self.p["b2"]

    def proba(self, X):
        return softmax(self.forward(X)[1])

    def fit(self, X, y, steps=300, lr=1e-2, wd=1e-4):
        st = {}
        Y = np.eye(self.p["b2"].shape[0])[y]
        for _ in range(steps):
            h, z = self.forward(X)
            dz = (softmax(z) - Y) / len(y)
            dh = (dz @ self.p["W2"].T) * (h > 0)
            g = {"W2": h.T @ dz + wd * self.p["W2"], "b2": dz.sum(0), "W1": X.T @ dh + wd * self.p["W1"],
                 "b1": dh.sum(0)}
            _adam(self.p, g, st, lr)
        return self

    def input_grad(self, X, y):
        """Gradient of the cross-entropy w.r.t. the INPUT (for FGSM)."""
        h, z = self.forward(X)
        dz = softmax(z) - np.eye(z.shape[1])[y]
        dh = (dz @ self.p["W2"].T) * (h > 0)
        return dh @ self.p["W1"].T

# ----------------------------------------------------------------------------------------------- dimensionality reduction

class Reducers:
    """All DR methods, fitted on the training split; K latent dimensions for PCA / SRP / UAE / TAE."""

    def __init__(self, Xtr, ytr, K=16, seed=0):
        r = np.random.default_rng(seed)
        D = Xtr.shape[1]
        self.K = K
        self.mu = Xtr.mean(0)
        _, S, Vt = np.linalg.svd(Xtr - self.mu, full_matrices=False)
        self.R_pca = Vt[:K].T
        self.pca_explained = float((S[:K] ** 2).sum() / (S ** 2).sum())
        v = np.sqrt(D)                                                # SRP, eq. (1): density 1 / v
        u = r.random((D, K))
        self.R_srp = np.where(u < 1 / (2 * v), np.sqrt(v / K), np.where(u < 1 / v, -np.sqrt(v / K), 0.0))
        self.W_uae = r.normal(0, np.sqrt(2 / D), (D, K))              # untrained encoder: random ReLU layer
        self.b_uae = r.normal(0, 0.1, K)
        self.tae = self._train_autoencoder(Xtr, K, seed)
        self.clf = MLP(D, 64, 10, seed).fit(Xtr, ytr, steps=400)

    @staticmethod
    def _train_autoencoder(X, K, seed, steps=600):
        r = np.random.default_rng(seed + 1)
        D = X.shape[1]
        p = {"We": r.normal(0, np.sqrt(1 / D), (D, K)), "be": np.zeros(K), "Wd": r.normal(0, np.sqrt(1 / K), (K, D)),
             "bd": np.zeros(D)}
        st = {}
        for _ in range(steps):
            h = np.tanh(X @ p["We"] + p["be"])
            out = h @ p["Wd"] + p["bd"]
            d_out = 2 * (out - X) / len(X)
            dh = (d_out @ p["Wd"].T) * (1 - h ** 2)
            _adam(p, {"Wd": h.T @ d_out, "bd": d_out.sum(0), "We": X.T @ dh, "be": dh.sum(0)}, st, 5e-3)
        return p

    def transform(self, X, method):
        if method == "NoRed":
            return X
        if method == "PCA":
            return (X - self.mu) @ self.R_pca
        if method == "SRP":
            return X @ self.R_srp
        if method == "UAE":
            return np.maximum(0, X @ self.W_uae + self.b_uae)
        if method == "TAE":
            return np.tanh(X @ self.tae["We"] + self.tae["be"])
        if method == "BBSDs":
            return self.clf.proba(X)
        if method == "BBSDh":
            return self.clf.proba(X).argmax(1)
        raise ValueError(method)

# ----------------------------------------------------------------------------------------------- two-sample tests

def ks_bonferroni(A, B):
    """One KS test per dimension; the aggregated p-value is min(p) * K (Bonferroni), capped at 1."""
    A, B = np.atleast_2d(A.T).T, np.atleast_2d(B.T).T
    ps = [stats.ks_2samp(A[:, j], B[:, j]).pvalue for j in range(A.shape[1])]
    return float(min(1.0, min(ps) * len(ps)))


def mmd2_unbiased(K, m):
    """Unbiased squared MMD from the joint kernel matrix K (first m rows = sample 1), eq. (3)."""
    Kxx, Kyy, Kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    n = K.shape[0] - m
    return ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
            - 2 * Kxy.mean())


def mmd_test(A, B, perms=100, seed=0):
    """MMD^2 with kernel exp(-||x - x'||^2 / sigma), sigma = median squared distance of the pooled sample; the p-value
    is the share of label permutations whose MMD^2 is at least the observed one."""
    Z = np.vstack([A, B])
    sq = (Z ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    sigma = np.median(D2[np.triu_indices(len(Z), 1)]) or 1.0
    K = np.exp(-D2 / sigma)
    m = len(A)
    obs = mmd2_unbiased(K, m)
    r = np.random.default_rng(seed)
    count = 0
    for _ in range(perms):
        idx = r.permutation(len(Z))
        count += mmd2_unbiased(K[np.ix_(idx, idx)], m) >= obs
    return float((count + 1) / (perms + 1))


def chi2_test(a, b, classes=10):
    """Pearson chi-squared test of homogeneity on predicted-class counts, eq. (5)."""
    O = np.array([np.bincount(a, minlength=classes), np.bincount(b, minlength=classes)])
    O = O[:, O.sum(0) > 0]
    if O.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(O)[1])


def domain_classifier_test(A, B, seed=0, return_model=False):
    """Split source and target in halves; train a classifier source (0) vs target (1) on the first halves; binomial
    test of its accuracy on the second halves against chance (0.5)."""
    r = np.random.default_rng(seed)
    ia, ib = r.permutation(len(A)), r.permutation(len(B))
    ha, hb = len(A) // 2, len(B) // 2
    Xtr = np.vstack([A[ia[:ha]], B[ib[:hb]]])
    ytr = np.r_[np.zeros(ha, int), np.ones(hb, int)]
    Xte = np.vstack([A[ia[ha:]], B[ib[hb:]]])
    yte = np.r_[np.zeros(len(A) - ha, int), np.ones(len(B) - hb, int)]
    clf = MLP(A.shape[1], 32, 2, seed).fit(Xtr, ytr, steps=200)
    correct = int((clf.proba(Xte).argmax(1) == yte).sum())
    p = float(stats.binomtest(correct, len(yte), 0.5).pvalue)
    return (p, clf) if return_model else p


METHODS = [("NoRed", "univ"), ("PCA", "univ"), ("SRP", "univ"), ("UAE", "univ"), ("TAE", "univ"), ("BBSDs", "univ"),
           ("BBSDh", "chi2"), ("Classif", "bin"),
           ("NoRed", "mmd"), ("PCA", "mmd"), ("SRP", "mmd"), ("UAE", "mmd"), ("TAE", "mmd"), ("BBSDs", "mmd")]


def detect(red, Xs, Xt, method, test, perms=100, seed=0):
    """p-value of the two-sample test for one DR method / test pair."""
    if method == "Classif":
        return domain_classifier_test(Xs, Xt, seed)
    A, B = red.transform(Xs, method), red.transform(Xt, method)
    if test == "chi2":
        return chi2_test(A, B)
    if test == "univ":
        return ks_bonferroni(A, B)
    return mmd_test(A, B, perms, seed)

# ----------------------------------------------------------------------------------------------- shifts

GN = {"s_gn": 0.05, "m_gn": 0.2, "l_gn": 1.0}                         # noise std on [0, 1] pixels (our choice)
IMG = {"s_img": (10, 0.05, 0.1), "m_img": (40, 0.2, 0.2), "l_img": (90, 0.4, 0.4)}   # rotation, translation, zoom


def image_transform(x, rot, trans, zoom, r):
    img = x.reshape(8, 8)
    img = ndimage.rotate(img, r.uniform(-rot, rot), reshape=False, order=1, mode="constant")
    img = ndimage.shift(img, r.uniform(-trans, trans, 2) * 8, order=1, mode="constant")
    z = 1 + r.uniform(0, zoom)
    big = ndimage.zoom(img, z, order=1)
    o = (big.shape[0] - 8) // 2
    return np.clip(big[o:o + 8, o:o + 8], 0, 1).ravel()


def apply_shift(X, y, shift, delta, red=None, seed=0):
    """Return a shifted copy of the target pool (X, y). Fraction `delta` of samples is affected (for ko: a fraction
    delta of class 0 is removed)."""
    r = np.random.default_rng(seed)
    X, y = X.copy(), y.copy()
    if shift == "none":
        return X, y
    if shift in ("ko", "m_img+ko"):
        if shift == "m_img+ko":
            X, y = apply_shift(X, y, "m_img", 0.5, red, seed + 1)
        zeros = np.where(y == 0)[0]
        drop = r.choice(zeros, int(round(delta * len(zeros))), replace=False)
        keep = np.setdiff1d(np.arange(len(y)), drop)
        return X[keep], y[keep]
    if shift == "oz+m_img":
        X, y = X[y == 0], y[y == 0]
        return apply_shift(X, y, "m_img", delta, red, seed + 1)
    hit = r.random(len(y)) < delta
    if shift in GN:
        X[hit] = np.clip(X[hit] + r.normal(0, GN[shift], X[hit].shape), 0, 1)
    elif shift in IMG:
        for i in np.where(hit)[0]:
            X[i] = image_transform(X[i], *IMG[shift], r)
    elif shift == "adv":                                            # FGSM against the label classifier
        g = red.clf.input_grad(X[hit], y[hit])
        X[hit] = np.clip(X[hit] + 0.15 * np.sign(g), 0, 1)
    else:
        raise ValueError(shift)
    return X, y

# ----------------------------------------------------------------------------------------------- experiments

def detection_rate(red, Xva, Xte_shifted, method, test, n, reps=10, perms=100, alpha=0.05, seed=0):
    """Share of `reps` random draws of n source and n target samples in which the test rejects H0 at alpha."""
    hits = 0
    for k in range(reps):
        r = np.random.default_rng(seed * 1000 + k)
        A = Xva[r.choice(len(Xva), n, replace=False)]
        B = Xte_shifted[r.choice(len(Xte_shifted), min(n, len(Xte_shifted)), replace=False)]
        hits += detect(red, A, B, method, test, perms, seed * 1000 + k) < alpha
    return hits / reps


def most_anomalous(Xs, Xt, yt, red, top=10, seed=0):
    """Train the domain classifier on all available source / target samples; rank target samples by P(target);
    the malignancy check = label-classifier accuracy on the top-ranked ones (labels assumed obtainable for them)."""
    _, clf = domain_classifier_test(Xs, Xt, seed, return_model=True)
    score = clf.proba(Xt)[:, 1]
    order = np.argsort(-score)[:top]
    acc_top = float((red.clf.proba(Xt[order]).argmax(1) == yt[order]).mean())
    return order, score[order], acc_top


REPORTED = {
    "setup": "MNIST and CIFAR-10; ResNet-18 label and domain classifiers; K = 32 latent dims for PCA / SRP / UAE / "
             "TAE; alpha = 0.05; target sample sizes 10 ... 10,000 (kernel tests only up to 1,000); 5 random splits",
    "Table 1a (univariate, all shifts)": "detection accuracy at 10 / 100 / 1,000 / 10,000 samples: NoRed 0.03 / 0.36 / "
                                         "0.54 / 0.72, BBSDs 0.19 / 0.47 / 0.70 / 0.79, BBSDh (chi2) 0.03 / 0.22 / "
                                         "0.46 / 0.57, Classif (binomial) 0.01 / 0.21 / 0.51 / 0.67",
    "findings": "BBSDs best overall (univariate); UAE best for multivariate tests; multiple univariate tests with "
                "Bonferroni comparable to multivariate MMD despite the heavy correction; the domain classifier is bad "
                "with <= 100 samples but catches up; large shifts detectable with ~20 samples, small ones need orders "
                "of magnitude more; 10% affected samples hard to detect, 50% / 100% detectable with 50 samples",
    "Table 1b (BBSDs, univariate)": "easy: l_gn, l_img, m_img+ko, oz+m_img (and adv); hard even with many samples: "
                                    "s_gn, m_gn, ko",
    "MNIST original split": "not i.i.d.: domain classifier's most anomalous samples were mostly 6s (training 6s "
                            "rotated slightly right, test 6s more open and centred); KS p = 2.7e-10 vs Bonferroni "
                            "threshold 6.3e-5; the shift was judged harmless by the malignancy check",
}
