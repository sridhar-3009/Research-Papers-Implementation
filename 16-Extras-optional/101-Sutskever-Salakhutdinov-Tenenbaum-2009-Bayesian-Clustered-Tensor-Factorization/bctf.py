"""Modelling Relational Data using Bayesian Clustered Tensor Factorization (Sutskever, Salakhutdinov, Tenenbaum;
NIPS 2009).

  data          triples (a, r, b) with truth values t (binary, shifted to mean 0)
  factorization each object a has a LEFT vector a_L and a RIGHT vector a_R (d-dim), each relation r a d x d matrix R;
                prediction  t ~ N(a_L^T R b_R, sigma^2)   (Gaussian likelihood: conjugate, so Gibbs sampling is easy)
  clustering    a Chinese Restaurant Process partitions the objects (and, separately, the relations); each cluster has
                its own mean and diagonal covariance with a Normal-Inverse-Gamma prior (mu | Sigma ~ N(0, Sigma),
                sigma_d^2 ~ IG(alpha, 1)), and members' vectors are drawn from it -> objects in a cluster get similar
                vectors, so predictions are mostly determined by clusters, but individual vectors can still differ
  inference     MAP by conjugate gradient to initialise; then MCMC: collapsed Gibbs over cluster assignments, sample the
                cluster means / variances, sample every a_L (Gaussian, factorises over objects because a_L and a_R are
                separate), every a_R, every R (the paper uses hybrid Monte Carlo for R; with our small d the Gaussian
                conditional is sampled exactly), and the noise variance; predictions average over samples
  baselines     MAP tensor factorization, BTF (= BCTF with everything in one cluster), and an IRM-like block model
                (cluster objects and relations, predict each block's mean)

The paper's datasets (Animals, Kinship, UML, MovieLens, ConceptNet) are not used here; instead relational data with a
PLANTED cluster + low-rank structure is generated, so the recovered clusters can be checked against the truth.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize

# ----------------------------------------------------------------------------------------------- planted data

def planted_data(n_obj=60, n_rel=6, k_obj=4, k_rel=2, d=3, noise=0.5, seed=0):
    """Objects in k_obj clusters and relations in k_rel clusters; true vectors = cluster mean + small jitter;
    t(a, r, b) = 1 if a_L^T R b_R + noise > threshold (about 30% true). Returns the full binary tensor and labels."""
    r = np.random.default_rng(seed)
    co, cr = r.integers(0, k_obj, n_obj), r.integers(0, k_rel, n_rel)
    mL, mR, mRel = r.normal(0, 1, (k_obj, d)), r.normal(0, 1, (k_obj, d)), r.normal(0, 1, (k_rel, d, d))
    aL = mL[co] + 0.25 * r.standard_normal((n_obj, d))
    aR = mR[co] + 0.25 * r.standard_normal((n_obj, d))
    R = mRel[cr] + 0.25 * r.standard_normal((n_rel, d, d))
    score = np.einsum("ad,rde,be->arb", aL, R, aR) + noise * r.standard_normal((n_obj, n_rel, n_obj))
    T = (score > np.quantile(score, 0.7)).astype(float)
    return T, co, cr


def split(T, test_frac=0.1, train_frac=1.0, seed=0):
    """Random test set (test_frac of all entries); of the rest, keep train_frac as observed training data."""
    r = np.random.default_rng(seed)
    idx = np.array(np.unravel_index(np.arange(T.size), T.shape)).T
    perm = r.permutation(len(idx))
    n_test = int(test_frac * len(idx))
    test, rest = idx[perm[:n_test]], idx[perm[n_test:]]
    train = rest[:int(train_frac * len(rest))]
    return train, test

# ----------------------------------------------------------------------------------------------- metrics

def rmse(pred, t):
    return float(np.sqrt(((pred - t) ** 2).mean()))


def pr_auc(score, y):
    """Area under the precision-recall curve (average precision), the paper's 'AUC'."""
    order = np.argsort(-score, kind="stable")
    yy = y[order]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / max(yy.sum(), 1))

# ----------------------------------------------------------------------------------------------- MAP factorization

def predict(aL, R, aR, idx):
    return np.einsum("nd,nde,ne->n", aL[idx[:, 0]], R[idx[:, 1]], aR[idx[:, 2]])


def fit_map(train, t, n_obj, n_rel, d, lam=0.1, seed=0, maxiter=500):
    """Penalised least squares  sum (t - a_L^T R b_R)^2 / 2 + lam / 2 ||theta||^2  by conjugate gradient."""
    r = np.random.default_rng(seed)
    shapes = [(n_obj, d), (n_rel, d, d), (n_obj, d)]
    sizes = [int(np.prod(s)) for s in shapes]
    x0 = 0.3 * r.standard_normal(sum(sizes))

    def unpack(x):
        out, i = [], 0
        for s, n in zip(shapes, sizes):
            out.append(x[i:i + n].reshape(s)); i += n
        return out

    def f(x):
        aL, R, aR = unpack(x)
        A, Rr, B = aL[train[:, 0]], R[train[:, 1]], aR[train[:, 2]]
        RB = np.einsum("nde,ne->nd", Rr, B)
        e = np.einsum("nd,nd->n", A, RB) - t
        gL = np.zeros_like(aL); np.add.at(gL, train[:, 0], e[:, None] * RB)
        gR = np.zeros_like(aR); np.add.at(gR, train[:, 2], e[:, None] * np.einsum("nde,nd->ne", Rr, A))
        gM = np.zeros_like(R); np.add.at(gM, train[:, 1], e[:, None, None] * A[:, :, None] * B[:, None, :])
        cost = 0.5 * (e ** 2).sum() + 0.5 * lam * (x ** 2).sum()
        return cost, np.concatenate([gL.ravel(), gM.ravel(), gR.ravel()]) + lam * x

    res = minimize(f, x0, jac=True, method="CG", options={"maxiter": maxiter})
    return unpack(res.x)

# ----------------------------------------------------------------------------------------------- MCMC pieces

def niw_post(X, alpha, beta=1.0):
    """Normal-Inverse-Gamma posterior for the members X (n x D) of one cluster, prior mu | s2 ~ N(0, s2),
    s2 ~ IG(alpha, beta) per dimension. Returns (kappa_n, mu_n, alpha_n, beta_n) per dimension."""
    n = len(X)
    s = X.sum(0) if n else np.zeros(X.shape[1])
    ss = (X ** 2).sum(0) if n else np.zeros(X.shape[1])
    kn = 1.0 + n
    mun = s / kn
    an = alpha + n / 2
    bn = beta + 0.5 * (ss - s ** 2 / kn)
    return kn, mun, an, bn


def sample_cluster_params(X, labels, K, alpha, rng, beta=1.0):
    D = X.shape[1]
    mu, var = np.zeros((K, D)), np.ones((K, D))
    for k in range(K):
        kn, mun, an, bn = niw_post(X[labels == k], alpha, beta)
        var[k] = 1 / rng.gamma(an, 1 / bn)
        mu[k] = mun + np.sqrt(var[k] / kn) * rng.standard_normal(D)
    return mu, var


def log_predictive(x, Xk, alpha, beta=1.0):
    """log p(x | members Xk) under the collapsed Normal-Inverse-Gamma model (Student-t per dimension)."""
    kn, mun, an, bn = niw_post(Xk, alpha, beta)
    scale = np.sqrt(bn * (kn + 1) / (an * kn))
    return float(stats.t.logpdf(x, 2 * an, mun, scale).sum())


def gibbs_assignments(X, labels, alpha, alpha_dp, rng, beta=1.0):
    """One collapsed Gibbs sweep over the CRP assignments of the rows of X."""
    labels = labels.copy()
    for i in rng.permutation(len(X)):
        labels[i] = -1
        ks, counts = np.unique(labels[labels >= 0], return_counts=True)
        logp = [np.log(c) + log_predictive(X[i], X[labels == k], alpha, beta) for k, c in zip(ks, counts)]
        logp.append(np.log(alpha_dp) + log_predictive(X[i], X[:0], alpha, beta))
        logp = np.array(logp)
        p = np.exp(logp - logp.max()); p /= p.sum()
        j = rng.choice(len(p), p=p)
        labels[i] = ks[j] if j < len(ks) else (labels.max() + 1)
    _, labels = np.unique(labels, return_inverse=True)                # relabel 0..K-1
    return labels


def sample_vectors(design, t, owner, n_owner, prior_mu, prior_var, noise_var, rng):
    """Gaussian conditional for each owner's vector v: t_n ~ N(v . design_n, noise_var), v ~ N(prior_mu, diag(prior_var)).
    design: (n_obs, D); owner: (n_obs,) index of the vector each observation depends on."""
    D = design.shape[1]
    P = np.zeros((n_owner, D, D))
    np.add.at(P, owner, design[:, :, None] * design[:, None, :] / noise_var)
    P += np.eye(D)[None] / prior_var[:, None, :]
    h = np.zeros((n_owner, D))
    np.add.at(h, owner, design * t[:, None] / noise_var)
    h += prior_mu / prior_var
    L = np.linalg.cholesky(P)
    mean = np.linalg.solve(P, h[:, :, None])[:, :, 0]
    z = rng.standard_normal((n_owner, D))
    return mean + np.linalg.solve(L.transpose(0, 2, 1), z[:, :, None])[:, :, 0]


def run_mcmc(train, t, n_obj, n_rel, d, clustered=True, sweeps=60, burn=20, seed=0, alpha=2.0, alpha_dp=1.0,
             init=None, test=None, init_clusters=10, beta=None):
    """BTF (clustered=False: one cluster each) or BCTF. Returns the posterior-mean predictions for `test` and the last
    object / relation partitions."""
    rng = np.random.default_rng(seed)
    aL, R, aR = init if init is not None else fit_map(train, t, n_obj, n_rel, d, seed=seed)
    aL, R, aR = aL.copy(), R.copy(), aR.copy()
    # the paper samples its hyperparameters; here the inverse-Gamma scale is tied to the size of the MAP vectors
    # (a fraction of their variance), so 'a cluster' means 'much tighter than the overall spread'
    b_obj = beta if beta is not None else 0.05 * float(np.var(np.hstack([aL, aR])))
    b_rel = beta if beta is not None else 0.05 * float(np.var(R))
    # single-site Gibbs rarely SPLITS a big cluster (the paper adds split-merge moves); we start from many small
    # random clusters instead and let the sampler merge them
    co = rng.integers(0, init_clusters, n_obj) if clustered else np.zeros(n_obj, int)
    cr = rng.integers(0, min(init_clusters, n_rel), n_rel) if clustered else np.zeros(n_rel, int)
    _, co = np.unique(co, return_inverse=True)
    _, cr = np.unique(cr, return_inverse=True)
    noise_var = 1.0
    preds, n_kept = np.zeros(len(test)), 0
    for s in range(sweeps):
        Xo = np.hstack([aL, aR])
        Xr = R.reshape(n_rel, -1)
        if clustered:
            co = gibbs_assignments(Xo, co, alpha, alpha_dp, rng, b_obj)
            cr = gibbs_assignments(Xr, cr, alpha, alpha_dp, rng, b_rel)
        mo, vo = sample_cluster_params(Xo, co, co.max() + 1, alpha, rng, b_obj)
        mr, vr = sample_cluster_params(Xr, cr, cr.max() + 1, alpha, rng, b_rel)
        # a_L given everything else
        design = np.einsum("nde,ne->nd", R[train[:, 1]], aR[train[:, 2]])
        aL = sample_vectors(design, t, train[:, 0], n_obj, mo[co][:, :d], vo[co][:, :d], noise_var, rng)
        design = np.einsum("nde,nd->ne", R[train[:, 1]], aL[train[:, 0]])
        aR = sample_vectors(design, t, train[:, 2], n_obj, mo[co][:, d:], vo[co][:, d:], noise_var, rng)
        design = (aL[train[:, 0]][:, :, None] * aR[train[:, 2]][:, None, :]).reshape(len(train), -1)
        R = sample_vectors(design, t, train[:, 1], n_rel, mr[cr], vr[cr], noise_var, rng).reshape(n_rel, d, d)
        e = t - predict(aL, R, aR, train)
        noise_var = 1 / rng.gamma(1 + len(e) / 2, 1 / (1 + 0.5 * (e ** 2).sum()))
        if s >= burn:
            preds += predict(aL, R, aR, test)
            n_kept += 1
    return preds / max(n_kept, 1), co, cr

# ----------------------------------------------------------------------------------------------- IRM-like block model

def kmeans(X, k, rng, iters=50):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


def block_model(train, t, n_obj, n_rel, k_obj, k_rel, test, seed=0):
    """A cluster-only model in the spirit of the IRM: cluster objects by their observed relation profiles and relations
    by theirs (k-means), then predict each (object cluster, relation cluster, object cluster) block's smoothed mean."""
    rng = np.random.default_rng(seed)
    full = np.zeros((n_obj, n_rel, n_obj)); obs = np.zeros_like(full)
    full[tuple(train.T)] = t; obs[tuple(train.T)] = 1
    mean_fill = lambda s, c: s / np.maximum(c, 1)
    prof_o = np.hstack([mean_fill(full.sum(2), obs.sum(2)), mean_fill(full.sum(0).T, obs.sum(0).T)])
    prof_r = np.hstack([mean_fill(full.sum(2).T, obs.sum(2).T), mean_fill(full.sum(0), obs.sum(0))])
    co, cr = kmeans(prof_o, k_obj, rng), kmeans(prof_r, k_rel, rng)
    S = np.zeros((k_obj, k_rel, k_obj)); C = np.zeros_like(S)
    np.add.at(S, (co[train[:, 0]], cr[train[:, 1]], co[train[:, 2]]), t)
    np.add.at(C, (co[train[:, 0]], cr[train[:, 1]], co[train[:, 2]]), 1)
    block = S / (C + 1)                                                # smoothed towards 0 (the data mean)
    return block[co[test[:, 0]], cr[test[:, 1]], co[test[:, 2]]], co, cr


def adjusted_rand(a, b):
    """Adjusted Rand index between two partitions (1 = identical, ~0 = chance)."""
    from math import comb
    ua, ub = np.unique(a), np.unique(b)
    M = np.array([[np.sum((a == i) & (b == j)) for j in ub] for i in ua])
    s_ij = sum(comb(int(x), 2) for x in M.ravel())
    s_a = sum(comb(int(x), 2) for x in M.sum(1))
    s_b = sum(comb(int(x), 2) for x in M.sum(0))
    tot = comb(len(a), 2)
    exp = s_a * s_b / tot
    mx = (s_a + s_b) / 2
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 1.0

# ----------------------------------------------------------------------------------------------- one full comparison

def compare(train_frac=1.0, d=3, seed=0, sweeps=60, burn=20, n_obj=60, n_rel=6):
    T, co_true, cr_true = planted_data(n_obj=n_obj, n_rel=n_rel, seed=seed)
    mean = T.mean()
    train, test = split(T, 0.1, train_frac, seed)
    t = T[tuple(train.T)] - mean
    y = T[tuple(test.T)]
    res = {}
    init = fit_map(train, t, n_obj, n_rel, d, seed=seed)
    p = predict(*init, test) + mean
    res["MAP"] = (rmse(p, y), pr_auc(p, y))
    pb, _, _ = run_mcmc(train, t, n_obj, n_rel, d, clustered=False, sweeps=sweeps, burn=burn, seed=seed, init=init,
                        test=test)
    res["BTF"] = (rmse(pb + mean, y), pr_auc(pb, y))
    pc, co, cr = run_mcmc(train, t, n_obj, n_rel, d, clustered=True, sweeps=sweeps, burn=burn, seed=seed, init=init,
                          test=test)
    res["BCTF"] = (rmse(pc + mean, y), pr_auc(pc, y))
    pi, co_b, _ = block_model(train, t, n_obj, n_rel, 4, 2, test, seed)
    res["block model (IRM-like)"] = (rmse(pi + mean, y), pr_auc(pi, y))
    clusters = {"BCTF objects": (int(co.max() + 1), adjusted_rand(co, co_true)),
                "BCTF relations": (int(cr.max() + 1), adjusted_rand(cr, cr_true)),
                "block model objects": (4, adjusted_rand(co_b, co_true))}
    return res, clusters, len(train)


REPORTED = {
    "Table 1 (RMSE / AUC = area under the precision-recall curve)":
        "Animals: MAP20 0.467 / 0.78, BTF20 0.337 / 0.85, BCTF20 0.331 / 0.86, IRM 0.382 / 0.75, MRC - / 0.81; "
        "Kinship: MAP20 0.122 / 0.82, MAP40 0.110 / 0.90, BCTF40 0.108 / 0.90, IRM 0.140 / 0.66, MRC 0.85; "
        "UML: MAP40 0.024 / 0.98, BCTF40 0.024 / 0.98, IRM 0.054 / 0.70, MRC 0.98; "
        "MovieLens RMSE: MAP20 0.899, MAP40 0.933, BTF20 0.835, BCTF20 0.836; "
        "ConceptNet: MAP20 0.536 / 0.57, MAP40 0.614 / 0.48, BTF40 0.267 / 0.94, BCTF40 0.260 / 0.94",
    "findings": "BCTF beats IRM and MRC; on dense Kinship / UML MAP is as good as the Bayesian models (many more "
                "observations than parameters); on the small Animals dataset BTF beats MAP and BCTF beats BTF; on the "
                "sparse MovieLens / ConceptNet MAP overfits badly and the fully Bayesian models are much better",
    "inference": "MAP by conjugate gradient to initialise; collapsed Gibbs + split-merge for the partitions; exact "
                 "Gaussian sampling of a_L and a_R (factorises over objects because of the two vectors); hybrid Monte "
                 "Carlo for R (10 leapfrog steps of 1e-5); Metropolis-Hastings for hyperparameters",
    "datasets": "Animals (50 x 85), Kinship (104 people, 26 relations), UML (135 terms, 49 relations), MovieLens "
                "(1,000,209 ratings), ConceptNet (7,000 objects, 19 relations, 82,062 true facts + 2x random negatives)",
}
