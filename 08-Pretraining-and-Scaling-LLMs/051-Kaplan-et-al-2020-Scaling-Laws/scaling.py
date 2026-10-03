"""Scaling Laws for Neural Language Models (Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu &
Amodei, 2020).

Test loss L (nats/token, WebText2, BPE 50,257) of decoder-only Transformers as a function of
  N = non-embedding parameters ~ 12 n_layer d_model^2                                                  (Eq. 2.1)
  D = dataset tokens,  C ~ 6 N B S = training compute (B batch tokens, S steps; PF-day = 8.64e19 FLOPs)
  L(N)      = (N_c / N)^a_N,      a_N ~ 0.076, N_c ~ 8.8e13                                                 (1.1)
  L(D)      = (D_c / D)^a_D,      a_D ~ 0.095, D_c ~ 5.4e13                                                 (1.2)
  L(C_min)  = (C_c / C_min)^a_C,  a_C ~ 0.050, C_c ~ 3.1e8 PF-days                                          (1.3)
  B_crit(L) = B* / L^(1/a_B),     B* ~ 2e8 tokens, a_B ~ 0.21                                               (1.4)
  L(N, D)   = [ (N_c/N)^(a_N/a_D) + D_c/D ]^a_D         (Table 2: a_N 0.076, a_D 0.103, N_c 6.4e13, D_c 1.8e13) (1.5)
  L(N, S)   = (N_c/N)^a_N + (S_c/S_min)^a_S             (Table 3: a_N 0.077, a_S 0.76, N_c 6.5e13, S_c 2.1e3)  (1.6)
  S_min = S / (1 + B_crit/B),  C_min = C / (1 + B/B_crit)                                                (5.4, 5.5)
  compute-efficient allocation: N ~ C^(a_C/a_N), B ~ C^(a_C/a_B), S ~ C^(a_C/a_S),
      a_C = 1 / (1/a_S + 1/a_B + 1/a_N)                                                                     (1.7, 1.8)
  empirically N_opt = 1.3e9 C^0.73 params, B = 2.0e6 C^0.24 tokens, S = 5.4e3 C^0.03, D = 2e10 C^0.27 (Table 5)
  overfitting: dL = L(N,D)/L(N,inf) - 1 ~ (1 + (N/N_c)^(a_N/a_D) D_c/D)^a_D - 1; avoid with D >~ 5e3 N^0.74 (4.3, 4.4)
"""

import math

import numpy as np

# ---------------------------------------------------------------------------------------------------- constants
A_N, N_C = 0.076, 8.8e13
A_D, D_C = 0.095, 5.4e13
A_CMIN, C_CMIN = 0.050, 3.1e8                                                       # PF-days
A_C_FIXED_BATCH, C_C = 0.057, 1.6e7
A_B, B_STAR = 0.21, 2e8
ND = dict(a_N=0.076, a_D=0.103, N_c=6.4e13, D_c=1.8e13)                              # Table 2
NS = dict(a_N=0.077, a_S=0.76, N_c=6.5e13, S_c=2.1e3)                                # Table 3
PF_DAY = 8.64e19


# ---------------------------------------------------------------------------------------------------- the laws
def L_of_N(N):
    return (N_C / N) ** A_N


def L_of_D(D):
    return (D_C / D) ** A_D


def L_of_Cmin(C_pf_days):
    return (C_CMIN / C_pf_days) ** A_CMIN


def L_of_ND(N, D, p=ND):
    return ((p["N_c"] / N) ** (p["a_N"] / p["a_D"]) + p["D_c"] / D) ** p["a_D"]


def L_of_NS(N, S_min, p=NS):
    return (p["N_c"] / N) ** p["a_N"] + (p["S_c"] / S_min) ** p["a_S"]


def B_crit(L):
    return B_STAR / L ** (1 / A_B)


def S_min(S, B, L):
    return S / (1 + B_crit(L) / B)


def C_min(C, B, L):
    return C / (1 + B / B_crit(L))


def overfit_penalty(N, D, p=ND):
    """Eq. 4.3: the relative excess loss from finite data."""
    return (1 + (N / p["N_c"]) ** (p["a_N"] / p["a_D"]) * p["D_c"] / D) ** p["a_D"] - 1


def tokens_to_avoid_overfitting(N):
    """Eq. 4.4: D >~ 5e3 N^0.74 keeps the overfitting penalty below the ~0.02 seed-to-seed noise."""
    return 5e3 * N ** 0.74


def S_stop_lower_bound(N, D):
    """Eq. 5.7: early stopping happens no earlier than S_c / (L(N, D) - L(N, inf))^(1/a_S)."""
    gap = L_of_ND(N, D) - L_of_ND(N, float("inf"))
    return NS["S_c"] / gap ** (1 / NS["a_S"])


# ---------------------------------------------------------------------------------------------------- sizes and compute
def non_embedding_params(n_layer, d_model):
    """Eq. 2.1 with d_attn = d_ff / 4 = d_model: N = 2 d n (2 d_attn + d_ff) = 12 n d^2."""
    return 12 * n_layer * d_model ** 2


def forward_flops_per_token(N, n_layer, n_ctx, d_attn):
    """Eq. 2.2: C_forward ~ 2N + 2 n_layer n_ctx d_attn (multiply-adds counted as 2 FLOPs)."""
    return 2 * N + 2 * n_layer * n_ctx * d_attn


def training_compute(N, tokens):
    """C ~ 6 N D: forward (2N) + backward (4N) per token."""
    return 6 * N * tokens


def allocation_exponents(a_N=NS["a_N"], a_S=NS["a_S"], a_B=A_B):
    """Eq. 1.8: the compute exponent and how N, B, S should scale with compute."""
    a_C = 1 / (1 / a_S + 1 / a_B + 1 / a_N)
    return {"a_C_min": a_C, "N": a_C / a_N, "B": a_C / a_B, "S": a_C / a_S}


def table5(C_pf_days):
    """Table 5's empirical compute-efficient recipe for a budget in PF-days."""
    return {"N_opt": 1.3e9 * C_pf_days ** 0.73, "B": 2.0e6 * C_pf_days ** 0.24, "S_min": 5.4e3 * C_pf_days ** 0.03,
            "D_opt": 2e10 * C_pf_days ** 0.27}


def loss_at_compute(N, C_pf_days, iters=100):
    """L(N, S_min) when compute C_min (PF-days) is spent on a model of size N:
    C_min = 6 N E_min and E_min = B_crit(L) S_min, so S_min depends on the loss we reach. Solve
    L = L(N, S_min(L)) by bisection: the left side rises with L, the right side falls (larger L -> smaller B_crit ->
    more steps -> lower loss), so there is exactly one crossing."""
    E = C_pf_days * PF_DAY / (6 * N)                                               # tokens processed (minimum)
    lo, hi = L_of_NS(N, 1e30), 1e4
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if mid - L_of_NS(N, E / B_crit(mid)) > 0:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def compute_optimal_N(C_pf_days, grid=None):
    """The model size minimising loss for a compute budget, by search over N (Section 6.1, Figure 14)."""
    grid = np.logspace(5, 13, 400) if grid is None else grid
    losses = [loss_at_compute(N, C_pf_days) for N in grid]
    k = int(np.argmin(losses))
    return grid[k], losses[k]


# ---------------------------------------------------------------------------------------------------- fitting
def fit_power_law(x, y):
    """Fit y = (x_c / x)^a by least squares in log space: log y = a log x_c - a log x. Returns (a, x_c)."""
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    a = -slope
    return a, math.exp(intercept / a)


def fit_L_ND(N, D, L, steps=4000, lr=0.05):
    """Fit Eq. 1.5's four parameters by gradient descent on squared log error (initial guess from the paper)."""
    import torch
    N, D, L = (torch.tensor(np.asarray(v), dtype=torch.float64) for v in (N, D, L))
    theta = torch.tensor([math.log(0.076), math.log(0.103), math.log(6.4e13), math.log(1.8e13)],
                         dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr)
    for _ in range(steps):
        aN, aD, Nc, Dc = theta.exp()
        pred = ((Nc / N) ** (aN / aD) + Dc / D) ** aD
        loss = ((pred.log() - L.log()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    aN, aD, Nc, Dc = theta.exp().tolist()
    return {"a_N": aN, "a_D": aD, "N_c": Nc, "D_c": Dc}


# ---------------------------------------------------------------------------------------------------- why power laws
def spectrum_toy(n_features=4000, alpha=1.5, seed=0):
    """A minimal mechanism (not from this paper): data whose 'patterns' have power-law importance lambda_k ~ k^-alpha.
    A model of size N captures the N most important; the rest is error. Returns (lambda, w*)."""
    rng = np.random.default_rng(seed)
    lam = np.arange(1, n_features + 1) ** -alpha
    w = rng.standard_normal(n_features)
    return lam, w


def spectrum_loss_vs_N(lam, w, Ns):
    """Infinite data, capacity N: the irreducible error is sum_{k > N} lambda_k w_k^2 ~ N^-(alpha - 1)."""
    return np.array([float((lam[N:] * w[N:] ** 2).sum()) for N in Ns])


def spectrum_loss_vs_D(lam, w, Ds, ridge=1e-3, n_test=4000, seed=1):
    """Unlimited capacity, D samples: ridge regression on x ~ N(0, diag lambda), y = w.x; returns test MSE."""
    rng = np.random.default_rng(seed)
    K = len(lam)
    Xt = rng.standard_normal((n_test, K)) * np.sqrt(lam)
    out = []
    for D in Ds:
        X = rng.standard_normal((D, K)) * np.sqrt(lam)
        y = X @ w
        if D < K:                                                                 # dual form: (X X^T + r I)^-1
            what = X.T @ np.linalg.solve(X @ X.T + ridge * np.eye(D), y)
        else:
            what = np.linalg.solve(X.T @ X + ridge * np.eye(K), X.T @ y)
        out.append(float(((Xt @ (what - w)) ** 2).mean()))
    return np.array(out)
