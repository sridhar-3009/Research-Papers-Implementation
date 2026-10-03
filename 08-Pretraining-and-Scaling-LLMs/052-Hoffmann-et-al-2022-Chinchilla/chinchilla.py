"""Training Compute-Optimal Large Language Models (Chinchilla; Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai,
Rutherford, de Las Casas, Hendricks, Welbl, Clark, Hennigan, Noland, Millican, van den Driessche, Damoc, Guy,
Osindero, Simonyan, Elsen, Rae, Vinyals & Sifre, NeurIPS 2022).

Question: given a FLOP budget C ~ 6 N D, what model size N and number of training tokens D minimise the loss?
Three approaches, one answer: N_opt ~ C^a and D_opt ~ C^b with a ~ b ~ 0.5 (Table 2), i.e. grow parameters and
data EQUALLY, about 20 tokens per parameter, unlike Kaplan et al.'s N ~ C^0.73.
  Approach 1: envelope of many training curves (models 70M-10B, 4 cosine lengths each): a = 0.50, b = 0.50
  Approach 2: IsoFLOP profiles: at 9 fixed budgets, vary N; fit a parabola in log N; take its minimum: a = 0.49
  Approach 3: fit the parametric loss  L(N, D) = E + A / N^alpha + B / D^beta                          (Eq. 2)
              (E = 1.69, A = 406.4, B = 410.7, alpha = 0.34, beta = 0.28; Eq. 10) by Huber(delta = 1e-3) on
              log-loss with L-BFGS from a grid of starts (Eq. 3, 11); the frontier is closed-form (Eq. 4):
              N_opt = G (C/6)^a, D_opt = G^-1 (C/6)^b, G = (alpha A / (beta B))^(1/(alpha+beta)),
              a = beta / (alpha + beta), b = alpha / (alpha + beta)  -> a = 0.46, b = 0.54
Chinchilla = 70B parameters on 1.4T tokens, the same compute as Gopher (280B on 300B tokens, 5.76e23 FLOPs);
it beats Gopher, GPT-3, Jurassic-1 and MT-NLG (MMLU 67.6% vs Gopher's 60.0%).
FLOPs (Appendix F): full count including embeddings and attention; the backward pass is 2x the forward pass.
"""

import math

import numpy as np
import torch

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28                              # Eq. 10
GOPHER_FLOPS = 5.76e23

TABLE_3 = {4e8: (1.92e19, 8.0e9), 1e9: (1.21e20, 20.2e9), 10e9: (1.23e22, 205.1e9), 67e9: (5.76e23, 1.5e12),
           175e9: (3.85e24, 3.7e12), 280e9: (9.90e24, 5.9e12), 520e9: (3.43e25, 11.0e12), 1e12: (1.27e26, 21.2e12),
           10e12: (1.30e28, 216.2e12)}                                              # params: (FLOPs, tokens), Appr. 1


# ---------------------------------------------------------------------------------------------------- the law
def loss(N, D, p=None):
    E_, A_, B_, a_, b_ = p or (E, A, B, ALPHA, BETA)
    return E_ + A_ / N ** a_ + B_ / D ** b_


def frontier(C, p=None):
    """Eq. 4: the closed-form compute-optimal N and D under C = 6 N D."""
    _, A_, B_, a_, b_ = p or (E, A, B, ALPHA, BETA)
    G = (a_ * A_ / (b_ * B_)) ** (1 / (a_ + b_))
    a, b = b_ / (a_ + b_), a_ / (a_ + b_)
    return G * (C / 6) ** a, (C / 6) ** b / G


def frontier_numeric(C, p=None, grid=None):
    """Minimise L(N, C / 6N) over a log grid of N (checks Eq. 4)."""
    Ns = np.logspace(6, 14, 4001) if grid is None else grid
    Ls = loss(Ns, C / (6 * Ns), p)
    k = int(np.argmin(Ls))
    return Ns[k], C / (6 * Ns[k]), Ls[k]


def kaplan_N_opt(C_flops):
    """Kaplan et al.'s Table 5 recipe (N_opt = 1.3e9 C^0.73 with C in PF-days), for comparison."""
    return 1.3e9 * (C_flops / 8.64e19) ** 0.73


# ---------------------------------------------------------------------------------------------------- FLOPs
def flops_appendix_f(seq_len, vocab, d_model, n_layers, ffw, n_heads, kq_size):
    """Appendix F's per-sequence training FLOPs (forward + 2x backward), counting embeddings and attention."""
    att_width = kq_size * n_heads
    embed = 2 * seq_len * vocab * d_model
    attn = (2 * 3 * seq_len * d_model * att_width                                  # Q, K, V projections
            + 2 * seq_len * seq_len * att_width                                     # K @ Q logits
            + 3 * n_heads * seq_len * seq_len                                       # softmax
            + 2 * seq_len * seq_len * att_width                                     # softmax @ V
            + 2 * seq_len * att_width * d_model)                                    # output projection
    dense = 2 * seq_len * (d_model * ffw + d_model * ffw)
    logits = 2 * seq_len * d_model * vocab
    forward = embed + n_layers * (attn + dense) + logits
    return 3 * forward


def params_count(vocab, d_model, n_layers, ffw, n_heads, kq_size):
    """Parameters including embeddings (the paper counts them): embedding + per layer (attention + dense)."""
    att_width = kq_size * n_heads
    return vocab * d_model + n_layers * (4 * d_model * att_width + 2 * d_model * ffw)


# ---------------------------------------------------------------------------------------------------- fitting
def huber(x, delta):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x ** 2, delta * (ax - 0.5 * delta))


def fit_parametric(N, D, L, delta=1e-3, grid=None, steps=200):
    """Approach 3 (Eq. 11): minimise sum Huber_delta( LSE(a - alpha log N, b - beta log D, e) - log L ) with L-BFGS
    from a grid of initialisations; returns (E, A, B, alpha, beta) of the best fit."""
    N, D, L = (torch.tensor(np.asarray(v, dtype=float), dtype=torch.float64) for v in (N, D, L))
    grid = grid or [(a0, b0, e0, al, be) for a0 in (0, 5, 10) for b0 in (0, 5, 10) for e0 in (-1, 0, 1)
                    for al in (0.0, 0.5, 1.0) for be in (0.0, 0.5, 1.0)]
    best, best_val = None, float("inf")
    for init in grid:
        th = torch.tensor(init, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            a, b, e, al, be = th
            pred = torch.logsumexp(torch.stack([a - al * N.log(), b - be * D.log(), e.expand_as(N)]), 0)
            obj = huber(pred - L.log(), delta).sum()
            obj.backward()
            return obj
        try:
            val = opt.step(closure).item()
        except RuntimeError:
            continue
        if math.isfinite(val) and val < best_val:
            best_val, best = val, th.detach().clone()
    a, b, e, al, be = best.tolist()
    return math.exp(e), math.exp(a), math.exp(b), al, be


def approach1_envelope(curves):
    """Approach 1: curves = {N: [(flops, loss), ...]}. For each FLOP level, the model whose curve is lowest there.
    Returns [(C, N_best, D_best)] and the fitted exponent a in N_opt ~ C^a."""
    Cs = np.logspace(np.log10(max(c[0][0] for c in curves.values())),
                     np.log10(min(c[-1][0] for c in curves.values())), 30)
    pts = []
    for C in Cs:
        cands = []
        for N, curve in curves.items():
            f, l = zip(*curve)
            if f[0] <= C <= f[-1]:
                cands.append((float(np.interp(np.log(C), np.log(f), l)), N))
        if cands:
            l, N = min(cands)
            pts.append((C, N, C / (6 * N)))
    a = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0]
    return pts, a


def approach2_isoflop(isoflop):
    """Approach 2: isoflop = {C: [(N, loss), ...]}. Fit loss as a parabola in log N for each budget; its minimum is
    N_opt(C). Returns [(C, N_opt)] and the exponent a."""
    pts = []
    for C, rows in isoflop.items():
        x = np.log([r[0] for r in rows]); y = np.array([r[1] for r in rows])
        c2, c1, _ = np.polyfit(x, y, 2)
        pts.append((C, math.exp(-c1 / (2 * c2))))
    a = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0]
    return pts, a


# ---------------------------------------------------------------------------------------------------- synthetic runs
def synthetic_curves(Ns, D_max, noise=0.0, seed=0, n_points=40, p=None):
    """Training curves from the parametric law (the loss after D tokens of a cosine schedule matched to D):
    a stand-in for the paper's hundreds of runs, so Approaches 1-3 can be exercised end to end."""
    rng = np.random.default_rng(seed)
    out = {}
    for N in Ns:
        Ds = np.logspace(8, np.log10(D_max), n_points)
        out[N] = [(6 * N * D, loss(N, D, p) * math.exp(noise * rng.standard_normal())) for D in Ds]
    return out


def synthetic_isoflop(budgets, n_per=9, noise=0.0, seed=0, p=None):
    rng = np.random.default_rng(seed)
    out = {}
    for C in budgets:
        N_star, _ = frontier(C, p)
        Ns = N_star * np.logspace(-1, 1, n_per)
        out[C] = [(N, loss(N, C / (6 * N), p) * math.exp(noise * rng.standard_normal())) for N in Ns]
    return out
