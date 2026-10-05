"""Automation and New Tasks: How Technology Displaces and Reinstates Labor (Acemoglu & Restrepo, Journal of Economic
Perspectives 2019).

  task model     production combines tasks z in [N - 1, N]; tasks z <= I are AUTOMATED (done by capital), tasks z > I
                 by labour; labour has a comparative advantage in higher tasks (productivity gamma_L(z) = exp(a z),
                 capital gamma_K(z) = 1). With a CES aggregator (elasticity sigma) output reduces to (Eq. 1)
                   Y = Pi(I, N) [ Gamma^(1/sigma) (A_L L)^((sigma-1)/sigma) + (1 - Gamma)^(1/sigma) (A_K K)^((sigma-1)/sigma) ]^(sigma/(sigma-1))
                 Gamma(I, N) = the TASK CONTENT of production (labour's share of tasks, productivity-adjusted); for
                 sigma = 1, Gamma = N - I. Labour share (Eq. 2):  s_L = 1 / (1 + (1 - Gamma)/Gamma (A_L R / (A_K W))^(1 - sigma))
  effects        automation (I up):  labour demand = productivity effect + DISPLACEMENT effect (always lowers s_L);
                 new tasks (N up):   labour demand = productivity effect + REINSTATEMENT effect (always raises s_L);
                 factor-augmenting:  productivity effect + (small) substitution effect
                 'so-so' automation: capital only marginally better than labour -> tiny productivity gain, displacement
                 dominates, wages FALL
  decomposition  change in aggregate wage bill = productivity + composition + substitution + change in task content;
                 task content (per industry) = % change in labour share - substitution effect; split into displacement
                 (industries with negative change, 5-year moving averages) and reinstatement (positive change)

This file solves the one-sector task model for any sigma, measures each effect, and implements the paper's wage-bill
decomposition, validated on synthetic industries with PLANTED automation and new-task shocks.
"""

import numpy as np

# ----------------------------------------------------------------------------------------------- the task model

def _int_exp(c, lo, hi):
    """integral of exp(c z) dz from lo to hi."""
    return (hi - lo) if abs(c) < 1e-12 else (np.exp(c * hi) - np.exp(c * lo)) / c


def task_content(I, N, sigma=0.8, a=1.0):
    """Gamma(I, N) and the TFP term Pi(I, N) for gamma_L(z) = exp(a z), gamma_K(z) = 1."""
    if abs(sigma - 1) < 1e-9:
        lab = N - I
        return lab, np.exp(a * (I + N) / 2 * 0)                          # sigma = 1: Gamma = N - I (TFP not used)
    c = a * (sigma - 1)
    K_int = (I - (N - 1))                                               # integral of gamma_K^(sigma-1) = 1
    L_int = _int_exp(c, I, N)
    tot = K_int + L_int
    return L_int / tot, tot ** (1 / (sigma - 1))


def equilibrium(I, N, K=1.0, L=1.0, AL=1.0, AK=1.0, sigma=0.8, a=1.0):
    """Output, wage, rental rate and labour share with fixed K and L (Eq. 1 and 2)."""
    G, Pi = task_content(I, N, sigma, a)
    if abs(sigma - 1) < 1e-9:
        Y = (AL * L) ** G * (AK * K) ** (1 - G)                          # Cobb-Douglas limit (TFP normalised)
        sL = G
    else:
        e = (sigma - 1) / sigma
        tl = G ** (1 / sigma) * (AL * L) ** e
        tk = (1 - G) ** (1 / sigma) * (AK * K) ** e
        Y = Pi * (tl + tk) ** (1 / e)
        sL = tl / (tl + tk)
    return {"Y": float(Y), "W": float(sL * Y / L), "R": float((1 - sL) * Y / K), "labor share": float(sL),
            "Gamma": float(G)}


BASE = dict(I=0.4, N=1.0)


def automation_pays(I=0.4, N=1.0, a=1.0, **kw):
    """The model assumes capital is used in all automated tasks, i.e. capital is cheaper than labour at the marginal
    task I:  R / A_K  <  W / (A_L exp(a I)).  Returns (labour cost, capital cost) per unit of task I."""
    e = equilibrium(I, N, a=a, **kw)
    return e["W"] / (kw.get("AL", 1.0) * np.exp(a * I)), e["R"] / kw.get("AK", 1.0)


def effects(change, base=BASE, size=0.02, K=3.0, **kw):
    """Log changes in output, labour share and the wage bill for a small technology change:
    'automation' (I + size), 'new tasks' (N + size), 'labour-augmenting' (A_L * (1 + size)).
    Wage bill change = productivity effect (d ln Y) + task content / substitution effect (d ln s_L)."""
    kw = dict(kw, K=K)
    b = equilibrium(**base, **kw)
    if change == "automation":
        n = equilibrium(I=base["I"] + size, N=base["N"], **kw)
    elif change == "new tasks":
        n = equilibrium(I=base["I"], N=base["N"] + size, **kw)
    elif change == "labour-augmenting":
        kw2 = dict(kw); kw2["AL"] = kw.get("AL", 1.0) * (1 + size)
        n = equilibrium(**base, **kw2)
    else:
        raise ValueError(change)
    dY = np.log(n["Y"] / b["Y"])
    dS = np.log(n["labor share"] / b["labor share"])
    return {"productivity effect (d ln Y)": float(dY), "change in labour share (d ln s_L)": float(dS),
            "wage bill (d ln W L)": float(dY + dS), "wage (d ln W)": float(np.log(n["W"] / b["W"]))}


def so_so_automation(AK_values=(0.5, 1.0, 2.0, 4.0, 8.0), K=3.0, **kw):
    """Automation's effect on the wage and labour share as the productivity of the new capital (A_K) varies, with the
    marginal unit costs of labour and capital at task I. Low A_K = 'so-so' technology: capital only somewhat cheaper
    than labour at the automated tasks."""
    out = {}
    for AK in AK_values:
        e = effects("automation", AK=AK, K=K, **kw)
        lab, cap = automation_pays(**BASE, AK=AK, K=K, **kw)
        out[AK] = {"wage": e["wage (d ln W)"], "labour share": e["change in labour share (d ln s_L)"],
                   "productivity": e["productivity effect (d ln Y)"], "labour cost": lab, "capital cost": cap}
    return out

# ----------------------------------------------------------------------------------------------- the decomposition

def labour_share_eq2(G, rel_price, sigma):
    """Eq. 2 with rel_price = (W / A_L) / (R / A_K), the effective price of labour relative to capital."""
    return 1 / (1 + (1 - G) / G * rel_price ** (sigma - 1))


def synthetic_industries(n_ind=20, years=31, sigma=0.8, seed=0, auto_rate=0.010, new_rate=0.008, both_prob=0.0):
    """Industries whose task content Gamma_i moves through PLANTED automation (Gamma down) and new-task (Gamma up)
    shocks, with effective relative factor prices drifting and value-added shares reallocating; GDP per capita grows
    1.5% a year. If both_prob > 0, some industry-years get both kinds of shock at once (the paper's lower-bound
    caveat). Returns arrays over (years, industries) and the planted displacement / reinstatement."""
    r = np.random.default_rng(seed)
    G = r.uniform(0.4, 0.8, n_ind)
    rel = np.ones(n_ind)
    chi = r.dirichlet(np.ones(n_ind) * 5)
    Gs, sLs, rels, chis = [], [], [], []
    plant_disp, plant_rein = np.zeros((years, n_ind)), np.zeros((years, n_ind))
    for t in range(years):
        if t > 0:
            kind = r.random(n_ind)
            d_auto = np.where(kind < 0.5, r.exponential(2 * auto_rate, n_ind), 0.0)
            d_new = np.where(kind >= 0.5, r.exponential(2 * new_rate, n_ind), 0.0)
            both = r.random(n_ind) < both_prob
            d_new = np.where(both, d_new + r.exponential(2 * new_rate, n_ind), d_new)
            d_auto = np.where(both, d_auto + r.exponential(2 * auto_rate, n_ind), d_auto)
            G = np.clip(G - d_auto * G + d_new * (1 - G), 0.05, 0.95)
            plant_disp[t], plant_rein[t] = d_auto, d_new
            rel = rel * np.exp(r.normal(0.01, 0.02, n_ind))              # labour gets relatively dearer
            chi = chi * np.exp(r.normal(0, 0.03, n_ind)); chi /= chi.sum()
        Gs.append(G.copy()); rels.append(rel.copy()); chis.append(chi.copy())
        sLs.append(labour_share_eq2(G, rel, sigma))
    gdp = np.exp(0.015 * np.arange(years))
    return {"Gamma": np.array(Gs), "labor share": np.array(sLs), "rel price": np.array(rels), "va share": np.array(chis),
            "gdp": gdp, "sigma": sigma, "planted displacement": plant_disp, "planted reinstatement": plant_rein}


def decompose(data, sigma_assumed=None, window=5):
    """The paper's accounting, year by year (log points):
       wage bill growth  = productivity (d ln GDP) + composition + substitution + task content
       composition       = sum_i s_i d chi_i / s_L
       substitution_i    = (1 - s_i)(1 - sigma) d ln(rel price_i)       (from Eq. 2)
       task content_i    = d ln s_i - substitution_i
       aggregate pieces  = wage-bill weights chi_i s_i / s_L
    Displacement / reinstatement = the negative / positive parts of each industry's task-content change over
    `window`-year moving averages, aggregated with the same weights."""
    sigma = data["sigma"] if sigma_assumed is None else sigma_assumed
    s, chi, rel, gdp = data["labor share"], data["va share"], data["rel price"], data["gdp"]
    agg = (chi * s).sum(1)
    T = len(agg)
    out = {k: np.zeros(T - 1) for k in ("wage bill", "productivity", "composition", "substitution", "task content")}
    tc_ind = np.zeros((T - 1, s.shape[1]))
    for t in range(1, T):
        w = chi[t - 1] * s[t - 1] / agg[t - 1]
        dls = np.log(s[t] / s[t - 1])
        sub = (1 - s[t - 1]) * (1 - sigma) * np.log(rel[t] / rel[t - 1])
        tc_ind[t - 1] = dls - sub
        out["wage bill"][t - 1] = np.log(gdp[t] * agg[t] / (gdp[t - 1] * agg[t - 1]))
        out["productivity"][t - 1] = np.log(gdp[t] / gdp[t - 1])
        out["composition"][t - 1] = (s[t - 1] * (chi[t] - chi[t - 1])).sum() / agg[t - 1]
        out["substitution"][t - 1] = (w * sub).sum()
        out["task content"][t - 1] = (w * tc_ind[t - 1]).sum()
    # displacement / reinstatement from moving averages of industry task-content changes
    weights = np.array([chi[t] * s[t] / agg[t] for t in range(T - 1)])
    disp, rein = np.zeros(T - 1), np.zeros(T - 1)
    for t in range(T - 1):
        lo = max(0, t - window + 1)
        ma = tc_ind[lo:t + 1].mean(0)
        disp[t] = (weights[t] * np.minimum(ma, 0)).sum()
        rein[t] = (weights[t] * np.maximum(ma, 0)).sum()
    out["displacement"], out["reinstatement"] = disp, rein
    return out


def planted_task_content(data):
    """The true weighted task-content changes implied by the planted Gamma paths (for validation):
    d ln s_L from Gamma alone = (1 - s) d ln((Gamma / (1 - Gamma)))."""
    G, s, chi = data["Gamma"], data["labor share"], data["va share"]
    agg = (chi * s).sum(1)
    tc = []
    for t in range(1, len(G)):
        w = chi[t - 1] * s[t - 1] / agg[t - 1]
        d = (1 - s[t - 1]) * np.log((G[t] / (1 - G[t])) / (G[t - 1] / (1 - G[t - 1])))
        tc.append((w * d).sum())
    return np.array(tc)


def planted_split(data):
    """Year-by-year planted displacement and reinstatement (wage-bill weighted, first order): automation lowers
    Gamma by d_auto * Gamma, new tasks raise it by d_new * (1 - Gamma); each moves ln s_L by (1 - s) dGamma /
    (Gamma (1 - Gamma))."""
    G, s, chi = data["Gamma"], data["labor share"], data["va share"]
    agg = (chi * s).sum(1)
    disp, rein = [], []
    for t in range(1, len(G)):
        w = chi[t - 1] * s[t - 1] / agg[t - 1]
        k = (1 - s[t - 1]) / (G[t - 1] * (1 - G[t - 1]))
        disp.append((w * k * -data["planted displacement"][t] * G[t - 1]).sum())
        rein.append((w * k * data["planted reinstatement"][t] * (1 - G[t - 1])).sum())
    return np.array(disp), np.array(rein)


REPORTED = {
    "framework": "automation -> displacement effect (always reduces the labour share) + productivity effect; new tasks "
                 "-> reinstatement effect (always raises the labour share and labour demand) + productivity effect; "
                 "factor-augmenting technologies act mostly via productivity (sigma close to 1); 'so-so technologies' "
                 "bring displacement with modest productivity gains",
    "1947-1987": "wage bill per capita grew 2.5% a year, mostly productivity (2.4%); small net change in task content, "
                 "but displacement -0.48% a year offset by reinstatement +0.47% a year",
    "1987-2017": "wage bill per capita grew only 1.33% a year; productivity effect 1.54% a year; task content shifted "
                 "against labour by 0.35% a year (~10% lower labour demand cumulatively); reinstatement fell to 0.35% "
                 "and displacement rose to 0.7% a year; in manufacturing displacement ~1.1% a year (~30% cumulative)",
    "method": "sigma = 0.8 (Oberfield and Raval 2014); A_L / A_K grows with labour productivity (2% a year 1947-87, "
              "1.46% 1987-2017); task content = % change in labour share - substitution effect; displacement / "
              "reinstatement from five-year moving averages of negative / positive industry changes (lower bounds)",
    "history": "agriculture's labour share fell from 33% to 17% (about 1850-1870) with mechanisation, while industry's rose from 47% "
               "(1850) to 55% (1890) -- reinstatement from new factory and clerical tasks",
}
