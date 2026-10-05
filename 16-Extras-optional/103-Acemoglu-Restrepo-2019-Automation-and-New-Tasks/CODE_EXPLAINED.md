# The code, explained simply

How the code in this folder implements Acemoglu & Restrepo's task model and wage-bill decomposition (JEP 2019).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `tasks.py` | the task model (task content, TFP, equilibrium wages, rents and labor share for any σ), the effects of automation / new tasks / factor-augmenting technology, so-so automation, synthetic industries with planted shocks, and the paper's decomposition |
| `experiments.py` | the so-so boundary, decomposition accuracy, σ sensitivity, window and simultaneous-shock bias (E1–E4) |
| `demo.py` | the model's effects, so-so automation, σ robustness, and the decomposition check (instant) |
| `test_tasks.py` | 4 quick tests (~0.1 seconds) |

**Run it** (from `16-Extras-optional/103-Acemoglu-Restrepo-2019-Automation-and-New-Tasks`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `tasks.py`

### The model
| Name | What it does |
|---|---|
| `task_content(I, N, sigma, a)` | with γ_L(z) = e^(az) and γ_K = 1: the capital integral is I − (N − 1) and the labor integral is ∫_I^N e^(a(σ−1)z) dz (closed form); Γ = labor / total; Π = total^(1/(σ−1)); for σ = 1, Γ = N − I |
| `equilibrium(I, N, K, L, AL, AK, sigma)` | Eq. 1 for Y; the labor share from the two CES terms; W = s_L Y / L and R = (1 − s_L) Y / K (fixed factor supplies, competitive markets) |
| `automation_pays` | unit costs at the marginal task: W/(A_L e^(aI)) for labor vs R/A_K for capital (the model assumes capital is cheaper) |
| `effects(change, size, K)` | log changes in Y (productivity effect), s_L (displacement / reinstatement / substitution) and the wage bill, for a step in I, N or A_L (baseline I = 0.4, N = 1, K = 3) |
| `so_so_automation(AK_values)` | the automation step for each capital productivity A_K, with unit costs |

### The decomposition
| Name | What it does |
|---|---|
| `labour_share_eq2(G, rel_price, sigma)` | Eq. 2 with the effective relative price (W/A_L)/(R/A_K) |
| `synthetic_industries(…, both_prob)` | per year and industry: either an automation shock (Γ falls by d·Γ) or a new-task shock (Γ rises by d·(1 − Γ)), optionally both; the relative price drifts +1%/yr; value-added shares reallocate; GDP grows 1.5%/yr. Labor shares come from Eq. 2 |
| `decompose(data, sigma_assumed, window)` | per year: wage bill = Δ ln GDP + composition Σ s_i Δχ_i / s_L + wage-bill-weighted substitution and task content (task content = Δ ln s_i − (1 − s_i)(1 − σ) Δ ln rel price_i); displacement and reinstatement from `window`-year moving averages of each industry's task content, split by sign |
| `planted_task_content`, `planted_split` | the true task-content change, and its automation / new-task parts, from the planted Γ paths (first order) |

`REPORTED` holds the framework and the paper's 1947–1987 and 1987–2017 numbers, the method, and the historical example, checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | the A_K where automation starts to raise wages, for σ ∈ {0.5, 0.8, 1, 1.5} |
| `e2` | estimated vs planted task content, displacement and reinstatement for 0.5× to 4× shocks |
| `e3` | assumed σ ∈ {0.6, 0.8, 1, 1.2} with truth 0.8 |
| `e4` | estimated / planted ratios for windows 1–10 years and simultaneous-shock shares 0–0.5 |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_task_content_limits_and_monotonicity` | Γ = N − I when σ = 1; Γ falls with I and rises with N |
| `test_equilibrium_shares_add_up_and_eq2` | W·L + R·K = Y; the equilibrium labor share matches Eq. 2 |
| `test_automation_lowers_labour_share_for_any_sigma_and_so_so_lowers_wages` | the signs hold for 4 values of σ; so-so (A_K = 0.5) lowers wages while A_K = 8 raises them; automation is cost-minimising throughout |
| `test_decomposition_adds_up_and_bounds` | the pieces add up to the wage bill (within 0.2 points); task content is recovered within 0.1 points; the estimated displacement and reinstatement are lower bounds of the planted ones |

---

## 5. Try it yourself

1. Add a second sector (agriculture vs industry) and reproduce the 19th-century story: mechanisation lowers agriculture's labor share while composition and new tasks raise aggregate labor demand.
2. Make automation's productivity gain depend on the wage (cheaper labor → smaller gains), and show how weak reinstatement makes automation less productive, a point the paper makes.
3. Let K respond to the rental rate (capital accumulation) and recompute the long-run effect of automation on wages.
4. Feed in an industry panel of real labor shares (e.g. BEA data) and run `decompose`.
5. Classify tasks by exposure to an AI system and simulate "AI as automation" vs "AI that creates new tasks".
