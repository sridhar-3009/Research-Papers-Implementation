# The code, explained simply

How the code in this folder implements Bayesian Clustered Tensor Factorization (Sutskever, Salakhutdinov & Tenenbaum 2009) in numpy.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `bctf.py` | planted relational data, train/test splitting, RMSE and PR-AUC, MAP tensor factorization by conjugate gradient, the full MCMC sampler (CRP clustering, Normal–Inverse-Gamma cluster parameters, exact Gaussian draws for a_L, a_R and R, noise variance), an IRM-like block model, the adjusted Rand index, and a one-call comparison |
| `experiments.py` | density, dimensionality, cluster-recovery and sampler-length sweeps (E1–E4) |
| `demo.py` | the comparison at 100%, 10% and 3% observed, with cluster recovery (~7 seconds) |
| `test_bctf.py` | 4 quick tests (~1 second) |

**Run it** (from `16-Extras-optional/101-Sutskever-Salakhutdinov-Tenenbaum-2009-Bayesian-Clustered-Tensor-Factorization`; needs scipy):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `bctf.py`

### Data and metrics
| Name | What it does |
|---|---|
| `planted_data(n_obj, n_rel, k_obj, k_rel, d)` | cluster labels; cluster-mean vectors and matrices plus jitter 0.25; score a_Lᵀ R b_R + noise; the top 30% are true |
| `split(T, test_frac, train_frac)` | random test entries; a fraction of the remaining entries is kept as training data |
| `rmse`, `pr_auc`, `adjusted_rand` | metrics (PR-AUC = average precision, the paper's "AUC") |

### MAP
`fit_map(train, t, n_obj, n_rel, d, lam)` minimises ½Σ(t − a_Lᵀ R b_R)² + (λ/2)‖θ‖² with `scipy.optimize.minimize(method="CG")`, using analytic gradients accumulated with `np.add.at`.

### MCMC
| Name | What it does |
|---|---|
| `niw_post(X, alpha, beta)` | conjugate posterior (κ_n, μ_n, α_n, β_n) per dimension for a cluster's members (prior μ \| σ² ~ N(0, σ²), σ² ~ IG(α, β)) |
| `sample_cluster_params` | draw each cluster's variances and means from that posterior |
| `log_predictive`, `gibbs_assignments` | collapsed CRP Gibbs sweep; predictive = product of Student-t densities per dimension; new-cluster probability ∝ α_DP |
| `sample_vectors(design, t, owner, …)` | batched exact draws: per owner, precision = prior precision + Σ xxᵀ/σ², mean by solve, noise via the Cholesky factor |
| `run_mcmc(…, clustered, init_clusters, beta)` | MAP init; then per sweep: assignments (BCTF only; starting from many random clusters), cluster parameters, a_L (design R b_R), a_R (design Rᵀ a_L), R (design vec(a_L b_Rᵀ)), and noise variance; averages test predictions after burn-in. The prior scale β defaults to 0.05 × the MAP vectors' variance (separately for objects and relations) |

### Baselines and comparison
| Name | What it does |
|---|---|
| `kmeans`, `block_model` | IRM-like: k-means on objects' and relations' mean observed profiles, then block means smoothed toward 0 |
| `compare(train_frac, d, seed)` | runs MAP, BTF (`clustered=False`), BCTF and the block model on one planted dataset; returns (RMSE, PR-AUC) per model and ARI of the recovered clusters |

`REPORTED` holds Table 1, the main findings, the inference details and the dataset sizes, checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | training fraction 1 → 0.01 × 5 seeds |
| `e2` | d ∈ {2, 3, 5, 8} at 10% |
| `e3` | object and relation ARI vs density and prior scale |
| `e4` | 20, 60 or 150 sweeps |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_planted_data_and_split` | 30% positives; test and training sets don't overlap |
| `test_metrics` | PR-AUC, ARI (relabelling invariance) and RMSE by hand |
| `test_gaussian_conditional_sampler_matches_posterior_mean` | averaged draws recover the true coefficients of a Bayesian linear regression |
| `test_bayesian_beats_map_when_sparse` | at 3% density both BTF and BCTF beat MAP by > 0.2 RMSE, and BTF beats the block model |

---

## 5. Try it yourself

1. Add split-merge moves for the cluster assignments and check whether the relation clusters at 10% stop merging.
2. Sample the prior scale β with Metropolis–Hastings (as the paper does for its hyperparameters) instead of tying it to the MAP vectors.
3. Replace the Gaussian likelihood with a logistic one and sample with hybrid Monte Carlo.
4. Plot the covariance of the learned a_L vectors, sorted by cluster (the paper's Figure 3).
5. Run on the Kinship data (104 people, 26 relations) if you can download it, and compare with Table 1.
