"""Reproduce the experiments of Auto-Encoding Variational Bayes (Kingma & Welling 2014).

  E1  Figure 2 (top): MNIST, AEVB vs wake-sleep, Nz in {3, 5, 10, 20, 200}, 500 hidden units; the lower bound
      on train and test vs the number of training samples evaluated.
  E2  Figure 2 (bottom): Frey Face (Gaussian decoder with sigmoid means), Nz in {2, 5, 10, 20}, 200 hidden units.
  E3  Figure 3: marginal log-likelihood (Appendix D estimator) of AEVB, wake-sleep and Monte Carlo EM (HMC),
      Nz = 3, 100 hidden units, N_train = 1000 and 50000.
  E4  Figure 4: learned 2-D manifolds of MNIST and Frey Face.
  E5  Figure 5: random samples for Nz = 2, 5, 10, 20.
  E6  Section 2.3 claims: gradient variance of estimator A vs B, and L = 1 sample being enough for M = 100.

Settings (Section 5): weights ~ N(0, 0.01); Adagrad, global step size chosen from {0.01, 0.02, 0.1} by the training
bound after the first few iterations; minibatch M = 100, L = 1; weight decay = prior p(theta) = N(0, I) (MAP).
MNIST pixels are binarized (> 0.5) for the Bernoulli decoder; the paper does not say how it handled grey values.

!! HEAVY: the paper runs to 10^8 training samples (~1700 epochs) per curve. Not run on the author's laptop.
       python3 experiments.py --quick          # short curves, ~30-60 minutes on CPU
       python3 experiments.py --only e3
       python3 experiments.py --report-only
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import torch

from vae import VAE, hmc, log_joint, marginal_likelihood_appendix_d, wake_sleep_losses

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
FIG = HERE / "figures"


def mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(DATA, train=False, download=True)
    f = lambda d: (d.data.float().view(-1, 784) / 255 > 0.5).float()
    return f(tr), f(te)


def frey():
    """1965 faces of 28 x 20 pixels from Roweis' page (the paper's footnote 3); last 10% held out."""
    import scipy.io
    d = DATA / "frey"; d.mkdir(parents=True, exist_ok=True)
    path = d / "frey_rawface.mat"
    if not path.exists():
        urllib.request.urlretrieve("https://cs.nyu.edu/~roweis/data/frey_rawface.mat", path)
    X = torch.tensor(scipy.io.loadmat(path)["ff"].T, dtype=torch.float32) / 255      # 1965 x 560
    g = torch.Generator().manual_seed(0)
    X = X[torch.randperm(len(X), generator=g)]
    k = int(0.9 * len(X))
    return X[:k], X[k:]


# ---------------------------------------------------------------------------------------------------- training
def pick_lr(make, step_fn, X, a):
    """Section 5: try each Adagrad step size for a few iterations, keep the best training bound."""
    best = None
    for lr in (0.01, 0.02, 0.1):
        torch.manual_seed(0)
        model, opts = make(lr)
        for _ in range(a.probe_iters):
            step_fn(model, opts, X[torch.randint(0, len(X), (100,))], len(X))
        with torch.no_grad():
            score = model.elbo(X[:1000], L=1).mean().item()
        if best is None or score > best[0]:
            best = (score, lr)
    return best[1]


def make_aevb(x_dim, z_dim, hidden, decoder):
    def make(lr):
        m = VAE(x_dim, z_dim, hidden, decoder)
        return m, [torch.optim.Adagrad(m.parameters(), lr=lr)]
    return make


def make_ws(x_dim, z_dim, hidden, decoder):
    def make(lr):
        m = VAE(x_dim, z_dim, hidden, decoder)
        return m, [torch.optim.Adagrad(m.dec.parameters(), lr=lr), torch.optim.Adagrad(m.enc.parameters(), lr=lr)]
    return make


def aevb_step(m, opts, x, N):
    loss = -m.map_objective(x, N)
    opts[0].zero_grad(); loss.backward(); opts[0].step()


def ws_step(m, opts, x, N):
    wake, sleep = wake_sleep_losses(m, x)
    prior = 0.5 * sum((p ** 2).sum() for p in m.parameters()) / N              # same MAP weight decay
    opts[0].zero_grad(); opts[1].zero_grad()
    (wake + sleep + prior).backward()                                            # the two losses touch disjoint params
    opts[0].step(); opts[1].step()


def curve(make, step_fn, Xtr, Xte, a):
    """Train with M = 100; every `a.eval_every` samples record the average lower bound on 1000 train / test points."""
    lr = pick_lr(make, step_fn, Xtr, a)
    torch.manual_seed(0)
    model, opts = make(lr)
    pts, seen, t0 = [], 0, time.time()
    while seen < a.budget:
        for idx in torch.randperm(len(Xtr)).split(100):
            step_fn(model, opts, Xtr[idx], len(Xtr))
            seen += len(idx)
            if seen % a.eval_every < 100:
                with torch.no_grad():
                    pts.append({"samples": seen, "train": model.elbo(Xtr[:1000], L=1).mean().item(),
                                "test": model.elbo(Xte[:1000], L=1).mean().item()})
            if seen >= a.budget:
                break
    return {"lr": lr, "curve": pts, "seconds": time.time() - t0}, model


def plot_curves(res, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    keys = sorted({k.split("|")[0] for k in res}, key=lambda s: int(s.split("=")[1]))
    fig, ax = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3), squeeze=False)
    for axx, k in zip(ax[0], keys):
        for alg, c in (("AEVB", "tab:red"), ("wake-sleep", "tab:blue")):
            r = res.get(f"{k}|{alg}")
            if not r:
                continue
            s = [p["samples"] for p in r["curve"]]
            axx.plot(s, [p["train"] for p in r["curve"]], c=c, ls="--", label=f"{alg} (train)")
            axx.plot(s, [p["test"] for p in r["curve"]], c=c, label=f"{alg} (test)")
        axx.set_xscale("log"); axx.set_title(k); axx.set_xlabel("# training samples evaluated")
    ax[0][0].set_ylabel("lower bound L"); ax[0][0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=100); plt.close(fig)


def figure2(Xtr, Xte, zs, hidden, decoder, a, name):
    out = {}
    for z in zs:
        for alg, mk, st in (("AEVB", make_aevb, aevb_step), ("wake-sleep", make_ws, ws_step)):
            r, _ = curve(mk(Xtr.shape[1], z, hidden, decoder), st, Xtr, Xte, a)
            out[f"Nz={z}|{alg}"] = r
            last = r["curve"][-1]
            print(f"  {name} Nz={z} {alg}: lr {r['lr']}, train {last['train']:.1f}, test {last['test']:.1f}", flush=True)
    plot_curves(out, name)
    return out


def e1(a):
    Xtr, Xte = mnist()
    return figure2(Xtr, Xte, a.mnist_z, 500, "bernoulli", a, "e1_mnist")


def e2(a):
    Xtr, Xte = frey()
    return figure2(Xtr, Xte, a.frey_z, 200, "gaussian", a, "e2_frey")


# ---------------------------------------------------------------------------------------------------- Monte Carlo EM
def train_mcem(Xtr, a, z_dim=3, hidden=100):
    """Appendix E: no encoder. Per minibatch: 10 HMC leapfrog steps from each datapoint's persistent posterior
    sample, step size tuned towards 90% acceptance, then 5 decoder updates with that sample. Adagrad."""
    torch.manual_seed(0)
    m = VAE(784, z_dim, hidden)
    opt = torch.optim.Adagrad(m.dec.parameters(), lr=a.mcem_lr)
    Z = torch.randn(len(Xtr), z_dim)                                             # persistent chains
    step, seen = 0.05, 0
    while seen < a.budget_e3:
        for idx in torch.randperm(len(Xtr)).split(100):
            x = Xtr[idx]
            s, acc = hmc(log_joint(m, x), Z[idx], 1, leapfrog=10, step=step)
            Z[idx] = s[-1]
            step *= 1.02 if acc > 0.9 else 0.98                                    # crude automatic tuning
            for _ in range(5):
                loss = -(m.log_px_given_z(x, Z[idx])).mean() + 0.5 * sum((p ** 2).sum() for p in m.dec.parameters()) / len(Xtr)
                opt.zero_grad(); loss.backward(); opt.step()
            seen += len(idx)
            if seen >= a.budget_e3:
                break
    return m


def e3(a):
    Xtr_full, Xte = mnist()
    out = {}
    for n in a.e3_sizes:
        Xtr = Xtr_full[:n]
        aa = argparse.Namespace(**{**vars(a), "budget": a.budget_e3})
        models = {"AEVB": curve(make_aevb(784, 3, 100, "bernoulli"), aevb_step, Xtr, Xte, aa)[1],
                  "wake-sleep": curve(make_ws(784, 3, 100, "bernoulli"), ws_step, Xtr, Xte, aa)[1]}
        if n <= 1000 or not a.quick:                                               # MCEM is slow on 50k
            models["MCEM"] = train_mcem(Xtr, a)
        for name, m in models.items():
            res = {}
            for split, X in (("train", Xtr[:a.e3_eval]), ("test", Xte[:a.e3_eval])):
                est, acc = marginal_likelihood_appendix_d(m, X, L=50, burn=50, leapfrog=4, step=0.05)
                res[split] = est.mean().item(); res["hmc accept"] = acc
                if name != "MCEM":                                                 # a cross-check (needs an encoder)
                    res[f"{split} importance-sampled"] = m.importance_log_likelihood(X, K=1000).mean().item()
            out[f"N_train={n}|{name}"] = res
            print(f"  E3 N={n} {name}: {res}", flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- figures 4-5
def grid_image(X, n_rows, h, w):
    X = X.reshape(-1, h, w)
    n_cols = len(X) // n_rows
    return X[: n_rows * n_cols].reshape(n_rows, n_cols, h, w).permute(0, 2, 1, 3).reshape(n_rows * h, n_cols * w)


def save_image(img, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.imsave(FIG / name, img.numpy(), cmap="gray")


def e4(a):
    out = {}
    for name, (Xtr, Xte), dec, hidden, shape in (("mnist", mnist(), "bernoulli", 500, (28, 28)),
                                                  ("frey", frey(), "gaussian", 200, (28, 20))):
        r, m = curve(make_aevb(Xtr.shape[1], 2, hidden, dec), aevb_step, Xtr, Xte, a)
        save_image(grid_image(m.manifold(20 if name == "mnist" else 10), 20 if name == "mnist" else 10, *shape),
                   f"e4_manifold_{name}.png")
        out[name] = {"final test bound": r["curve"][-1]["test"], "figure": f"figures/e4_manifold_{name}.png"}
    return out


def e5(a):
    Xtr, Xte = mnist()
    out = {}
    for z in (2, 5, 10, 20):
        r, m = curve(make_aevb(784, z, 500, "bernoulli"), aevb_step, Xtr, Xte, a)
        torch.manual_seed(0)
        save_image(grid_image(m.sample(100), 10, 28, 28), f"e5_samples_nz{z}.png")
        out[f"Nz={z}"] = {"final test bound": r["curve"][-1]["test"], "figure": f"figures/e5_samples_nz{z}.png"}
    return out


# ---------------------------------------------------------------------------------------------------- E6
def grad_vector(m, x, estimator, L):
    m.zero_grad()
    (-m.elbo(x, L=L, estimator=estimator).mean()).backward()
    return torch.cat([p.grad.flatten() for p in m.enc.parameters()])               # phi: where the trick matters


def e6(a):
    Xtr, Xte = mnist()
    aa = argparse.Namespace(**{**vars(a), "budget": a.budget // 4})
    _, m = curve(make_aevb(784, 20, 500, "bernoulli"), aevb_step, Xtr, Xte, aa)
    out = {}
    for M in (1, 10, 100):
        for L in (1, 10):
            for est in ("A", "B"):
                torch.manual_seed(0)
                x = Xtr[torch.randint(0, len(Xtr), (M,))]
                G = torch.stack([grad_vector(m, x, est, L) for _ in range(a.e6_reps)])
                out[f"M={M} L={L} estimator {est}"] = {"total variance of grad phi": G.var(0).sum().item(),
                                                       "norm of mean grad": G.mean(0).norm().item()}
                print("  E6", M, L, est, out[f"M={M} L={L} estimator {est}"], flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- driver
def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k, t in (("e1", "E1: Figure 2, MNIST"), ("e2", "E2: Figure 2, Frey Face"), ("e3", "E3: Figure 3, marginal likelihood"),
                 ("e4", "E4: Figure 4, 2-D manifolds"), ("e5", "E5: Figure 5, samples"), ("e6", "E6: estimator variance")):
        if R.get(k):
            body = R[k]
            if k in ("e1", "e2"):                                                  # final points only
                body = {kk: {"lr": v["lr"], **v["curve"][-1]} for kk, v in body.items()}
            L += [f"## {t}", "", "```", json.dumps(body, indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 7)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.budget, a.eval_every, a.probe_iters = 10 ** 7, 10 ** 6, 50
    a.mnist_z, a.frey_z = (3, 5, 10, 20, 200), (2, 5, 10, 20)
    a.budget_e3, a.e3_sizes, a.e3_eval, a.mcem_lr = 2 * 10 ** 7, (1000, 50000), 1000, 0.02
    a.e6_reps = 200
    if a.quick:
        a.budget, a.eval_every, a.probe_iters = 3 * 10 ** 5, 5 * 10 ** 4, 20
        a.mnist_z, a.frey_z = (3, 20), (2, 10)
        a.budget_e3, a.e3_sizes, a.e3_eval = 10 ** 5, (1000,), 100
        a.e6_reps = 30
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
