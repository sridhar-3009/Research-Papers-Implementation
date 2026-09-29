"""Reproduce and test the claims of "Efficient BackProp".

  E1  Section 5.2, Figs 9-12: the linear network; learning rates vs lambda_max
  E2  Section 5.3: non-zero-mean inputs create a huge eigenvalue and slow learning
  E3  Section 4.1: stochastic beats batch on redundant data
  E4  Sections 4.3-4.6: each trick removed in turn, on real handwritten digits
  E5  Section 9.2, Figs 24-25: predicting the best learning rate from lambda_max
  E6  Section 9.1: stochastic diagonal Levenberg-Marquardt vs tuned SGD
  E7  Section 8, Figs 20-21: the Hessian's eigenvalue spread and layer-to-layer spread

Data: two Gaussians (Figure 10) and scikit-learn's built-in 8x8 handwritten digits
(1,797 images; the paper used its own digit sets, which aren't available).

Run:  python3 experiments.py      (~1-2 minutes; writes results.md and figures/)
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

from efficient_backprop import (MLP, InputTransform, lms_batch, lms_hessian, online_eigenvalue,
                                power_method, stochastic_diag_lm_rates, train_batch, train_sgd,
                                two_gaussians)

HERE = Path(__file__).parent


def digits(n_train, seed=0):
    """scikit-learn digits: returns raw train/test inputs, labels."""
    d = load_digits()
    idx = np.random.default_rng(seed).permutation(len(d.target))
    tr, te = idx[:n_train], idx[1000:]
    return d.data[tr], d.target[tr], d.data[te], d.target[te]


def targets(y, lo=-1.0, hi=1.0):
    T = np.full((len(y), 10), lo)
    T[np.arange(len(y)), y] = hi
    return T


def accuracy(net, X, y):
    return float((net.predict(X).argmax(1) == y).mean())


# ---------------------------------------------------------------------------

def e1_linear():
    X, D = two_gaussians(100, rng=0)
    cov = np.cov(X.T, bias=True)
    H = lms_hessian(X)
    rows = []
    for lr in (1.5, 1.9, 2.1, 2.3, 2.5):
        h, _ = lms_batch(X, D, lr, epochs=40)
        rows.append((lr, h[0], h[-1], h[-1] > h[0]))
    # stochastic LMS, eta = 0.2 (Figure 12)
    Xb, w = np.c_[X, np.ones(len(X))], np.zeros(3)
    rng = np.random.default_rng(0)
    for p in rng.integers(0, 100, 1000):                  # 10 epochs of 100 updates
        w += 0.2 * (D[p, 0] - Xb[p] @ w) * Xb[p]
    sgd_mse = float(0.5 * np.mean((D[:, 0] - Xb @ w) ** 2))
    best = float(0.5 * np.mean((D[:, 0] - Xb @ np.linalg.lstsq(Xb, D[:, 0], rcond=None)[0]) ** 2))
    return np.linalg.eigvalsh(cov), np.linalg.eigvalsh(H), rows, sgd_mse, best


def e2_mean_shift(shift=3.0):
    """Same data, inputs shifted by +3. Epochs of batch LMS (at eta = 1/lambda_max,
    the fastest safe single rate) until the MSE is within 1% of the best possible."""
    out = []
    for s in (0.0, shift):
        X, D = two_gaussians(100, rng=0)
        X = X + s
        ev = np.linalg.eigvalsh(lms_hessian(X))
        Xb = np.c_[X, np.ones(len(X))]
        best = 0.5 * np.mean((D[:, 0] - Xb @ np.linalg.lstsq(Xb, D[:, 0], rcond=None)[0]) ** 2)
        h, _ = lms_batch(X, D, 1 / ev.max(), epochs=20000)
        epochs = next((i for i, v in enumerate(h) if v <= 1.01 * best), None)
        out.append((s, ev, ev.max() / ev.min(), epochs))
    return out


def e3_redundant(epochs=5, seeds=3):
    """1000 training patterns that are secretly 10 copies of 100 patterns.
    Best learning rate chosen separately for each method."""
    Xr, yr, _, _ = digits(100)
    tf = InputTransform().fit(Xr)
    X, D = tf(np.tile(Xr, (10, 1))), targets(np.tile(yr, 10))
    res = {}
    for name, lrs in (("stochastic", (0.003, 0.01, 0.03)), ("batch", (0.1, 0.3, 1.0))):
        best = None
        for lr in lrs:
            curves = []
            for s in range(seeds):
                net = MLP([64, 30, 10], rng=s)
                curves.append(train_sgd(net, X, D, epochs, lr, rng=s) if name == "stochastic"
                              else train_batch(net, X, D, epochs, lr))
            curve = np.mean(curves, axis=0)
            if best is None or curve[-1] < best[1][-1]:
                best = (lr, curve)
        res[name] = best
    return res


def e4_recipe(epochs=3, seeds=3):
    """Each configuration gets its best learning rate from a grid, averaged over seeds."""
    Xtr, ytr, Xte, yte = digits(1000)
    tf = InputTransform().fit(Xtr)
    configs = [
        ("full recipe (all tricks)", dict(norm=True, sig="lecun_tanh", tg=(-1, 1), init="lecun")),
        ("inputs NOT normalized (raw 0-16)", dict(norm=False, sig="lecun_tanh", tg=(-1, 1), init="lecun")),
        ("logistic sigmoid, targets 0/1", dict(norm=True, sig="logistic", tg=(0, 1), init="lecun")),
        ("targets at the asymptotes (+-1.7159)", dict(norm=True, sig="lecun_tanh", tg=(-1.7159, 1.7159), init="lecun")),
        ("initial weights too small (+-0.01)", dict(norm=True, sig="lecun_tanh", tg=(-1, 1), init=0.01)),
        ("initial weights too big (+-3)", dict(norm=True, sig="lecun_tanh", tg=(-1, 1), init=3.0)),
    ]
    rows = []
    for name, c in configs:
        A, B = (tf(Xtr), tf(Xte)) if c["norm"] else (Xtr, Xte)
        best = (0.0, None)
        for lr in (0.0003, 0.001, 0.003, 0.01, 0.03):
            accs = []
            for s in range(seeds):
                net = MLP([64, 30, 10], sigmoid=c["sig"], init=c["init"], rng=s)
                train_sgd(net, A, targets(ytr, *c["tg"]), epochs, lr, rng=s)
                accs.append(accuracy(net, B, yte))
            best = max(best, (float(np.mean(accs)), lr))
        rows.append((name, *best))
    return rows


def e5_predicted_rate():
    """Figures 24-25: lambda_max by the power method, the on-line estimate, and MSE
    after training at (ratio x predicted eta_opt), for batch and stochastic learning."""
    Xtr, ytr, _, _ = digits(300)
    tf = InputTransform().fit(Xtr)
    X, D = tf(Xtr), targets(ytr)
    lam, _ = power_method(MLP([64, 30, 10], rng=1), X, D, iters=60, rng=0)
    online = online_eigenvalue(MLP([64, 30, 10], rng=1), X, D, presentations=1500, gamma=0.01, rng=0)
    ratios = (0.1, 0.25, 0.5, 1, 1.5, 2, 3)
    batch = [train_batch(MLP([64, 30, 10], rng=1), X, D, 50, r / lam)[-1] for r in ratios]
    stoch = [train_sgd(MLP([64, 30, 10], rng=1), X, D, 5, r / lam, rng=0)[-1] for r in ratios]
    return lam, online, ratios, batch, stoch


def e6_sdlm(epochs=6):
    Xtr, ytr, _, _ = digits(1000)
    tf = InputTransform().fit(Xtr)
    X, D = tf(Xtr), targets(ytr)
    sgd = {lr: train_sgd(MLP([64, 30, 10], rng=1), X, D, epochs, lr, rng=0) for lr in (0.003, 0.01, 0.03)}
    sdlm = {}
    for eta, mu in ((0.1, 0.1), (0.3, 0.1), (1.0, 1.0)):
        net = MLP([64, 30, 10], rng=1)
        h = [net.cost(X, D)]
        for ep in range(epochs):
            rates = stochastic_diag_lm_rates(net, X[:200], eta=eta, mu=mu)   # re-estimated each epoch
            h += train_sgd(net, X, D, 1, rng=ep, per_weight_lr=rates)[1:]
        sdlm[(eta, mu)] = h
    return sgd, sdlm


def full_hessian(net, X, D, h=1e-5):
    w = net.get()
    H = np.zeros((w.size, w.size))
    for i in range(w.size):
        e = np.zeros_like(w); e[i] = h
        net.set(w + e); g1 = net.flat_gradient(X, D)
        net.set(w - e); g2 = net.flat_gradient(X, D)
        H[:, i] = (g1 - g2) / (2 * h)
    net.set(w)
    return (H + H.T) / 2


def e7_hessian():
    """Train a 64-10-10 net briefly, then look at its Hessian (Figs 20-21), with
    normalized inputs and with raw (non-centred) inputs."""
    Xtr, ytr, _, _ = digits(300)
    tf = InputTransform().fit(Xtr)
    out = {}
    for name, A in (("normalized inputs", tf(Xtr)), ("raw inputs 0-16", Xtr / 16 * 1.0 + 0.0)):
        net = MLP([64, 10, 10], rng=0)
        train_sgd(net, A, targets(ytr), 2, 0.01 if name.startswith("norm") else 0.003, rng=0)
        ev = np.linalg.eigvalsh(full_hessian(net, A, targets(ytr)))
        hW, _ = net.diag_hessian(A)
        out[name] = (ev, [float(h.mean()) for h in hW])
    return out


# ---------------------------------------------------------------------------

def save_figures(e1_rows, e5, e7, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.mkdir(exist_ok=True)

    X, D = two_gaussians(100, rng=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for lr in (1.5, 1.9, 2.1, 2.5):
        h, _ = lms_batch(X, D, lr, epochs=15)
        ax.semilogy(h, marker="o", ms=3, label=f"eta = {lr}")
    ax.set_xlabel("epoch (batch)"); ax.set_ylabel("MSE"); ax.set_ylim(1e-2, 1e3)
    ax.set_title("Fig. 11 reproduced: batch LMS. Paper's bound 2/0.84 = 2.38,\n"
                 "true bound 2/lambda_max = 2.0 (the bias adds eigenvalue ~1)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path / "fig11_lms_learning_rates.png", dpi=130); plt.close(fig)

    lam, online, ratios, batch, stoch = e5
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(online, lw=0.8); axes[0].axhline(lam, color="k", ls="--", label=f"power method {lam:.1f}")
    axes[0].set_xlabel("pattern presentations"); axes[0].set_ylabel("estimated lambda_max")
    axes[0].set_title("Fig. 24 reproduced: on-line eigenvalue estimate", fontsize=9); axes[0].legend(fontsize=8)
    axes[1].plot(ratios, batch, "o-", label="batch, 50 updates")
    axes[1].plot(ratios, stoch, "s-", label="stochastic, 5 epochs")
    axes[1].axvline(1, color="k", ls=":"); axes[1].set_yscale("log")
    axes[1].set_xlabel("learning rate / predicted optimal (1/lambda_max)"); axes[1].set_ylabel("final MSE")
    axes[1].set_title("Fig. 25 reproduced (64-30-10 net, 300 digits)", fontsize=9); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path / "fig24_25_learning_rate_prediction.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (ev, _) in e7.items():
        ax.semilogy(np.sort(np.abs(ev))[::-1], label=name)
    ax.set_xlabel("eigenvalue order"); ax.set_ylabel("|eigenvalue|")
    ax.set_title("Fig. 19 reproduced: Hessian spectrum of a 64-10-10 net", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path / "fig19_hessian_spectrum.png", dpi=130); plt.close(fig)


def main():
    L = ["# Efficient BackProp: reproduced results", "", "Generated by `python3 experiments.py`.", ""]

    cov_ev, H_ev, rows, sgd_mse, best = e1_linear()
    L += ["## E1. The linear network (Section 5.2, Figures 9-12)", "",
          f"- Input covariance eigenvalues: **{cov_ev[1]:.3f} and {cov_ev[0]:.3f}** (paper: 0.84 and 0.036).",
          f"- But the network also has a **bias**. Its Hessian (Eq. 29, bias as an always-1 input) has eigenvalues "
          f"**{', '.join(f'{v:.3f}' for v in H_ev)}**, so lambda_max = {H_ev.max():.3f} and the real limit is "
          f"2/lambda_max = **{2 / H_ev.max():.2f}**, not the paper's 2/0.84 = 2.38.", "",
          "| eta | MSE at start | MSE after 40 epochs | diverged? |", "|---|---|---|---|"]
    for lr, a, b, div in rows:
        L.append(f"| {lr} | {a:.3f} | {b:.3g} | {'**yes**' if div else 'no'} |")
    L += ["", f"- Stochastic LMS with eta = 0.2 (Figure 12): MSE {sgd_mse:.4f} after 10 epochs "
          f"(best possible {best:.4f}): close, but it keeps fluctuating around the minimum.", ""]

    L += ["## E2. Non-zero-mean inputs (Section 5.3)", "",
          "| inputs | Hessian eigenvalues | condition number | epochs to within 1% of best |", "|---|---|---|---|"]
    for s, ev, cond, ep in e2_mean_shift():
        L.append(f"| {'centred' if s == 0 else f'shifted by +{s:g}'} | {', '.join(f'{v:.3f}' for v in ev)} | "
                 f"{cond:.0f} | {ep if ep is not None else '> 20000'} |")
    L.append("")

    r3 = e3_redundant()
    L += ["## E3. Stochastic vs batch on redundant data (Section 4.1)", "",
          "1,000 training patterns = 10 copies of the same 100 digits. MSE after each epoch (best learning rate for each):", "",
          "| method | best lr | " + " | ".join(f"epoch {i}" for i in range(6)) + " |", "|---|---|" + "---|" * 6]
    for name, (lr, curve) in r3.items():
        L.append(f"| {name} | {lr} | " + " | ".join(f"{v:.3f}" for v in curve) + " |")
    L.append("")

    L += ["## E4. The tricks, one at a time (Sections 4.3-4.6)", "",
          "64-30-10 net on 1,000 digits, 3 epochs of SGD, test accuracy on 797 digits,",
          "best learning rate from {0.0003, ..., 0.03} for each row, mean of 3 seeds.", "",
          "| configuration | test accuracy | best lr |", "|---|---|---|"]
    for name, acc, lr in e4_recipe():
        L.append(f"| {name} | {acc:.3f} | {lr} |")
    L.append("")

    e5 = e5_predicted_rate()
    lam, online, ratios, batch, stoch = e5
    L += ["## E5. Predicting the best learning rate (Section 9.2, Figures 24-25)", "",
          f"- Power method (Eq. 60): lambda_max = **{lam:.2f}**, so eta_opt = 1/lambda_max = {1 / lam:.4f}.",
          f"- On-line estimate (Eq. 64): {online[99]:.1f} after 100 presentations, mean of the last 500 = {np.mean(online[-500:]):.1f}.", "",
          "| lr / predicted | batch: MSE after 50 updates | stochastic: MSE after 5 epochs |", "|---|---|---|"]
    for r, b, s in zip(ratios, batch, stoch):
        L.append(f"| {r} | {b:.3f} | {s:.3f} |")
    L.append("")

    sgd, sdlm = e6_sdlm()
    L += ["## E6. Stochastic diagonal Levenberg-Marquardt (Section 9.1)", "",
          "64-30-10 net, 1,000 digits, training MSE after each epoch. Second derivatives re-estimated each epoch on 200 patterns.", "",
          "| method | " + " | ".join(f"epoch {i}" for i in range(1, 7)) + " |", "|---|" + "---|" * 6]
    for lr, h in sgd.items():
        L.append(f"| SGD, lr {lr} | " + " | ".join(f"{v:.3f}" for v in h[1:]) + " |")
    for (eta, mu), h in sdlm.items():
        L.append(f"| SDLM, eta {eta}, mu {mu} | " + " | ".join(f"{v:.3f}" for v in h[1:]) + " |")
    L.append("")

    e7 = e7_hessian()
    L += ["## E7. The Hessian of a trained net (Section 8, Figures 19-21)", "",
          "64-10-10 net after 2 epochs; full Hessian by finite differences (760 weights).", "",
          "| inputs | largest eigenvalue | 11th largest | ratio 1st/11th | eigenvalues > 1% of largest | mean d2E/dw2, layer 1 | layer 2 |",
          "|---|---|---|---|---|---|---|"]
    for name, (ev, per_layer) in e7.items():
        s = np.sort(np.abs(ev))[::-1]
        L.append(f"| {name} | {s[0]:.3f} | {s[10]:.4f} | {s[0] / s[10]:.1f} | {int((s > 0.01 * s[0]).sum())} of {len(s)} | "
                 f"{per_layer[0]:.5f} | {per_layer[1]:.5f} |")

    text = "\n".join(L) + "\n"
    (HERE / "results.md").write_text(text)
    print(text)
    save_figures(rows, e5, e7, HERE / "figures")
    print("Saved results.md and figures/")


if __name__ == "__main__":
    main()
