"""Reproduce Sutskever, Martens, Dahl & Hinton (2013), at reduced scale.

  E1  Section 2.1 / Theorem 2.1 / appendix Fig. 2: CM vs NAG on an oblong 2-D quadratic
  E2  Section 3, Table 1: deep MNIST autoencoder, momentum type (CM / NAG) x mu_max,
      schedule of Eq. (5), sparse initialization, best learning rate per setting
  E3  Table 2: lowering mu to 0.9 for the last updates ("fine convergence")
  E4  Table 3: the scale of the sparse initialization
  E5  Section 4, Table 5: RNN on the addition problem, mu0 in {0, 0.9, 0.98}, CM vs NAG

Scale: the paper ran 750,000 updates (autoencoders) and 50,000 (RNNs). Here the
autoencoder gets 15,000 updates of 200 and the RNN 4,000 updates of 100, with a
shorter sequence (T = 50). Absolute errors are therefore higher; we compare the
PATTERN across settings. Because the run is 50x shorter, the momentum schedule of
Eq. (5) is compressed in time (period updates/120 instead of 250) so that mu
still climbs to ~0.996 by the end - otherwise mu_max 0.99 and 0.995 would never
be reached.

Run:  python3 experiments.py            (~1-1.5 hours; saves results.json/.md, figures/)
      python3 experiments.py --quick    (~5 minutes)
      python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from momentum import RNN, Autoencoder, Momentum, addition_problem, autoencoder_losses, mu_schedule, quadratic_run

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(8)


# ---------------------------------------------------------------------------

def e1_quadratic():
    """Appendix Fig. 2: same learning rate and momentum, oblong quadratic.
    Curvatures 1 (flat direction) and 100 (steep direction). eps * lambda_steep = 1.
    Also records the case eps * lambda = 1.5, where NAG diverges and CM doesn't."""
    A, b = np.diag([1.0, 100.0]), np.zeros(2)
    lr, mu = 0.01, 0.9
    cm15 = quadratic_run(A, b, [-10, 1], 0.015, 0.95, 100, False)
    nag15 = quadratic_run(A, b, [-10, 1], 0.015, 0.95, 100, True)
    cm = quadratic_run(A, b, [-10, 1], lr, mu, 100, False)
    nag = quadratic_run(A, b, [-10, 1], lr, mu, 100, True)
    # effective momentum per direction (Theorem 2.1): mu (1 - lambda * lr)
    eff = [mu * (1 - lam * lr) for lam in (1, 100)]
    # oscillation in the steep direction = number of sign changes of x2
    flips = lambda p: int(np.sum(np.diff(np.sign(p[:, 1])) != 0))
    dist = lambda p: float(np.linalg.norm(p[-1]))
    return dict(cm_path=cm.tolist(), nag_path=nag.tolist(), eff_mu=eff,
                cm_flips=flips(cm), nag_flips=flips(nag), cm_final=dist(cm), nag_final=dist(nag),
                cm_steep_max_after10=float(np.abs(cm[10:, 1]).max()), nag_steep_max_after10=float(np.abs(nag[10:, 1]).max()),
                big_lr=dict(cm_final=float(np.linalg.norm(cm15[-1])), nag_final=float(np.linalg.norm(nag15[-1]))))


def mnist_train():
    tr = torchvision.datasets.MNIST(DATA, train=True, download=True)
    return (tr.data.reshape(-1, 784).float() / 255).to(DEV)


def train_autoencoder(X, lr, mu_max, nesterov, updates, final_mu=None, final_updates=0, si_scale=1.0, seed=0):
    """Minibatches of 200, momentum schedule of Eq. (5). Returns the training
    squared error (Table 1's measure) on a fixed 10,000-image subset."""
    torch.manual_seed(seed)
    model = Autoencoder(si_scale=si_scale, seed=seed).to(DEV)
    opt = Momentum(model.parameters(), lr, 0.0, nesterov)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for t in range(updates):
        opt.mu = mu_schedule(t, mu_max, period=max(1, updates // 120))   # compressed Eq. (5)
        if final_mu is not None and t >= updates - final_updates:
            opt.mu = min(opt.mu, final_mu)                  # Table 2: lower mu at the end
        idx = torch.randint(0, len(X), (200,), generator=gen).to(DEV)
        xb = X[idx]
        loss = opt.step(lambda: autoencoder_losses(model, xb)[0])
        if not np.isfinite(loss):
            return float("inf")
    with torch.no_grad():
        se = [float(autoencoder_losses(model, X[i:i + 2000])[1]) for i in range(0, 10000, 2000)]
    return float(np.mean(se))


def e2_e3_e4(quick):
    X = mnist_train()
    updates = 600 if quick else 15000
    lrs = (0.001, 0.003, 0.01) if not quick else (0.003,)
    table1 = {}
    settings = [("SGD", 0.0, False)] + [(f"{m}{'N' if n else 'M'}", m, n) for m in (0.9, 0.99, 0.995) for n in (True, False)]
    for name, mu_max, nes in settings:
        best = (np.inf, None)
        for lr in lrs:
            t0 = time.time()
            err = train_autoencoder(X, lr, mu_max, nes, updates)
            print(f"autoencoder {name:7s} lr {lr:<6} squared error {err:.3f} ({time.time() - t0:.0f}s)", flush=True)
            best = min(best, (err, lr))
        table1[name] = dict(err=best[0], lr=best[1])
    # Table 2: NAG 0.99 with/without lowering mu to 0.9 for the final 10% of updates
    lr = table1["0.99N"]["lr"]
    before = table1["0.99N"]["err"]
    after = train_autoencoder(X, lr, 0.99, True, updates, final_mu=0.9, final_updates=updates // 10)
    # Table 3: sparse-init scale multipliers (NAG, mu_max 0.99)
    table3 = {s: train_autoencoder(X, lr, 0.99, True, updates, si_scale=s) for s in (0.25, 0.5, 1, 2, 4)}
    return dict(table1=table1, table2=dict(before=before, after=after), table3=table3, updates=updates)


def train_rnn(mu0, nesterov, lr, updates, T, seed):
    torch.manual_seed(seed)
    model = RNN(2, 100, 1, radius=1.1, in_scale=0.1, seed=seed)   # addition: no distractors -> 0.1
    opt = Momentum(model.parameters(), lr, 0.9, nesterov)
    gen = torch.Generator().manual_seed(seed)
    for t in range(updates):
        opt.mu = 0.9 if t < 1000 else mu0                           # Section 4.2's schedule
        x, y = addition_problem(100, T, gen)
        loss = opt.step(lambda: ((model(x) - y) ** 2).mean())
        if not np.isfinite(loss):
            return 1.0
    xt, yt = addition_problem(2000, T, torch.Generator().manual_seed(999))
    with torch.no_grad():
        return float(((model(xt) - yt).abs() > 0.04).float().mean())   # zero-one loss


def e5_rnn(quick):
    updates, T = (500, 30) if quick else (4000, 50)
    seeds = (0,) if quick else (0, 1)
    out = {}
    # the "biases" baseline: predict the mean (always 1.0 before centring)
    xt, yt = addition_problem(2000, T, torch.Generator().manual_seed(999))
    out["biases"] = float((yt.abs() > 0.04).float().mean())
    for mu0 in (0.0, 0.9, 0.98):
        for nes in ((True, False) if mu0 > 0 else (True,)):
            name = "0" if mu0 == 0 else f"{mu0}{'N' if nes else 'M'}"
            best = None
            for lr in ((1e-3, 3e-4) if not quick else (1e-3,)):
                errs = [train_rnn(mu0, nes, lr, updates, T, s) for s in seeds]
                e = float(np.mean(errs))
                print(f"rnn {name:6s} lr {lr:<7} zero-one error {e:.3f}", flush=True)
                if best is None or e < best[0]:
                    best = (e, lr)
            out[name] = dict(err=best[0], lr=best[1])
    return dict(results=out, updates=updates, T=T)


# ---------------------------------------------------------------------------

def report(r):
    e1 = r["e1"]
    L = ["# Sutskever et al. (2013): reproduced results", "", "Generated by `python3 experiments.py`.", "",
         "## E1. CM vs NAG on an oblong quadratic (Section 2.1, appendix Fig. 2)", "",
         "Curvatures 1 and 100, learning rate 0.01, mu = 0.9, both methods identical otherwise.", "",
         f"- Theorem 2.1: NAG's effective momentum = mu (1 - lambda eps) = **{e1['eff_mu'][0]:.3f}** in the flat direction, "
         f"**{e1['eff_mu'][1]:.3f}** in the steep one (CM uses 0.9 in both).",
         f"- Sign flips in the steep direction over 100 steps: CM **{e1['cm_flips']}**, NAG **{e1['nag_flips']}**.",
         f"- Largest |x_steep| after step 10: CM {e1['cm_steep_max_after10']:.3f}, NAG {e1['nag_steep_max_after10']:.3f}.",
         f"- Distance to the minimum after 100 steps: CM {e1['cm_final']:.4f}, NAG {e1['nag_final']:.4f}.",
         f"- **But** with learning rate 0.015 and mu 0.95 (eps * lambda = 1.5): CM ends at distance {e1['big_lr']['cm_final']:.3f}, "
         f"NAG at {e1['big_lr']['nag_final']:.3g}: **NAG diverges**. Its effective momentum mu(1 - lambda eps) = -0.475 is negative.", ""]
    a = r["ae"]
    L += [f"## E2. Deep MNIST autoencoder (Table 1), {a['updates']:,} updates of 200 (Eq. 5 schedule compressed in time)", "",
          "Training squared error (sum over 784 pixels, averaged over images), best learning rate per column.", "",
          "| setting | squared error | best lr |", "|---|---|---|"]
    for k, v in a["table1"].items():
        L.append(f"| {k} | {v['err']:.2f} | {v['lr']} |")
    L += ["", "Paper (750,000 updates): SGD 2.1; NAG 0.9/0.99/0.995 = 1.0/0.73/0.75; CM 0.9/0.99/0.995 = 1.0/0.77/0.84.", "",
          "## E3. Lowering mu at the end (Table 2)", "",
          f"NAG, mu_max 0.99: squared error **{a['table2']['before']:.2f}** without, **{a['table2']['after']:.2f}** "
          "with mu lowered to 0.9 for the last 10% of updates. (Paper: 1.20 -> 0.73.)", "",
          "## E4. Scale of the sparse initialization (Table 3)", "",
          "| SI scale multiplier | " + " | ".join(str(k) for k in a["table3"]) + " |", "|---|" + "---|" * len(a["table3"]),
          "| squared error | " + " | ".join(f"{v:.2f}" for v in a["table3"].values()) + " |", "",
          "Paper (Curves dataset): 16, 16, 0.074, 0.083, 0.35: too small is catastrophic, 2x is fine, 4x is worse.", ""]
    rn = r["rnn"]
    L += [f"## E5. RNN, addition problem (Table 5), T = {rn['T']}, {rn['updates']:,} updates of 100", "",
          "Zero-one error (|prediction - target| > 0.04) on 2,000 new sequences; mean of seeds; best lr per column.", "",
          "| setting | error | best lr |", "|---|---|---|"]
    for k, v in rn["results"].items():
        L.append(f"| {k} | {v:.3f} | – |" if k == "biases" else f"| {k} | {v['err']:.3f} | {v['lr']} |")
    L += ["", "Paper (T = 80, 50,000 updates): biases 0.82; mu0 0: 0.39; NAG 0.9/0.98/0.995: 0.02/0.21/0.00025; "
          "CM 0.9/0.98/0.995: 0.43/0.62/0.036."]
    return "\n".join(L) + "\n"


def figures(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (HERE / "figures").mkdir(exist_ok=True)
    e1 = r["e1"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    xs = np.linspace(-11, 1, 200); ys = np.linspace(-1.5, 1.5, 200)
    Xg, Yg = np.meshgrid(xs, ys)
    ax.contour(Xg, Yg, 0.5 * (Xg ** 2 + 100 * Yg ** 2), levels=15, colors="lightgrey")
    for name, key, c in (("classical momentum", "cm_path", "tab:red"), ("Nesterov", "nag_path", "tab:blue")):
        p = np.array(e1[key])
        ax.plot(p[:, 0], p[:, 1], "-o", ms=2, lw=1, color=c, label=name)
    ax.set_xlabel("flat direction (curvature 1)"); ax.set_ylabel("steep direction (curvature 100)")
    ax.set_title("Appendix Fig. 2 reproduced: same learning rate and momentum", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "fig2_cm_vs_nag_quadratic.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        r = json.loads((HERE / "results.json").read_text())
    else:
        r = {"e1": e1_quadratic(), "ae": e2_e3_e4(args.quick), "rnn": e5_rnn(args.quick)}
        (HERE / "results.json").write_text(json.dumps(r))
    text = report(r)
    (HERE / "results.md").write_text(text)
    print(text)
    figures(r)


if __name__ == "__main__":
    main()
