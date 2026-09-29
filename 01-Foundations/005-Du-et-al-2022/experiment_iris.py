"""Reproduce the Iris experiment of Section 12.1 (Table 1 and Figure 4).

Setup, as described in the paper:
  - Iris: 150 patterns, 4 features, 3 classes. 80% train / 20% test.
  - A 4-4-3 MLP: logistic hidden layer, linear output layer.
  - Targets: +1 for the true class, -1 for the others.
  - Prediction: the output node with the largest value wins.
  - Performance goal MSE = 0.001, at most 1000 epochs, 50 independent runs.

Details the paper doesn't give, chosen here:
  - ONE fixed stratified split (40 train / 10 test per class) for all 50 runs; only
    the initial weights change between runs. The paper doesn't say this, but its
    Table 1 implies it: LM and BFGS have accuracy std 0.000, i.e. every run scored
    the same on the test set. Use --resplit to draw a new split every run instead.
  - Inputs are scaled to [-1, 1] using the training set's min and max.

Run:  python3 experiment_iris.py            (all 8 methods, 50 runs)
      python3 experiment_iris.py --runs 5   (quick check)
"""

import argparse
import time
from pathlib import Path

import numpy as np

from mlp import MLP
from optimizers import train

HERE = Path(__file__).parent

# The 8 algorithms in the paper's Table 1, in the paper's order.
PAPER_METHODS = {"rprop": "RP", "lm": "LM", "bfgs": "BFGS", "oss": "OSS",
                 "scg": "SCG", "cgb": "CGB", "cgf": "CGF", "cgp": "CGP"}

# Table 1 of the paper: mean epochs, training MSE, accuracy (%), mean time (s).
PAPER_TABLE_1 = {"RP": (990.94, 0.025, 96.00, 0.7372), "LM": (238.54, 0.007, 100.00, 0.3186),
                 "BFGS": (154.72, 0.015, 93.33, 0.2619), "OSS": (999.94, 0.027, 96.53, 1.1087),
                 "SCG": (903.16, 0.016, 95.40, 0.7462), "CGB": (439.48, 0.028, 95.27, 0.5456),
                 "CGF": (562.64, 0.021, 95.87, 0.6820), "CGP": (573.38, 0.021, 96.27, 0.6962)}


def load_iris():
    """Return X (150, 4) features and labels (150,) in {0, 1, 2}."""
    data = np.genfromtxt(HERE / "iris.csv", delimiter=",", skip_header=1, dtype=str)
    X = data[:, :4].astype(float)
    names, labels = np.unique(data[:, 4], return_inverse=True)
    return X, labels


def split_and_scale(X, labels, rng):
    """Stratified 80/20 split, inputs scaled to [-1, 1], targets coded +1/-1."""
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        cut = int(0.8 * len(idx))
        train_idx += list(idx[:cut])
        test_idx += list(idx[cut:])
    lo, hi = X[train_idx].min(0), X[train_idx].max(0)
    scale = lambda A: 2 * (A - lo) / (hi - lo) - 1
    Y = -np.ones((len(X), labels.max() + 1))
    Y[np.arange(len(X)), labels] = 1
    return (scale(X[train_idx]), Y[train_idx], labels[train_idx],
            scale(X[test_idx]), labels[test_idx])


def run(methods, runs, seed=0, resplit=False):
    X, labels = load_iris()
    results = {m: {"epochs": [], "mse": [], "acc": [], "time": [], "curve": None} for m in methods}
    for r in range(runs):
        split_seed = seed + r if resplit else seed
        Xtr, Ytr, ltr, Xte, lte = split_and_scale(X, labels, np.random.default_rng(split_seed))
        init = MLP([4, 4, 3], hidden="logistic", output="linear", rng=seed + r).get_params()
        for m in methods:
            net = MLP([4, 4, 3], hidden="logistic", output="linear")
            net.set_params(init.copy())               # every method starts from the SAME weights
            t0 = time.perf_counter()
            history = train(net, Xtr, Ytr, method=m, epochs=1000, goal=1e-3, rng=seed + r)
            elapsed = time.perf_counter() - t0
            acc = np.mean(net.predict(Xte).argmax(1) == lte) * 100
            res = results[m]
            res["epochs"].append(len(history) - 1)
            res["mse"].append(history[-1])
            res["acc"].append(acc)
            res["time"].append(elapsed)
            if r == 0:
                res["curve"] = history
    return results


def table(results):
    lines = ["| Algorithm | Mean epochs | Training MSE | Test accuracy (%) | Acc. std | Mean time (s) |"
             " Paper: epochs / MSE / acc (%) |",
             "|---|---|---|---|---|---|---|"]
    for m, res in results.items():
        name = PAPER_METHODS.get(m, m.upper())
        paper = PAPER_TABLE_1.get(name)
        paper_s = f"{paper[0]:.0f} / {paper[1]:.3f} / {paper[2]:.2f}" if paper else "n/a"
        lines.append(f"| {name} | {np.mean(res['epochs']):.2f} | {np.mean(res['mse']):.3f} | "
                     f"{np.mean(res['acc']):.2f} | {np.std(np.array(res['acc']) / 100):.3f} | "
                     f"{np.mean(res['time']):.4f} | {paper_s} |")
    return "\n".join(lines)


def plot(results, path):
    """Figure 4: training error vs epoch for one run, log-log axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, res in results.items():
        curve = res["curve"]
        ax.loglog(np.arange(1, len(curve) + 1), curve, label=PAPER_METHODS.get(m, m.upper()))
    ax.set_xlabel("Number of training epochs")
    ax.set_ylabel("Training MSE")
    ax.set_title("4-4-3 MLP on Iris: training error vs epochs (run 1)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--methods", nargs="+", default=list(PAPER_METHODS))
    ap.add_argument("--resplit", action="store_true", help="new train/test split every run")
    ap.add_argument("--no-save", action="store_true", help="print only; don't write results.md / figure")
    args = ap.parse_args()

    results = run(args.methods, args.runs, resplit=args.resplit)
    md = table(results)
    print(md)
    if not args.no_save:
        (HERE / "results.md").write_text(
            f"# Iris results ({args.runs} runs, {'new split each run' if args.resplit else 'one fixed split'})\n\n"
            f"Generated by `python3 experiment_iris.py`.\n\n{md}\n"
            "\nTimes are from this machine and this Python code; the paper used MATLAB, "
            "so compare times only within a column.\n")
        plot(results, HERE / "figure4_learning_curves.png")
        print("\nSaved results.md and figure4_learning_curves.png")


if __name__ == "__main__":
    main()
