"""Reproduce the figures and conclusions of Rosenblatt (1958).

  Figure 4   Pa vs retinal area R                      (theory, Eq. 1)
  Figure 5   Pc vs R for non-overlapping stimuli       (theory, Eq. 2)
  Figure 6   Pc vs overlap C                           (theory, Eqs. 2-3)
  Figure 7   ideal environment: P_r falls as more is learned; Sigma vs mu
  Figure 10  alpha vs gamma systems with unequal amounts of training
  Figure 11  differentiated environment: P_r and P_g meet at one asymptote
  + the generalization condition Pc12 < Pa < Pc11, distributed memory
    (conclusion 9) and bivalent trial-and-error learning (conclusion 7).

Run:  python3 experiments.py      (~15 seconds; writes results.md and figures/)
"""

from pathlib import Path

import numpy as np

from perceptron import Photoperceptron
from theory import Pa, Pc, Pc_min, overlap_to_LG

HERE = Path(__file__).parent
N_POINTS = 400                      # a 20 x 20 retina


# ---------------------------------------------------------------------------
# Stimulus environments
# ---------------------------------------------------------------------------

def ideal_environment(n, rng, R=0.5):
    """'Ideal environment': random dot patterns, each point lit with probability R.
    No two stimuli are related, so there is nothing to generalize from."""
    return rng.random((n, N_POINTS)) < R


PROTOTYPES = np.random.default_rng(1958).random((10, N_POINTS)) < 0.5


def prototype_classes(n, rng, flip=0.3, n_classes=2):
    """'Differentiated environment': class k = noisy copies of prototype k
    (each point flipped with probability `flip`), so members of a class are
    similar to each other and different from the other class."""
    labels = rng.integers(0, n_classes, n)
    S = PROTOTYPES[labels] ^ (rng.random((n, N_POINTS)) < flip)
    return S, labels


def shape_classes(n, rng, jitter=2):
    """Squares vs circles (both ~49 points) placed with a random shift of up to
    `jitter` pixels: the square-circle problem of the paper's Figure 11."""
    side = 20
    yy, xx = np.mgrid[:side, :side]
    labels = rng.integers(0, 2, n)
    S = np.zeros((n, N_POINTS), bool)
    for k, lab in enumerate(labels):
        cy, cx = side // 2 + rng.integers(-jitter, jitter + 1, 2)
        if lab == 0:
            img = (abs(yy - cy) <= 3) & (abs(xx - cx) <= 3)
        else:
            img = (yy - cy) ** 2 + (xx - cx) ** 2 <= 16
        S[k] = img.ravel()
    return S, labels


def mean_pc(A1, A2):
    """Measured Pc between two groups of stimuli: the average, over stimulus pairs,
    of P(A-unit fires for the 2nd | it fired for the 1st)."""
    both = (A1[:, None, :] & A2[None, :, :]).sum(-1)
    return float((both / np.maximum(A1.sum(-1)[:, None], 1)).mean())


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def fig4_data(R=np.linspace(0.01, 0.8, 40)):
    return {
        "(a) theta = 1, varying inhibitory mix":
            {f"x={x}, y={y}": [Pa(r, x, y, 1) for r in R] for x, y in [(10, 0), (8, 2), (6, 4), (5, 5)]},
        "(b) x = 10, y = 0, varying theta":
            {f"theta={t}": [Pa(r, 10, 0, t) for r in R] for t in [1, 2, 4, 6, 8, 10]},
        "(c) ~50% inhibitory, varying size and theta":
            {f"x={x}, y={x}, theta={t}": [Pa(r, x, x, t) for r in R] for x, t in [(5, 1), (5, 2), (5, 3), (10, 2), (10, 4)]},
    }, R


def fig5_data(R=np.linspace(0.02, 0.5, 25)):
    curves = {}
    for t in [1, 2, 4, 6]:
        curves[f"theta={t}"] = [Pc(r, *overlap_to_LG(r, 0.0), 10, 0, t) for r in R]
    return curves, R


def fig6_data(C=np.linspace(0, 1, 21)):
    curves = {}
    for R, style in [(0.5, "R=.5"), (0.2, "R=.2")]:
        for t in [2, 5, 10]:
            curves[f"{style}, theta={t}"] = [Pc(R, *overlap_to_LG(R, c), 10, 0, t) for c in C]
    return curves, C


def fig7_ideal(ns=(10, 30, 100, 300, 1000), NA=2000, seeds=3):
    """P_r (recall of the learned stimuli) and P_g (new stimuli) vs n_sr, the
    number of stimuli learned per response, in the ideal environment."""
    out = {}
    for system in ["alpha", "gamma"]:
        for mode in ["sigma", "mu"]:
            pr, pg = [], []
            for n in ns:
                r_vals, g_vals = [], []
                for s in range(seeds):
                    rng = np.random.default_rng(100 + s)
                    p = Photoperceptron(N_POINTS, NA, x=5, y=5, theta=3, rng=s)
                    S = ideal_environment(2 * n, rng)
                    lab = np.repeat([0, 1], n)
                    p.train_forced(S, lab, system)
                    r_vals.append(p.p_correct(S, lab, mode))
                    g_vals.append(p.p_correct(ideal_environment(400, rng), rng.integers(0, 2, 400), mode))
                pr.append(np.mean(r_vals))
                pg.append(np.mean(g_vals))
            out[f"{system}, {mode}"] = (pr, pg)
    return out, list(ns)


def fig10_variable(mean_ns=(20, 50, 100, 200), NR=10, NA=5000, seeds=3):
    """Unequal training: each response gets a random number of stimuli (between
    0.5x and 1.5x the mean). The gamma system keeps the total value of each
    source-set fixed, so heavily trained responses can't take over."""
    out = {k: [] for k in ["alpha, sigma", "alpha, mu", "gamma, sigma", "gamma, mu"]}
    for m in mean_ns:
        scores = {k: [] for k in out}
        for s in range(seeds):
            rng = np.random.default_rng(200 + s)
            counts = rng.integers(int(0.5 * m), int(1.5 * m) + 1, NR)
            lab = np.repeat(np.arange(NR), counts)
            S = ideal_environment(len(lab), rng)
            for system in ["alpha", "gamma"]:
                p = Photoperceptron(N_POINTS, NA, n_resp=NR, x=5, y=5, theta=3, rng=s)
                p.train_forced(S, lab, system)
                for mode in ["sigma", "mu"]:
                    scores[f"{system}, {mode}"].append(p.p_correct(S, lab, mode))
        for k in out:
            out[k].append(np.mean(scores[k]))
    return out, list(mean_ns)


def fig11_differentiated(ns=(2, 5, 10, 20, 50, 100, 300), NAs=(100, 400, 2000), flip=0.3, seeds=3):
    """Differentiated environment: P_r falls and P_g rises toward the SAME
    asymptote, and more A-units push that asymptote toward 1."""
    out = {}
    for NA in NAs:
        pr, pg = [], []
        for n in ns:
            r_vals, g_vals = [], []
            for s in range(seeds):
                rng = np.random.default_rng(300 + s)
                p = Photoperceptron(N_POINTS, NA, x=5, y=5, theta=3, rng=s)
                S, lab = prototype_classes(2 * n, rng, flip)
                T, lt = prototype_classes(400, rng, flip)
                p.train_forced(S, lab, "gamma")
                r_vals.append(p.p_correct(S, lab))
                g_vals.append(p.p_correct(T, lt))
            pr.append(np.mean(r_vals))
            pg.append(np.mean(g_vals))
        out[NA] = (pr, pg)
    return out, list(ns)


def pick_theta(R, x=5, y=5, target=0.05):
    """Choose the threshold whose theoretical Pa (Eq. 1) is closest to `target`
    for stimuli lighting a proportion R of the retina. The paper stresses that
    Pa must sit near a good value for the stimulus size (Figure 4 discussion)."""
    return min(range(-x, x + 1), key=lambda t: abs(Pa(R, x, y, t) - target))


def generalization_condition(NA=2000, n=300):
    """The paper's condition for better-than-chance generalization, Pc12 < Pa < Pc11,
    measured on four environments next to the P_g actually reached.

    The paper says that if the condition holds, performance will be better than
    chance, and if it doesn't, improvement "may not" occur: it is sufficient, not
    necessary. What these runs show matters most is the gap Pc11 - Pc12: how much
    more A-unit activity two members of the SAME class share than members of
    DIFFERENT classes."""
    rows = []
    envs = [("noisy prototypes, 30% noise", lambda k, rng: prototype_classes(k, rng, 0.3)),
            ("squares vs circles, fixed position", lambda k, rng: shape_classes(k, rng, 0)),
            ("squares vs circles, shifted up to 2 px", lambda k, rng: shape_classes(k, rng, 2)),
            ("squares vs circles, shifted up to 4 px", lambda k, rng: shape_classes(k, rng, 4))]
    for name, make in envs:
        rng = np.random.default_rng(7)
        S, lab = make(2 * n, rng)
        theta = pick_theta(S.mean())
        p = Photoperceptron(N_POINTS, NA, x=5, y=5, theta=theta, rng=1)
        A = p.activate(S)
        a0, a1 = A[lab == 0][:60], A[lab == 1][:60]
        pa = float(A.mean())
        pc11 = mean_pc(a0, a0[::-1])
        pc12 = mean_pc(a0, a1)
        p.train_forced(S, lab, "gamma")
        T, lt = make(400, rng)
        rows.append((name, theta, pc12, pa, pc11, pc12 < pa < pc11, p.p_correct(T, lt)))
    return rows


def distributed_memory(fractions=(0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95), NA=2000):
    """Conclusion 9: removing part of the association system gives a gradual,
    general loss, not the loss of particular memories."""
    rng = np.random.default_rng(9)
    p = Photoperceptron(N_POINTS, NA, n_resp=5, x=5, y=5, theta=3, rng=9)
    S, lab = prototype_classes(500, rng, flip=0.3, n_classes=5)
    T, lt = prototype_classes(500, rng, flip=0.3, n_classes=5)
    p.train_forced(S, lab, "gamma")
    order = rng.permutation(NA)
    res = []
    for f in fractions:
        p.alive[:] = True
        p.alive[order[:int(f * NA)]] = False
        per_class = [p.p_correct(T[lt == k], lt[lt == k]) for k in range(5)]
        res.append((f, p.p_correct(T, lt), min(per_class), max(per_class)))
    p.alive[:] = True
    return res


def bivalent(epochs=8, NA=1000):
    """Conclusion 7: trial-and-error learning with reward and punishment.
    Here each of 4 responses must be learned for noisy copies of its prototype."""
    rng = np.random.default_rng(11)
    p = Photoperceptron(N_POINTS, NA, n_resp=4, x=5, y=5, theta=3, rng=11)
    S, lab = prototype_classes(400, rng, flip=0.35, n_classes=4)
    T, lt = prototype_classes(400, rng, flip=0.35, n_classes=4)
    errors = p.train_bivalent(S, lab, epochs=epochs)
    return errors, len(S), p.p_correct(T, lt)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_figures(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.mkdir(exist_ok=True)

    panels, R = fig4_data()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, curves) in zip(axes, panels.items()):
        for label, ys in curves.items():
            ax.plot(R, ys, label=label)
        ax.set_title(title, fontsize=9); ax.set_xlabel("R (proportion of retina lit)"); ax.set_ylabel("Pa")
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1.05); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("Figure 4: Pa as a function of retinal area illuminated")
    fig.tight_layout(); fig.savefig(path / "fig4_Pa.png", dpi=130); plt.close(fig)

    curves, R = fig5_data()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, ys in curves.items():
        ax.plot(R, ys, label=label)
    ax.set_title("Figure 5: Pc vs R, non-overlapping stimuli (x=10, y=0)", fontsize=10)
    ax.set_xlabel("R"); ax.set_ylabel("Pc"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path / "fig5_Pc_nonoverlap.png", dpi=130); plt.close(fig)

    curves, C = fig6_data()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, ys in curves.items():
        ax.plot(C, ys, "-" if "R=.5" in label else "--", label=label)
    ax.axhline(0, color="k", lw=.5)
    ax.set_title("Figure 6: Pc vs overlap C (x=10, y=0; solid R=.5, dashed R=.2)", fontsize=10)
    ax.set_xlabel("C (proportion of overlap between stimuli)"); ax.set_ylabel("Pc")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path / "fig6_Pc_overlap.png", dpi=130); plt.close(fig)

    data, ns = fig7_ideal()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (pr, pg) in data.items():
        ax.semilogx(ns, pr, "o-", label=f"P_r {label}")
    ax.semilogx(ns, data["gamma, mu"][1], "k:", label="P_g (any system)")
    ax.set_title("Figure 7: ideal environment (random stimuli)", fontsize=10)
    ax.set_xlabel("n_sr (stimuli learned per response)"); ax.set_ylabel("probability correct")
    ax.set_ylim(0.4, 1.02); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path / "fig7_ideal.png", dpi=130); plt.close(fig)

    data, ns = fig11_differentiated()
    fig, ax = plt.subplots(figsize=(6, 4))
    for NA, (pr, pg) in data.items():
        line, = ax.semilogx(ns, pr, "o-", label=f"P_r, N_A={NA}")
        ax.semilogx(ns, pg, "s--", color=line.get_color(), label=f"P_g, N_A={NA}")
    ax.set_title("Figure 11: differentiated environment (noisy prototypes)", fontsize=10)
    ax.set_xlabel("n_sr (stimuli learned per response)"); ax.set_ylabel("probability correct")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path / "fig11_differentiated.png", dpi=130); plt.close(fig)


def main():
    lines = ["# Rosenblatt (1958): reproduced results", "", "Generated by `python3 experiments.py`.", ""]

    data, ns = fig7_ideal()
    lines += ["## Ideal environment (Figure 7, conclusions 1-3)", "",
              "P_r = recall of the learned stimuli, P_g = new random stimuli.", "",
              "| system | " + " | ".join(f"n_sr={n}" for n in ns) + " |",
              "|---|" + "---|" * len(ns)]
    for label, (pr, pg) in data.items():
        lines.append(f"| P_r {label} | " + " | ".join(f"{v:.3f}" for v in pr) + " |")
    lines.append("| P_g gamma, mu | " + " | ".join(f"{v:.3f}" for v in data["gamma, mu"][1]) + " |")

    data, ns = fig10_variable()
    lines += ["", "## Unequal training, 10 responses (Figure 10)", "",
              "Each response learns a random number of stimuli (0.5x-1.5x the mean). Probability the",
              "correct response beats all 9 others.", "",
              "| system | " + " | ".join(f"mean n_sr={n}" for n in ns) + " |",
              "|---|" + "---|" * len(ns)]
    for label, vals in data.items():
        lines.append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    data, ns = fig11_differentiated()
    lines += ["", "## Differentiated environment (Figure 11, conclusions 4-5)", "",
              "Two classes = noisy copies (30% of points flipped) of two random prototypes.", "",
              "| N_A | | " + " | ".join(f"n_sr={n}" for n in ns) + " |",
              "|---|---|" + "---|" * len(ns)]
    for NA, (pr, pg) in data.items():
        lines.append(f"| {NA} | P_r | " + " | ".join(f"{v:.3f}" for v in pr) + " |")
        lines.append(f"| {NA} | P_g | " + " | ".join(f"{v:.3f}" for v in pg) + " |")

    lines += ["", "## When can it generalize? Pc12 < Pa < Pc11", "",
              "The condition is sufficient, not necessary; the gap Pc11 - Pc12 is what decides.", "",
              "Threshold chosen per environment so that the theoretical Pa is about 0.05.", "",
              "| environment | theta | Pc12 | Pa | Pc11 | gap Pc11-Pc12 | condition met | P_g |",
              "|---|---|---|---|---|---|---|---|"]
    for name, theta, pc12, pa, pc11, ok, pg in generalization_condition():
        lines.append(f"| {name} | {theta} | {pc12:.3f} | {pa:.3f} | {pc11:.3f} | {pc11 - pc12:.3f} | "
                     f"{'yes' if ok else 'no'} | {pg:.3f} |")

    lines += ["", "## Distributed memory (conclusion 9)", "",
              "5 classes learned, then a random part of the A-units is removed.", "",
              "| A-units removed | P_g overall | worst class | best class |", "|---|---|---|---|"]
    for f, overall, worst, best in distributed_memory():
        lines.append(f"| {f:.0%} | {overall:.3f} | {worst:.3f} | {best:.3f} |")

    errors, n, pg = bivalent()
    lines += ["", "## Bivalent trial-and-error learning (conclusion 7)", "",
              f"4 classes, {n} training stimuli. Mistakes per pass: {errors}. P_g on new stimuli: {pg:.3f}."]

    text = "\n".join(lines) + "\n"
    (HERE / "results.md").write_text(text)
    print(text)
    save_figures(HERE / "figures")
    print("Saved results.md and figures/")


if __name__ == "__main__":
    main()
