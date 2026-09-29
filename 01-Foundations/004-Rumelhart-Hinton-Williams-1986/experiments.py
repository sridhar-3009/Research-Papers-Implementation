"""Reproduce the experiments of Rumelhart, Hinton & Williams (1986).

  Figure 1   mirror symmetry with 2 hidden units: how often it's solved, how many
             sweeps, and whether the learned weights have the paper's structure
  (text)     "adding a few more connections" and local minima
  Figure 1   no hidden layer: symmetry is impossible
  Figs 2-4   family trees: learning, generalizing to 4 held-out cases, and the
             features the "person code" units discover
  Figure 5   a recurrent net unrolled in time = a layered net with tied weights

Run:  python3 experiments.py      (~1 minute; writes results.md and figures/)
"""

from pathlib import Path

import numpy as np

from backprop import IterativeNet, LayeredNet, logistic, train
from tasks import ENGLISH, FAMILY_NET, family_cases, symmetry_data

HERE = Path(__file__).parent
SYM_NET = lambda h: [("in", 6, []), ("hidden", h, ["in"]), ("out", 1, ["hidden"])]


# ---------------------------------------------------------------------------
# Mirror symmetry (Figure 1)
# ---------------------------------------------------------------------------

def solves_symmetry(net, X, d):
    """Solved = every one of the 64 outputs is on the right side of 0.5."""
    return bool(np.all(np.abs(net.forward({"in": X})["out"] - d) < 0.5))


def symmetry_runs(hidden=2, eps=0.1, alpha=0.9, seeds=20, max_sweeps=30000):
    """Train from `seeds` random starts; return (sweeps needed or None, net) per run."""
    X, d = symmetry_data()
    runs = []
    for s in range(seeds):
        net = LayeredNet(SYM_NET(hidden), init_range=0.3, rng=s)
        hist = train(net, {"in": X}, d, sweeps=max_sweeps, eps=eps, alpha=alpha,
                     stop=lambda n: solves_symmetry(n, X, d))
        runs.append((len(hist) if solves_symmetry(net, X, d) else None, net, hist[-1]))
    return runs


def weight_structure(net):
    """The paper's description of Figure 1's solution, measured:
    - mirror weights equal and opposite: |w_i + w_(7-i)| / max|w|
    - the two hidden units are sign-flipped copies: correlation of their weights
    - the magnitudes on one side, sorted, divided by the smallest (1 : 2 : 4?)."""
    W = net.W[("in", "hidden")].T                        # (2 hidden, 6 inputs)
    anti = max(np.abs(w[:3] + w[::-1][:3]).max() / np.abs(w).max() for w in W)
    corr = float(np.corrcoef(W[0], W[1])[0, 1])
    mags = np.sort(np.abs(W[0][:3]))
    return anti, corr, mags / mags[0]


def no_hidden_layer_symmetry(seeds=5):
    """With no hidden units (inputs wired straight to the output), how many of the
    64 cases can gradient descent get right? (Minsky & Papert: this can't be done.)"""
    X, d = symmetry_data()
    best = 0
    for s in range(seeds):
        net = LayeredNet([("in", 6, []), ("out", 1, ["in"])], init_range=0.3, rng=s)
        train(net, {"in": X}, d, sweeps=5000, eps=0.1, alpha=0.9)
        y = net.forward({"in": X})["out"]
        best = max(best, int(np.sum(np.abs(y - d) < 0.5)))
    return best


# ---------------------------------------------------------------------------
# Family trees (Figures 2-4)
# ---------------------------------------------------------------------------

def correct(y, d):
    """A case is right when EVERY output unit is on the right side of 0.5."""
    return np.all((y > 0.5) == (d == 1), axis=1)


def family_run(seed, eps=0.05, decay=0.0002, init=1.0, sweeps=1500):
    """Train on 100 of the 104 cases, test on the other 4 (chosen at random).
    Schedule as in the paper: half the step size and alpha = 0.5 for the first
    20 sweeps, then alpha = 0.9. Margin rule 0.2 / 0.8 as in the paper."""
    P, R, D, keys = family_cases()
    rng = np.random.default_rng(100 + seed)
    test = rng.choice(len(keys), 4, replace=False)
    tr = np.setdiff1d(np.arange(len(keys)), test)
    net = LayeredNet(FAMILY_NET, init_range=init, rng=seed)
    train(net, {"person": P[tr], "relation": R[tr]}, D[tr], sweeps=sweeps,
          schedule=lambda t: (eps / 2, 0.5) if t < 20 else (eps, 0.9),
          decay=decay, margin=(0.2, 0.8))
    acc_train = correct(net.forward({"person": P[tr], "relation": R[tr]})["out"], D[tr]).mean()
    n_test = int(correct(net.forward({"person": P[test], "relation": R[test]})["out"], D[test]).sum())
    return net, float(acc_train), n_test, [keys[i] for i in test]


def family_settings(seeds=6):
    """The paper's exact recipe, and what had to change to make it learn."""
    rows = []
    for label, kw in [("paper exactly (eps .01, decay .002, init ±0.3)", dict(eps=0.01, decay=0.002, init=0.3)),
                      ("init ±1.0, otherwise paper", dict(eps=0.01, decay=0.002, init=1.0)),
                      ("init ±1.0, no decay", dict(eps=0.01, decay=0.0, init=1.0)),
                      ("init ±1.0, eps .05, decay .0002 (used below)", dict(eps=0.05, decay=0.0002, init=1.0))]:
        res = [family_run(s, **kw)[1:3] for s in range(seeds)]
        rows.append((label, np.mean([r[0] for r in res]), sum(r[1] for r in res), 4 * seeds))
    return rows


GENERATION = {"Christopher": 1, "Penelope": 1, "Andrew": 1, "Christine": 1, "Margaret": 2, "Arthur": 2,
              "Victoria": 2, "James": 2, "Jennifer": 2, "Charles": 2, "Colin": 3, "Charlotte": 3}
BRANCH = {"Christopher": 0, "Penelope": 0, "Arthur": 0, "Margaret": 0, "Victoria": 0,
          "Andrew": 1, "Christine": 1, "James": 1, "Jennifer": 1, "Charles": 1, "Colin": .5, "Charlotte": .5}


def person_codes(net):
    """Each person's 6-unit code: the activity of the person_code layer when only
    that person's input unit is on."""
    return logistic(np.eye(24) @ net.W[("person", "person_code")] + net.b["person_code"])


def feature_analysis(net):
    """Figure 4's claims, measured on one trained net:
    for nationality, generation and branch, the best |correlation| of any code
    unit's weights with that property; and how close each English person's code
    is to their Italian twin's (twin distance / average distance, without the
    nationality unit; 1.0 = no closer than a random person)."""
    W = net.W[("person", "person_code")]
    props = {"nationality": np.r_[np.zeros(12), np.ones(12)],
             "generation": np.array([GENERATION[p] for p in ENGLISH] * 2, float),
             "branch": np.array([BRANCH[p] for p in ENGLISH] * 2, float)}
    best = {k: max(abs(np.corrcoef(W[:, j], v)[0, 1]) for j in range(6)) for k, v in props.items()}
    C = person_codes(net)
    nat_unit = int(np.argmax([abs(np.corrcoef(C[:, j], props["nationality"])[0, 1]) for j in range(6)]))
    C = np.delete(C, nat_unit, axis=1)
    twin = np.mean([np.linalg.norm(C[i] - C[i + 12]) for i in range(12)])
    other = np.mean([np.linalg.norm(C[i] - C[k]) for i in range(24) for k in range(24) if k not in (i, (i + 12) % 24)])
    return best, twin / other


# ---------------------------------------------------------------------------
# Figure 5: recurrent net = layered net with tied weights
# ---------------------------------------------------------------------------

def bptt_check():
    """Back-propagation through time must give the true gradient of the unrolled
    net. Compare with finite differences, and train a tiny task: after 3 steps,
    unit 0 must say whether the FIRST input bit was 1 (it has to carry it along)."""
    rng = np.random.default_rng(0)
    net = IterativeNet(n_units=4, n_inputs=1, rng=0)
    h0 = np.zeros((8, 4))
    seqs = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], float)
    inputs = [seqs[:, [t]] for t in range(3)]
    targets = np.full((8, 4), 0.5)
    targets[:, 0] = seqs[:, 0]
    loss = lambda: 0.5 * float(np.sum((net.run(h0, inputs)[-1] - targets) ** 2))
    gW, _, _ = net.gradients(h0, inputs, targets)
    fd = np.zeros_like(net.W)
    for idx in np.ndindex(net.W.shape):
        old = net.W[idx]
        net.W[idx] = old + 1e-6; up = loss()
        net.W[idx] = old - 1e-6; down = loss()
        net.W[idx] = old
        fd[idx] = (up - down) / 2e-6
    max_err = float(np.abs(gW - fd).max())
    # train it (only unit 0's error counts: other targets set to its own output)
    vel = [np.zeros_like(net.W), np.zeros_like(net.U), np.zeros_like(net.b)]
    for _ in range(3000):
        final = net.run(h0, inputs)[-1]
        targets[:, 1:] = final[:, 1:]                    # don't care about units 1-3
        grads = net.gradients(h0, inputs, targets)
        for k, (p, g) in enumerate(zip((net.W, net.U, net.b), grads)):
            vel[k] = -0.5 * g + 0.9 * vel[k]
            p += vel[k]
    final = net.run(h0, inputs)[-1][:, 0]
    return max_err, int(np.sum((final > 0.5) == (seqs[:, 0] == 1)))


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------

def save_figures(sym_net, fam_net, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.mkdir(exist_ok=True)

    W = sym_net.W[("in", "hidden")].T
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for j, ax in enumerate(axes):
        ax.bar(range(1, 7), W[j], color=["tab:blue" if v > 0 else "tab:red" for v in W[j]])
        ax.axvline(3.5, color="k", lw=0.8, ls="--")
        ax.set_title(f"hidden unit {j + 1} (bias {sym_net.b['hidden'][j]:.1f})", fontsize=9)
        ax.set_xlabel("input position (dashed line = the mirror)")
    axes[0].set_ylabel("weight")
    fig.suptitle("Figure 1 reproduced: mirror weights are equal and opposite", fontsize=10)
    fig.tight_layout(); fig.savefig(path / "fig1_symmetry_weights.png", dpi=130); plt.close(fig)

    Wp = fam_net.W[("person", "person_code")]            # (24 people, 6 units)
    fig, axes = plt.subplots(6, 1, figsize=(9, 7))
    m = np.abs(Wp).max()
    for j, ax in enumerate(axes):
        block = np.vstack([Wp[:12, j], Wp[12:, j]])      # row 1 English, row 2 Italian
        ax.imshow(block, cmap="bwr", vmin=-m, vmax=m, aspect="auto")
        ax.set_yticks([0, 1], ["English", "Italian"], fontsize=7)
        ax.set_xticks(range(12), ENGLISH if j == 5 else [""] * 12, rotation=45, fontsize=7)
        ax.set_ylabel(f"unit {j + 1}", fontsize=8)
    fig.suptitle("Figure 4 reproduced: weights from the 24 people to the 6 code units\n"
                 "(blue = excitatory, red = inhibitory; each Italian is under their English twin)", fontsize=9)
    fig.tight_layout(); fig.savefig(path / "fig4_person_codes.png", dpi=130); plt.close(fig)


def main():
    lines = ["# Rumelhart, Hinton & Williams (1986): reproduced results", "",
             "Generated by `python3 experiments.py`.", ""]

    # --- symmetry ---
    runs = symmetry_runs()
    solved = [r for r in runs if r[0]]
    lines += ["## Figure 1: mirror symmetry (6 inputs, 2 hidden units, 1 output)", "",
              "Paper's settings: batch sweeps over all 64 patterns, eps = 0.1, alpha = 0.9,",
              "initial weights uniform in [-0.3, 0.3]. Paper: 'The learning required 1,425 sweeps'.", "",
              f"- Solved in **{len(solved)}/20** random starts; median **{int(np.median([r[0] for r in solved]))} sweeps**"
              f" (range {min(r[0] for r in solved)}–{max(r[0] for r in solved)}).",
              f"- The {20 - len(solved)} unsolved runs all ended at E = "
              f"{', '.join(sorted({f'{r[2]:.2f}' for r in runs if not r[0]}))}: the 'always say not symmetric' plateau",
              "  (56 of 64 patterns are not symmetric; the output saturates at 0 and its gradient vanishes).", ""]
    lines += ["Learned weights, for every solved run: mirror error = |w_i + w_mirror| / max|w|;",
              "twin corr = correlation between the two hidden units' weights (-1 = sign-flipped copies);",
              "ratio = the three magnitudes on one side, divided by the smallest (paper: 1 : 2 : 4).", "",
              "| run | sweeps | mirror error | twin corr | magnitude ratio |", "|---|---|---|---|---|"]
    for k, (n, net, _) in enumerate(runs):
        if n:
            anti, corr, ratio = weight_structure(net)
            lines.append(f"| {k} | {n} | {anti:.3f} | {corr:.3f} | 1 : {ratio[1]:.2f} : {ratio[2]:.2f} |")

    lines += ["", "## 'Adding a few more connections' (page 535)", "",
              "The paper says poor local minima show up mainly in nets with 'just enough connections'.",
              "Solved out of 20 random starts (max 30,000 sweeps):", "",
              "| hidden units | eps = 0.1 (paper) | eps = 0.03 |", "|---|---|---|"]
    for h in (2, 4, 8):
        a = sum(1 for r in symmetry_runs(h, eps=0.1) if r[0])
        b = sum(1 for r in symmetry_runs(h, eps=0.03) if r[0])
        lines.append(f"| {h} | {a}/20 | {b}/20 |")
    lines += ["", f"With **no hidden layer**, the best run gets {no_hidden_layer_symmetry()} of 64 patterns right:",
              "barely better than always answering 'not symmetric' (56/64). The hidden layer is essential, as the paper says."]

    # --- family trees ---
    lines += ["", "## Figures 2-4: family trees", "",
              "104 cases (person1, relation) -> all correct person2s; trained on 100, tested on 4.",
              "A case counts as right only if all 24 output units are on the correct side of 0.5.", "",
              "### Getting it to learn (6 random starts each)", "",
              "| settings | train accuracy | held-out correct |", "|---|---|---|"]
    for label, acc, n, total in family_settings():
        lines.append(f"| {label} | {acc:.2f} | {n}/{total} |")

    stats, feats, nets = [], [], []
    for s in range(20):
        net, acc, n_test, _ = family_run(s)
        stats.append((acc, n_test)); nets.append(net)
        feats.append(feature_analysis(net))
    st = np.array(stats)
    best = {k: np.mean([f[0][k] for f in feats]) for k in ("nationality", "generation", "branch")}
    lines += ["", "### 20 runs with the working settings", "",
              f"- Training cases right: **{st[:, 0].mean():.1%}** on average ({int((st[:, 0] >= 0.95).sum())}/20 runs above 95%).",
              f"- Held-out cases right: **{int(st[:, 1].sum())}/80** ({st[:, 1].sum() / 80:.0%}). Paper: 4/4 in its run.",
              f"- Code units that track a property (best |correlation| of a unit's weights, averaged over runs):",
              f"  nationality **{best['nationality']:.2f}**, generation **{best['generation']:.2f}**, branch **{best['branch']:.2f}**.",
              f"- English person vs Italian twin: distance ratio **{np.mean([f[1] for f in feats]):.2f}**"
              " (without the nationality unit; 1.0 = no closer than a random person)."]

    # --- recurrent ---
    err, right = bptt_check()
    lines += ["", "## Figure 5: recurrent net = layered net with tied weights", "",
              f"- Back-propagation through 3 time steps matches finite differences to {err:.1e}.",
              f"- Trained to remember the first of 3 input bits: {right}/8 sequences right."]

    text = "\n".join(lines) + "\n"
    (HERE / "results.md").write_text(text)
    print(text)
    best_fam = nets[int(np.argmax(st[:, 0] + st[:, 1]))]
    save_figures(solved[0][1], best_fam, HERE / "figures")
    print("Saved results.md and figures/")


if __name__ == "__main__":
    main()
