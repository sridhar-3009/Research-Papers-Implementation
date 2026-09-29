"""Tests: each checks an equation or a result of Rumelhart, Hinton & Williams (1986).

Run with:  python3 -m pytest -q
"""

import numpy as np
import pytest

from backprop import LayeredNet, train
from experiments import bptt_check, family_run, solves_symmetry, weight_structure
from tasks import family_cases, family_facts, symmetry_data


def finite_difference(net, inputs, d, margin=None, h=1e-6):
    """dE/dw measured directly: nudge each weight up and down, see how E changes."""
    out = {}
    for k, p in net.params().items():
        g = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + h; up = net.error(inputs, d, margin)
            p[idx] = old - h; down = net.error(inputs, d, margin)
            p[idx] = old
            g[idx] = (up - down) / (2 * h)
        out[k] = g
    return out


# ---- Equations (1)-(7): the backward pass gives the true gradient ----

@pytest.mark.parametrize("margin", [None, (0.2, 0.8)])
def test_backprop_matches_finite_differences(margin):
    # Two input groups, a skip connection (in_a -> out), several layers: the
    # general layered net of page 533.
    spec = [("in_a", 3, []), ("in_b", 2, []), ("h1", 4, ["in_a"]), ("h2", 3, ["h1", "in_b"]),
            ("out", 2, ["h2", "in_a"])]
    rng = np.random.default_rng(0)
    net = LayeredNet(spec, init_range=1.0, rng=1)
    inputs = {"in_a": rng.random((5, 3)), "in_b": rng.random((5, 2))}
    d = (rng.random((5, 2)) > 0.5) * 1.0
    g = net.gradients(inputs, d, margin)
    fd = finite_difference(net, inputs, d, margin)
    for k in g:
        assert np.allclose(g[k], fd[k], atol=1e-6), k


def test_error_is_half_sum_of_squares():
    # Eq. (3)
    net = LayeredNet([("in", 2, []), ("out", 1, ["in"])], rng=0)
    X, d = np.eye(2), np.array([[1.0], [0.0]])
    y = net.forward({"in": X})["out"]
    assert net.error({"in": X}, d) == pytest.approx(0.5 * np.sum((y - d) ** 2))


def test_margin_rule_ignores_close_enough_outputs():
    # Figure 4 caption: no error if an "on" unit is above 0.8 or an "off" unit below 0.2.
    y = np.array([[0.9, 0.1, 0.7, 0.3]])
    d = np.array([[1.0, 0.0, 1.0, 0.0]])
    err = LayeredNet.output_error(y, d, margin=(0.2, 0.8))
    assert np.allclose(err, [[0.0, 0.0, -0.3, 0.3]])


def test_momentum_and_weight_decay():
    # Eq. (9) with alpha, and "decrementing every weight by 0.2%": decay shrinks weights.
    X, d = symmetry_data()
    norms = []
    for decay in (0.0, 0.01):
        net = LayeredNet([("in", 6, []), ("hidden", 2, ["in"]), ("out", 1, ["hidden"])], rng=3)
        train(net, {"in": X}, d, sweeps=300, eps=0.1, alpha=0.9, decay=decay)
        norms.append(sum(np.sum(p ** 2) for p in net.params().values()))
    assert norms[1] < norms[0]


# ---- Figure 1: mirror symmetry ----

def test_symmetry_data():
    X, d = symmetry_data()
    assert X.shape == (64, 6) and d.sum() == 8           # 8 of the 64 vectors are symmetric


def test_learns_symmetry_with_the_papers_solution():
    X, d = symmetry_data()
    net = LayeredNet([("in", 6, []), ("hidden", 2, ["in"]), ("out", 1, ["hidden"])], init_range=0.3, rng=3)
    hist = train(net, {"in": X}, d, sweeps=5000, eps=0.1, alpha=0.9, stop=lambda n: solves_symmetry(n, X, d))
    assert solves_symmetry(net, X, d) and len(hist) < 2000
    anti, corr, ratio = weight_structure(net)
    assert anti < 0.05                  # mirror weights equal and opposite
    assert corr < -0.99                 # the two hidden units are sign-flipped copies
    assert ratio == pytest.approx([1, 2, 4], rel=0.1)    # this run finds exactly 1 : 2 : 4
    assert np.all(net.b["hidden"] < 0) and net.b["out"][0] > 0   # biases as in the caption


def test_no_hidden_layer_cannot_do_symmetry():
    X, d = symmetry_data()
    net = LayeredNet([("in", 6, []), ("out", 1, ["in"])], init_range=0.3, rng=0)
    train(net, {"in": X}, d, sweeps=3000, eps=0.1, alpha=0.9)
    assert not solves_symmetry(net, X, d)


def test_smaller_steps_avoid_the_plateau():
    # With eps = 0.03, bigger nets solve symmetry from every start we try.
    X, d = symmetry_data()
    for s in range(3):
        net = LayeredNet([("in", 6, []), ("hidden", 8, ["in"]), ("out", 1, ["hidden"])], init_range=0.3, rng=s)
        train(net, {"in": X}, d, sweeps=15000, eps=0.03, alpha=0.9, stop=lambda n: solves_symmetry(n, X, d))
        assert solves_symmetry(net, X, d)


# ---- Figures 2-4: family trees ----

def test_family_trees_have_104_cases():
    P, R, D, keys = family_cases()
    assert len(keys) == 104                               # "the 104 possible triples"
    assert P.shape == (104, 24) and R.shape == (104, 12)
    colin_aunts = {b for a, r, b in family_facts() if a == "Colin" and r == "aunt"}
    assert colin_aunts == {"Jennifer", "Margaret"}        # "Colin has two aunts" (Fig. 3)


def test_trees_are_isomorphic():
    # Figure 2: the Italian tree is an exact copy of the English one. Renaming
    # each English person to their Italian twin must turn the English facts into
    # exactly the Italian facts.
    from tasks import ENGLISH, ITALIAN
    to_it = dict(zip(ENGLISH, ITALIAN))
    facts = set(family_facts())
    eng = {f for f in facts if f[0] in ENGLISH}
    ita = facts - eng
    assert {(to_it[a], r, to_it[b]) for a, r, b in eng} == ita


def test_family_net_learns_training_cases():
    net, acc_train, n_test, _ = family_run(seed=0)
    assert acc_train > 0.95


def test_papers_exact_family_recipe_stalls():
    # Reproduction finding: with small initial weights (+-0.3) and the paper's
    # eps/decay, the five-layer net never gets going (see EXPLAINED.md).
    _, acc_train, _, _ = family_run(seed=0, eps=0.01, decay=0.002, init=0.3)
    assert acc_train == 0.0


# ---- Figure 5: recurrent nets ----

def test_backprop_through_time():
    err, right = bptt_check()
    assert err < 1e-6 and right == 8
