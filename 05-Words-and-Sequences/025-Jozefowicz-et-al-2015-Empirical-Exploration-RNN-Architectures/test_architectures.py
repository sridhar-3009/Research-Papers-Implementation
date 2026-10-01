"""Tests for Jozefowicz, Zaremba & Sutskever (2015). About a second.

Run with:  python3 -m pytest -q
"""

import random

import pytest
import torch

from architectures import (CELLS, GRU, LSTM, MUT, GraphCell, Search, SequenceModel, fitness, gru_graph, lstm_graph,
                           mutate)
from tasks import VOCAB, arithmetic, encode, memorization, xml, xml_is_well_formed

torch.manual_seed(0)


# ---- the hand-written cells ----

def test_every_cell_runs():
    x = torch.randn(3, 8)
    for name, make in CELLS.items():
        cell = make(8)
        h, state = cell(x, cell.init_state(3))
        assert h.shape == (3, 8), name


def test_lstm_without_forget_gate_never_forgets():
    cell = LSTM(4, no_forget=True)
    h, (h1, c1) = cell(torch.randn(2, 4), cell.init_state(2))
    _, (h2, c2) = cell(torch.zeros(2, 4), (h1, c1))
    a_i, a_j, _, _ = cell.lin(torch.cat([torch.zeros(2, 4), h1], -1)).chunk(4, -1)
    assert torch.allclose(c2, c1 + torch.tanh(a_i) * torch.sigmoid(a_j))        # c' = c + i * j  (f = 1)


def test_forget_bias_of_one_opens_the_forget_gate():
    plain, biased = LSTM(16), LSTM(16, forget_bias=1.0)
    with torch.no_grad():
        for c in (plain, biased):
            c.lin.weight.mul_(0.01)
            c.lin.bias[2 * 16:3 * 16].sub_(c.lin.bias[2 * 16:3 * 16] - c.forget_bias)   # bias = exactly 0 or 1
    f = lambda c: torch.sigmoid(c.lin(torch.zeros(1, 32))[0, 32:48]).mean().item()
    assert f(plain) == pytest.approx(0.5, abs=0.01)                                   # 'effectively sets the forget gate to 0.5'
    assert f(biased) == pytest.approx(0.731, abs=0.01)                                # sigmoid(1)
    assert 0.5 ** 20 < 1e-5 < 0.731 ** 20                                            # 20 steps: vanishing vs surviving


def test_forget_bias_survives_sequence_model_init():
    m = SequenceModel(CELLS["LSTM-b"], 10, 8, 10)
    assert m.cells[0].lin.bias[16:24].mean().item() > 0.5


def test_gru_update_rule():
    cell = GRU(5)
    h0 = torch.randn(2, 5); x = torch.randn(2, 5)
    h1, _ = cell(x, (h0,))
    r, z = torch.sigmoid(cell.gates(torch.cat([x, h0], -1))).chunk(2, -1)
    ht = torch.tanh(cell.W_xh(x) + cell.W_hh(r * h0))
    assert torch.allclose(h1, z * h0 + (1 - z) * ht)


def test_mut1_update_formula():
    # MUT1: z = sigm(W_xz x + b_z) (no h!), r = sigm(W_xr x + W_hr h + b_r),
    #       h' = tanh(W_hh (r * h) + tanh(x) + b_h) * z + h * (1 - z)
    cell = MUT(6, which=1)
    x = torch.randn(2, 6)
    for h in (torch.zeros(2, 6), torch.randn(2, 6)):
        z = torch.sigmoid(cell.W_xz(x))
        r = torch.sigmoid(cell.W_xr(x) + cell.W_hr(h))
        expected = torch.tanh(cell.W_hh(r * h) + torch.tanh(x)) * z + h * (1 - z)
        assert torch.allclose(cell(x, (h,))[0], expected, atol=1e-6)


# ---- graphs ----

def test_lstm_graph_equals_the_hand_written_lstm():
    n = 4
    hand, gcell = LSTM(n), GraphCell(lstm_graph(), n)
    # copy weights: the graph has W_x and W_h per gate (pre-activations in order i, j, f, o)
    lins = [gcell.lins[k] for k in sorted(gcell.lins, key=int)]
    W, b = hand.lin.weight, hand.lin.bias
    with torch.no_grad():
        for g in range(4):
            lx, lh = lins[2 * g], lins[2 * g + 1]
            lx.weight.copy_(W[g * n:(g + 1) * n, :n]); lx.bias.copy_(b[g * n:(g + 1) * n])
            lh.weight.copy_(W[g * n:(g + 1) * n, n:]); lh.bias.zero_()
    x, st = torch.randn(3, n), (torch.randn(3, n), torch.randn(3, n))
    h1, s1 = hand(x, st)
    h2, s2 = gcell(x, st)
    assert torch.allclose(h1, h2, atol=1e-6) and torch.allclose(s1[1], s2[1], atol=1e-6)


def test_gru_graph_equals_the_hand_written_gru():
    n = 4
    hand, gcell = GRU(n), GraphCell(gru_graph(), n)
    lins = [gcell.lins[k] for k in sorted(gcell.lins, key=int)]       # r_x, r_h, z_x, z_h, h_x, h_h(r*h)
    Wg, bg = hand.gates.weight, hand.gates.bias
    with torch.no_grad():
        for g in range(2):
            lins[2 * g].weight.copy_(Wg[g * n:(g + 1) * n, :n]); lins[2 * g].bias.copy_(bg[g * n:(g + 1) * n])
            lins[2 * g + 1].weight.copy_(Wg[g * n:(g + 1) * n, n:]); lins[2 * g + 1].bias.zero_()
        lins[4].weight.copy_(hand.W_xh.weight); lins[4].bias.copy_(hand.W_xh.bias)
        lins[5].weight.copy_(hand.W_hh.weight); lins[5].bias.zero_()
    x, h = torch.randn(3, n), torch.randn(3, n)
    assert torch.allclose(hand(x, (h,))[0], gcell(x, (h,))[0], atol=1e-6)


def test_mutations_always_give_valid_runnable_graphs():
    rng = random.Random(0)
    g = lstm_graph()
    changed = 0
    for _ in range(200):
        child = mutate(g, rng)
        assert child.is_valid()
        cell = GraphCell(child, 4)
        h, st = cell(torch.randn(2, 4), cell.init_state(2))
        assert h.shape == (2, 4) and len(st) == len(child.state_inputs())
        changed += [n["op"] for n in child.nodes] != [n["op"] for n in g.nodes]
        g = child if rng.random() < 0.3 else g
    assert changed > 100                                                     # mutations really change things


# ---- the search ----

def test_fitness_is_the_worst_task_relative_to_gru():
    gru = {"arith": 0.8, "xml": 0.5, "ptb": 0.1}
    assert fitness({"arith": 0.88, "xml": 0.45, "ptb": 0.11}, gru) == pytest.approx(0.9)
    assert fitness(gru, gru) == 1.0


def test_search_adds_only_architectures_that_pass_every_stage():
    rng = random.Random(1)
    gru = {"t1": 1.0, "t2": 1.0}
    good = lambda graph, task, k: 1.2                                        # a fake evaluator
    s = Search(good, lambda g: True, ["t1", "t2"], gru, rng=rng, top=5)
    for _ in range(30):
        s.step()
    assert len(s.pool) == 5 and s.score(s.pool[0]) == pytest.approx(1.2)
    bad = Search(lambda g, t, k: 0.5, lambda g: True, ["t1", "t2"], gru, rng=rng)
    results = {bad.step() for _ in range(20)}
    assert "added" not in results and "failed t1" in results                # below 90% of the GRU on task 1


# ---- tasks ----

def test_task_strings():
    rng = random.Random(2)
    s, start = memorization(rng)
    assert s[:5] == s[6:11] and s[5] == "=" and start == 6
    for _ in range(50):
        s, start = arithmetic(rng)
        q = "".join(ch for ch in s[:start - 1] if ch not in "abcdefghijklmnopqrstuvwxyz")
        assert s[start - 1] == "=" and str(eval(q.replace("--", "+"))) == s[start:-1]
        x, _ = xml(rng)
        assert xml_is_well_formed(x)
        assert all(c in VOCAB for c in s + x) and len(encode(s)) == len(s)
