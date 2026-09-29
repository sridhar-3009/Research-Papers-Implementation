"""Tests: each one checks a claim or figure from the paper.

Run with:  python3 -m pytest -q
pytest runs every function whose name starts with test_. `assert` fails the test
if its condition is False.
"""

import random
from itertools import product   # product([0,1], repeat=2) -> (0,0) (0,1) (1,0) (1,1)

import pytest

import circuits
from mp_neuron import Network, Neuron
from tpe import And, AndNot, Delay, Or, Var, compile_tpe, evaluate


def after_one_tick(net: Network, a: int, b: int, out: str = "3") -> int:
    """Present a, b at tick 0 and read the output at tick 1.

    A gate answers one tick later, because of the synaptic delay.
    """
    return net.run({"1": [a], "2": [b]}, steps=2)[out][1]


# ---- Figure 1a–d: the four building blocks ----

def test_delay():
    # The output is the input shifted one tick to the right.
    h = circuits.delay().run({"1": [1, 0, 1, 1, 0]}, steps=6)
    assert h["2"] == [0, 1, 0, 1, 1, 0]


# parametrize runs this test three times, once per (net, expected truth table) pair.
@pytest.mark.parametrize("net, table", [
    (circuits.or_gate(),      {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}),
    (circuits.and_gate(),     {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}),
    (circuits.and_not_gate(), {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 0}),
])
def test_gates(net, table):
    # Try all four input combinations and compare with the truth table.
    for (a, b), want in table.items():
        assert after_one_tick(net, a, b) == want


def test_inhibition_is_a_veto_not_a_negative_weight():
    # Assumption 4: one inhibitor blocks firing no matter how much excitation arrives.
    # Even 100 endings of excitation can't beat a single inhibitor.
    n = Neuron(threshold=2, excite={"x": 100}, inhibit={"y"})
    assert n.fires({"x": 1, "y": 1}) == 0   # inhibitor on  -> silent
    assert n.fires({"x": 1, "y": 0}) == 1   # inhibitor off -> fires


# ---- Figure 1e: the heat/cold illusion ----

def heat_cold_run(cold, heat=(), steps=8):
    """Run the heat/cold net. Input "1" = heat receptor, "2" = cold receptor."""
    return circuits.heat_cold().run({"1": list(heat), "2": list(cold)}, steps=steps)


def test_brief_cold_feels_hot():
    # The illusion: cold for one tick is felt as HEAT.
    h = heat_cold_run(cold=[1])
    assert h["3"] == [0, 0, 0, 1, 0, 0, 0, 0]   # heat felt at tick 3
    assert not any(h["4"])                      # cold never felt


def test_long_cold_feels_only_cold():
    # Held cold is felt as cold, with no flash of heat first.
    h = heat_cold_run(cold=[1] * 8)
    assert h["4"] == [0, 0, 1, 1, 1, 1, 1, 1]   # cold from tick 2
    assert not any(h["3"])                      # "no preliminary warmth"


def test_real_heat_is_felt_after_one_tick():
    # Real heat goes straight through the OR neuron: felt after one tick.
    h = heat_cold_run(cold=[], heat=[1])
    assert h["3"][1] == 1


def test_two_tick_cold_gives_cold_then_heat():
    # Not in the paper: cold for exactly 2 ticks is felt as cold, then as heat.
    h = heat_cold_run(cold=[1, 1])
    assert h["4"] == [0, 0, 1, 0, 0, 0, 0, 0]
    assert h["3"] == [0, 0, 0, 0, 1, 0, 0, 0]


def test_heat_cold_matches_the_papers_formulas_for_every_input():
    # Brute force: try EVERY heat/cold pattern over 6 ticks (2^12 = 4096 patterns)
    # and check the net against the formulas printed on page 9.
    T = 6
    for bits in product([0, 1], repeat=2 * T):
        heat, cold = list(bits[:T]), list(bits[T:])   # first 6 bits heat, last 6 cold
        h = circuits.heat_cold().run({"1": heat, "2": cold}, steps=T)
        # N1(t), N2(t) as in the paper; before tick 0 counts as silent.
        N1 = lambda t: heat[t] if t >= 0 else 0
        N2 = lambda t: cold[t] if t >= 0 else 0
        for t in range(T):
            # N3(t) = N1(t-1) OR [N2(t-3) AND NOT N2(t-2)]
            assert h["3"][t] == int(N1(t - 1) or (N2(t - 3) and not N2(t - 2)))
            # N4(t) = N2(t-2) AND N2(t-1)
            assert h["4"][t] == int(N2(t - 2) and N2(t - 1))


# ---- Figure 1h: two spikes in a row ----

def test_two_in_a_row():
    # Input fires at ticks 0, 2, 3. Only 2-3 is "two in a row", so the output
    # fires once, at tick 4 (one tick after the second spike).
    h = circuits.two_in_a_row().run({"1": [1, 0, 1, 1, 0, 0]}, steps=6)
    assert h["2"] == [0, 0, 0, 0, 1, 0]


# ---- Figure 1i: learning replaced by a loop ----

def test_learned_association():
    #        tick: 0  1  2  3  4  5  6  7  8
    one = {"1": [1, 0, 0, 1, 0, 0, 1, 0, 0],   # 1 fires alone at 0 and 6, with 2 at 3
           "2": [0, 0, 0, 1, 0, 0, 0, 0, 0]}   # 2 fires once, at tick 3 (the pairing)
    h = circuits.learned_association().run(one, steps=9)
    assert h["3"][1] == 0   # before pairing: 1 alone does nothing
    assert h["3"][4] == 1   # pairing at tick 3: fired by 2
    assert h["L"][5:] == [1, 1, 1, 1]   # memory stays on forever
    assert h["3"][7] == 1   # after pairing: 1 alone now fires 3


# ---- Theorem X: loops give memory and rhythm ----

def test_latch_remembers_forever():
    # P fires once at tick 2; M turns on at tick 3 and never turns off.
    h = circuits.latch().run({"P": [0, 0, 1, 0, 0, 0, 0]}, steps=7)
    assert h["M"] == [0, 0, 0, 1, 1, 1, 1]


def test_latch_with_reset():
    # Set at tick 0 -> on from tick 1. Reset at tick 3 -> off from tick 4.
    h = circuits.latch(with_reset=True).run(
        {"P": [1, 0, 0, 0, 0, 0], "R": [0, 0, 0, 1, 0, 0]}, steps=6)
    assert h["M"] == [0, 1, 1, 1, 0, 0]


def test_always_so_far_switches_off_for_good():
    # P fails at tick 3, so A goes off at tick 4 and stays off,
    # even though P comes back at ticks 4 and 5.
    h = circuits.always_so_far().run({"P": [1, 1, 1, 0, 1, 1]}, steps=6, initial={"A": 1})
    assert h["A"] == [1, 1, 1, 1, 0, 0]


def test_clock_fires_every_n_ticks():
    # A ring of 3 started at c0: each neuron fires every 3rd tick, one after another.
    h = circuits.clock(3).run({}, steps=10, initial={"c0": 1})
    assert h["c0"] == [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert h["c1"] == [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]


# ---- Theorem II: every TPE can be built as a net ----

def random_tpe(rng: random.Random, depth: int) -> object:
    """Make a random formula over inputs p, q, r, at most `depth` levels deep."""
    # Stop at a plain input when out of depth, or sometimes earlier (25%) for variety.
    if depth == 0 or rng.random() < 0.25:
        return Var(rng.choice("pqr"))
    kind = rng.choice([Delay, Or, And, AndNot])
    if kind is Delay:
        return Delay(random_tpe(rng, depth - 1))
    return kind(random_tpe(rng, depth - 1), random_tpe(rng, depth - 1))


def test_compiled_net_equals_formula():
    # Theorem II checked on 300 random formulas x 20 random input runs each:
    # the compiled net must give the formula's answer, `lag` ticks late.
    rng = random.Random(1943)   # fixed seed, so the test is the same on every run
    T = 12
    for _ in range(300):
        e = random_tpe(rng, depth=4)
        net, out, lag = compile_tpe(e)
        for _ in range(20):
            inputs = {v: [rng.randint(0, 1) for _ in range(T)] for v in net.inputs}
            h = net.run(inputs, steps=T)
            for t in range(lag, T):
                # The net at tick t equals the formula at tick t - lag.
                assert h[out][t] == evaluate(e, inputs, t - lag), (e, inputs, t)


def test_compiler_handles_repeated_input():
    # "p AND p" must behave like p. (Regression test: an early version merged the
    # two connections from p into one ending, so this gate never fired.)
    net, out, lag = compile_tpe(And(Var("p"), Var("p")))
    assert net.run({"p": [1, 0]}, steps=3)[out][lag] == 1


# ---- Theorem III and XOR ----

def test_silence_in_means_silence_out():
    # Theorem III: nothing a net computes can be true when every input is off.
    # Run 200 random compiled nets with no input at all: the output never fires.
    rng = random.Random(0)
    for _ in range(200):
        net, out, _ = compile_tpe(random_tpe(rng, depth=4))
        assert not any(net.run({}, steps=12)[out])


# XOR = (p AND NOT q) OR (q AND NOT p). It's false when both are off, so by
# Theorem III a net CAN build it.
XOR = Or(AndNot(Var("p"), Var("q")), AndNot(Var("q"), Var("p")))


def test_xor_needs_two_layers():
    # The compiled XOR net has two layers (lag 2) and gives the right truth table.
    net, out, lag = compile_tpe(XOR)
    assert lag == 2
    for a, b in product([0, 1], repeat=2):
        assert net.run({"p": [a], "q": [b]}, steps=lag + 1)[out][lag] == a ^ b   # ^ is XOR


def test_no_single_neuron_computes_xor():
    # Search every single neuron: endings 0-4 per input, any inhibition, thresholds 1-8.
    # None can do XOR, which is why XOR needs a second layer.
    for ea, eb, th in product(range(5), range(5), range(1, 9)):
        for inh in [set(), {"a"}, {"b"}, {"a", "b"}]:
            n = Neuron(th, {"a": ea, "b": eb}, inh)
            if all(n.fires({"a": a, "b": b}) == a ^ b for a, b in product([0, 1], repeat=2)):
                pytest.fail(f"found XOR neuron: {n}")
