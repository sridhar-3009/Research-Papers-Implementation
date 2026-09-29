"""Watch the paper's nets run tick by tick.  Run:  python3 demo.py   (# = fires)"""

import circuits
from mp_neuron import show
from tpe import AndNot, Or, Var, compile_tpe


def section(title: str) -> None:
    """Print a heading between demos."""
    print(f"\n=== {title} ===")


# Brief cold: the cold receptor (2) fires only at tick 0.
# Expect: neuron 3 ("feel heat") fires at tick 3, and 4 ("feel cold") never fires.
section("Heat/cold illusion (Fig 1e): brief cold -> feels HOT")
h = circuits.heat_cold().run({"2": [1]}, steps=7)
print(show(h, ["2", "a", "b", "4", "3"]))
print("2 = cold in, 4 = feel cold, 3 = feel heat")

# Long cold: the cold receptor fires at every tick.
# Expect: 4 fires from tick 2 on, and 3 never fires (no warmth first).
section("Heat/cold illusion (Fig 1e): long cold -> feels only COLD")
h = circuits.heat_cold().run({"2": [1] * 7}, steps=7)
print(show(h, ["2", "a", "b", "4", "3"]))

# Input 1 fires at ticks 0, 3 and 6; input 2 fires only at tick 3 (the pairing).
# Expect: 1 alone at tick 0 does nothing; after the pairing, L stays on,
# and 1 alone at tick 6 fires 3 (seen at tick 7).
section("Learning as a loop (Fig 1i): 1 alone works only after pairing with 2")
h = circuits.learned_association().run(
    {"1": [1, 0, 0, 1, 0, 0, 1, 0], "2": [0, 0, 0, 1, 0, 0, 0, 0]}, steps=8)
print(show(h, ["1", "2", "L", "3"]))

# P fires once at tick 2. Expect: M turns on at tick 3 and stays on forever.
section("Memory latch (Theorem X)")
print(show(circuits.latch().run({"P": [0, 0, 1]}, steps=8)))

# Start one spike in c0. Expect: it goes round c0 -> c1 -> c2 -> c0 ... forever.
section("Clock: ring of 3 (Theorem X)")
print(show(circuits.clock(3).run({}, steps=10, initial={"c0": 1})))

# XOR = (p AND NOT q) OR (q AND NOT p). No single neuron can do it; the compiler
# builds a 2-layer net. Read the answer `lag` ticks after the inputs.
section("XOR compiled from a formula (Theorem II) - needs 2 layers")
xor = Or(AndNot(Var("p"), Var("q")), AndNot(Var("q"), Var("p")))
net, out, lag = compile_tpe(xor)
for a in (0, 1):
    for b in (0, 1):
        print(f"p={a} q={b} -> {net.run({'p': [a], 'q': [b]}, steps=lag + 1)[out][lag]}")
