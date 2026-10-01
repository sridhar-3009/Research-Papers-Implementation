"""A light tour of Jozefowicz, Zaremba & Sutskever (2015). No training; about a second.

    python3 demo.py
"""

import random

import torch

from architectures import CELLS, GraphCell, LSTM, SequenceModel, lstm_graph, mutate
from tasks import arithmetic, xml

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. The paper's tasks")
rng = random.Random(0)
for _ in range(2):
    s, start = arithmetic(rng)
    print(f"arithmetic: {s[:start]}  ->  {s[start:]}")
samples = [xml(rng)[0] for _ in range(1000)]
print(f"xml       : {next(x for x in samples if len(x) > 40)[:110]}")
print(f"            ({sum(x == '' for x in samples) / 10:.0f}% of XML samples are EMPTY: the paper's rule stops at once")
print("             whenever the first coin flip says 'close' and no tag is open)")

line("2. Why the forget-gate bias matters (Section 2.2)")
for bias in (0.0, 1.0, 2.0):
    f = torch.sigmoid(torch.tensor(bias)).item()
    print(f"forget bias {bias}: forget gate ~ {f:.3f} at init  ->  a memory keeps {f ** 20:.2e} of itself after 20 steps, "
          f"{f ** 100:.2e} after 100")
print("-> small random init makes f ~ 0.5: 'a vanishing gradient with a factor of 0.5 per timestep'.")
print("   A bias of 1-2 keeps memories (and gradients) alive from the start.")

line("3. The candidate architectures, measured (64 units, 40-symbol vocabulary)")
for name, make in CELLS.items():
    m = SequenceModel(make, 40, 64, 40)
    print(f"{name:7s} {sum(p.numel() for p in m.parameters()):7,} parameters")

line("4. Architecture search: a few random mutations of the LSTM graph (Section 3.2)")
g = lstm_graph()
print(f"LSTM graph: {len(g.nodes)} nodes, ops: {[n['op'] for n in g.nodes]}")
r = random.Random(3)
for k in range(4):
    child = mutate(g, r)
    diff = [(i, a["op"], b["op"]) for i, (a, b) in enumerate(zip(g.nodes, child.nodes)) if a["op"] != b["op"]]
    out, _ = GraphCell(child, 8)(torch.randn(2, 8), GraphCell(child, 8).init_state(2))
    print(f"mutant {k + 1}: {len(child.nodes)} nodes, first changed ops {diff[:3]}, runs -> output {tuple(out.shape)}")
print("-> every mutant is still a valid cell with the same state interface (h, c); the search keeps the")
print("   100 best by Eq. (1): min over tasks of (accuracy / GRU's accuracy).")
