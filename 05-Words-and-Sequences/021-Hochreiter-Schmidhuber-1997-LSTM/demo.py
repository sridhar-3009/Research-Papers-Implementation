"""A light tour of the 1997 LSTM. No training; a couple of seconds.

    python3 demo.py
"""

import numpy as np
import torch

from lstm import LSTM1997Torch, VanillaRNN, df, error_scaling, max_sigmoid_factor
from tasks import adding_problem, embedded_reber

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Section 3.1: an error sent q steps back is multiplied by f'(net) * w at every step")
for w in (1.0, 3.0, 3.99, 6.0):
    print(f"logistic unit, |w| = {w:4}: best-case factor per step 0.25|w| = {max_sigmoid_factor(w):.3f}  "
          f"-> after 100 steps x{error_scaling([max_sigmoid_factor(w)] * 100):.2e}")
print("-> below |w| = 4 the error MUST shrink exponentially. (Large weights don't save you: they push")
print("   units into saturation, where f' -> 0 even faster.) A linear unit with w = 1: x1.0 forever.")

line("2. How much of an error at the END reaches the FIRST input? (untrained nets, real numbers)")
print("lag | plain RNN | LSTM, default init (|s| at end) | LSTM, input-gate biases -3,-6 (|s| at end)")
for lag in (10, 50, 100, 200):
    xs = torch.randn(lag, 1, 2)
    x = xs.clone().requires_grad_()
    VanillaRNN(2, 8, 1, init=0.5)(x)[-1].sum().backward()
    row = [f"{x.grad[0].norm().item():.1e}"]
    for bias in (None, np.array([-3.0, -6.0])):
        x = xs.clone().requires_grad_()
        out, st = LSTM1997Torch(2, 2, 2, 1, truncate=True, init=0.5, in_gate_bias=bias)(x, return_states=True)
        out[-1].sum().backward()
        row.append(f"{x.grad[0].norm().item():.1e} ({st[-1].abs().max().item():4.1f})")
    print(f"{lag:3d} | {row[0]:>9} | {row[1]:>31} | {row[2]:>30}")
print("-> plain RNN: the signal from step 1 vanishes completely (0 in float32 after ~50 steps).")
print("   LSTM, default init: it ALSO fades, but for a different reason, 'internal state drift'")
print("   (Section 4): with no forget gate, s keeps adding inputs (|s| reaches ~44), h(s) saturates")
print("   and h'(s) -> 0 at the cell OUTPUT. The paper's remedy, negative input-gate biases, keeps s")
print("   small, and then the signal barely fades over 200 steps: the CEC itself passes it with factor 1.0.")


line("3. The constant error carrousel stores a value: write once, read 1,000 steps later")
net = LSTM1997Torch(1, 1, 1, 1, dtype=torch.float64)
with torch.no_grad():
    net.W_in.zero_(); net.W_in[0, 0] = 50.0; net.W_in[0, -1] = -25.0         # input gate opens only when x = 1
    net.W_c.zero_(); net.W_c[0, -1] = 1.0
xs = torch.zeros(1000, 1, 1, dtype=torch.float64)
xs[0, 0, 0] = 1.0
_, states = net(xs, return_states=True)
print(f"state after the write: {states[0].item():.6f}   after 999 more steps: {states[-1].item():.6f}")
print("-> s(t) = s(t-1) + y_in g(net_c): with the input gate closed nothing is added; nothing decays")
print("   (no forget gate in 1997: the cell only forgets if it is reset between sequences).")

line("4. The paper's tasks")
s, X, Y = embedded_reber(np.random.default_rng(1))
print("embedded Reber string:", "".join(s))
print(f"  second symbol '{s[1]}' must be remembered across {len(s) - 4} steps to predict '{s[-2]}'")
x, target = adding_problem(100, np.random.default_rng(0))
marked = np.where(x[:, 1] == 1)[0]
print(f"adding problem: {len(x)} pairs, marked positions {marked.tolist()}, values {np.round(x[marked, 0], 3).tolist()}")
print(f"  target 0.5 + (X1 + X2)/4 = {target:.4f}; the net sees the answer's ingredients >= {len(x) - marked.max()} steps before it must answer")
