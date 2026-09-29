"""A light tour of Kingma & Ba (2015). Only tiny NumPy problems; runs in about a second.

    python3 demo.py
"""

import numpy as np

from optimizers import AdaDelta, AdaGrad, AdaMax, Adam, RMSProp, SGDNesterov


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


# 1. The first Adam step is alpha * sign(g), whatever the size of g
line("1. The step size does not depend on the gradient's size (Section 2.1)")
for g in (0.001, 1.0, 1000.0):
    p = Adam(alpha=0.1).step([np.zeros(1)], [np.array([g])])
    print(f"gradient {g:>8}:  first step = {p[0][0]:+.4f}")
print("-> always -alpha = -0.1. Adam moves by about alpha, not by alpha * gradient.")

# 2. Bias correction
line("2. Why bias correction (Section 3)")
opt = Adam(alpha=0.001, beta1=0.9, beta2=0.999)
p = [np.zeros(1)]
print(" t   m_t      m_hat    v_t        v_hat")
for t in range(1, 6):
    p = opt.step(p, [np.array([2.0])])                     # constant gradient 2
    m, v = opt.m[0][0], opt.v[0][0]
    print(f"{t:2d}  {m:.4f}   {m / (1 - 0.9 ** t):.4f}   {v:.6f}   {v / (1 - 0.999 ** t):.4f}")
print("-> m_t and v_t start near 0 (they were initialized to 0). Dividing by (1 - beta^t)")
print("   gives the true values 2 and 4 immediately.")

g = [np.array([1.0])] * 3
for bc in (True, False):
    o, p = Adam(alpha=0.01, beta1=0.0, beta2=0.9999, bias_correction=bc), [np.zeros(1)]
    steps = []
    for gi in g:
        new = o.step(p, [gi]); steps.append(abs(new[0][0] - p[0][0])); p = new
    print(f"bias correction {str(bc):5}: first 3 step sizes = {np.round(steps, 4)}")
print("-> without it, beta2 = 0.9999 makes the first steps ~100x too big (Figure 4's instability).")

# 3. A badly scaled problem: every optimizer from the paper
line("3. f(x, y) = 0.5 (x^2 + 100 y^2) from (3, 2), 200 steps")
A = np.diag([1.0, 100.0])
for name, opt in [("SGDNesterov", SGDNesterov(0.005, 0.9)), ("AdaGrad", AdaGrad(0.5)),
                  ("RMSProp", RMSProp(0.05)), ("AdaDelta", AdaDelta()),
                  ("Adam", Adam(0.1)), ("AdaMax", AdaMax(0.1))]:
    p = [np.array([3.0, 2.0])]
    for _ in range(200):
        p = opt.step(p, [A @ p[0]])
    print(f"{name:12s} final loss {0.5 * p[0] @ A @ p[0]:.2e}   at {np.round(p[0], 4)}")
print("-> SGD needed a hand-picked lr (0.005; 0.03 diverges on the steep y-direction).")
print("   Adam/AdaMax/AdaGrad used large alphas safely: they rescale each coordinate.")
print("   RMSProp (fixed alpha, no momentum) keeps jittering at a distance ~alpha.")
print("   AdaDelta hardly moves in 200 steps: its first steps are only ~sqrt(eps) = 0.001.")

# 4. Noisy gradients: Adam's momentum averages the noise away
line("4. Noisy gradients: minimize 0.5 x^2 when each gradient has noise of std 5")
rng = np.random.default_rng(0)
for name, opt in [("RMSProp (no momentum)", RMSProp(0.01)), ("Adam", Adam(0.01)),
                  ("Adam, alpha/sqrt(t)", Adam(0.1, decay_sqrt_t=True))]:
    p, tail = [np.array([5.0])], []
    for t in range(3000):
        p = opt.step(p, [p[0] + rng.normal(0, 5, 1)])
        if t >= 2500:
            tail.append(abs(p[0][0]))
    print(f"{name:22s} mean |x| over the last 500 steps: {np.mean(tail):.3f}")
print("-> Adam's momentum (m_t) averages the noise: it ends ~5x closer than RMSProp.")
print("   alpha/sqrt(t) (the schedule of Section 4's theorem) is safe in theory but slow here:")
print("   the steps shrink before x has reached 0. Decay schedules need tuning too.")
