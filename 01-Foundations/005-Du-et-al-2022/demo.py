"""A guided tour of the survey, one small experiment per section.

Run:  python3 demo.py
(For the paper's Table 1 / Figure 4 experiment, run experiment_iris.py.)
"""

import numpy as np

from experiment_iris import load_iris, split_and_scale
from mlp import MLP
from optimizers import train
from perceptron import accuracy, lms_train, perceptron_train, pocket_train

# The 4 input patterns of a 2-input logic gate.
X_LOGIC = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
section("Section 3.2 - Perceptron learning: converges only if linearly separable")
for name, y in [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])]:
    w, th, ok, ep, errs = perceptron_train(X_LOGIC, np.array(y), max_epochs=100, rng=0)
    status = f"converged in {ep} epochs" if ok else f"did NOT converge in {ep} epochs"
    print(f"{name:4s}: {status}; mistakes in last 5 epochs = {errs[-5:]}")
print("XOR can't be split by one straight line, so the perceptron keeps cycling forever.")

section("Section 3.2 - Pocket algorithm: best possible single line for XOR")
w, th, acc = pocket_train(X_LOGIC, np.array([0, 1, 1, 0]), steps=500, rng=0)
print(f"pocket accuracy on XOR = {acc:.2f}  (3 of 4 is the best any single line can do)")

section("Section 3.3 - LMS / Adaline: minimizes squared error even without a perfect line")
y_pm = np.array([-1, 1, 1, 1.0])                        # OR, coded -1/+1
w, th, hist = lms_train(X_LOGIC, y_pm, lr=0.05, epochs=300, rng=0)
A = np.c_[X_LOGIC, np.ones(4)]
w_ls = np.linalg.lstsq(A, y_pm, rcond=None)[0]         # the exact least-squares answer
print(f"LMS weights       = {np.r_[w, th].round(3)}")
print(f"least-squares     = {w_ls.round(3)}")
print(f"OR classified by sign(net): {np.sign(X_LOGIC @ w + th).astype(int)} (target {y_pm.astype(int)})")

# ---------------------------------------------------------------------------
section("Section 4 - An MLP with one hidden layer solves XOR")
Y_xor = np.array([[0], [1], [1], [0]], dtype=float)
for method in ["gd", "gdm", "lm"]:
    net = MLP([2, 3, 1], hidden="logistic", output="logistic", rng=3)
    h = train(net, X_LOGIC, Y_xor, method=method, epochs=5000, goal=1e-3, lr=2.0)
    out = net.predict(X_LOGIC).ravel()
    print(f"{method:4s}: {len(h) - 1:5d} epochs, outputs {out.round(2)}  (target [0 1 1 0])")
print("gd = plain BP, gdm = BP + momentum (Eq. 32), lm = Levenberg-Marquardt (Section 8.1.2)")

# ---------------------------------------------------------------------------
section("Section 5 - Generalization: weight decay and early stopping vs overfitting")
rng = np.random.default_rng(1)
x_tr = np.sort(rng.uniform(-1, 1, 20))[:, None]
y_tr = np.sin(np.pi * x_tr) + rng.normal(0, 0.25, x_tr.shape)     # 20 noisy samples
x_va = rng.uniform(-1, 1, (20, 1))
y_va = np.sin(np.pi * x_va) + rng.normal(0, 0.25, x_va.shape)     # noisy validation set
x_te = np.linspace(-1, 1, 200)[:, None]
y_te = np.sin(np.pi * x_te)                                       # the true function
test_mse = lambda net: float(np.mean((net.predict(x_te) - y_te) ** 2))

nets = {}
for label, kw in [("no regularization", {}),
                  ("weight decay 1e-3", {"decay": 1e-3}),
                  ("early stopping", {"val": (x_va, y_va)})]:
    net = MLP([1, 30, 1], hidden="tanh", output="linear", rng=0)   # far too big for 20 points
    train(net, x_tr, y_tr, method="lm", epochs=300, goal=0, **kw)
    nets[label] = net
    print(f"{label:18s}: train MSE {np.mean((net.predict(x_tr) - y_tr) ** 2):.4f}   "
          f"test MSE vs true sine {test_mse(net):.4f}   weight norm {np.linalg.norm(net.get_params()):.1f}")
print("No regularization fits the noise (tiny train error, bigger test error).")

# ---------------------------------------------------------------------------
section("Section 10.1 - Fault tolerance: what if a hidden node dies?")
X, labels = load_iris()
Xtr, Ytr, ltr, Xte, lte = split_and_scale(X, labels, np.random.default_rng(0))


def acc_with_single_faults(net):
    """Average test accuracy over every possible single dead hidden node."""
    accs = []
    for j in range(net.sizes[1]):
        net.masks[0][:] = 1
        net.masks[0][j] = 0                          # open-node fault: output stuck at 0
        accs.append(np.mean(net.predict(Xte).argmax(1) == lte))
    net.masks[0][:] = 1
    return float(np.mean(accs)), float(np.min(accs))


for label, kw in [("normal online BP", {}), ("BP + fault injection", {"fault_rate": 0.2})]:
    scores = []
    for seed in range(5):
        net = MLP([4, 10, 3], hidden="logistic", output="linear", rng=seed)
        train(net, Xtr, Ytr, method="sgd", epochs=300, goal=0, lr=0.02, rng=seed, **kw)
        healthy = np.mean(net.predict(Xte).argmax(1) == lte)
        mean_f, worst_f = acc_with_single_faults(net)
        scores.append((healthy, mean_f, worst_f))
    h, m, wst = np.mean(scores, axis=0)
    print(f"{label:21s}: healthy {h:.1%}   one node dead: mean {m:.1%}, worst {wst:.1%}   (5 runs)")
print("Training with random node faults spreads the work across nodes, so losing one hurts less.")
