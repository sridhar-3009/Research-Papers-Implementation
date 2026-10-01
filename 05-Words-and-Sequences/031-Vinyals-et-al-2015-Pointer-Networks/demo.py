"""A light tour of Pointer Networks (Vinyals, Fortunato & Jaitly 2015). About 15-20 seconds.

    python3 demo.py
"""

import numpy as np
import torch

from ptrnet import PtrNet, beam_search, targets
from tasks import A1, A2, A3, convex_hull, held_karp, hull_metrics, sample_points, tour_length

torch.manual_seed(0)
rng = np.random.default_rng(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def train(m, make_batch, steps, lr=1e-2):
    opt = torch.optim.Adam(m.parameters(), lr)
    for _ in range(steps):
        P, C = make_batch()
        loss = -m.log_prob(P, C).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
    return m.eval()


line("1. The output dictionary = the input: train on lists of 5-10 numbers, test on longer ones (sorting)")


def sort_batch(lo=5, hi=10):
    n = int(torch.randint(lo, hi + 1, ()))
    X = torch.rand(64, n, 1)
    return X, targets((X[..., 0].argsort(1) + 1).tolist())


ptr = train(PtrNet(64, in_dim=1), sort_batch, 600)
for n in (5, 10, 15, 20):
    exact = pos = 0
    for _ in range(100):
        x = torch.rand(n, 1)
        p, t = beam_search(ptr, x), (x[:, 0].argsort() + 1).tolist()
        exact += p == t
        pos += sum(a == b for a, b in zip(p, t)) / n
    tag = "  <- never seen this length" if n > 10 else ""
    print(f"  n = {n:2d}: exactly sorted {exact:3d}%, positions right {pos:5.1f}%{tag}")
print("-> one model handles any n (its softmax has n + 1 entries). A seq2seq softmax has a FIXED number of")
print("   classes, so it cannot even name position 15 if it was built for n = 10. Accuracy fades with length:")
print("   this small, briefly trained model learned a rough rule, not a perfect algorithm.")

line("2. Convex hull, n = 5 (Table 1 setting, much smaller): Ptr-Net vs the seq2seq baseline")


def hull_batch(n=5):
    Ps = [sample_points(n, rng) for _ in range(64)]
    return torch.tensor(Ps, dtype=torch.float), targets([convex_hull(P) for P in Ps])


test = [sample_points(5, rng) for _ in range(200)]
models = {}
for mode in ("ptr", "seq2seq"):
    torch.manual_seed(0)
    models[mode] = m = train(PtrNet(128, mode=mode, n_max=5), hull_batch, 1200)
    r = hull_metrics(test, [beam_search(m, torch.tensor(P, dtype=torch.float)) for P in test])
    area = r["area"] if r["area"] == "FAIL" else f"{r['area']:.1f}%"
    print(f"  {mode:8s}: exact hull {r['accuracy']:5.1f}%, area coverage {area}")
for P in test[:3]:
    print(f"  true hull {convex_hull(P)}   Ptr-Net {beam_search(models['ptr'], torch.tensor(P, dtype=torch.float))}")
print("-> 'FAIL' means more than 1% of outputs are not simple polygons (the paper's rule). The paper trained on 1M")
print("   examples; here ~77k in a few seconds, so the numbers are far below Table 1. Training is fragile at this")
print("   scale: the loss sits on a plateau and then suddenly drops; with lr 3e-3 instead of 1e-2 the Ptr-Net was")
print("   still stuck (0%) after the same budget.")

line("3. TSP, n = 8: exact (Held-Karp) vs the approximate algorithms used to label big-n training data")
lens = {k: [] for k in ("optimal", "A1 greedy edge", "A2 NN + 2-opt", "A3 Christofides + 2-opt")}
for _ in range(20):
    P = sample_points(8, rng)
    lens["optimal"].append(tour_length(P, held_karp(P)))
    lens["A1 greedy edge"].append(tour_length(P, A1(P)))
    lens["A2 NN + 2-opt"].append(tour_length(P, A2(P)))
    lens["A3 Christofides + 2-opt"].append(tour_length(P, A3(P)))
for k, v in lens.items():
    print(f"  {k:24s}: mean tour length {np.mean(v):.3f}")
print("-> like Table 2's n = 10 row (2.87 / 3.07 / 2.87 / 2.87), the cheap heuristic is a few % worse and 2-opt")
print("   closes most of the gap. The Ptr-Net is trained to imitate whichever of these labelled its data.")
