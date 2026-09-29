"""A light tour of Ioffe & Szegedy (2015). Only tiny NumPy arrays; runs in about a second.

    python3 demo.py
"""

import numpy as np

from batchnorm import bn_backward, bn_forward, fuse_for_inference, normalize_outside_gradient, population_stats

rng = np.random.default_rng(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Algorithm 1: every feature -> mean 0, variance 1, then gamma, beta")
x = np.column_stack([rng.normal(50, 10, 8), rng.normal(-3, 0.01, 8)])      # two features, very different scales
y, cache = bn_forward(x, gamma=np.ones(2), beta=np.zeros(2))
print("input  mean", np.round(x.mean(0), 3), " std", np.round(x.std(0), 3))
print("output mean", np.round(y.mean(0), 3), " std", np.round(y.std(0), 3))
y2, _ = bn_forward(x, gamma=np.array([2.0, 0.5]), beta=np.array([1.0, -1.0]))
print("with gamma = [2, 0.5], beta = [1, -1]: mean", np.round(y2.mean(0), 3), " std", np.round(y2.std(0), 3))
print("   (feature 2's std is ~0.91, not 1: its variance 5e-5 is close to eps = 1e-5, which is added to it)")
print("-> the network LEARNS the mean (beta) and scale (gamma) it wants; the old layers can't shift them.")

line("2. Section 2: normalizing outside the gradient step")
u, target = rng.normal(size=50), rng.normal(1.0, 1.0, size=50)
for inside in (False, True):
    b, loss = normalize_outside_gradient(u, target, steps=200, inside=inside)
    print(f"gradient {'includes' if inside else 'ignores '} E[x]:  b after 50/100/200 steps = "
          f"{b[49]:7.2f} {b[99]:7.2f} {b[199]:7.2f}   loss {loss[0]:.3f} -> {loss[-1]:.3f}")
print("-> ignoring the normalization in the gradient, b grows forever and the loss never moves.")
print("   (b cancels inside x - E[x], so no b can change the loss; the correct gradient knows this: it is 0.)")
print("   That's why BN is part of the network and is backpropagated through.")

line("3. The backward pass removes the gradient's mean and its x_hat component")
dy = rng.normal(size=(8, 2))
_, cache = bn_forward(x, np.ones(2), np.zeros(2), eps=0)
dx, _, _ = bn_backward(dy, cache)
print("sum of dl/dy over the batch:", np.round(dy.sum(0), 3))
print("sum of dl/dx over the batch:", np.round(dx.sum(0), 10))
print("-> the gradient cannot shift the batch mean: BN would undo that shift anyway.")

line("4. Section 3.3: scaling the weights by 10 changes nothing")
u, W = rng.normal(size=(32, 5)), rng.normal(size=(5, 3))
for a in (1.0, 10.0):
    out, _ = bn_forward(u @ (a * W), np.ones(3), np.zeros(3))
    print(f"W x {a:4}:  first output row = {np.round(out[0], 4)}")
print("-> big weights don't blow up the activations, and their gradient shrinks as 1/a (stabilizing).")

line("5. Algorithm 2: inference uses population statistics")
batches = [rng.normal(3.0, 2.0, size=(10, 1)) for _ in range(2000)]
mean, var = population_stats(batches)
print(f"E[x] = {mean[0]:.3f} (true 3), unbiased Var[x] = {var[0]:.3f} (true 4); "
      f"plain average of batch variances = {np.mean([b.var() for b in batches]):.3f} (biased by 9/10)")
a, c = fuse_for_inference(np.array([1.5]), np.array([0.2]), mean, var)
print(f"at test time BN is just y = {a[0]:.4f} x + {c[0]:.4f}: deterministic, and it can be folded into W")
