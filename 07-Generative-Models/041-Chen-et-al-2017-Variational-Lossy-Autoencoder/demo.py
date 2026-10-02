"""A ~10-second tour of the Variational Lossy Autoencoder.

  1. Information preference (Section 2.2), on a toy with one GLOBAL bit and LOCAL copying:
     factorized decoder -> z carries a lot; window-1 decoder -> z carries ~ the global bit; full decoder -> ~ nothing
  2. Bits-back coding: naive two-part code vs bits-back code on the trained window-1 model (Eqs. 5-7)
  3. The receptive fields we control: exact A x B windows (Section 4.3) and the paper's 6-layer 3x3 stack
  4. AF prior = IAF posterior (Eqs. 12-14): the same number, computed both ways
"""

import math
import time

import torch
import torch.nn as nn

from vlae import AFPrior, LocalPixelCNN, code_lengths, receptive_field, toy_data, toy_true_log_prob, train_toy

torch.manual_seed(0)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. Information preference: how many nats does z carry, depending on what the decoder can see?")
X, Xt = toy_data(20000), toy_data(2000)
H = -toy_true_log_prob(Xt).mean().item()
print(f"  true entropy of the data H = {H:.3f} nats per sequence; the global bit alone is worth at most ln 2 = 0.693")
print(f"  {'decoder p(x|z)':>34} {'-bound':>7} {'gap to H':>9} {'KL = nats in z':>15}")
models = {}
for w, name in ((0, "factorized (sees only z)"), (1, "window 1 (sees x_(i-1))"), (15, "full autoregressive (all x_<i)")):
    m = train_toy(w, 400, 0, X)
    with torch.no_grad():
        nb, kl = -m.elbo(Xt, 4).mean().item(), m.kl(Xt).mean().item()
    models[w] = m
    print(f"  {name:>34} {nb:7.3f} {nb - H:9.3f} {kl:15.3f}")
print("  -> the factorized decoder must route everything correlated through z. The window-1 decoder models the local")
print("     copying itself, so z keeps (about) only the global bit. The full decoder can infer g from the history, so")
print("     z is pushed toward carrying nothing (its KL is still shrinking after this short run; at the optimum it is 0).")

# --------------------------------------------------------------------------------------------- 2
section("2. Bits-back coding on the trained window-1 model (nats per sequence)")
with torch.no_grad():
    rec, lp, lq = models[1].terms(Xt, 4)
c = code_lengths(rec, lp, lq)
print(f"  naive two-part code  E[-log p(z) - log p(x|z)]          = {c['naive']:8.3f}")
print(f"  refund: the sender's random choice of z is worth H(q)    = {c['refund = H(q)']:8.3f}")
print(f"  bits-back code       naive - refund = -ELBO              = {c['bits-back']:8.3f}")
print(f"  ideal (Shannon)      H(data)                             = {H:8.3f}")
print("  -> bits-back makes the VAE a real code whose length is -ELBO = -log p(x) + KL(q || true posterior): any posterior")
print("     mismatch is paid for, which is exactly why information that the decoder can model locally avoids z.")

# --------------------------------------------------------------------------------------------- 3
section("3. Receptive fields (X = pixels the prediction for 'o' can see), 12 x 12 image")


def show(rf, centre):
    for r in range(centre[0] - 6, centre[0] + 2):
        row = "".join("o" if (r, c) == centre else ("X" if rf[r, c] else ".") for c in range(12))
        print("    " + row)


for A, B in ((4, 2), (7, 4)):
    print(f"  window {A}x{B}: {A}-wide, {B}-tall block above + {(A - 1) // 2} pixel(s) to the left")
    show(receptive_field(LocalPixelCNN(1, 12, 0, 8, window=(A, B)), pixel=(8, 6)), (8, 6))
rf = receptive_field(LocalPixelCNN(1, 12, 0, 8, window=None, layers=6), pixel=(8, 6))
print(f"  the paper's MNIST decoder: 6 masked 3x3 layers ({rf.sum().item()} pixels, note PixelCNN's blind spot on the right)")
show(rf, (8, 6))
print("  -> anything visible inside the window (stroke texture, local smoothness) is modelled by the decoder;")
print("     only what needs a wider view (global shape) is left for z: a lossy code by design.")

# --------------------------------------------------------------------------------------------- 4
section("4. AF prior vs IAF posterior: two readings of the same bound")
torch.manual_seed(1)
prior = AFPrior(3, T=1, hidden=16)
for p in prior.parameters():
    nn.init.normal_(p, 0, 0.5)
q = torch.distributions.Normal(torch.tensor([0.3, -0.2, 0.5]), torch.tensor([0.5, 0.8, 0.3]))
z = q.sample((1,))
dec = lambda v: -(v ** 2).sum(-1)                                              # any log p(x|z)
with torch.no_grad():
    L_af = dec(z) + prior.log_prob(z) - q.log_prob(z).sum(-1)
    eps, logdet = prior.inverse(z)
    L_iaf = dec(z) - 0.5 * (3 * math.log(2 * math.pi) + (eps ** 2).sum(-1)) - (q.log_prob(z).sum(-1) - logdet)
    t = time.time(); prior.log_prob(torch.randn(256, 3)); t_density = time.time() - t
    t = time.time(); prior.sample(256); t_sample = time.time() - t
print(f"  bound read as 'AF prior on z':                     {L_af.item():.6f}")
print(f"  bound read as 'IAF posterior on eps = f^-1(z)':     {L_iaf.item():.6f}")
print(f"  AF prior density (training): one parallel pass {1000 * t_density:.2f} ms; sampling (generation): sequential "
      f"{1000 * t_sample:.2f} ms")
print("  -> identical numbers, identical training cost; but the AF version's generator p(x|f(eps)) is deeper,")
print("     consistent with Table 1: AF prior 79.30 nats vs the equivalent IAF posterior 79.88.")

print(f"\n(total {time.time() - T0:.1f} s)")
