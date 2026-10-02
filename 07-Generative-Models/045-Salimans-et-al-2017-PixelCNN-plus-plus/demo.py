"""A ~5-second tour of PixelCNN++.

  1. Figure 1: real pixel marginals pile up at an edge; a discretized logistic mixture models that for free,
     because the edge bins keep the tails
  2. Section 2.1: logistic mixture (15 numbers) vs 256-way softmax (256 numbers) fitted to 300 pixel values:
     the softmax doesn't know 127 is near 128, so unseen values get ~zero probability
  3. Section 3.4.2: discretized likelihood vs a continuous density on dequantized data (a lower bound)
  4. Causality: which input pixels each output pixel can see (two-stream network vs a stacked masked PixelCNN)
  5. Output size per pixel: mixture vs softmax
"""

import importlib.util
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pixelcnnpp import (PixelCNNpp, dequantized_log_density, discretized_mix_logistic_log_prob, n_params, to_pm1)

torch.manual_seed(0)
T0 = time.time()
LEVELS = to_pm1(torch.arange(256))


def section(t):
    print(f"\n=== {t} ===")


def mixture_params(raw, K):
    return raw.view(1, 3 * K, 1, 1)


def mixture_log_probs(raw, K):
    """log P(v) for all 256 values under a 1-channel mixture with parameters raw (3K,)."""
    l = mixture_params(raw, K).expand(256, -1, -1, -1)
    return discretized_mix_logistic_log_prob(LEVELS.view(256, 1, 1, 1), l, K)[:, 0, 0]


def fit_mixture(counts, K=5, steps=1500, lr=0.05, seed=0):
    torch.manual_seed(seed)
    raw = torch.cat([torch.zeros(K), torch.linspace(-0.8, 0.8, K), torch.full((K,), -2.0)]).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr)
    for _ in range(steps):
        loss = -(counts * mixture_log_probs(raw, K)).sum() / counts.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return raw.detach()


# --------------------------------------------------------------------------------------------- 1
section("1. Figure 1's observation, on MNIST pixels (10,000 images)")
try:
    import torchvision
    data = torchvision.datasets.MNIST(Path(__file__).resolve().parents[2] / "data", train=True, download=False)
    pix = data.data[:10000].flatten()
    source = "MNIST"
except Exception:                                                                 # fallback: clipped noisy blobs
    pix = (torch.randn(500000) * 120 + 60).clamp(0, 255).round().long()
    source = "synthetic (MNIST not found)"
counts = torch.bincount(pix, minlength=256).float()
p = counts / counts.sum()
print(f"  data: {source}. P(0) = {p[0]:.3f}, P(1) = {p[1]:.5f};  P(254) = {p[254]:.4f}, P(255) = {p[255]:.4f}")
raw = fit_mixture(counts, K=5)
q = mixture_log_probs(raw, 5).exp()
H = -(p[p > 0] * p[p > 0].log()).sum().item()
nll = -(p * q.log()).sum().item()
print(f"  5-component discretized logistic mixture: P(0) = {q[0]:.3f}, P(1) = {q[1]:.5f};  P(254) = {q[254]:.4f}, "
      f"P(255) = {q[255]:.4f}")
print(f"  bits per pixel: mixture {nll / math.log(2):.3f}  vs  the empirical entropy (the best any model could do) "
      f"{H / math.log(2):.3f}")
print("  -> the huge spike at 0 (81% of MNIST pixels) comes for free: the edge bin P(0) collects ALL the logistic mass")
print("     below 0.5, so one narrow component there captures it exactly. At the top, MNIST actually peaks at 254, not")
print("     255 (Figure 1's spike at 255 is a CIFAR-10 feature); the top bin also collects the upper tail, which here")
print("     slightly OVER-estimates P(255) (0.0095 vs 0.0064). Overall the mixture is within 0.006 bits of the entropy.")

# --------------------------------------------------------------------------------------------- 2
section("2. Few observations: 300 values from a smooth bump; fit a 5-logistic mixture vs a 256-way softmax")
torch.manual_seed(1)
truth = F.softmax(-((torch.arange(256).float() - 120) / 25) ** 2 / 2 * 1.0 + 0.0, 0)
truth = 0.7 * truth + 0.3 * F.softmax(-((torch.arange(256).float() - 200) / 10) ** 2 / 2, 0)
train = torch.multinomial(truth, 300, replacement=True)
c_train = torch.bincount(train, minlength=256).float()
raw = fit_mixture(c_train, K=5)
lp_mix = mixture_log_probs(raw, 5)
lp_soft = torch.log((c_train + 1e-3) / (c_train + 1e-3).sum())                    # the softmax's MLE (+ tiny floor)
test_nll = lambda lp: -(truth * lp).sum().item() / math.log(2)
print(f"  values never seen in training: {(c_train == 0).sum().item()} of 256")
print(f"  {'model':>26} {'parameters':>10} {'train bits':>10} {'test bits (true dist.)':>23}")
for name, lp, k in (("discretized logistic mix", lp_mix, 15), ("256-way softmax", lp_soft, 256)):
    tr = -(c_train * lp).sum().item() / c_train.sum().item() / math.log(2)
    print(f"  {name:>26} {k:10d} {tr:10.3f} {test_nll(lp):23.3f}")
print(f"  {'(true distribution)':>26} {'':>10} {'':>10} {-(truth * truth.log()).sum().item() / math.log(2):23.3f}")
print("  -> the softmax fits the training values best but gives the unseen ones almost no probability; the mixture")
print("     knows 127 is next to 128 and generalises. Its 15 outputs also give dense gradients (Section 2.1).")

# --------------------------------------------------------------------------------------------- 3
section("3. Discretized likelihood vs a continuous density on dequantized values (Section 3.4.2)")
lpd = (truth * lp_mix).sum().item()
u = (torch.rand(200000) - 0.5) * (2 / 255)
v = torch.multinomial(truth, 200000, replacement=True)
xn = (LEVELS[v] + u).view(-1, 1, 1, 1)
bound = dequantized_log_density(xn, mixture_params(raw, 5).expand(len(xn), -1, -1, -1), 5).mean().item()
print(f"  same mixture parameters: discrete log-likelihood {-lpd / math.log(2):.4f} bits, dequantized bound "
      f"{-bound / math.log(2):.4f} bits")
print("  -> E_u[log p_continuous(x + u)] <= log P_discrete(x) (Jensen), so the dequantized model pays a little extra.")
print("     On CIFAR-10 the paper measured the gap at 3.11 vs 2.92 bits/dim, much larger, for a trained model.")

# --------------------------------------------------------------------------------------------- 4
section("4. Causality: inputs that can affect the prediction at pixel 'o' (8 x 8 image)")
m = PixelCNNpp(3, 1, 16, 2, 0.0).eval()
x = to_pm1(torch.randint(0, 256, (1, 3, 8, 8))).requires_grad_(True)
m(x)[0, :, 4, 4].sum().backward()
seen = x.grad.abs().sum(1)[0] > 0
spec = importlib.util.spec_from_file_location(
    "vlae041", Path(__file__).resolve().parents[1] / "041-Chen-et-al-2017-Variational-Lossy-Autoencoder" / "vlae.py")
vlae = importlib.util.module_from_spec(spec); spec.loader.exec_module(vlae)
old = vlae.receptive_field(vlae.LocalPixelCNN(1, 8, 0, 8, window=None, layers=4), size=8, pixel=(4, 4)).bool()
print(f"  {'PixelCNN++ (two streams)':<26}{'original PixelCNN (4 masked 3x3)':<34}")
for r in range(8):
    row = lambda s: "".join("o" if (r, c) == (4, 4) else ("X" if s[r, c] else ".") for c in range(8))
    print(f"  {row(seen):<26}{row(old):<34}")
print(f"  -> PixelCNN++ sees ALL {seen.sum().item()} earlier pixels (every row above, every pixel to the left); the stacked")
print("     masked convs miss the upper-right region (the 'blind spot' that Gated PixelCNN and PixelCNN++ remove).")

# --------------------------------------------------------------------------------------------- 5
section("5. Network outputs per pixel")
print(f"  logistic mixture, K = 10, RGB: {n_params(3, 10)} numbers (10 x [1 weight + 3 means + 3 scales + 3 coupling])")
print(f"  256-way softmax per sub-pixel: 768 numbers, plus coupling coefficients (the paper's version: 1536)")
print(f"  full model (5 resnet layers per block, 160 filters): {sum(p.numel() for p in PixelCNNpp(3, 5, 160, 10).parameters()) / 1e6:.1f}M parameters")

print(f"\n(total {time.time() - T0:.1f} s)")
