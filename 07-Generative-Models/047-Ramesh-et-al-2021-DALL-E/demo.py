"""A ~20-second tour of DALL-E (the paper's two stages and its scale tricks, at toy size).

  1. The sizes: why compress images into tokens first, and the 12-billion-parameter arithmetic
  2. The logit-Laplace likelihood (Appendix A.3): no probability wasted outside the valid pixel range
  3. Stage 1 in miniature: a dVAE with gumbel-softmax; the KL weight beta and codebook usage (Section 2.1, footnote 4)
  4. Stage 2 in miniature: one transformer over [caption tokens ; image tokens], with row/column/conv masks;
     can it draw caption combinations it never saw? (zero-shot composition) and image completion (Section 3.3)
  5. Reranking (Section 2.6): best of N samples under a scorer
  6. Training at scale: per-resblock gradient scaling vs fp16 underflow (2.4), PowerSGD compression (2.5)
"""

import math
import time

import torch

torch.set_num_threads(2)
from dalle import (DALLE, DVAE, PowerSGD, codebook_perplexity, cosine_schedule, dvae_loss,  # noqa: E402
                   fp16_underflow_fraction, logit_laplace_log_prob, phi, phi_inv, powersgd_compression_rate,
                   transformer_params)

torch.manual_seed(0)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. Sizes")
print(f"  a 256x256 RGB image is {256 * 256 * 3:,} numbers; the dVAE turns it into 32 x 32 = 1,024 tokens (a 192x "
      f"shorter context)")
print(f"  one stream = up to 256 text tokens + 1,024 image tokens = 1,280 positions")
print(f"  transformer d_model = 3968, 64 layers: ~12 d^2 per layer -> {transformer_params(3968, 64) / 1e9:.1f}B "
      f"parameters (the paper: 12 billion)")
print("  -> pixels as tokens would mean a 196,608-long context, with most capacity spent on high-frequency detail.")

# --------------------------------------------------------------------------------------------- 2
section("2. Logit-Laplace vs Laplace for a pixel near the edge of its range (mu = 0.95 in [0, 1])")
lap = torch.distributions.Laplace(torch.tensor(0.95), torch.tensor(0.1))
outside = 1 - (lap.cdf(torch.tensor(1.0)) - lap.cdf(torch.tensor(0.0))).item()
x = torch.linspace(1e-6, 1 - 1e-6, 200001)
inside = torch.trapezoid(logit_laplace_log_prob(x, torch.logit(torch.tensor(0.95)), torch.tensor(math.log(0.5))).exp(), x).item()
print(f"  Laplace(0.95, 0.1): {100 * outside:.1f}% of its probability lies outside [0, 1] (impossible pixel values)")
print(f"  logit-Laplace: integrates to {inside:.4f} over (0, 1), so nothing is wasted")
print("  -> the dVAE also maps pixels into (0.1, 0.9) first (Eq. 3) so the 1/(x(1-x)) factor stays finite.")

# --------------------------------------------------------------------------------------------- 3
section("3. A tiny dVAE: 16x16 images made of a 4x4 grid of 5 colours -> 4x4 tokens from K = 16 codes")
PAL = torch.tensor([[20, 20, 20], [230, 40, 40], [40, 200, 60], [50, 80, 230], [240, 220, 50]]).float()


def colour_grid(n):
    g = torch.randint(0, 5, (n, 4, 4))
    x = PAL[g].permute(0, 3, 1, 2).repeat_interleave(4, 2).repeat_interleave(4, 3)
    return (x + torch.randn_like(x) * 8).clamp(0, 255), g


print(f"  {'beta':>5} {'seed':>4} {'pixel error (0-255)':>20} {'codes used (perplexity)':>24} {'one colour per code':>20}")
for beta in (0.0, 6.6):
    for seed in (0, 1):
        torch.manual_seed(seed)
        d = DVAE(K=16, width=16, down=4)
        opt = torch.optim.Adam(d.parameters(), 3e-3)
        S = 250
        for s in range(S):
            x, _ = colour_grid(16)
            tau = cosine_schedule(s, S, 1.0, 1 / 16)                               # 1 -> 1/16, cosine
            b = cosine_schedule(s, 60, 0.0, beta)                                  # 0 -> beta, cosine
            loss = dvae_loss(d, phi(x), tau, b)
            opt.zero_grad(); loss.backward(); opt.step()
        x, g = colour_grid(256)
        with torch.no_grad():
            z = d.tokens(phi(x))
            rec = phi_inv(d.reconstruct(phi(x)))
        purity = sum(torch.bincount(g[z == k], minlength=5).max().item() for k in range(16) if (z == k).any()) / z.numel()
        print(f"  {beta:5.1f} {seed:4d} {(rec - x).abs().mean().item():20.1f} {codebook_perplexity(z, 16):24.2f} "
              f"{100 * purity:19.0f}%")
print("  -> with beta = 0 the codebook collapses (about 1 code used), as the paper's footnote 4 warns; beta = 6.6 keeps")
print("     several codes alive, closer to the 5 colours. Reconstruction quality still varies a lot run to run at this")
print("     tiny scale (the paper trains 3 million updates).")

# --------------------------------------------------------------------------------------------- 4
section("4. One transformer over [caption ; image tokens]: zero-shot composition (4x4 token images)")
H = W = 4
COLOURS = ["red", "green", "blue", "yellow"]
PLACES = ["top-left", "top-right", "bottom-left", "bottom-right"]


def scene(colours, places):
    """Caption = [colour word, place word]; image = a 2x2 block of that colour token in that quadrant."""
    img = torch.zeros(len(colours), H, W, dtype=torch.long)
    for b, (c, q) in enumerate(zip(colours.tolist(), places.tolist())):
        r, cc = divmod(q, 2)
        img[b, 2 * r:2 * r + 2, 2 * cc:2 * cc + 2] = c
    return torch.stack([colours, places + 5], 1), img.view(len(colours), -1)


held_out = [(1, 3), (2, 0), (3, 1), (4, 2)]                                       # never seen during training
seen = [(c, q) for c in range(1, 5) for q in range(4) if (c, q) not in held_out]


def batch(n):
    idx = torch.randint(0, len(seen), (n,))
    return scene(torch.tensor([seen[i][0] for i in idx]), torch.tensor([seen[i][1] for i in idx]))


models = []
print("  held-out captions: " + ", ".join(f"'{COLOURS[c - 1]} {PLACES[q]}'" for c, q in held_out))
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    m = DALLE(9, 5, 2, H, W, d=64, layers=4, heads=4)                              # masks: row, column, row, conv
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(300):
        t, im = batch(64)
        loss, _, _ = m.loss(t, torch.full((64,), 2), im)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    res = []
    for pairs in (seen, held_out):
        ok = 0
        for c, q in pairs:
            t, im = scene(torch.full((10,), c), torch.full((10,), q))
            ok += (m.generate(t, torch.full((10,), 2)) == im).all(1).sum().item()
        res.append(ok / (10 * len(pairs)))
    per = []
    for c, q in held_out:
        t, im = scene(torch.full((10,), c), torch.full((10,), q))
        per.append((m.generate(t, torch.full((10,), 2)) == im).all(1).float().mean().item())
    models.append(m)
    print(f"  seed {seed}: exact images for seen captions {100 * res[0]:.0f}%, for held-out captions {100 * res[1]:.0f}% "
          f"(per held-out caption: {' '.join(f'{100 * p:.0f}%' for p in per)})")
print("  -> seen captions are drawn almost perfectly; NEW combinations work only sometimes and differently per seed.")
print("     The paper reports the same flavour at 12B scale: composition 'performs inconsistently' (Section 3.3).")

t, im = scene(torch.tensor([2] * 10), torch.tensor([0] * 10))                       # 'green top-left' (held out)
full = [100 * (mm.generate(t, torch.full((10,), 2)) == im).all(1).float().mean().item() for mm in models]
comp = [100 * (mm.generate(t, torch.full((10,), 2), prefix=im[:, :4]) == im).all(1).float().mean().item()
        for mm in models]                                                          # give the first image row
print("  image completion, held-out 'green top-left': from scratch " + "/".join(f"{v:.0f}%" for v in full)
      + " exact; given the first image row " + "/".join(f"{v:.0f}%" for v in comp) + " (seeds 0/1/2)")
print("  -> a partial image pins down what the caption alone could not: the model continues the half-drawn object")
print("     (Section 3.3's image-to-image translation works the same way, with the top of the grid as the prompt).")

# --------------------------------------------------------------------------------------------- 5
section("5. Reranking: draw N samples for a held-out caption, keep the best under a scorer")
torch.manual_seed(0)
best_case = None                                                                   # a (model, caption) the model gets
for mi, mm in enumerate(models):                                                   # only partly right
    for c, q in held_out:
        t, im = scene(torch.full((64,), c), torch.full((64,), q))
        samples = mm.generate(t, torch.full((64,), 2))
        score = (samples == im).float().mean(1)                                    # stand-in for CLIP: token agreement
        exact = (score == 1).float().mean().item()
        if 0 < exact < 1 and (best_case is None or abs(exact - 0.5) < abs(best_case[0] - 0.5)):
            best_case = (exact, mi, c, q, score)
exact, mi, c, q, score = best_case
print(f"  model seed {mi}, held-out caption '{COLOURS[c - 1]} {PLACES[q]}': {100 * exact:.0f}% of single samples exact")
for N in (1, 4, 16, 64):
    best = torch.stack([score[i:i + N].max() for i in range(0, 64, N)]).mean().item()
    print(f"  best of N = {N:2d}: average agreement with the caption {100 * best:.0f}%")
print("  -> picking the best of many samples raises quality (Figure 9c; the paper uses N = 512 and a contrastive")
print("     model). Our scorer is an oracle that knows the true image; CLIP plays that role without knowing it.")

# --------------------------------------------------------------------------------------------- 6
section("6. Training at scale")
torch.manual_seed(0)
sizes = (1e1, 1e-2, 1e-5, 1e-8, 1e-11)                                          # early -> late resblocks
grads = [torch.randn(10000) * s for s in sizes]
lost = lambda g: 100 * (fp16_underflow_fraction(g) + torch.isinf(g.half()).float().mean().item())
glob = 2.0 ** 10                                                                   # largest scale with no overflow
print("  gradient size per resblock:       " + "  ".join(f"{s:7.0e}" for s in sizes))
print("  % lost (underflow to 0 or overflow to inf) in fp16:")
print("    one global loss scale 2^10:     " + "  ".join(f"{lost(g * glob):6.1f}%" for g in grads))
print("    per-resblock scales:            " + "  ".join(
    f"{lost(g * 2.0 ** (14 - math.floor(math.log2(g.abs().max().item())))):6.1f}%" for g in grads))
print("  -> one scale cannot fit gradients spanning 12 orders of magnitude: big enough to save the last block, it would")
print("     overflow the first. A separate scale per resblock (Figure 4) keeps every block representable.")
print("  PowerSGD compression rate 1 - 5r/(8 d_model): " + ", ".join(
    f"r={r}, d={d}: {100 * powersgd_compression_rate(r, d):.0f}%" for r, d in ((512, 1920), (640, 2688), (896, 3968))))
comp_ = PowerSGD((512, 512), r=8)
g = torch.randn(512, 512)
approx, sent = comp_.compress(g)
print(f"  a 512x512 gradient at rank 8 sends {sent:,} numbers instead of {g.numel():,}; error feedback carries the "
      f"remainder ({(g - approx).norm() / g.norm():.0%} of this random full-rank matrix) into the next step.")

print(f"\n(total {time.time() - T0:.1f} s)")
