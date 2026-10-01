"""A light tour of the Transformer (Vaswani et al. 2017). About 8 seconds.

    python3 demo.py
"""

import math

import torch

from transformer import (BOS, EOS, Transformer, attention, beam_search, label_smoothed_loss, noam_lr,
                         sinusoidal_positions)

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Scaled dot-product attention by hand (Eq. 1)")
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
out, w = attention(Q, K, V)
print("  scores Q K^T / sqrt(2):\n", (Q @ K.T / math.sqrt(2)).numpy().round(3))
print("  weights (softmax of each row):\n", w.numpy().round(3))
print("  outputs (weighted averages of the values):\n", out.numpy().round(2))
print("-> query 1 matches keys 1 and 3, so it mostly reads values 1 and 3.")

line("2. Why divide by sqrt(d_k)? (footnote 4)")
for d_k in (4, 64, 512):
    q, k = torch.randn(d_k), torch.randn(6, d_k)
    raw, scaled = torch.softmax(k @ q, 0), torch.softmax(k @ q / math.sqrt(d_k), 0)
    print(f"  d_k = {d_k:3d}: largest weight unscaled {raw.max():.3f}, scaled {scaled.max():.3f}")
print("-> dot products of random unit-variance vectors have variance d_k; unscaled, the softmax saturates")
print("   (one weight ~1, the rest ~0) and its gradients vanish. Dividing by sqrt(d_k) restores variance 1.")

line("3. Sinusoidal positions encode RELATIVE offsets")
pe = sinusoidal_positions(200, 64)
for k in (1, 5, 20):
    dots = [round((pe[p] @ pe[p + k]).item(), 3) for p in (0, 37, 150)]
    print(f"  PE(pos) . PE(pos + {k:2d}) at pos = 0, 37, 150: {dots}")
print("-> the similarity depends only on the offset k, never on where you are: PE(pos + k) is a fixed")
print("   rotation of PE(pos), so 'attend 5 positions back' is one linear rule for every position.")

line("4. Table 1: cost per layer and path length (n = sequence length, d = 512)")
for n in (50, 500, 5000):
    print(f"  n = {n:5d}: self-attention n^2 d = {n * n * 512:>14,}   recurrent n d^2 = {n * 512 * 512:>14,}"
          f"   path: attention 1, RNN {n}")
print("-> attention is cheaper than an RNN while n < d (typical sentences), connects any two positions in ONE")
print("   step, and has no sequential dependency, so all positions train in parallel.")

line("5. Train a tiny Transformer to reverse sequences (2 layers, d_model 64, 4 heads)")
V = 12
m = Transformer(V, V, N=2, d_model=64, d_ff=128, h=4, dropout=0.0)
opt = torch.optim.Adam(m.parameters(), 1e-3, betas=(0.9, 0.98), eps=1e-9)
for step in range(1, 401):
    src = torch.randint(3, V, (32, 8))
    tgt = torch.cat([torch.full((32, 1), BOS), src.flip(1), torch.full((32, 1), EOS)], 1)
    loss = label_smoothed_loss(m(src, tgt[:, :-1]), tgt[:, 1:])
    opt.zero_grad(); loss.backward(); opt.step()
m.eval()
test = torch.randint(3, V, (50, 8), generator=torch.Generator().manual_seed(3))
right = sum(beam_search(m, test[i:i + 1], beam=4) == test[i].flip(0).tolist() for i in range(50))
q = torch.full((V,), 0.1 / V); q[3] += 0.9
print(f"  {right}/50 test sequences reversed exactly (beam 4, length penalty 0.6)")
print(f"  final training loss {loss.item():.3f}; with label smoothing 0.1 the best possible is "
      f"{-(q * q.log()).sum().item():.3f} (the entropy of the smoothed target)")
src = test[:1]
m(src, torch.tensor([[BOS] + src[0].flip(0).tolist()]))
heads = m.dec[-1].cross_attn.weights[0]                        # (h, T_out, S)
best = max(range(heads.shape[0]), key=lambda i: sum(heads[i, t, 7 - t].item() for t in range(8)))
print(f"  decoder-encoder attention of head {best} in the last layer (rows = output step, columns = source):")
for t in range(8):
    print("   " + " ".join(" #" if heads[best, t, j] > 0.5 else " +" if heads[best, t, j] > 0.2 else " ." for j in range(8)))
print("-> one head has learned the anti-diagonal: output step t reads source position 7 - t.")

line("6. The learning-rate schedule (Eq. 3, d_model = 512, warmup 4000)")
for s in (100, 1000, 4000, 16000, 100000):
    print(f"  step {s:6d}: lr = {noam_lr(s):.6f}")
print("-> linear warm-up to a peak at step 4000, then decay like 1/sqrt(step).")
