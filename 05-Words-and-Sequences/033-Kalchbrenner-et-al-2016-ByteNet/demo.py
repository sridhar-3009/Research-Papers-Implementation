"""A light tour of ByteNet (Kalchbrenner et al. 2016). About 15 seconds.

    python3 demo.py
"""

import time

import torch
import torch.nn.functional as F

from bytenet import EOS, ByteNet, ByteNetLM, dilations, receptive_field, saliency


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Dilation: the receptive field doubles with every layer (Section 3.5)")
for n in (1, 2, 3, 4, 5, 10, 30):
    r = dilations(n)
    print(f"  {n:2d} blocks, kernel 3, rates {str(r[:5]) + ('...' if n > 5 else ''):22s}: sees "
          f"{receptive_field(3, r):4d} positions   (without dilation: {receptive_field(3, [1] * n):3d})")
print("-> 30 blocks (the paper's LM) see 373 characters by this formula; the paper says 315 (we could not")
print("   reproduce that number). Any two characters inside the window are linked by a path of <= 30 layers.")

line("2. A shallow stack can only learn what its receptive field covers: copy the character 10 steps back")
V, T, L = 8, 30, 10
for max_rate, name in ((16, "dilated 1,2,4"), (1, "undilated 1,1,1")):
    torch.manual_seed(0)
    m = ByteNetLM(V, d=32, n_blocks=3, k=3, block="relu", dropout=0, max_rate=max_rate)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(250):
        x = torch.randint(3, V, (16, T))
        for t in range(L, T):
            x[:, t] = x[:, t - L]
        loss = F.cross_entropy(m(x[:, :-1])[:, L:].reshape(-1, V), x[:, L + 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  3 blocks, {name:16s}: sees {m.receptive_field:2d} positions -> loss {loss.item():.3f} "
          f"(guessing: {torch.log(torch.tensor(5.0)).item():.3f})")
print("-> same depth, same parameters: dilation makes the 10-step dependency reachable, so it is learned.")

line("3. Translation with dynamic unfolding: a character substitution cipher (Section 3.2)")
torch.manual_seed(0)
perm = torch.cat([torch.zeros(3, dtype=torch.long), torch.randperm(V - 3) + 3])


def batch(N):
    S = int(torch.randint(4, 12, ()))
    src = torch.randint(3, V, (N, S))
    return src, torch.cat([perm[src], torch.full((N, 1), EOS)], 1)


m = ByteNet(V, V, d=32, n_blocks=3, k=3, a=1.0, b=1)
opt = torch.optim.Adam(m.parameters(), 3e-3)
for _ in range(250):
    s, t = batch(32)
    loss = -m.log_prob(s, t).mean()
    opt.zero_grad(); loss.backward(); opt.step()
s, t = batch(100)
pred = m.greedy(s, max_len=20)
right = sum(pred[i, :t.shape[1]].tolist() == t[i].tolist() for i in range(100))
print(f"  {right}/100 test strings translated exactly (including stopping with EOS)")
print(f"  source {s[0].tolist()} -> {pred[0, :t.shape[1]].tolist()} (target {t[0].tolist()})")

line("4. Which inputs did each output depend on? Gradient saliency (Figure 6)")
S_map, T_map = saliency(m, s[:1], t[:1])
print("  source position with the largest gradient, for each output character:", S_map.argmax(1).tolist())
print("-> output i depends most on source i: the alignment, read off from gradients (no attention needed).")

line("5. Linear time: one forward pass of the decoder for longer and longer sequences")
m = ByteNetLM(50, d=32, n_blocks=10, block="relu", dropout=0).eval()
for n in (500, 1000, 2000, 4000):
    x = torch.randint(0, 50, (1, n))
    t0 = time.time()
    with torch.no_grad():
        m(x)
    dt = time.time() - t0
    print(f"  length {n:5d}: {1000 * dt:6.1f} ms   attention would compute {n * n:>11,} source-target scores")
print("-> convolutions touch each position a constant number of times: time grows linearly with length,")
print("   and every position is computed in parallel during training (no recurrence).")
