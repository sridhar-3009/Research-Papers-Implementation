"""A light tour of Sutskever's thesis (2013), Chapters 3-4. A few seconds.

    python3 demo.py
"""

import numpy as np
import torch

from hf_rnn import RNN, HFStructural
from rtrbm import RTRBM
from tasks import addition, bouncing_balls, memorization, temporal_order

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Chapter 3: the RTRBM, an RNN whose every step is an RBM")
m = RTRBM(n_visible=5, n_hidden=3, init_std=0.5, seed=0)
m.W = np.random.default_rng(1).normal(0, 0.8, (3, 5))
vs = (np.random.default_rng(2).random((4, 5)) < 0.5).astype(float)
rs = m.infer(vs)
print("frames v_t:      ", vs.astype(int).tolist())
print("inferred r_t:    ", np.round(rs[1:], 3).tolist())
print(f"exact log P(v_1..v_4) = {m.log_prob_exact(vs):.4f} (sum of 4 tiny-RBM log-probabilities, by enumeration)")
print("-> inference is ONE deterministic forward pass (no sampling, no smoothing): that's what makes")
print("   the RTRBM trainable, unlike the TRBM whose posterior is intractable.")

line("2. Chapter 4: the long-memory ('pathological') problems")
X, Y, M, _ = addition(30, 1, np.random.default_rng(0))
marks = torch.nonzero(X[:, 0, 1]).ravel().tolist()
print(f"addition, T=30: {len(X)} steps, marked steps {marks}, values {[round(X[t, 0, 0].item(), 3) for t in marks]}, "
      f"target (u_I + u_J)/2 = {Y[-1, 0, 0].item():.3f}")
X, Y, M, _ = temporal_order(30, 1, np.random.default_rng(1))
print("temporal order: symbols", X[:, 0].argmax(-1).add(1).tolist(), "-> class", Y[-1, 0].item())
X, Y, M, _ = memorization(10, 1, np.random.default_rng(2))
print("5-bit memorization: inputs ", X[:, 0].argmax(-1).tolist())
print("                    targets", Y[:, 0].tolist(), "(2 = blank, 3 = trigger in the inputs)")

line("3. Structural damping penalizes changes that would scramble the hidden states")
rnn = RNN(4, 20, 4)
hf = HFStructural(rnn, "ce", lam=1.0, mu=1.0)
data = memorization(10, 16, np.random.default_rng(3))[:3]
P = sum(p.numel() for p in rnn.parameters())
names = [n for n, _ in rnn.named_parameters()]
sizes = [p.numel() for p in rnn.parameters()]
for target in ("W_hh", "W_oh"):
    v = torch.zeros(P)
    i = names.index(target)
    start = sum(sizes[:i])
    v[start:start + sizes[i]] = torch.randn(sizes[i])
    v = v / v.norm()
    with_s = v @ hf.curvature_product(data, v, structural=True)
    without = v @ hf.curvature_product(data, v, structural=False)
    print(f"unit-length change of {target}: curvature without structural damping {without:.4f}, with {with_s:.4f}")
print("-> a change of the hidden-to-output weights W_oh doesn't move the hidden states: no extra")
print("   penalty. A change of W_hh, applied again at every step, can move them a lot: structural")
print("   damping makes it 'expensive' so HF takes small steps there, without having to raise lambda")
print("   (which would slow down ALL directions).")

line("4. A few HF steps on short memorization: with vs without structural damping")
for structural in (True, False):
    torch.manual_seed(1)
    rnn = RNN(4, 20, 4)
    hf = HFStructural(rnn, "ce", lam=0.1 if structural else 0.3, mu=1 / 30 if structural else 0.0, max_cg=50)
    data = memorization(5, 32, np.random.default_rng(4))[:3]
    losses = [hf.objective(data).item()]
    for _ in range(6):
        losses.append(hf.step(data, data, structural)[1])
    print(f"{'structural' if structural else 'plain     '} HF: loss " + " -> ".join(f"{l:.3f}" for l in losses))
print("(tiny problem: both work; the thesis found structural damping essential only for T > 50 memorization)")

line("5. Chapter 3's video data: bouncing balls")
video = bouncing_balls(3, res=12, n_balls=2, rng=np.random.default_rng(5))
for t, frame in enumerate(video):
    print(f"frame {t}:")
    for row in frame.reshape(12, 12)[::2]:
        print("   " + "".join(" .:#"[min(3, int(p * 4))] for p in row))
