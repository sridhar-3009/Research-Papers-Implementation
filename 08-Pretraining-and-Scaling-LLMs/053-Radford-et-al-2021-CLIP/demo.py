"""CLIP in a few seconds: a tiny image encoder + bag-of-words text encoder trained contrastively on coloured MNIST
with made-up captions, then classified ZERO-SHOT from prompts (no labels ever used for training)."""

import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import CBOW, CLIP, SmallConvNet, bag_of_words_loss, clip_loss, count_params, zero_shot_predict, zero_shot_weights

torch.manual_seed(0); random.seed(0); torch.set_num_threads(1)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------------------------------- data
COLOURS = ["red", "green", "blue"]
COL = torch.tensor([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.2, 0.3, 1.0]])
DIGITS = "zero one two three four five six seven eight nine".split()
CAPTIONS = ["a photo of the number {d} in {c}", "a {c} handwritten {d}", "the digit {d} written in {c} ink",
            "{d} , {c}", "a picture of a {c} {d}"]
HELD_OUT = {(7, 0), (3, 1), (5, 2)}                                     # (digit, colour) pairs never captioned


def load(n_train=12000, n_test=2000):
    try:
        import torchvision
        root = Path(__file__).resolve().parents[2] / "data"
        tr = torchvision.datasets.MNIST(root, train=True, download=False)
        te = torchvision.datasets.MNIST(root, train=False, download=False)
        f = lambda d, n: (F.avg_pool2d(d.data[:n, None].float() / 255, 2), d.targets[:n])  # 14x14 for speed
        return *f(tr, n_train), *f(te, n_test), "MNIST"
    except Exception:
        protos = torch.rand(10, 1, 14, 14).round()
        mk = lambda n: (lambda y: ((protos[y] + 0.3 * torch.randn(n, 1, 14, 14)).clamp(0, 1), y))(torch.randint(0, 10, (n,)))
        return *mk(n_train), *mk(n_test), "synthetic digits (MNIST not found)"


xtr, ytr, xte, yte, source = load()
ctr = torch.randint(0, 3, (len(ytr),))
keep = torch.tensor([(int(d), int(c)) not in HELD_OUT for d, c in zip(ytr, ctr)])
xtr, ytr, ctr = xtr[keep], ytr[keep], ctr[keep]
cte = torch.randint(0, 3, (len(yte),))
paint = lambda x, c: x * COL[c][:, :, None, None]
words = sorted({w for t in CAPTIONS + ["a photo of the digit {d}", "a photo of something {c}"] for w in t.split()} |
               set(DIGITS) | set(COLOURS) | {"number", "colour", "photo", "of", "the"})
vocab = {w: i + 1 for i, w in enumerate(w for w in words if "{" not in w)}
L = 9


def encode(texts):
    ids = [[vocab[w] for w in t.split()][:L] for t in texts]
    return (torch.tensor([s + [0] * (L - len(s)) for s in ids]),)


def captions(y, c):
    return [random.choice(CAPTIONS).format(d=DIGITS[int(a)], c=COLOURS[int(b)]) for a, b in zip(y, c)]


section(f"0. Data: {len(ytr)} {source} images painted red/green/blue, each with a template caption")
print("  e.g.", captions(ytr[:3], ctr[:3]))
print(f"  held out from training: {sorted((DIGITS[d], COLOURS[c]) for d, c in HELD_OUT)} (never seen together)")

# ---------------------------------------------------------------------------------------------------- loss
section("1. The symmetric contrastive loss (Figure 3)")
N = 128
print(f"  random embeddings -> loss ~ log N = log {N} = {math.log(N):.2f};  "
      f"actual: {clip_loss(torch.randn(N, N) * 0.01).item():.2f}")
print(f"  logit scale starts at exp(t) = 1/0.07 = {1 / 0.07:.2f} and is clipped at 100 (CLIP ends there)")

# ---------------------------------------------------------------------------------------------------- training
section("2. Contrastive pre-training on (image, caption) pairs")


def train(objective, steps=800, B=128):
    torch.manual_seed(0); random.seed(0)
    m = CLIP(SmallConvNet(3, 16), CBOW(len(vocab) + 1, 64), d_embed=64)
    head = nn.Linear(m.image_encoder.out_dim, len(vocab) + 1)
    opt = torch.optim.Adam(list(m.parameters()) + list(head.parameters()), 3e-3)
    for s in range(steps):
        i = torch.randint(0, len(ytr), (B,))
        x, (ids,) = paint(xtr[i], ctr[i]), encode(captions(ytr[i], ctr[i]))
        loss = clip_loss(m(x, ids)) if objective == "contrastive" else bag_of_words_loss(m.image_encoder(x), head, ids, len(vocab) + 1)
        opt.zero_grad(); loss.backward(); opt.step()
    return m, head, loss.item()


t = time.time()
model, _, last = train("contrastive")
print(f"  {count_params(model):,} parameters, 800 steps x batch 128, {time.time() - t:.1f}s; final loss {last:.3f} "
      f"(chance {math.log(128):.2f}); logit scale now {model.logit_scale().item():.1f}")

# ---------------------------------------------------------------------------------------------------- zero-shot
section("3. Zero-shot classification from prompts (Section 3.1)")
xt = paint(xte, cte)
model.eval()
acc = lambda W, y: (zero_shot_predict(model, xt, W) == y).float().mean().item()
W_name = zero_shot_weights(model, encode, DIGITS, ["{}"])
W_one = zero_shot_weights(model, encode, DIGITS, ["a photo of the digit {}"])
W_ens = zero_shot_weights(model, encode, DIGITS, [t.replace("{d}", "{}").replace("{c}", c) for t in CAPTIONS for c in COLOURS])
print(f"  digits, class name only ('seven')           : {acc(W_name, yte):.1%}")
print(f"  digits, one prompt ('a photo of the digit seven'): {acc(W_one, yte):.1%}")
print(f"  digits, ensemble of 15 prompts (embeddings averaged): {acc(W_ens, yte):.1%}")
W_col = zero_shot_weights(model, encode, COLOURS, ["a photo of something {}"])
print(f"  colours, one prompt                         : {acc(W_col, cte):.1%}")
print("  -> no classifier was trained: the text encoder WROTE the classifier's weights from the class names")
print("  -> the ensemble beats the single prompt, as in the paper; the bare name does about as well here because our")
print("     bag-of-words text encoder just averages words, so filler words only dilute the digit word")

held = torch.tensor([(int(d), int(c)) in HELD_OUT for d, c in zip(yte, cte)])
pred = zero_shot_predict(model, xt, W_ens)
print(f"  held-out combinations (e.g. a red seven, never captioned): {(pred[held] == yte[held]).float().mean():.1%} "
      f"on {int(held.sum())} images vs {(pred[~held] == yte[~held]).float().mean():.1%} on the rest")
print("  -> clearly above chance (10%) but far below the seen pairs: composing an unseen colour+digit is only partial")

# ---------------------------------------------------------------------------------------------------- objective
section("4. Contrastive vs predictive (bag-of-words) objective, same steps (Figure 2)")
bow_model, head, _ = train("bag-of-words")
with torch.no_grad():
    logp = F.log_softmax(head(bow_model.image_encoder(xt)), -1)
    digit_ids = torch.tensor([vocab[d] for d in DIGITS])
    bow_acc = (logp[:, digit_ids].argmax(-1) == yte).float().mean().item()
print(f"  bag-of-words predictor, zero-shot by 'which digit word is most likely': {bow_acc:.1%}")
print(f"  contrastive (ensemble, above)                                     : {acc(W_ens, yte):.1%}")
print("  -> same steps, same image encoder: the contrastive model is far ahead here, in the paper's direction. Caveat:")
print("     the predictor wastes capacity on colour/filler words and is read out differently, so this is not the paper's")
print("     controlled 4x-efficiency measurement (see experiments.py e1)")

# ---------------------------------------------------------------------------------------------------- robustness
section("5. The paper's robustness numbers (Figure 13): ResNet-101 vs zero-shot CLIP, same ImageNet accuracy")
rows = [("ImageNet", 76.2, 76.2), ("ImageNetV2", 64.3, 70.1), ("ImageNet-R", 37.7, 88.9), ("ObjectNet", 32.6, 72.3),
        ("ImageNet Sketch", 25.2, 60.2), ("ImageNet-A", 2.7, 77.1)]
for name, r, c in rows:
    print(f"  {name:16s} ResNet-101 {r:5.1f}   CLIP {c:5.1f}   ({c - r:+5.1f})")
shift_r = sum(r for _, r, _ in rows[1:]) / 5; shift_c = sum(c for _, _, c in rows[1:]) / 5
print(f"  mean over the 5 shifts: {shift_r:.1f} vs {shift_c:.1f}; drop from ImageNet {76.2 - shift_r:.1f} vs {76.2 - shift_c:.1f} points")
print("  (the paper's 'up to 75%' is about effective robustness on 7 shifts, a different summary of the same idea)")

print(f"\nTotal time: {time.time() - T0:.1f}s")
