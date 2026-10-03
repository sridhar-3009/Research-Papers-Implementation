"""CLIP's experiments (Radford et al. 2021) at small scale.

Data: CIFAR-10 / CIFAR-100 images with NOISY synthetic captions (random template, a synonym of the class name in 30%
of captions, 0-3 random distractor words, sometimes a colour adjective computed from the image), so the text is
web-like rather than a clean label. Images under torchvision corruptions (blur, noise, greyscale, edge/sketch) give
small 'distribution shifts'.

  E1  Figure 2: zero-shot test accuracy vs training images seen for (a) a Transformer captioner predicting the exact
      caption, (b) a bag-of-words predictor, (c) a bag-of-words CONTRASTIVE model, (d) the full contrastive CLIP.
  E2  Section 3.1.4: zero-shot accuracy with the bare class name, 'a photo of a {}.', and the ensemble of all
      templates averaged in embedding space (paper: +1.3% and +3.5%, ~5% together on ImageNet).
  E3  Figure 6 / Section 3.2: linear probe on frozen CLIP features with k = 1, 2, 4, 8, 16 labelled examples per
      class vs zero-shot; how many shots does zero-shot equal (paper: ~4-shot linear probe on average)?
  E4  Section 3.3 / Figure 13: effective robustness. Train a supervised CNN of the same size on the labels; evaluate
      both on 4 corruption 'shifts'; compare the drop from clean accuracy for supervised, CLIP linear probe and
      zero-shot CLIP (paper: adapting to ImageNet with a linear probe REDUCES robustness).
  E5  Ablations: fixed temperature (0.07, 1.0) vs learned; nonlinear vs linear projection; batch size 64 vs 512 at
      equal images seen (contrastive learning wants many negatives).

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import (CBOW, CLIP, SmallConvNet, TextTransformer, bag_of_words_loss, clip_loss, linear_probe,
                  zero_shot_predict, zero_shot_weights)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a picture of the {}.", "a low resolution photo of a {}.",
             "a close-up photo of a {}.", "a bright photo of a {}.", "a cropped photo of the {}.", "art of the {}.",
             "a photo of my {}.", "a good photo of a {}.", "a photo of the small {}.", "a photo of the large {}."]
SYNONYMS = {"airplane": "plane", "automobile": "car", "bird": "birdie", "cat": "kitty", "deer": "stag",
            "dog": "puppy", "frog": "toad", "horse": "pony", "ship": "boat", "truck": "lorry"}
DISTRACT = "today my new trip lovely look at this wow nice from our holiday shot the weekend".split()
COLOURS = ["red", "green", "blue", "dark", "bright"]


# ---------------------------------------------------------------------------------------------------- data
class Tok:
    def __init__(self, texts):
        words = sorted({w for t in texts for w in self.split(t)})
        self.ids = {w: i + 3 for i, w in enumerate(words)}                     # 0 pad, 1 <s>, 2 <eos>

    @staticmethod
    def split(t):
        return t.lower().replace(".", " .").replace(",", " ,").split()

    def __len__(self):
        return len(self.ids) + 3

    def __call__(self, texts, L=20):
        seqs = [[1] + [self.ids.get(w, 0) for w in self.split(t)][:L - 2] + [2] for t in texts]
        eos = torch.tensor([len(s) - 1 for s in seqs])
        return torch.tensor([s + [0] * (L - len(s)) for s in seqs]), eos


def colour_word(img):
    m = img.mean((1, 2))
    if img.mean() < 0.3: return "dark"
    if img.mean() > 0.6: return "bright"
    return COLOURS[int(m.argmax())]


def caption(img, name, rng):
    t = rng.choice(TEMPLATES).format(SYNONYMS.get(name, name) if rng.random() < 0.3 else name)
    if rng.random() < 0.4:
        t = t.replace("a photo", f"a {colour_word(img)} photo")
    words = t.split()
    for _ in range(rng.randint(0, 3)):
        words.insert(rng.randint(0, len(words)), rng.choice(DISTRACT))
    return " ".join(words)


def setup(a):
    import torchvision
    ds = getattr(torchvision.datasets, a.dataset)
    tr, te = ds(DATA, train=True, download=True), ds(DATA, train=False, download=True)
    f = lambda d, n: (torch.tensor(d.data[:n]).permute(0, 3, 1, 2).float() / 255, torch.tensor(d.targets[:n]))
    xtr, ytr = f(tr, a.n_train); xte, yte = f(te, a.n_test)
    names = [c.replace("_", " ") for c in tr.classes]
    rng = random.Random(0)
    caps = [caption(x, names[int(y)], rng) for x, y in zip(xtr, ytr)]
    tok = Tok(caps + [t.format(n) for t in TEMPLATES for n in names] + names)
    return xtr, ytr, caps, xte, yte, names, tok


def shifts(x):
    import torchvision.transforms.functional as TF
    grey = x.mean(1, keepdim=True).expand_as(x)
    edge = (x - TF.gaussian_blur(x, 5)).abs().mul(4).clamp(0, 1)
    return {"clean": x, "blur": TF.gaussian_blur(x, 5), "noise": (x + 0.15 * torch.randn_like(x)).clamp(0, 1),
            "greyscale": grey, "sketch (edges)": 1 - edge.mean(1, keepdim=True).expand_as(x)}


# ---------------------------------------------------------------------------------------------------- models
class Captioner(nn.Module):
    """Figure 2's 'transformer language model': predict the caption's tokens given the image feature as a prefix."""

    def __init__(self, img, V, d):
        super().__init__()
        self.img, self.proj = img, nn.Linear(img.out_dim, d)
        self.text = TextTransformer(V, d, layers=2, heads=4, ctx=21)
        self.out = nn.Linear(d, V)

    def logits(self, x, ids):
        h = self.text.tok(ids) + self.text.pos[:ids.shape[1]]
        h = torch.cat([self.proj(self.img(x))[:, None], h], 1)
        T = h.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        return self.out(self.text.ln(self.text.blocks(h, mask=mask)))[:, :-1]

    def nll(self, x, ids):
        lg = self.logits(x, ids)
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids.reshape(-1), ignore_index=0, reduction="none") \
            .view(ids.shape).sum(1)


def build(kind, V, a, nonlinear=False, temp=None):
    torch.manual_seed(0)
    img = SmallConvNet(3, a.width, attn_pool=True)
    if kind == "captioner":
        return Captioner(img, V, a.d).to(DEV)
    if kind == "bow-predict":
        m = nn.Module(); m.img, m.head = img, nn.Linear(img.out_dim, V)
        return m.to(DEV)
    txt = CBOW(V, a.d) if kind == "bow-contrastive" else TextTransformer(V, a.d, layers=a.layers, heads=4, ctx=20)
    m = CLIP(img, txt, d_embed=a.d)
    if nonlinear:
        m.W_i = nn.Sequential(nn.Linear(img.out_dim, a.d), nn.ReLU(), nn.Linear(a.d, a.d, bias=False))
        m.W_t = nn.Sequential(nn.Linear(txt.out_dim, a.d), nn.ReLU(), nn.Linear(a.d, a.d, bias=False))
    if temp is not None:
        m.t.data.fill_(math.log(1 / temp)); m.t.requires_grad_(False)
    return m.to(DEV)


def loss_of(kind, m, x, ids, eos):
    if kind == "captioner":
        return m.nll(x, ids).mean() / 10
    if kind == "bow-predict":
        return bag_of_words_loss(m.img(x), m.head, ids, m.head.out_features)
    return clip_loss(m(x, ids, eos))


@torch.no_grad()
def zero_shot(kind, m, xte, yte, names, tok, templates):
    m.eval()
    if kind == "captioner":                                                     # pick the most likely caption
        scores = []
        for n in names:
            ids, _ = tok([templates[0].format(n)])
            scores.append(torch.cat([-m.nll(xte[i:i + 500].to(DEV), ids.expand(len(xte[i:i + 500]), -1).to(DEV))
                                     for i in range(0, len(xte), 500)]))
        acc = (torch.stack(scores, 1).argmax(1).cpu() == yte).float().mean().item()
    elif kind == "bow-predict":
        lp = torch.cat([F.log_softmax(m.head(m.img(xte[i:i + 500].to(DEV))), -1) for i in range(0, len(xte), 500)])
        acc = (lp[:, [tok.ids[n.split()[-1]] for n in names]].argmax(1).cpu() == yte).float().mean().item()
    else:
        enc = lambda texts: tuple(t.to(DEV) for t in tok(texts))
        W = zero_shot_weights(m, enc, names, templates)
        acc = (torch.cat([zero_shot_predict(m, xte[i:i + 500].to(DEV), W) for i in range(0, len(xte), 500)]).cpu()
               == yte).float().mean().item()
    m.train()
    return acc


def train(kind, m, data, a, steps, batch=None, evals=0):
    xtr, ytr, caps, xte, yte, names, tok = data
    batch = batch or a.batch
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], a.lr, weight_decay=0.2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    curve = []
    for s in range(1, steps + 1):
        i = torch.randint(0, len(ytr), (batch,))
        ids, eos = tok([caps[j] for j in i])
        x = xtr[i] + 0 if not a.augment else torch.where(torch.rand(len(i), 1, 1, 1) < 0.5, xtr[i], xtr[i].flip(3))
        loss = loss_of(kind, m, x.to(DEV), ids.to(DEV), eos.to(DEV))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if evals and s % max(1, steps // evals) == 0:
            curve.append((s * batch, zero_shot(kind, m, xte, yte, names, tok, TEMPLATES[:1])))
            print(f"    {kind} images={s * batch} zero-shot={curve[-1][1]:.3f}", flush=True)
    return curve


@torch.no_grad()
def features(m, x):
    m.eval()
    return torch.cat([m.encode_image(x[i:i + 500].to(DEV)).cpu() for i in range(0, len(x), 500)])


# ---------------------------------------------------------------------------------------------------- experiments
def e1(a):
    data = setup(a)
    out = {}
    for kind in ("captioner", "bow-predict", "bow-contrastive", "clip"):
        m = build(kind, len(data[-1]), a)
        out[kind] = train(kind, m, data, a, a.steps, evals=10)
    return {"curves (images seen, zero-shot acc)": out, "paper": "contrastive BoW 4x more efficient than BoW "
            "prediction, which is 3x more efficient than the Transformer LM"}


def trained_clip(a, data):
    path = HERE / f"clip_{a.dataset}_{'quick' if a.quick else 'full'}.pt"
    m = build("clip", len(data[-1]), a)
    if path.exists():
        m.load_state_dict(torch.load(path, map_location=DEV))
    else:
        train("clip", m, data, a, a.steps)
        torch.save(m.state_dict(), path)
    return m


def e2(a):
    data = setup(a)
    m = trained_clip(a, data)
    _, _, _, xte, yte, names, tok = data
    zs = lambda T: zero_shot("clip", m, xte, yte, names, tok, T)
    return {"bare class name": zs(["{}"]), "'a photo of a {}.'": zs(TEMPLATES[:1]),
            f"ensemble of {len(TEMPLATES)} templates": zs(TEMPLATES), "paper (ImageNet)": "+1.3% prompt, +3.5% ensemble"}


def e3(a):
    data = setup(a)
    xtr, ytr, _, xte, yte, names, tok = data
    m = trained_clip(a, data)
    Ftr, Fte = features(m, xtr), features(m, xte)
    out = {"zero-shot": zero_shot("clip", m, xte, yte, names, tok, TEMPLATES)}
    g = torch.Generator().manual_seed(0)
    for k in (1, 2, 4, 8, 16):
        accs = []
        for seed in range(a.probe_seeds):
            idx = torch.cat([torch.nonzero(ytr == c).flatten()[torch.randperm(int((ytr == c).sum()), generator=g)[:k]]
                             for c in range(len(names))])
            accs.append((linear_probe(Ftr[idx], ytr[idx], Fte, C=a.C) == yte).float().mean().item())
        out[f"{k}-shot linear probe"] = sum(accs) / len(accs)
    out["all-data linear probe"] = (linear_probe(Ftr, ytr, Fte, C=a.C) == yte).float().mean().item()
    out["paper"] = "zero-shot CLIP matches a 4-shot linear probe on average across 20 datasets"
    return out


def e4(a):
    data = setup(a)
    xtr, ytr, _, xte, yte, names, tok = data
    m = trained_clip(a, data)
    torch.manual_seed(0)
    sup = nn.Sequential(SmallConvNet(3, a.width, attn_pool=True), nn.Linear(2 * a.width, len(names))).to(DEV)
    opt = torch.optim.AdamW(sup.parameters(), a.lr, weight_decay=0.05)
    for _ in range(a.steps):
        i = torch.randint(0, len(ytr), (a.batch,))
        loss = F.cross_entropy(sup(xtr[i].to(DEV)), ytr[i].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    sup.eval()
    Ftr = features(m, xtr)
    out = {}
    for name, xs in shifts(xte).items():
        with torch.no_grad():
            sup_acc = (torch.cat([sup(xs[i:i + 500].to(DEV)) for i in range(0, len(xs), 500)]).argmax(1).cpu() == yte).float().mean().item()
        probe_acc = (linear_probe(Ftr, ytr, features(m, xs), C=a.C) == yte).float().mean().item()
        zs_acc = zero_shot("clip", m, xs, yte, names, tok, TEMPLATES)
        out[name] = {"supervised": sup_acc, "CLIP linear probe": probe_acc, "CLIP zero-shot": zs_acc}
        print("  E4", name, out[name], flush=True)
    for k in ("supervised", "CLIP linear probe", "CLIP zero-shot"):
        shifted = [out[s][k] for s in out if s != "clean"]
        out[f"{k}: drop from clean"] = out["clean"][k] - sum(shifted) / len(shifted)
    return out


def e5(a):
    data = setup(a)
    _, _, _, xte, yte, names, tok = data
    V, out = len(tok), {}
    for label, kw in (("learned temperature", {}), ("fixed T = 0.07", {"temp": 0.07}), ("fixed T = 1.0", {"temp": 1.0}),
                      ("nonlinear projection", {"nonlinear": True})):
        m = build("clip", V, a, **kw)
        train("clip", m, data, a, a.steps)
        out[label] = zero_shot("clip", m, xte, yte, names, tok, TEMPLATES)
        print("  E5", label, out[label], flush=True)
    seen = a.steps * a.batch
    for B in (64, 512):
        m = build("clip", V, a)
        train("clip", m, data, a, max(1, seen // B), batch=B)
        out[f"batch {B} (same images seen)"] = zero_shot("clip", m, xte, yte, names, tok, TEMPLATES)
        print("  E5 batch", B, out[f"batch {B} (same images seen)"], flush=True)
    return out


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        L += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    a = ap.parse_args()
    a.n_train, a.n_test, a.width, a.d, a.layers = 50000, 10000, 64, 256, 4
    a.steps, a.batch, a.lr, a.C, a.probe_seeds, a.augment = 20000, 512, 1e-3, 1.0, 5, True
    if a.quick:
        a.n_train, a.n_test, a.width, a.d, a.layers = 2000, 500, 8, 32, 1
        a.steps, a.batch, a.probe_seeds = 30, 32, 1
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
