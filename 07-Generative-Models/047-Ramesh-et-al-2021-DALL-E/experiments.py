"""DALL-E (Ramesh et al. 2021) at laptop-to-single-GPU scale.

The paper trains a 12B transformer on 250M internet image-text pairs; none of that is reproducible here. Instead we
build a synthetic captioned world ('a red circle at top left and a blue square at bottom right', 32x32 images) where
every claim can be measured exactly, and test the paper's design choices on it:

  E1  Stage 1: dVAE (32x32 -> 8x8 tokens, K = 64) with the paper's schedules; beta in {0, 1, 6.6}: reconstruction
      error and codebook perplexity (footnote 4: larger beta -> better codebook usage).
  E2  Stage 2: the text+image transformer on dVAE tokens; zero-shot accuracy on held-out attribute combinations
      (colour x shape x position never seen together), judged by decoding and reading the image back with
      exact pixel checks.
  E3  Sparse attention (row / column / conv) vs dense causal attention: validation loss at equal steps.
  E4  Loss weighting: 1/8 text + 7/8 image (the paper) vs 1/2 + 1/2.
  E5  Learned per-position padding tokens vs a fixed (zero, untrained) padding embedding: loss on in-distribution
      captions vs out-of-distribution 4-object captions (cf. Section 2.2: learned padding gave higher validation
      loss but better out-of-distribution behaviour; the paper's alternative was -inf masking, approximated here).
  E6  Reranking (Figure 9c): a small contrastive image-text model (CLIP-style, trained on the same data) picks the
      best of N in {1, 2, 4, ..., 64} samples; caption-match accuracy vs N.
  E7  PowerSGD (Table 1): training with compressed gradients at ranks {2, 4, 8, 16, full}; loss gap vs uncompressed.
  E8  Optional: MS-COCO captions if a local copy is supplied with --coco (not downloaded automatically, ~19 GB).

!! HEAVY (minutes to hours). Not run on the author's laptop.
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

from dalle import DALLE, DVAE, PowerSGD, codebook_perplexity, cosine_schedule, dvae_loss, phi, phi_inv

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------------------------------- the world
COLOURS = {"red": (230, 40, 40), "green": (40, 200, 60), "blue": (50, 80, 230), "yellow": (240, 220, 50),
           "white": (240, 240, 240)}
SHAPES = ["square", "circle", "triangle"]
PLACES = ["top left", "top right", "bottom left", "bottom right"]
WORDS = ["<pad>", "a", "at", "and"] + list(COLOURS) + SHAPES + [w for p in PLACES for w in p.split()]
WORDS = list(dict.fromkeys(WORDS))
VOCAB = {w: i for i, w in enumerate(WORDS)}
HELD_OUT = {("red", "triangle"), ("blue", "circle"), ("yellow", "square")}       # never shown together in training


def draw(objs, size=32):
    """objs: list of (colour, shape, place). Returns a uint8-valued float image (3, size, size) on dark grey."""
    img = torch.full((3, size, size), 25.0)
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    masks = {"square": (yy >= 3) & (yy < 13) & (xx >= 3) & (xx < 13),
             "circle": ((yy - 7.5) ** 2 + (xx - 7.5) ** 2) <= 25,
             "triangle": (yy >= 3) & (yy < 13) & ((xx - 7.5).abs() <= (yy - 3) / 2 + 0.5)}
    for colour, shape, place in objs:
        r, c = divmod(PLACES.index(place), 2)
        sub = img[:, 16 * r:16 * r + 16, 16 * c:16 * c + 16]
        sub[:, masks[shape]] = torch.tensor(COLOURS[colour], dtype=torch.float)[:, None]
    return img


def caption(objs):
    return " and ".join(f"a {c} {s} at {p}" for c, s, p in objs)


def sample_scene(rng, allow_held_out=False, n_max=2):
    places = rng.sample(PLACES, rng.randint(1, n_max))
    objs = []
    for p in places:
        while True:
            c, s = rng.choice(list(COLOURS)), rng.choice(SHAPES)
            if allow_held_out or (c, s) not in HELD_OUT:
                break
        objs.append((c, s, p))
    return objs


def encode_text(text, T):
    ids = [VOCAB[w] for w in text.split()][:T]
    return torch.tensor(ids + [0] * (T - len(ids))), len(ids)


def dataset(n, T, seed, held_out_only=False):
    rng = random.Random(seed)
    xs, ts, ls, scenes = [], [], [], []
    while len(xs) < n:
        objs = sample_scene(rng, allow_held_out=held_out_only)
        if held_out_only and not any((c, s) in HELD_OUT for c, s, _ in objs):
            continue
        t, L = encode_text(caption(objs), T)
        xs.append(draw(objs)); ts.append(t); ls.append(L); scenes.append(objs)
    return torch.stack(xs), torch.stack(ts), torch.tensor(ls), scenes


def scene_correct(img, objs):
    """Exact check: every described object's pixels have the described colour (within 40 per channel)."""
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    core = {"square": (yy >= 5) & (yy < 11) & (xx >= 5) & (xx < 11), "circle": ((yy - 7.5) ** 2 + (xx - 7.5) ** 2) <= 9,
            "triangle": (yy >= 8) & (yy < 12) & ((xx - 7.5).abs() <= 1.5)}
    for colour, shape, place in objs:
        r, c = divmod(PLACES.index(place), 2)
        sub = img[:, 16 * r:16 * r + 16, 16 * c:16 * c + 16][:, core[shape]]
        if (sub - torch.tensor(COLOURS[colour], dtype=torch.float)[:, None]).abs().max() > 40:
            return False
    return True


# ---------------------------------------------------------------------------------------------------- stage 1
def train_dvae(X, a, beta, seed=0):
    torch.manual_seed(seed)
    d = DVAE(K=a.K, width=a.dvae_width, down=4).to(DEV)
    opt = torch.optim.AdamW(d.parameters(), a.lr_dvae, weight_decay=1e-4)
    for s in range(a.dvae_steps):
        x = phi(X[torch.randint(0, len(X), (a.batch,))]).to(DEV)
        tau = cosine_schedule(s, a.dvae_steps * 0.5, 1.0, 1 / 16)
        b = cosine_schedule(s, max(1, a.dvae_steps // 20), 0.0, beta)
        for g in opt.param_groups:
            g["lr"] = cosine_schedule(s, a.dvae_steps, a.lr_dvae, a.lr_dvae / 80)
        loss = dvae_loss(d, x, tau, b)
        opt.zero_grad(); loss.backward(); opt.step()
    return d.eval()


@torch.no_grad()
def dvae_metrics(d, X):
    x = phi(X).to(DEV)
    z = torch.cat([d.tokens(x[k:k + 256]) for k in range(0, len(x), 256)])
    rec = torch.cat([phi_inv(d.reconstruct(x[k:k + 256])) for k in range(0, len(x), 256)]).cpu()
    return {"pixel MAE": (rec - X).abs().mean().item(), "codebook perplexity": codebook_perplexity(z.cpu(), d.K)}


def e1(a):
    X, _, _, _ = dataset(a.n_train, a.T, 0)
    Xv, _, _, _ = dataset(1000, a.T, 1)
    out = {}
    for beta in (0.0, 1.0, 6.6):
        d = train_dvae(X, a, beta)
        out[f"beta={beta}"] = dvae_metrics(d, Xv)
        print("  E1", beta, out[f"beta={beta}"], flush=True)
        if beta == 6.6:
            torch.save(d.state_dict(), HERE / "dvae.pt")
    return out


# ---------------------------------------------------------------------------------------------------- stage 2
def get_dvae(a, X):
    d = DVAE(K=a.K, width=a.dvae_width, down=4).to(DEV)
    if (HERE / "dvae.pt").exists():
        d.load_state_dict(torch.load(HERE / "dvae.pt", map_location=DEV))
        return d.eval()
    return train_dvae(X, a, 6.6)


@torch.no_grad()
def tokenize(d, X):
    return torch.cat([d.tokens(phi(X[k:k + 256]).to(DEV)).flatten(1).cpu() for k in range(0, len(X), 256)])


def train_prior(Z, Tx, L, a, sparse=True, w_text=1 / 8, mask_padding=False, compress_rank=None, steps=None, seed=0):
    torch.manual_seed(seed)
    m = DALLE(len(WORDS), a.K, a.T, 8, 8, d=a.d, layers=a.layers, heads=a.heads, sparse=sparse).to(DEV)
    if mask_padding:                                                               # alternative: a fixed zero padding
        with torch.no_grad():
            m.pad_emb.zero_()
        m.pad_emb.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), a.lr, betas=(0.9, 0.96), weight_decay=4.5e-2)
    comps = {n: PowerSGD(p.shape, compress_rank, seed=i) for i, (n, p) in enumerate(m.named_parameters())
             if compress_rank and p.dim() == 2 and min(p.shape) > compress_rank}
    losses = []
    for s in range(steps or a.steps):
        idx = torch.randint(0, len(Z), (a.batch,))
        loss, ct, ci = m.loss(Tx[idx].to(DEV), L[idx].to(DEV), Z[idx].to(DEV), w_text, 1 - w_text)
        opt.zero_grad(); loss.backward()
        for n, p in m.named_parameters():
            if n in comps and p.grad is not None:
                p.grad = comps[n].compress(p.grad.cpu())[0].to(DEV)
        opt.step()
        losses.append(ci)
    return m.eval(), losses


@torch.no_grad()
def val_loss(m, Z, Tx, L):
    tot = 0.0
    for k in range(0, len(Z), 128):
        _, _, ci = m.loss(Tx[k:k + 128].to(DEV), L[k:k + 128].to(DEV), Z[k:k + 128].to(DEV))
        tot += ci * len(Z[k:k + 128])
    return tot / len(Z)


@torch.no_grad()
def zero_shot_accuracy(m, d, a, n=200, N=1, scorer=None):
    """Fraction of held-out-combination captions whose (best of N) sample is drawn correctly."""
    _, Tx, L, scenes = dataset(n, a.T, 7, held_out_only=True)
    ok = 0
    for i in range(n):
        t, l = Tx[i:i + 1].repeat(N, 1).to(DEV), L[i:i + 1].repeat(N).to(DEV)
        toks = m.generate(t, l).view(N, 8, 8)
        imgs = phi_inv(torch.sigmoid(d.decode_tokens(toks)[:, :3])).cpu()
        pick = 0 if N == 1 else int(scorer(imgs, Tx[i:i + 1].repeat(N, 1), L[i:i + 1].repeat(N)).argmax())
        ok += scene_correct(imgs[pick], scenes[i])
    return ok / n


def e2(a):
    X, Tx, L, _ = dataset(a.n_train, a.T, 0)
    d = get_dvae(a, X)
    Z = tokenize(d, X)
    m, _ = train_prior(Z, Tx, L, a)
    torch.save(m.state_dict(), HERE / "prior.pt")
    Xs, Ts, Ls, sc = dataset(200, a.T, 3)
    seen_acc = sum(scene_correct(phi_inv(torch.sigmoid(d.decode_tokens(m.generate(Ts[i:i + 1].to(DEV), Ls[i:i + 1].to(DEV))
                                                                       .view(1, 8, 8))[:, :3])).cpu()[0], sc[i])
                   for i in range(200)) / 200
    return {"seen-combination accuracy": seen_acc, "held-out-combination (zero-shot) accuracy":
            zero_shot_accuracy(m, d, a)}


def e3(a):
    X, Tx, L, _ = dataset(a.n_train, a.T, 0)
    Xv, Tv, Lv, _ = dataset(1000, a.T, 1)
    d = get_dvae(a, X)
    Z, Zv = tokenize(d, X), tokenize(d, Xv)
    return {name: val_loss(train_prior(Z, Tx, L, a, sparse=sp)[0], Zv, Tv, Lv)
            for name, sp in (("sparse (row/column/conv)", True), ("dense causal", False))}


def e4(a):
    X, Tx, L, _ = dataset(a.n_train, a.T, 0)
    Xv, Tv, Lv, _ = dataset(1000, a.T, 1)
    d = get_dvae(a, X)
    Z, Zv = tokenize(d, X), tokenize(d, Xv)
    return {f"text weight {w}": val_loss(train_prior(Z, Tx, L, a, w_text=w)[0], Zv, Tv, Lv) for w in (1 / 8, 1 / 2)}


def e5(a):
    X, Tx, L, _ = dataset(a.n_train, a.T, 0)
    d = get_dvae(a, X)
    Z = tokenize(d, X)
    Xv, Tv, Lv, _ = dataset(1000, a.T, 1)
    rng = random.Random(5)
    ood = [[(rng.choice(list(COLOURS)), rng.choice(SHAPES), p) for p in PLACES] for _ in range(500)]   # 4 objects
    Xo = torch.stack([draw(o) for o in ood])
    To, Lo = zip(*[encode_text(caption(o), a.T) for o in ood])
    To, Lo = torch.stack(To), torch.tensor(Lo)
    out = {}
    for name, mp in (("learned padding tokens", False), ("fixed zero padding", True)):
        m, _ = train_prior(Z, Tx, L, a, mask_padding=mp)
        out[name] = {"in-distribution image loss": val_loss(m, tokenize(d, Xv), Tv, Lv),
                     "4-object captions (never seen)": val_loss(m, tokenize(d, Xo), To, Lo)}
    return out


class TinyCLIP(nn.Module):
    """A small contrastive image-text model trained on the same synthetic pairs (the paper uses CLIP)."""

    def __init__(self, T, d=128):
        super().__init__()
        self.img = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(64 * 8 * 8, d))
        self.emb = nn.Embedding(len(WORDS), d)
        self.pos = nn.Parameter(torch.randn(T, d) * 0.02)
        self.txt = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True), 2)
        self.scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode(self, x, t, L):
        mask = torch.arange(t.shape[1], device=t.device)[None, :] >= L[:, None]
        h = self.txt(self.emb(t) + self.pos, src_key_padding_mask=mask)
        h = (h * (~mask)[..., None]).sum(1) / L[:, None]
        return F.normalize(self.img(x / 255 - 0.5), dim=-1), F.normalize(h, dim=-1)

    def forward(self, x, t, L):
        i, s = self.encode(x, t, L)
        return (i * s).sum(-1)


def train_clip(X, Tx, L, a):
    torch.manual_seed(0)
    c = TinyCLIP(a.T).to(DEV)
    opt = torch.optim.Adam(c.parameters(), 3e-4)
    for _ in range(a.clip_steps):
        idx = torch.randint(0, len(X), (128,))
        i, s = c.encode(X[idx].to(DEV), Tx[idx].to(DEV), L[idx].to(DEV))
        logits = c.scale.exp() * i @ s.T
        lab = torch.arange(len(idx), device=DEV)
        loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
    return c.eval()


def e6(a):
    X, Tx, L, _ = dataset(a.n_train, a.T, 0)
    d = get_dvae(a, X)
    m = DALLE(len(WORDS), a.K, a.T, 8, 8, d=a.d, layers=a.layers, heads=a.heads).to(DEV)
    if (HERE / "prior.pt").exists():
        m.load_state_dict(torch.load(HERE / "prior.pt", map_location=DEV)); m.eval()
    else:
        m, _ = train_prior(tokenize(d, X), Tx, L, a)
    clip = train_clip(X, Tx, L, a)
    scorer = lambda imgs, t, l: clip(imgs.to(DEV), t.to(DEV), l.to(DEV)).cpu()
    return {f"N={N}": zero_shot_accuracy(m, d, a, n=a.n_rerank, N=N, scorer=scorer) for N in a.rerank_N}


def e7(a):
    X, Tx, L, _ = dataset(a.n_train, a.T, 0)
    d = get_dvae(a, X)
    Z = tokenize(d, X)
    out = {}
    base = None
    for r in (None, 16, 8, 4, 2):
        _, losses = train_prior(Z, Tx, L, a, compress_rank=r, steps=a.steps // 2)
        final = sum(losses[-50:]) / 50
        base = final if r is None else base
        out["uncompressed" if r is None else f"rank {r}"] = {"final train image loss": final,
                                                             "gap vs uncompressed": final - base}
        print("  E7", r, out, flush=True)
    return out


def e8(a):
    if not a.coco:
        return {"note": "pass --coco /path/to/coco (with annotations/captions_train2017.json and train2017/) to run"}
    return {"note": "MS-COCO support: tokenize captions with a BPE (e.g. the 033/034 code) and images with a dVAE "
                    "trained on 64x64 crops, then reuse train_prior; left as an exercise at this scale."}


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(R):
        L += [f"## {k.upper()}", "", "```", json.dumps(R[k], indent=1), "```", ""]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 9)])
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--coco", default=None)
    a = ap.parse_args()
    a.K, a.dvae_width, a.T, a.d, a.layers, a.heads = 64, 64, 24, 256, 8, 8
    a.n_train, a.batch, a.lr, a.lr_dvae = 50000, 64, 4.5e-4, 1e-3
    a.dvae_steps, a.steps, a.clip_steps, a.n_rerank, a.rerank_N = 20000, 20000, 5000, 200, (1, 2, 4, 8, 16, 32, 64)
    if a.quick:
        a.dvae_width, a.d, a.layers, a.heads, a.n_train = 16, 64, 2, 4, 2000
        a.dvae_steps, a.steps, a.clip_steps, a.n_rerank, a.rerank_N = 100, 100, 100, 5, (1, 4)
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6), ("e7", e7), ("e8", e8)):
            if a.only in (None, name):
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
