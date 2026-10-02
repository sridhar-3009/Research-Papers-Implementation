"""An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy, Beyer, Kolesnikov,
Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit & Houlsby, ICLR 2021).

  z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos      E: (P^2 C) x D, E_pos: (N+1) x D          (Eq. 1)
  z'_l = MSA(LN(z_{l-1})) + z_{l-1}                                                           (Eq. 2)
  z_l  = MLP(LN(z'_l)) + z'_l                                                                  (Eq. 3)
  y    = LN(z_L^0)                                                                              (Eq. 4)
  - an image H x W x C becomes N = HW / P^2 flattened patches (the 'words');
  - pre-training head: an MLP with one hidden layer; fine-tuning head: a ZERO-initialised linear layer;
  - fine-tuning at higher resolution keeps P, so N grows: the position embeddings are 2-D interpolated;
  - 'hybrid': the patches come from a CNN feature map (patch size 1x1);
  - self-supervised 'masked patch prediction' (Section 4.6 / Appendix B.1.2): corrupt 50% of the patches and
    predict each one's 3-bit mean colour (512 classes).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify(img, P):
    """img (B, C, H, W) -> (B, N, P*P*C): patches in row-major order, each flattened as (row, col, channel)."""
    B, C, H, W = img.shape
    x = img.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 3, 5, 1)       # B, gh, gw, P, P, C
    return x.reshape(B, (H // P) * (W // P), P * P * C)


def unpatchify(x, P, C, H, W):
    B = x.shape[0]
    x = x.reshape(B, H // P, W // P, P, P, C).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, C, H, W)


class Block(nn.Module):
    """Pre-LN Transformer encoder block (Eqs. 2-3); MLP = Linear -> GELU -> Linear."""

    def __init__(self, D, heads, mlp, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D, eps=1e-6), nn.LayerNorm(D, eps=1e-6)
        self.attn = nn.MultiheadAttention(D, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D, mlp), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp, D), nn.Dropout(dropout))
        self.last_attention = None

    def forward(self, z, keep_attention=False):
        h = self.ln1(z)
        a, w = self.attn(h, h, h, need_weights=keep_attention, average_attn_weights=False)
        if keep_attention:
            self.last_attention = w.detach()                                    # (B, heads, N+1, N+1)
        z = z + a
        return z + self.mlp(self.ln2(z))


class CNNStem(nn.Module):
    """A tiny stand-in for the hybrid's ResNet stem: its feature map is cut into 1x1 'patches'."""

    def __init__(self, C, width, downsample):
        super().__init__()
        layers, c = [], C
        for _ in range(int(math.log2(downsample))):
            layers += [nn.Conv2d(c, width, 3, stride=2, padding=1), nn.GroupNorm(1, width), nn.ReLU()]
            c = width
        self.net = nn.Sequential(*layers)
        self.out_dim = c

    def forward(self, img):
        f = self.net(img)                                                       # B, width, h, w
        return f.flatten(2).transpose(1, 2)                                     # B, h*w, width


class ViT(nn.Module):
    def __init__(self, image=224, patch=16, C=3, D=768, layers=12, heads=12, mlp=3072, classes=1000, dropout=0.0,
                 head="linear", hybrid_downsample=None):
        """head: 'mlp' (pre-training: one hidden layer, tanh) or 'linear' (fine-tuning, zero-initialised)."""
        super().__init__()
        self.P, self.C = patch, C
        if hybrid_downsample:
            self.stem = CNNStem(C, 64, hybrid_downsample)
            in_dim, self.grid = self.stem.out_dim, image // hybrid_downsample
        else:
            self.stem, in_dim, self.grid = None, patch * patch * C, image // patch
        self.E = nn.Linear(in_dim, D)                                           # patch embedding (Eq. 1)
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.randn(1, self.grid ** 2 + 1, D) * 0.02)   # learnable 1-D positions
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(D, heads, mlp, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(D, eps=1e-6)
        self.head = (nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, classes)) if head == "mlp"
                     else nn.Linear(D, classes))
        if head == "linear":
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def tokens(self, img):
        x = self.stem(img) if self.stem is not None else patchify(img, self.P)
        z = self.E(x)
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1)
        return self.drop(z + self.pos[:, : z.shape[1]])

    def forward(self, img, keep_attention=False, return_tokens=False):
        z = self.tokens(img)
        for b in self.blocks:
            z = b(z, keep_attention)
        y = self.ln(z)
        return y if return_tokens else self.head(y[:, 0])                      # Eq. 4: the [class] token

    def new_head(self, classes):
        """Fine-tuning: drop the pre-training head and attach a zero-initialised D x K layer (Section 3.2)."""
        D = self.ln.normalized_shape[0]
        self.head = nn.Linear(D, classes)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def resize_positions(self, new_grid):
        """Section 3.2: 2-D interpolation of the pre-trained position embeddings for a new grid of patches."""
        cls, grid = self.pos[:, :1], self.pos[:, 1:]
        D, g = grid.shape[-1], self.grid
        grid = grid.reshape(1, g, g, D).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        self.pos = nn.Parameter(torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(1, new_grid ** 2, D)], 1))
        self.grid = new_grid


# ---------------------------------------------------------------------------------------------------- Section 4.5
def position_similarity(model):
    """Figure 7 (centre): cosine similarity of every patch's position embedding with every other one."""
    p = F.normalize(model.pos[0, 1:].detach(), dim=-1)
    return p @ p.T


def embedding_filters_pca(model, k=28):
    """Figure 7 (left): the top principal components of the patch-embedding filters E (columns of E^T)."""
    W = model.E.weight.detach()                                                # D x (P^2 C)
    W = W - W.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return Vh[:k], S[:k]


def mean_attention_distance(model, img):
    """Figure 7 (right): for each layer and head, the attention-weighted average distance (in pixels) between the
    query patch and the patches it attends to (the [class] token is left out)."""
    model.eval()
    with torch.no_grad():
        model(img, keep_attention=True)
    g, P = model.grid, model.P
    r = torch.arange(g * g)
    xy = torch.stack([r // g, r % g], 1).float() * P
    dist = torch.cdist(xy, xy)                                                  # N x N pixel distances
    out = []
    for b in model.blocks:
        w = b.last_attention[:, :, 1:, 1:]                                      # B, h, N, N (patch to patch)
        w = w / w.sum(-1, keepdim=True)
        out.append((w * dist).sum(-1).mean((0, 2)))                             # per head
    return torch.stack(out)                                                     # layers x heads


# ---------------------------------------------------------------------------------------------------- self-supervision
def mean_colour_targets(img, P, bits=3):
    """Appendix B.1.2: each patch's mean colour, quantised to `bits` per channel -> one of 2^(3 bits) classes."""
    m = patchify(img, P).reshape(img.shape[0], -1, P * P, img.shape[1]).mean(2)        # B, N, C in [0, 1]
    q = (m.clamp(0, 1 - 1e-6) * (2 ** bits)).long()                                     # 0 .. 2^bits - 1
    return (q[..., 0] * (2 ** bits) + q[..., 1]) * (2 ** bits) + q[..., 2]


def corrupt_patches(x, rate=0.5, probs=(0.8, 0.1, 0.1), generator=None):
    """BERT-style corruption of patch vectors x (B, N, dim): choose `rate` of the patches; replace 80% by a
    learnable [mask] (here: zeros, filled in by the caller), 10% by another random patch, keep 10%.
    Returns (corrupted x, chosen mask, which were set to [mask])."""
    g = generator
    B, N, _ = x.shape
    chosen = torch.rand(B, N, generator=g) < rate
    r = torch.rand(B, N, generator=g)
    to_mask = chosen & (r < probs[0])
    to_rand = chosen & (r >= probs[0]) & (r < probs[0] + probs[1])
    out = x.clone()
    out[to_mask] = 0.0
    src = torch.randint(0, N, (B, N), generator=g)
    out[to_rand] = x[torch.arange(B)[:, None].expand(B, N)[to_rand], src[to_rand]]
    return out, chosen, to_mask


def count_params(m):
    return sum(p.numel() for p in m.parameters())
