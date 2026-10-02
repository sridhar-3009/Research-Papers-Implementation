"""A light tour of the Vision Transformer (Dosovitskiy et al. 2021). About 6 seconds.

    python3 demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit import ViT, count_params, mean_attention_distance, mean_colour_targets, patchify, position_similarity

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. An image becomes a sequence of 'words' (Eq. 1)")
for res, P in ((224, 16), (224, 32), (224, 14), (384, 16), (512, 16)):
    n = (res // P) ** 2
    print(f"  {res}x{res} image, {P}x{P} patches: N = {n:5d} patches of {P * P * 3:4d} numbers  "
          f"(+1 [class] token; attention over {n + 1:,}^2 pairs)")
print("-> smaller patches or higher resolution mean longer sequences: cost grows like N^2.")

line("2. Table 1: model sizes")
with torch.device("meta"):
    for name, kw in (("ViT-Base/16", {}), ("ViT-Large/16", dict(D=1024, layers=24, heads=16, mlp=4096)),
                     ("ViT-Huge/14", dict(patch=14, D=1280, layers=32, heads=16, mlp=5120))):
        print(f"  {name:13s}: {count_params(ViT(**kw)) / 1e6:6.1f}M parameters")
print("  (paper: 86M, 307M, 632M)")

line("3. Inductive bias: 'which of 16 cells holds the bright square?' (16x16 images, same small budget)")


def squares(N, g=None):
    img = torch.rand(N, 3, 16, 16, generator=g) * 0.3
    cell = torch.randint(0, 16, (N,), generator=g)
    mask = torch.zeros(N, 1, 16, 16)
    for i in range(N):
        r, c = divmod(cell[i].item(), 4)
        mask[i, :, 4 * r:4 * r + 4, 4 * c:4 * c + 4] = 1
    return img * (1 - mask) + mask, cell


test_x, test_y = squares(500, torch.Generator().manual_seed(1))
cnn = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 32, 3, padding=1),
                    nn.ReLU(), nn.Flatten(), nn.Linear(32 * 64, 16))
vit = ViT(16, 4, 3, D=64, layers=2, heads=4, mlp=128, classes=16, head="mlp")
for name, model in (("small CNN", cnn), ("small ViT (4x4 patches)", vit)):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(500):
        x, y = squares(64)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(test_x).argmax(-1) == test_y).float().mean()
    print(f"  {name:24s}: {100 * acc:5.1f}% after 500 updates ({count_params(model):,} parameters)")
print("-> a CNN has locality and 2-D position built in; a ViT starts with NO idea which patches are neighbours and")
print("   must learn it from data. That is why the paper's ViTs lose to ResNets on ImageNet alone (1.3M images)")
print("   and win only after JFT-300M (Figure 3): 'large scale training trumps inductive bias'.")

line("4. Has the ViT started to learn 2-D structure? (Figure 7, centre)")
S = position_similarity(vit)
r = torch.arange(16)
rr, cc = r // 4, r % 4
same_row = (rr[:, None] == rr[None]) & (r[:, None] != r[None])
same_col = (cc[:, None] == cc[None]) & (r[:, None] != r[None])
other = ~(same_row | same_col) & (r[:, None] != r[None])
print(f"  mean cosine similarity of position embeddings: same row {S[same_row].mean():+.3f}, same column "
      f"{S[same_col].mean():+.3f}, neither {S[other].mean():+.3f}")
print("-> not yet: after 500 updates on tiny images there is no row/column pattern (the 'neither' pairs are even a")
print("   bit MORE similar). The paper's crisp row/column structure (Figure 7) emerges only with far more data.")

line("5. Attention distance per head (Figure 7, right): pixels between a patch and what it attends to")
d = mean_attention_distance(vit, test_x[:64])
for l in range(d.shape[0]):
    print(f"  layer {l + 1}: " + "  ".join(f"{v:5.1f}" for v in d[l].tolist()))
xy = torch.stack([r // 4, r % 4], 1).float() * 4
print(f"  (uniform attention over all patches would give {torch.cdist(xy, xy).mean():.1f} px)")
print("-> every head is still near the uniform value: this small model attends almost evenly everywhere. The")
print("   paper's trained models show both local and global heads in early layers and global ones deeper.")

line("6. Self-supervised 'masked patch prediction' targets: 3-bit mean colour of each patch (512 classes)")
img = torch.zeros(1, 3, 8, 8)
img[0, 0, :4, :4] = 1.0; img[0, 1, :4, 4:] = 0.5; img[0, :, 4:, :4] = 1.0
print("  patch mean colours -> class:", mean_colour_targets(img, 4)[0].tolist(), "(red, half-green, white, black)")
print("-> the paper: masked patch prediction gave 79.9% on ImageNet for ViT-B/16, +2% over training from scratch,")
print("   but 4% behind supervised pre-training.")
