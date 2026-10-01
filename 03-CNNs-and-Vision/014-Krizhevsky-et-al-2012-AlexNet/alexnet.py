"""AlexNet - Krizhevsky, Sutskever & Hinton (2012) - written from the paper.

  AlexNet                   Section 3.5 / Figure 2: 5 conv + 3 fully connected layers, ~61M parameters,
                            the two-GPU split as conv groups, ReLU, LRN, overlapping max-pooling
  local_response_norm       Section 3.3, written by hand
  PaperDropout              Section 4.2: drop with p = 0.5 in training, multiply by 0.5 at test
  paper_sgd_step            Section 5: the exact update rule (momentum 0.9, weight decay 0.0005)
  random_crop_flip, ten_crop, rgb_pca, pca_color_augment     Section 4.1: data augmentation
  SmallCifarNet             a 4-layer CIFAR-10 net for the paper's small experiments (Figure 1, Section 3.3)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Section 3.3: local response normalization
# ---------------------------------------------------------------------------

def local_response_norm(a, k=2.0, n=5, alpha=1e-4, beta=0.75):
    """b^i = a^i / (k + alpha * sum_{j = i-n/2 .. i+n/2} (a^j)^2)^beta

    a: (N, C, H, W) after ReLU. For each position, the sum runs over n ADJACENT channels
    ("kernel maps"), clipped at 0 and C-1. A big activity in one map suppresses its
    neighbours at the same place: "lateral inhibition". No mean is subtracted, so the
    paper calls it "brightness normalization"."""
    N, C, H, W = a.shape
    sq = a ** 2
    padded = F.pad(sq, (0, 0, 0, 0, n // 2, n // 2))          # zeros beyond the first/last channel
    s = sum(padded[:, j:j + C] for j in range(n))              # sliding sum over n neighbouring channels
    return a / (k + alpha * s) ** beta


class LRN(nn.Module):
    def __init__(self, k=2.0, n=5, alpha=1e-4, beta=0.75):
        super().__init__()
        self.k, self.n, self.alpha, self.beta = k, n, alpha, beta

    def forward(self, x):
        return local_response_norm(x, self.k, self.n, self.alpha, self.beta)


# ---------------------------------------------------------------------------
# Section 4.2: dropout, the 2012 way
# ---------------------------------------------------------------------------

class PaperDropout(nn.Module):
    """Training: set each output to 0 with probability 0.5.
    Test: keep all outputs but multiply by 0.5 (so the expected input to the next layer matches).
    (Modern "inverted" dropout scales by 2 in training instead; the two are equivalent.)"""

    def __init__(self, p_drop=0.5):
        super().__init__()
        self.p = p_drop

    def forward(self, x):
        if self.training:
            return x * (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * (1 - self.p)


# ---------------------------------------------------------------------------
# Section 3.5: the network
# ---------------------------------------------------------------------------

class AlexNet(nn.Module):
    """Input 3x224x224 (mean-subtracted RGB).

      conv1  96 @ 11x11x3, stride 4  -> 96 x 55 x 55   ReLU, LRN, max-pool 3/2 -> 27x27
      conv2 256 @ 5x5x48  (2 groups) -> 256 x 27 x 27  ReLU, LRN, max-pool 3/2 -> 13x13
      conv3 384 @ 3x3x256 (full)     -> 384 x 13 x 13  ReLU
      conv4 384 @ 3x3x192 (2 groups) -> 384 x 13 x 13  ReLU
      conv5 256 @ 3x3x192 (2 groups) -> 256 x 13 x 13  ReLU, max-pool 3/2 -> 6x6
      fc6 9216 -> 4096  ReLU, dropout
      fc7 4096 -> 4096  ReLU, dropout
      fc8 4096 -> 1000  (softmax is in the loss)

    groups=2 is the two-GPU split: in conv2, conv4, conv5 each half of the kernels only sees
    the half of the previous layer's maps that lived "on the same GPU". conv3 and the fc layers
    see everything (that's where the GPUs communicate).

    Note: 224 with an 11x11 kernel at stride 4 gives (224-11)/4+1 = 54.25, not the paper's 55.
    Padding conv1 by 2 gives 55 (others crop 227x227 instead)."""

    def __init__(self, num_classes=1000, groups=2, lrn=True, overlap=True, act="relu", dropout=True):
        super().__init__()
        self.act = {"relu": F.relu, "tanh": torch.tanh}[act]
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, groups=groups)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1, groups=groups)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1, groups=groups)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.norm = LRN() if lrn else nn.Identity()
        # overlapping: 3x3 windows every 2 pixels; plain: 2x2 every 2 (same output size, Section 3.4)
        self.pool = nn.MaxPool2d(3, 2) if overlap else nn.MaxPool2d(2, 2)
        self.drop = PaperDropout() if dropout else nn.Identity()
        self.paper_init()

    def paper_init(self):
        """Section 5: weights ~ N(0, 0.01^2); biases 1 in conv2, conv4, conv5, fc6, fc7 (gives the
        ReLUs positive inputs early on), 0 elsewhere."""
        for name, m in self.named_children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 1.0 if name in ("conv2", "conv4", "conv5", "fc6", "fc7") else 0.0)

    def features(self, x):
        x = self.pool(self.norm(self.act(self.conv1(x))))
        x = self.pool(self.norm(self.act(self.conv2(x))))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(self.act(self.conv5(x)))
        return x

    def forward(self, x, return_hidden=False):
        x = self.features(x).flatten(1)
        x = self.drop(self.act(self.fc6(x)))
        h = self.drop(self.act(self.fc7(x)))                 # the 4096-d "last hidden layer" (Figure 4 right)
        out = self.fc8(h)
        return (out, h) if return_hidden else out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Section 5: the update rule
# ---------------------------------------------------------------------------

@torch.no_grad()
def paper_sgd_step(params, velocities, lr, momentum=0.9, weight_decay=0.0005):
    """v <- 0.9 v - 0.0005 * lr * w - lr * <dL/dw>
       w <- w + v
    Weight decay is written into the update (the paper found it lowers even the TRAINING error)."""
    for p, v in zip(params, velocities):
        if p.grad is None:
            continue
        v.mul_(momentum).add_(p, alpha=-weight_decay * lr).add_(p.grad, alpha=-lr)
        p.add_(v)


class DivideOnPlateau:
    """'divide the learning rate by 10 when the validation error rate stopped improving'.
    patience = how many evaluations without improvement to wait."""

    def __init__(self, lr=0.01, patience=2, factor=0.1):
        self.lr, self.patience, self.factor = lr, patience, factor
        self.best, self.bad = float("inf"), 0

    def update(self, val_error):
        if val_error < self.best - 1e-6:
            self.best, self.bad = val_error, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.lr *= self.factor
                self.bad = 0
        return self.lr


# ---------------------------------------------------------------------------
# Section 4.1: data augmentation
# ---------------------------------------------------------------------------

def random_crop_flip(img, size=224, generator=None):
    """img: (C, 256, 256). A random size x size patch, mirrored left-right half the time.
    With 256 and 224 there are 33 x 33 positions x 2 flips (the paper counts 32 x 32 x 2 = 2048)."""
    C, H, W = img.shape
    i = int(torch.randint(0, H - size + 1, (1,), generator=generator))
    j = int(torch.randint(0, W - size + 1, (1,), generator=generator))
    patch = img[:, i:i + size, j:j + size]
    if torch.rand(1, generator=generator).item() < 0.5:
        patch = patch.flip(-1)
    return patch


def ten_crop(img, size=224):
    """Test time: the 4 corner patches + the center patch, and their mirror images. (10, C, size, size).
    The network's softmax outputs on the ten are averaged."""
    C, H, W = img.shape
    c = ((H - size) // 2, (W - size) // 2)
    corners = [(0, 0), (0, W - size), (H - size, 0), (H - size, W - size), c]
    crops = [img[:, i:i + size, j:j + size] for i, j in corners]
    return torch.stack(crops + [p.flip(-1) for p in crops])


def rgb_pca(images):
    """images: (N, 3, H, W). Eigen-decomposition of the 3x3 covariance of RGB pixel values
    over the whole set. Returns eigenvalues (3,) and eigenvectors (3, 3) as columns."""
    px = images.permute(0, 2, 3, 1).reshape(-1, 3).double()
    cov = torch.cov(px.T)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return eigvals.float(), eigvecs.float()


def pca_color_augment(img, eigvals, eigvecs, std=0.1, generator=None):
    """Add [p1 p2 p3] [a1 l1, a2 l2, a3 l3]^T to EVERY pixel, a_i ~ N(0, 0.1^2), drawn once per
    image. It changes the colour and intensity of the illumination, not the object."""
    alpha = torch.randn(3, generator=generator) * std
    shift = eigvecs @ (alpha * eigvals)                        # (3,)
    return img + shift.view(3, 1, 1).to(img.dtype)


# ---------------------------------------------------------------------------
# A small network for CIFAR-10 (the paper's Figure 1 and Section 3.3 CIFAR experiments)
# ---------------------------------------------------------------------------

class SmallCifarNet(nn.Module):
    """A four-layer CNN in the style of the cuda-convnet CIFAR-10 nets the paper refers to
    (their exact file isn't in the paper): three 5x5 conv layers (32, 32, 64 maps), each
    followed by pooling, then a fully connected 10-way layer."""

    def __init__(self, act="relu", lrn=False, overlap=True):
        super().__init__()
        self.act = {"relu": F.relu, "tanh": torch.tanh}[act]
        self.c1, self.c2, self.c3 = nn.Conv2d(3, 32, 5, padding=2), nn.Conv2d(32, 32, 5, padding=2), nn.Conv2d(32, 64, 5, padding=2)
        self.fc = nn.Linear(64 * 4 * 4, 10)
        self.norm = LRN(k=2, n=5, alpha=1e-4, beta=0.75) if lrn else nn.Identity()
        # both give 32 -> 16 -> 8 -> 4, so the comparison is fair (Section 3.4: "equivalent dimensions")
        self.pool = nn.MaxPool2d(3, 2, padding=1) if overlap else nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.norm(self.pool(self.act(self.c1(x))))        # 32 -> 16
        x = self.norm(self.pool(self.act(self.c2(x))))        # -> 8
        x = self.pool(self.act(self.c3(x)))                   # -> 4
        return self.fc(x.flatten(1))
