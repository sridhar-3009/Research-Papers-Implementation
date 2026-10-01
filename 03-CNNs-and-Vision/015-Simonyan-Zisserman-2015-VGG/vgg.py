"""VGG - Simonyan & Zisserman (2015) - "Very Deep Convolutional Networks for Large-Scale Image Recognition".

  CONFIGS                 Table 1: networks A, A-LRN, B, C, D (VGG-16), E (VGG-19)
  VGG                     the network for any config; `width_div` and `fc` shrink it for small experiments
  count_params            Table 2, computed from the config (no memory needed)
  receptive_field         Section 2.3: two 3x3 = one 5x5, three 3x3 = one 7x7
  init_from_A             Section 3.1: start a deep net from the trained shallow net A
  to_fully_convolutional  Section 3.2: fc layers -> conv layers, for "dense" evaluation on any image size
  dense_predict           Section 3.2: class-score map -> average -> +flip
  scale_jitter, rescale_shorter_side, random_crop_flip, multi_crop    Sections 3.1-3.2, 4.3
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Table 1. Numbers are 3x3 conv layers (output channels); ("1x1", c) is a 1x1 conv;
# "M" is 2x2 max-pool, stride 2; "LRN" is AlexNet's local response normalization.
# ---------------------------------------------------------------------------

CONFIGS = {
    "A":     [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "A-LRN": [64, "LRN", "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B":     [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C":     [64, 64, "M", 128, 128, "M", 256, 256, ("1x1", 256), "M",
              512, 512, ("1x1", 512), "M", 512, 512, ("1x1", 512), "M"],
    "D":     [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E":     [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
              512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
PAPER_PARAMS_MILLIONS = {"A": 133, "A-LRN": 133, "B": 133, "C": 134, "D": 138, "E": 144}   # Table 2


def weight_layers(name):
    """Number of layers with weights: conv layers + 3 fully connected."""
    return sum(1 for v in CONFIGS[name] if v not in ("M", "LRN")) + 3


def _conv_spec(v):
    """-> (kernel size, out channels) for a config entry."""
    return (1, v[1]) if isinstance(v, tuple) else (3, v)


def count_params(name, num_classes=1000, input_size=224, fc=4096, width_div=1):
    """Table 2 from the config alone: sum of (k*k*C_in + 1) * C_out over conv layers, then the fc layers."""
    c_in, total, size = 3, 0, input_size
    for v in CONFIGS[name]:
        if v == "M":
            size //= 2
        elif v != "LRN":
            k, c = _conv_spec(v)
            c //= width_div
            total += (k * k * c_in + 1) * c
            c_in = c
    flat = c_in * size * size
    return total + (flat + 1) * fc + (fc + 1) * fc + (fc + 1) * num_classes


def receptive_field(n_layers, k=3):
    """A stack of n stride-1 k x k convs sees a (n(k-1)+1) x (n(k-1)+1) window of its input."""
    return n_layers * (k - 1) + 1


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class VGG(nn.Module):
    """Section 2.1: 3x3 convs with stride 1 and padding 1 (size preserved), ReLU after every one,
    2x2 max-pooling, then FC-4096, FC-4096, FC-classes (dropout 0.5 on the first two).

    width_div / fc / input_size shrink the network for small images (e.g. CIFAR: input_size=32,
    width_div=4, fc=512). With the defaults it is the paper's network."""

    def __init__(self, name="D", num_classes=1000, input_size=224, fc=4096, width_div=1, init="glorot",
                 dropout=0.5, device=None):
        super().__init__()
        self.name, self.input_size = name, input_size
        layers, c_in, size = [], 3, input_size
        for v in CONFIGS[name]:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2)); size //= 2
            elif v == "LRN":
                layers.append(nn.LocalResponseNorm(5, alpha=5 * 1e-4, beta=0.75, k=2.0))   # AlexNet's (torch scales alpha by 1/n)
            else:
                k, c = _conv_spec(v)
                c //= width_div
                layers += [nn.Conv2d(c_in, c, k, padding=k // 2, device=device), nn.ReLU(inplace=True)]
                c_in = c
        self.features = nn.Sequential(*layers)
        self.final_size, self.final_channels = size, c_in       # 7 x 7 x 512 for the paper's net
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * size * size, fc, device=device), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(fc, fc, device=device), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(fc, num_classes, device=device))
        if device != "meta":
            self.reset(init)

    def reset(self, init):
        """'paper'   : N(0, variance 1e-2) -> std 0.1, as WRITTEN in Section 3.1
           'paper001': N(0, std 0.01), how many re-implementations read it
           'glorot'  : Glorot & Bengio (2010), which the authors say works without pre-training
           Biases are 0 in all cases."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if init == "paper":
                    nn.init.normal_(m.weight, 0.0, 0.1)
                elif init == "paper001":
                    nn.init.normal_(m.weight, 0.0, 0.01)
                elif init == "glorot":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    raise ValueError(init)
                nn.init.zeros_(m.bias)

    def conv_layers(self):
        return [m for m in self.features if isinstance(m, nn.Conv2d)]

    def fc_layers(self):
        return [m for m in self.classifier if isinstance(m, nn.Linear)]

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Section 3.1: initialise a deep net from net A
# ---------------------------------------------------------------------------

@torch.no_grad()
def init_from_A(deep, net_a):
    """'we initialised the first four convolutional layers and the last three fully-connected layers
    with the layers of net A (the intermediate layers were initialised randomly)'.

    A's first four conv layers are 3->64, 64->128, 128->256, 256->256. A deeper net has extra
    layers in between (E: 3->64, 64->64, 64->128, 128->128, 128->256, 256->256, ...), so a layer
    of A can't simply go to the same POSITION. Our reading: each of A's first four layers goes into
    the deep net's next conv layer with the SAME SHAPE (E's 1st, 3rd, 5th, 6th). The paper doesn't
    spell this out. The three fc layers are identical in all configs and are copied directly.
    Returns the list of (A layer index, deep layer index) pairs copied."""
    dst = deep.conv_layers()
    pairs, start = [], 0
    for i, s in enumerate(net_a.conv_layers()[:4]):
        for j in range(start, len(dst)):
            if dst[j].weight.shape == s.weight.shape:
                dst[j].weight.copy_(s.weight); dst[j].bias.copy_(s.bias)
                pairs.append((i, j)); start = j + 1
                break
    for s, d in zip(net_a.fc_layers(), deep.fc_layers()):
        d.weight.copy_(s.weight); d.bias.copy_(s.bias)
    return pairs


# ---------------------------------------------------------------------------
# Section 3.2: dense (fully convolutional) evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def to_fully_convolutional(net):
    """'the first FC layer [becomes] a 7x7 conv. layer, the last two FC layers 1x1 conv. layers'.
    Same weights, just reshaped. On a 224 image the result equals the original net; on a bigger
    image it gives a MAP of class scores, one per position."""
    fc6, fc7, fc8 = net.fc_layers()
    k, c = net.final_size, net.final_channels
    conv6 = nn.Conv2d(c, fc6.out_features, k).to(fc6.weight.device)
    conv6.weight.copy_(fc6.weight.view(fc6.out_features, c, k, k)); conv6.bias.copy_(fc6.bias)
    conv7 = nn.Conv2d(fc7.in_features, fc7.out_features, 1).to(fc7.weight.device)
    conv7.weight.copy_(fc7.weight[:, :, None, None]); conv7.bias.copy_(fc7.bias)
    conv8 = nn.Conv2d(fc8.in_features, fc8.out_features, 1).to(fc8.weight.device)
    conv8.weight.copy_(fc8.weight[:, :, None, None]); conv8.bias.copy_(fc8.bias)
    return nn.Sequential(net.features, conv6, nn.ReLU(), conv7, nn.ReLU(), conv8)


@torch.no_grad()
def dense_predict(fcn, x, flip=True):
    """Class score map -> spatial average ('sum-pooled') -> softmax; averaged with the flipped image."""
    probs = F.softmax(fcn(x).mean((2, 3)), 1)
    if flip:
        probs = (probs + F.softmax(fcn(x.flip(-1)).mean((2, 3)), 1)) / 2
    return probs


# ---------------------------------------------------------------------------
# Sections 3.1, 3.2, 4.3: scales and crops
# ---------------------------------------------------------------------------

def rescale_shorter_side(img, S):
    """Isotropic rescale so the smallest side is S. img: (C, H, W)."""
    _, H, W = img.shape
    scale = S / min(H, W)
    size = (max(S, round(H * scale)), max(S, round(W * scale)))
    return F.interpolate(img[None], size=size, mode="bilinear", align_corners=False)[0]


def random_crop_flip(img, size=224, rng=random):
    _, H, W = img.shape
    i, j = rng.randint(0, H - size), rng.randint(0, W - size)
    out = img[:, i:i + size, j:j + size]
    return out.flip(-1) if rng.random() < 0.5 else out


def scale_jitter(img, s_min=256, s_max=512, size=224, rng=random):
    """Multi-scale training: S ~ U[s_min, s_max] for EACH image, rescale, then a random crop.
    The same object appears at many sizes: augmentation by scale."""
    S = rng.randint(s_min, s_max)
    return random_crop_flip(rescale_shorter_side(img, S), size, rng), S


def multi_crop(img, size=224, grid=5):
    """Section 3.2: a 5x5 regular grid of crops, plus their flips = 50 crops per scale."""
    _, H, W = img.shape
    ys = [round(i * (H - size) / (grid - 1)) for i in range(grid)]
    xs = [round(j * (W - size) / (grid - 1)) for j in range(grid)]
    crops = [img[:, y:y + size, x:x + size] for y in ys for x in xs]
    return torch.stack(crops + [c.flip(-1) for c in crops])
