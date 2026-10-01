"""ResNet - He, Zhang, Ren & Sun (2016), "Deep Residual Learning for Image Recognition".

  BasicBlock        Fig. 5 left: two 3x3 convs + shortcut, y = relu(F(x) + x)   (residual=False -> "plain")
  Bottleneck        Fig. 5 right: 1x1 (reduce) -> 3x3 -> 1x1 (expand x4) + shortcut
  Shortcut options  A: identity, zero-padded channels when they grow (no parameters)
                    B: 1x1 projection only when the shape changes, identity elsewhere
                    C: 1x1 projection on every block
  ImageNetResNet    Table 1: 18 / 34 / 50 / 101 / 152 layers (or their plain twins)
  CifarResNet       Section 4.2: 6n+2 layers, 16/32/64 filters, option A (20, 32, 44, 56, 110, 1202)
  count_macs        FLOPs (multiply-adds) as in Table 1. Build big nets inside `with torch.device("meta"):`
                    to get shapes without allocating any memory.
  layer_responses   Fig. 7: std of each 3x3 layer's output (after BN, before the addition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def he_init(m):
    """Weights as in He et al. (2015) [13]: N(0, 2/n) with n = k*k*in_channels (the 'forward' case)."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu"); nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Shortcuts (Section 3.3, "Identity vs. Projection Shortcuts")
# ---------------------------------------------------------------------------

class ZeroPadShortcut(nn.Module):
    """Option A: keep every other pixel (stride 2) and add zero channels. No parameters.
    The new channels get no shortcut signal at all ("no residual learning" for them)."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.stride, self.extra = stride, out_ch - in_ch

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, 0, self.extra)) if self.extra else x


def make_shortcut(in_ch, out_ch, stride, option):
    """Returns the module that carries x around the residual branch."""
    shape_changes = stride != 1 or in_ch != out_ch
    if option == "C" or (option == "B" and shape_changes):
        # Eqn. (2): y = F(x) + Ws x, Ws a 1x1 convolution (followed by BN, as in the released models)
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
    if shape_changes:                                        # option A
        return ZeroPadShortcut(in_ch, out_ch, stride)
    return nn.Identity()                                     # Eqn. (1): y = F(x) + x


# ---------------------------------------------------------------------------
# Building blocks (Fig. 5)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """conv3x3 - BN - ReLU - conv3x3 - BN, then (+ shortcut), then ReLU.
    'BN right after each convolution and before activation' (Section 3.4).
    residual=False removes the addition: the 'plain' network with identical layers."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, option="A", residual=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.residual = residual
        self.shortcut = make_shortcut(in_ch, out_ch, stride, option) if residual else None
        self.responses = []                                  # filled when recording (Fig. 7)
        self.record = False

    def forward(self, x):
        a = self.bn1(self.conv1(x))
        f = self.bn2(self.conv2(F.relu(a)))                  # F(x)
        if self.record:
            self.responses = [a.std().item(), f.std().item()]
        return F.relu(f + self.shortcut(x)) if self.residual else F.relu(f)


class Bottleneck(nn.Module):
    """1x1 conv reduces the channels (e.g. 256 -> 64), 3x3 conv works on the small tensor,
    1x1 conv restores them (64 -> 256). Same cost as a basic block, but much deeper nets.
    As in the original paper/Caffe models, the stride-2 downsampling is in the FIRST 1x1 conv.
    (torchvision's "ResNet v1.5" moves it to the 3x3 conv: a bit more accurate, 4.1 vs 3.8 GFLOPs.)"""
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, option="B", residual=True):
        super().__init__()
        out_ch = mid_ch * 4
        self.conv1, self.bn1 = nn.Conv2d(in_ch, mid_ch, 1, stride, bias=False), nn.BatchNorm2d(mid_ch)
        self.conv2, self.bn2 = nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_ch)
        self.conv3, self.bn3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        self.residual = residual
        self.shortcut = make_shortcut(in_ch, out_ch, stride, option) if residual else None
        self.responses, self.record = [], False

    def forward(self, x):
        a = self.bn1(self.conv1(x))
        b = self.bn2(self.conv2(F.relu(a)))
        f = self.bn3(self.conv3(F.relu(b)))
        if self.record:
            self.responses = [a.std().item(), b.std().item(), f.std().item()]
        return F.relu(f + self.shortcut(x)) if self.residual else F.relu(f)


# ---------------------------------------------------------------------------
# Table 1: ImageNet networks
# ---------------------------------------------------------------------------

IMAGENET = {18: (BasicBlock, [2, 2, 2, 2]), 34: (BasicBlock, [3, 4, 6, 3]),
            50: (Bottleneck, [3, 4, 6, 3]), 101: (Bottleneck, [3, 4, 23, 3]), 152: (Bottleneck, [3, 8, 36, 3])}
PAPER_GFLOPS = {18: 1.8, 34: 3.6, 50: 3.8, 101: 7.6, 152: 11.3}


class ImageNetResNet(nn.Module):
    """conv1 7x7/2 (64) -> 3x3 max-pool/2 -> conv2_x (56x56) -> conv3_x (28) -> conv4_x (14) -> conv5_x (7)
    -> global average pool -> 1000-d fc. Downsampling by stride 2 in conv3_1, conv4_1, conv5_1."""

    def __init__(self, depth=34, residual=True, option="B", num_classes=1000):
        super().__init__()
        block, counts = IMAGENET[depth]
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)
        stages, in_ch = [], 64
        for i, (width, n) in enumerate(zip((64, 128, 256, 512), counts)):
            for j in range(n):
                stride = 2 if (j == 0 and i > 0) else 1
                b = block(in_ch, width, stride, option, residual)
                stages.append(b)
                in_ch = width * block.expansion
        self.blocks = nn.Sequential(*stages)
        self.fc = nn.Linear(in_ch, num_classes)
        self.apply(he_init)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.blocks(x)
        return self.fc(x.mean((2, 3)))                       # global average pooling


# ---------------------------------------------------------------------------
# Section 4.2: CIFAR-10 networks
# ---------------------------------------------------------------------------

class CifarResNet(nn.Module):
    """3x3 conv (16) -> 2n layers at 32x32 (16 filters) -> 2n at 16x16 (32) -> 2n at 8x8 (64)
    -> global average pool -> 10-way fc. 6n+2 weight layers. Option A shortcuts (as in the paper)."""

    def __init__(self, n=3, residual=True, option="A", num_classes=10, widths=(16, 32, 64)):
        super().__init__()
        self.depth = 6 * n + 2
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        blocks, in_ch = [], widths[0]
        for i, w in enumerate(widths):
            for j in range(n):
                blocks.append(BasicBlock(in_ch, w, 2 if (j == 0 and i > 0) else 1, option, residual))
                in_ch = w
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_ch, num_classes)
        self.apply(he_init)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.fc(self.blocks(x).mean((2, 3)))


def weight_layers(model):
    """Convs on the main path + fc (shortcut projections are not counted, as in the paper)."""
    n = 1 + 1                                                # conv1 + fc
    for b in model.blocks:
        n += 3 if isinstance(b, Bottleneck) else 2
    return n


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Table 1 FLOPs: multiply-adds of every conv and fc layer, measured on meta tensors (no memory)
# ---------------------------------------------------------------------------

@torch.no_grad()
def count_macs(model, input_size=224):
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def fc_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(fc_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    model.eval()(torch.empty(1, 3, input_size, input_size, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    return total[0]


# ---------------------------------------------------------------------------
# Fig. 7: responses of the residual functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def layer_responses(model, x):
    """Std of every 3x3 layer's output after BN (before ReLU / addition), in order (Fig. 7 top).
    The paper's claim: residual functions are generally SMALL, smaller in deeper ResNets."""
    model.eval()
    out = []
    for b in model.blocks:
        b.record = True
    model(x)
    for b in model.blocks:
        out += b.responses
        b.record = False
    return out
