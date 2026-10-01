"""Zhang, Bengio, Hardt, Recht & Vinyals (2017), "Understanding deep learning requires rethinking
generalization" - the tools of the paper.

  Section 2.1  corrupt_labels, shuffle_pixels, random_pixels, gaussian_images: the randomization tests
  Appendix A   per_image_whitening, center_crop: the CIFAR-10 preprocessing
               SmallInception, SmallAlexNet, MLP: the CIFAR-10 models (Table 1 parameter counts, exactly)
  Section 2.2  rademacher_fit: how well a learner fits random +-1 labels (empirical Rademacher complexity)
  Section 4    finite_sample_network: Theorem 1, a 2-layer ReLU net with 2n + d weights that fits ANY labels
  Section 5    min_norm_interpolant, sgd_least_squares: SGD on a linear model finds the minimum-norm fit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Section 2.1: randomization tests
# ---------------------------------------------------------------------------

def corrupt_labels(y, p, num_classes=10, generator=None):
    """'independently with probability p, the label of each image is corrupted as a uniform random
    class' (the new class may, by chance, equal the old one). p = 1 gives completely random labels.
    The corruption is drawn ONCE and kept fixed across epochs."""
    y = y.clone()
    hit = torch.rand(len(y), generator=generator) < p
    y[hit] = torch.randint(0, num_classes, (int(hit.sum()),), generator=generator)
    return y


def shuffle_pixels(X, generator=None, perm=None):
    """ONE random permutation of the pixel positions, applied to every image (train AND test).
    Returns (shuffled images, permutation) so the same permutation can be reused on the test set."""
    N, C, H, W = X.shape
    if perm is None:
        perm = torch.randperm(H * W, generator=generator)
    return X.flatten(2)[:, :, perm].view(N, C, H, W), perm


def random_pixels(X, generator=None):
    """A DIFFERENT random permutation for each image: no spatial structure left at all."""
    N, C, H, W = X.shape
    flat = X.flatten(2)
    idx = torch.argsort(torch.rand(N, H * W, generator=generator), dim=1)       # one permutation per image
    return torch.gather(flat, 2, idx[:, None, :].expand(N, C, H * W)).view(N, C, H, W)


def gaussian_images(X, generator=None):
    """Images of pure Gaussian noise with the same per-channel mean and std as the dataset."""
    mean = X.mean((0, 2, 3), keepdim=True)
    std = X.std((0, 2, 3), keepdim=True)
    return mean + std * torch.randn(X.shape, generator=generator)


# ---------------------------------------------------------------------------
# Appendix A: preprocessing and models
# ---------------------------------------------------------------------------

def center_crop(X, size=28):
    H = X.shape[-1]
    o = (H - size) // 2
    return X[..., o:o + size, o:o + size]


def per_image_whitening(X):
    """TensorFlow's per_image_whitening: (x - mean) / max(std, 1/sqrt(num_elements)), per image."""
    n = X[0].numel()
    mean = X.mean((1, 2, 3), keepdim=True)
    std = X.std((1, 2, 3), keepdim=True, unbiased=False).clamp_min(1.0 / n ** 0.5)
    return (X - mean) / std


class ConvModule(nn.Module):
    """Figure 3: convolution -> batch norm -> ReLU. (The paper's parameter count includes conv
    biases but no BN parameters, so BN here has no learnable scale/shift.)"""

    def __init__(self, cin, cout, k, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding=k // 2)           # 'SAME' padding
        self.bn = nn.BatchNorm2d(cout, affine=False) if bn else nn.Identity()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    """Figure 3: a 1x1 conv module (ch1 filters) and a 3x3 conv module (ch3), concatenated."""

    def __init__(self, cin, ch1, ch3, bn=True):
        super().__init__()
        self.a, self.b = ConvModule(cin, ch1, 1, bn=bn), ConvModule(cin, ch3, 3, bn=bn)

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x)], 1)


class DownsampleModule(nn.Module):
    """Figure 3: a 3x3 stride-2 conv module (ch3 filters) and a 3x3 stride-2 max-pool, concatenated."""

    def __init__(self, cin, ch3, bn=True):
        super().__init__()
        self.conv = ConvModule(cin, ch3, 3, stride=2, bn=bn)

    def forward(self, x):
        return torch.cat([self.conv(x), F.max_pool2d(x, 3, 2, padding=1)], 1)     # 28 -> 14 -> 7


class SmallInception(nn.Module):
    """Figure 3 (28x28x3 input): Conv(96, 3x3) -> Inc(32+32) -> Inc(32+48) -> Down(80)
    -> Inc(112+48) -> Inc(96+64) -> Inc(80+80) -> Inc(48+96) -> Down(96)
    -> Inc(176+160) -> Inc(176+160) -> global mean pool -> FC 10.   1,649,402 parameters."""

    def __init__(self, bn=True):
        super().__init__()
        layers, c = [ConvModule(3, 96, 3, bn=bn)], 96
        for kind, *args in [("i", 32, 32), ("i", 32, 48), ("d", 80), ("i", 112, 48), ("i", 96, 64), ("i", 80, 80),
                            ("i", 48, 96), ("d", 96), ("i", 176, 160), ("i", 176, 160)]:
            if kind == "i":
                layers.append(InceptionModule(c, *args, bn=bn)); c = sum(args)
            else:
                layers.append(DownsampleModule(c, *args, bn=bn)); c = args[0] + c
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(c, 10)

    def forward(self, x):
        return self.fc(self.body(x).mean((2, 3)))


class SmallAlexNet(nn.Module):
    """Appendix A: two (conv 5x5 -> max-pool 3x3 -> LRN) modules, FC 384, FC 192, linear 10.
    With 64 channels per conv (TensorFlow's CIFAR-10 model) this gives exactly 1,387,786 parameters."""

    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(3, 64, 5, padding=2), nn.Conv2d(64, 64, 5, padding=2)
        self.norm = nn.LocalResponseNorm(4, alpha=0.001 / 9.0 * 4, beta=0.75, k=1.0)
        self.f1, self.f2, self.out = nn.Linear(7 * 7 * 64, 384), nn.Linear(384, 192), nn.Linear(192, 10)

    def forward(self, x):
        x = self.norm(F.max_pool2d(F.relu(self.c1(x)), 3, 2, padding=1))     # 28 -> 14
        x = self.norm(F.max_pool2d(F.relu(self.c2(x)), 3, 2, padding=1))     # 14 -> 7
        x = F.relu(self.f1(x.flatten(1)))
        return self.out(F.relu(self.f2(x)))


def MLP(hidden=1, width=512, inputs=28 * 28 * 3):
    """'MLP 1x512 means one hidden layer with 512 hidden units.' ReLU."""
    layers, n = [nn.Flatten()], inputs
    for _ in range(hidden):
        layers += [nn.Linear(n, width), nn.ReLU()]
        n = width
    return nn.Sequential(*layers, nn.Linear(n, 10))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Section 2.2: Rademacher complexity
# ---------------------------------------------------------------------------

def rademacher_fit(fit_and_predict, X, trials=20, generator=None):
    """Empirical Rademacher complexity of what a LEARNER can do (Eq. 1 with the sup replaced by
    training): for random signs sigma, train on (X, sigma) and measure (1/n) sum sigma_i h(x_i)
    with h in {-1, +1}. 1.0 = the learner can fit any sign pattern -> the bound says nothing."""
    vals = []
    for _ in range(trials):
        sigma = torch.randint(0, 2, (len(X),), generator=generator).float() * 2 - 1
        h = fit_and_predict(X, sigma).sign()
        vals.append((sigma * h).mean().item())
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Section 4, Theorem 1 (proof in Appendix C)
# ---------------------------------------------------------------------------

def finite_sample_network(X, y, generator=None):
    """A depth-2 ReLU network  c(x) = sum_j w_j * relu(<a, x> - b_j)  that equals y_i on every x_i.

    1. Pick a random direction a: the projections z_i = <a, x_i> are all different (almost surely).
    2. Sort them and put the thresholds b_j BETWEEN consecutive z's: b_1 < z_1 < b_2 < z_2 < ...
    3. The n x n matrix A_ij = relu(z_i - b_j) is lower-triangular with a positive diagonal,
       so it is invertible: solve A w = y.
    Weights: a (d) + b (n) + w (n) = 2n + d."""
    n, d = X.shape
    X = X.double()
    a = torch.randn(d, generator=generator, dtype=torch.float64)
    z = X @ a
    order = torch.argsort(z)
    zs = z[order]
    gaps = zs[1:] - zs[:-1]
    b = torch.empty(n, dtype=torch.float64)
    b[0] = zs[0] - 1.0
    b[1:] = zs[:-1] + gaps / 2                                # strictly between z_{j-1} and z_j
    A = torch.relu(zs[:, None] - b[None, :])                  # lower-triangular, positive diagonal
    w = torch.linalg.solve(A, y.double()[order])
    return a, b, w


def finite_sample_predict(a, b, w, X):
    return torch.relu((X.double() @ a)[:, None] - b[None, :]) @ w


# ---------------------------------------------------------------------------
# Section 5: implicit regularization in linear models
# ---------------------------------------------------------------------------

def min_norm_interpolant(X, y):
    """Solve X X^T alpha = y (Eq. 3) and set w = X^T alpha: the kernel trick 'in a roundabout
    fashion'. This w fits the data exactly and has the smallest L2 norm among all exact fits."""
    X, y = X.double(), y.double()
    alpha = torch.linalg.solve(X @ X.T, y)
    return X.T @ alpha


def sgd_least_squares(X, y, epochs=200, lr=None, generator=None):
    """Plain SGD on (1/2)(w.x_i - y_i)^2 starting from w = 0. Every update adds a multiple of
    some x_i, so w always stays in the span of the data: w = X^T alpha."""
    X, y = X.double(), y.double()
    n, d = X.shape
    lr = lr or 0.5 / (X ** 2).sum(1).max().item()
    w = torch.zeros(d, dtype=torch.float64)
    for _ in range(epochs):
        for i in torch.randperm(n, generator=generator):
            w -= lr * (X[i] @ w - y[i]) * X[i]
    return w
