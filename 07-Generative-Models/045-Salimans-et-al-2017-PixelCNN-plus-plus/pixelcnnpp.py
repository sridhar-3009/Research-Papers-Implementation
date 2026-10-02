"""PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications
(Salimans, Karpathy, Chen & Kingma, ICLR 2017).

  1) discretized logistic mixture likelihood per sub-pixel (Eqs. 1-2):
       P(x | pi, mu, s) = sum_i pi_i [ sigmoid((x + 0.5 - mu_i)/s_i) - sigmoid((x - 0.5 - mu_i)/s_i) ]
     with x - 0.5 -> -inf at x = 0 and x + 0.5 -> +inf at x = 255 (the edges keep the tails);
  2) conditioning on whole pixels: one mixture indicator per pixel, channels predicted in order with linearly
     coupled means (Eq. 3):  mu_g += alpha * r,  mu_b += beta * r + gamma * g
     (the paper prints 'gamma * b' in Eq. 3; the released code, and common sense, use g);
  3) downsampling (stride-2 convolutions) instead of dilation;
  4) long short-cut connections between corresponding layers of the down and up passes (U-net style);
  5) dropout on the residual path of each gated ResNet layer.
  Like the released code, pixel values live in [-1, 1]: value k in 0..255 sits at 2k/255 - 1, bin half-width 1/255.
  The network is the two-stream (downward 'u' + downward-and-rightward 'ul') causal architecture: every output
  pixel sees exactly the pixels above it and to its left, with no blind spot.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HALF_BIN = 1.0 / 255.0                                                           # half the width of one 8-bit bin


def to_pm1(k):
    """8-bit integers 0..255 -> centres in [-1, 1]."""
    return k.float() * 2 / 255 - 1


# ---------------------------------------------------------------------------------------------------- likelihood
def _log_bin_prob(x, mu, log_s):
    """log P(value x) under one discretized logistic(mu, s), stable, with the edge cases. x, mu, log_s broadcast."""
    centred = x - mu
    inv_s = torch.exp(-log_s)
    plus_in, min_in = inv_s * (centred + HALF_BIN), inv_s * (centred - HALF_BIN)
    cdf_delta = torch.sigmoid(plus_in) - torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)                                  # log sigmoid(plus_in), for x = 0
    log_one_minus_cdf_min = -F.softplus(min_in)                                   # log(1 - sigmoid(min_in)), x = 255
    mid_in = inv_s * centred                                                       # when the bin mass underflows:
    log_pdf_mid = mid_in - log_s - 2 * F.softplus(mid_in)                          # log density * bin width
    inner = torch.where(cdf_delta > 1e-5, torch.log(cdf_delta.clamp_min(1e-12)), log_pdf_mid - math.log(127.5))
    return torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, inner))


def n_params(C, K):
    """Network outputs per pixel: K mixture logits + per channel (mean, log-scale) + C(C-1)/2 coupling coefficients."""
    return K + K * C * 2 + K * (C * (C - 1) // 2)


def split_params(l, C, K):
    """l: (B, n_params, H, W) -> logits (B,K,H,W), means (B,C,K,H,W), log_scales, coeffs (B,C(C-1)/2,K,H,W)."""
    B, _, H, W = l.shape
    logit_pi = l[:, :K]
    rest = l[:, K:].view(B, -1, K, H, W)
    means, log_s = rest[:, :C], rest[:, C:2 * C].clamp(min=-7.0)
    coeffs = torch.tanh(rest[:, 2 * C:])
    return logit_pi, means, log_s, coeffs


def discretized_mix_logistic_log_prob(x, l, K):
    """log p(x) per pixel for x in [-1, 1] of shape (B, C, H, W), network output l (B, n_params(C, K), H, W).
    One mixture indicator per pixel; channel c's mean is shifted linearly by the earlier channels (Eq. 3)."""
    B, C, H, W = x.shape
    logit_pi, means, log_s, coeffs = split_params(l, C, K)
    xk = x.unsqueeze(2)                                                            # B, C, 1, H, W
    m = [means[:, 0]]
    if C == 3:
        m.append(means[:, 1] + coeffs[:, 0] * xk[:, 0])                             # mu_g + alpha r
        m.append(means[:, 2] + coeffs[:, 1] * xk[:, 0] + coeffs[:, 2] * xk[:, 1])   # mu_b + beta r + gamma g
    m = torch.stack(m, 1)                                                          # B, C, K, H, W
    log_probs = _log_bin_prob(xk, m, log_s).sum(1)                                 # B, K, H, W (channels multiply)
    return torch.logsumexp(log_probs + F.log_softmax(logit_pi, 1), 1)              # B, H, W


def sample_discretized_mix_logistic(l, C, K):
    """One sample per pixel: pick a component, then sample r, g, b in order with the coupled means."""
    logit_pi, means, log_s, coeffs = split_params(l, C, K)
    B, _, H, W = l.shape
    k = torch.distributions.Categorical(logits=logit_pi.permute(0, 2, 3, 1)).sample()    # B, H, W
    pick = lambda t: t.gather(1, k[:, None]).squeeze(1) if t.dim() == 4 else \
        t.gather(2, k[:, None, None].expand(-1, t.shape[1], 1, -1, -1)).squeeze(2)
    mu, ls, co = pick(means), pick(log_s), pick(coeffs) if C == 3 else None
    u = torch.rand(B, C, H, W).clamp(1e-5, 1 - 1e-5)
    v = mu + ls.exp() * (u.log() - (1 - u).log())                                  # logistic noise
    out = [v[:, 0].clamp(-1, 1)]
    if C == 3:
        out.append((v[:, 1] + co[:, 0] * out[0]).clamp(-1, 1))
        out.append((v[:, 2] + co[:, 1] * out[0] + co[:, 2] * out[1]).clamp(-1, 1))
    x = torch.stack(out, 1)
    return to_pm1(torch.round((x + 1) * 127.5))                                    # snap to the 256 levels


def softmax_log_prob(x, logits):
    """Ablation 3.4.1: a 256-way softmax per sub-pixel. logits (B, C, 256, H, W), x in [-1, 1]."""
    idx = torch.round((x + 1) * 127.5).long()
    return F.log_softmax(logits, 2).gather(2, idx.unsqueeze(2)).squeeze(2).sum(1)  # B, H, W


def dequantized_log_density(x_noisy, l, K):
    """Ablation 3.4.2: treat the dequantized pixel x + u (u ~ U(-bin/2, bin/2)) as continuous and score it with
    the CONTINUOUS logistic mixture density (no rounding), measured per unit of 8-bit intensity.
    E_u[this] is a lower bound on the discrete log-likelihood (Jensen)."""
    B, C, H, W = x_noisy.shape
    logit_pi, means, log_s, coeffs = split_params(l, C, K)
    xk = x_noisy.unsqueeze(2)
    m = [means[:, 0]]
    if C == 3:
        m.append(means[:, 1] + coeffs[:, 0] * xk[:, 0])
        m.append(means[:, 2] + coeffs[:, 1] * xk[:, 0] + coeffs[:, 2] * xk[:, 1])
    m = torch.stack(m, 1)
    z = (xk - m) * torch.exp(-log_s)
    log_pdf = z - log_s - 2 * F.softplus(z) + math.log(2 / 256)                     # density per 8-bit unit
    return torch.logsumexp(log_pdf.sum(1) + F.log_softmax(logit_pi, 1), 1)


def bits_per_dim(log_prob_sum, n_dims):
    return -log_prob_sum / (n_dims * math.log(2))


# ---------------------------------------------------------------------------------------------------- network
def concat_elu(x):
    return F.elu(torch.cat([x, -x], 1))


class DownShiftedConv(nn.Module):
    """Sees the rows ABOVE (kernel 2 x 3, padded so the output never sees its own row): the 'u' stream."""

    def __init__(self, cin, cout, k=(2, 3), stride=1):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(cin, cout, k, stride)

    def forward(self, x):
        kh, kw = self.k
        return self.conv(F.pad(x, ((kw - 1) // 2, (kw - 1) // 2, kh - 1, 0)))


class DownRightShiftedConv(nn.Module):
    """Sees rows above and pixels to the LEFT (kernel 2 x 2): the 'ul' stream."""

    def __init__(self, cin, cout, k=(2, 2), stride=1):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(cin, cout, k, stride)

    def forward(self, x):
        kh, kw = self.k
        return self.conv(F.pad(x, (kw - 1, 0, kh - 1, 0)))


class DownShiftedDeconv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(cin, cout, (2, 3), 2, output_padding=1)

    def forward(self, x):
        y = self.deconv(x)
        return y[:, :, :-1, 1:-1]                                                  # crop to keep causality


class DownRightShiftedDeconv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(cin, cout, (2, 2), 2, output_padding=1)

    def forward(self, x):
        y = self.deconv(x)
        return y[:, :, :-1, :-1]


def down_shift(x):
    return F.pad(x, (0, 0, 1, 0))[:, :, :-1]


def right_shift(x):
    return F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]


class GatedResnet(nn.Module):
    """Section 2.5: x + gated(conv(dropout(concat_elu(conv(concat_elu(x)) [+ skip])))), gate = a * sigmoid(b)."""

    def __init__(self, nf, conv, skip=0, dropout=0.5, cond=0):
        super().__init__()
        self.c1 = conv(2 * nf, nf)
        self.skip = nn.Conv2d(2 * skip * nf, nf, 1) if skip else None
        self.drop = nn.Dropout(dropout)
        self.c2 = conv(2 * nf, 2 * nf)
        self.cond = nn.Linear(cond, 2 * nf, bias=False) if cond else None           # class-conditional bias (3.2)

    def forward(self, x, a=None, h=None):
        c = self.c1(concat_elu(x))
        if a is not None and self.skip is not None:
            c = c + self.skip(concat_elu(a))
        c = self.c2(self.drop(concat_elu(c)))
        if h is not None and self.cond is not None:
            c = c + self.cond(h)[:, :, None, None]
        a_, b_ = c.chunk(2, 1)
        return x + a_ * torch.sigmoid(b_)


class PixelCNNpp(nn.Module):
    """3 resolutions down (nr_resnet gated layers each, stride-2 between) and 3 up, with every down-pass layer's
    output fed to the matching up-pass layer (the long short-cuts of Figure 2). shortcuts=False removes them
    (Section 3.4.3); downsample=False keeps full resolution throughout (Section 3.3's 'small' models)."""

    def __init__(self, C=3, nr_resnet=5, nf=160, K=10, dropout=0.5, shortcuts=True, downsample=True, n_classes=0):
        super().__init__()
        self.C, self.K, self.nr_resnet, self.shortcuts, self.downsample = C, K, nr_resnet, shortcuts, downsample
        cond = n_classes
        dc = lambda i, o: DownShiftedConv(i, o)
        drc = lambda i, o: DownRightShiftedConv(i, o)
        n_levels = 3
        self.u_init = DownShiftedConv(C + 1, nf, (2, 3))
        self.ul_init = nn.ModuleList([DownShiftedConv(C + 1, nf, (1, 3)), DownRightShiftedConv(C + 1, nf, (2, 1))])
        self.down_u = nn.ModuleList([nn.ModuleList([GatedResnet(nf, dc, 0, dropout, cond) for _ in range(nr_resnet)])
                                     for _ in range(n_levels)])
        self.down_ul = nn.ModuleList([nn.ModuleList([GatedResnet(nf, drc, 1, dropout, cond) for _ in range(nr_resnet)])
                                      for _ in range(n_levels)])
        sk = 1 if shortcuts else 0
        up_n = lambda lvl: nr_resnet + (0 if lvl == 0 else 1)
        self.up_u = nn.ModuleList([nn.ModuleList([GatedResnet(nf, dc, sk, dropout, cond) for _ in range(up_n(l))])
                                   for l in range(n_levels)])
        self.up_ul = nn.ModuleList([nn.ModuleList([GatedResnet(nf, drc, 1 + sk, dropout, cond) for _ in range(up_n(l))])
                                    for l in range(n_levels)])
        stride = 2 if downsample else 1
        self.down_u_s = nn.ModuleList([DownShiftedConv(nf, nf, (2, 3), stride) for _ in range(2)])
        self.down_ul_s = nn.ModuleList([DownRightShiftedConv(nf, nf, (2, 2), stride) for _ in range(2)])
        self.up_u_s = nn.ModuleList([DownShiftedDeconv(nf, nf) if downsample else DownShiftedConv(nf, nf)
                                     for _ in range(2)])
        self.up_ul_s = nn.ModuleList([DownRightShiftedDeconv(nf, nf) if downsample else DownRightShiftedConv(nf, nf)
                                      for _ in range(2)])
        self.out = nn.Conv2d(nf, n_params(C, K), 1)

    def forward(self, x, h=None):
        ones = torch.ones_like(x[:, :1])                                           # marks real pixels vs padding
        xp = torch.cat([x, ones], 1)
        u = [down_shift(self.u_init(xp))]
        ul = [down_shift(self.ul_init[0](xp)) + right_shift(self.ul_init[1](xp))]
        for lvl in range(3):                                                       # ---- down pass
            for k in range(self.nr_resnet):
                u.append(self.down_u[lvl][k](u[-1], h=h))
                ul.append(self.down_ul[lvl][k](ul[-1], a=u[-1], h=h))
            if lvl < 2:
                u.append(self.down_u_s[lvl](u[-1]))
                ul.append(self.down_ul_s[lvl](ul[-1]))
        uu, uul = u.pop(), ul.pop()
        for lvl in range(3):                                                       # ---- up pass
            for k in range(len(self.up_u[lvl])):
                su, sul = u.pop(), ul.pop()                                        # the matching down-pass layer
                if self.shortcuts:
                    uu = self.up_u[lvl][k](uu, a=su, h=h)
                    uul = self.up_ul[lvl][k](uul, a=torch.cat([uu, sul], 1), h=h)
                else:
                    uu = self.up_u[lvl][k](uu, h=h)
                    uul = self.up_ul[lvl][k](uul, a=uu, h=h)
            if lvl < 2:
                uu, uul = self.up_u_s[lvl](uu), self.up_ul_s[lvl](uul)
        return self.out(F.elu(uul))

    def log_prob(self, x, h=None):
        """Sum of per-pixel log-probabilities, per image (nats)."""
        return discretized_mix_logistic_log_prob(x, self(x, h), self.K).flatten(1).sum(1)

    @torch.no_grad()
    def sample(self, n, H, W, h=None):
        x = torch.zeros(n, self.C, H, W)
        for i in range(H):
            for j in range(W):
                l = self(x, h)
                x[:, :, i, j] = sample_discretized_mix_logistic(l[:, :, i:i + 1, j:j + 1], self.C, self.K)[:, :, 0, 0]
        return x


def count_params(m):
    return sum(p.numel() for p in m.parameters())
