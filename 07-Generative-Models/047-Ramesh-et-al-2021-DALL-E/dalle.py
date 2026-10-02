"""Zero-Shot Text-to-Image Generation (DALL-E; Ramesh, Pavlov, Goh, Gray, Voss, Radford, Chen & Sutskever,
ICML 2021).

  Stage 1 (dVAE): compress a 256x256x3 image into a 32x32 grid of tokens from a codebook of K = 8192 (192x fewer
           positions). Trained with the gumbel-softmax relaxation (temperature tau annealed 1 -> 1/16), a uniform
           prior, KL weight beta annealed 0 -> 6.6, and a logit-Laplace likelihood on pixels (Appendix A.3):
               f(x | mu, b) = 1 / (2 b x (1 - x)) * exp(-|logit(x) - mu| / b),   x in (0, 1)        (Eq. 2)
           with pixels mapped to (eps, 1 - eps) by phi(x) = (1 - 2 eps) x / 255 + eps, eps = 0.1   (Eq. 3)
  Stage 2 (prior): concatenate up to 256 BPE text tokens and the 1024 image tokens; one decoder-only transformer
           models the single stream autoregressively. Learned padding token per text position; image tokens get
           row + column embeddings; image-to-image attention uses row, column and (last layer) convolutional masks;
           loss = 1/8 text cross-entropy + 7/8 image cross-entropy (each normalised by its token count).
  Overall: an evidence lower bound (Eq. 1)  ln p(x, y) >= E_{q(z|x)}[ln p(x | y, z)] - beta KL(q(y, z | x) || p(y, z)).
  Scale tricks: per-resblock gradient scaling for 16-bit training (Section 2.4), PowerSGD gradient compression with
           compression rate 1 - 5r / (8 d_model) (Section 2.5), reranking samples with a contrastive model (2.6).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.1


# ---------------------------------------------------------------------------------------------------- pixels
def phi(x255):
    """Eq. 3: [0, 255] -> (eps, 1 - eps)."""
    return (1 - 2 * EPS) * x255 / 255 + EPS


def phi_inv(x):
    return ((x - EPS) / (1 - 2 * EPS) * 255).clamp(0, 255)


def logit_laplace_log_prob(x, mu, log_b):
    """Eq. 2, the logarithm: -log(2 b x (1 - x)) - |logit(x) - mu| / b, for x in (0, 1)."""
    b = log_b.exp()
    return -torch.log(2 * b * x * (1 - x)) - (torch.logit(x) - mu).abs() / b


# ---------------------------------------------------------------------------------------------------- dVAE
class ResBlock(nn.Module):
    """Bottleneck-style resblock; the residual branch is multiplied by a small constant for stable init (2.1)."""

    def __init__(self, cin, cout, post_gain=0.1):
        super().__init__()
        h = cout // 4
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.res = nn.Sequential(nn.ReLU(), nn.Conv2d(cin, h, 3, padding=1), nn.ReLU(), nn.Conv2d(h, h, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(h, h, 3, padding=1), nn.ReLU(), nn.Conv2d(h, cout, 1))
        self.post_gain = post_gain

    def forward(self, x):
        return self.skip(x) + self.post_gain * self.res(x)


class DVAE(nn.Module):
    """Encoder: 7x7 conv -> [resblocks, max-pool] x log2(down) -> resblocks -> 1x1 conv to K logits.
    Decoder: 1x1 conv from K one-hot (or relaxed) codes -> [resblocks, nearest upsample] -> 1x1 conv to 6 maps
    (mu and ln b of the logit-Laplace for R, G, B)."""

    def __init__(self, K=8192, C=3, width=64, down=8, blocks=1):
        super().__init__()
        self.K = K
        n = int(math.log2(down))
        enc = [nn.Conv2d(C, width, 7, padding=3)]
        for _ in range(n):
            enc += [ResBlock(width, width) for _ in range(blocks)] + [nn.MaxPool2d(2)]
        enc += [ResBlock(width, width) for _ in range(blocks)] + [nn.ReLU(), nn.Conv2d(width, K, 1)]
        self.encoder = nn.Sequential(*enc)
        dec = [nn.Conv2d(K, width, 1)]
        for _ in range(n):
            dec += [ResBlock(width, width) for _ in range(blocks)] + [nn.Upsample(scale_factor=2, mode="nearest")]
        dec += [ResBlock(width, width) for _ in range(blocks)] + [nn.ReLU(), nn.Conv2d(width, 2 * C, 1)]
        self.decoder = nn.Sequential(*dec)

    def logits(self, x):
        return self.encoder(x)                                                      # B, K, h, w

    def tokens(self, x):
        """Stage 2 uses argmax tokens (no gumbel noise; footnote 6)."""
        return self.logits(x).argmax(1)

    def decode_tokens(self, z):
        return self.decoder(F.one_hot(z, self.K).permute(0, 3, 1, 2).float())

    def elbo_terms(self, x, tau=1.0, hard=False):
        """x in (eps, 1 - eps). Returns (reconstruction log-likelihood per image, KL to the uniform prior per image,
        the relaxed codes)."""
        logits = self.logits(x)
        soft = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)                  # relaxed one-hot (2.1)
        out = self.decoder(soft)
        C = x.shape[1]
        rec = logit_laplace_log_prob(x, out[:, :C], out[:, C:].clamp(-5, 5)).flatten(1).sum(1)
        return rec, kl_to_uniform(logits).flatten(1).sum(1), soft

    def reconstruct(self, x):
        """Appendix A.3: x_hat = phi^-1(sigmoid(mu)), ignoring ln b."""
        out = self.decode_tokens(self.tokens(x))
        return torch.sigmoid(out[:, :x.shape[1]])


def kl_to_uniform(logits):
    """KL(q || Uniform(K)) per position = sum_k q_k log q_k + log K."""
    logq = F.log_softmax(logits, 1)
    return (logq.exp() * logq).sum(1) + math.log(logits.shape[1])


def dvae_loss(model, x, tau, beta):
    """Negative relaxed ELB divided by the number of pixel values (so the KL weight is effectively beta / 192 for
    the paper's 256x256x3 -> 32x32 shapes)."""
    rec, kl, _ = model.elbo_terms(x, tau)
    return (-(rec - beta * kl) / x[0].numel()).mean()


def cosine_schedule(step, total, start, end):
    """All dVAE schedules use a cosine shape (Appendix A.2)."""
    t = min(step / total, 1.0)
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))


def codebook_perplexity(tokens, K):
    """exp(entropy) of code usage: K if all codes are used equally, 1 if only one is used."""
    p = torch.bincount(tokens.flatten(), minlength=K).float()
    p = p[p > 0] / p.sum()
    return math.exp(-(p * p.log()).sum().item())


# ---------------------------------------------------------------------------------------------------- attention masks
def image_mask(kind, H, W, kernel=3):
    """(H*W) x (H*W) boolean: image position i (raster order) may attend to image position j.
      row:    the previous W + 1 tokens in raster order (ends at the same column of the previous row) + itself
      column: the same column in the current and previous rows
      conv:   causal kernel x kernel neighbourhood (rows above within reach, same row to the left) + itself"""
    N = H * W
    i = torch.arange(N)
    r, c = i // W, i % W
    causal = i[None, :] <= i[:, None]
    if kind == "row":
        return causal & (i[:, None] - i[None, :] <= W)
    if kind == "column":
        return causal & (c[:, None] == c[None, :])
    if kind == "conv":
        h = kernel // 2
        dr, dc = r[:, None] - r[None, :], (c[:, None] - c[None, :]).abs()
        return causal & (dr <= h) & (dc <= h)
    if kind == "dense":
        return causal
    raise ValueError(kind)


def full_mask(kind, T, H, W):
    """(T + H W)^2 boolean mask: text is causal, every image token sees all text, image-image uses `kind`."""
    N = T + H * W
    m = torch.zeros(N, N, dtype=torch.bool)
    m[:T, :T] = torch.tril(torch.ones(T, T, dtype=torch.bool))
    m[T:, :T] = True
    m[T:, T:] = image_mask(kind, H, W)
    return m


def layer_kinds(n_layers):
    """Appendix B.1: layer i in 1..n-1 uses 'column' if (i - 2) mod 4 == 0 else 'row'; the last layer 'conv'."""
    return ["column" if (i - 2) % 4 == 0 else "row" for i in range(1, n_layers)] + ["conv"]


# ---------------------------------------------------------------------------------------------------- transformer
class Block(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, allowed):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=~allowed, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class DALLE(nn.Module):
    """Decoder-only transformer over [text (T positions, learned per-position padding) ; image (H*W tokens)].
    Image embeddings = token + row + column embeddings (Figure 10). Input is shifted: position t predicts token t+1;
    a start-of-image embedding stands in front of the first image token."""

    def __init__(self, text_vocab, image_vocab, T, H, W, d=256, layers=8, heads=8, dropout=0.0, sparse=True):
        super().__init__()
        self.T, self.H, self.W, self.text_vocab, self.image_vocab = T, H, W, text_vocab, image_vocab
        self.text_emb = nn.Embedding(text_vocab, d)
        self.pad_emb = nn.Parameter(torch.randn(T, d) * 0.02)                      # one padding token per position
        self.text_pos = nn.Parameter(torch.randn(T, d) * 0.02)
        self.image_emb = nn.Embedding(image_vocab, d)
        self.row_emb, self.col_emb = nn.Embedding(H, d), nn.Embedding(W, d)
        self.boi = nn.Parameter(torch.randn(d) * 0.02)                              # start-of-image
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)])
        kinds = layer_kinds(layers) if sparse else ["dense"] * layers
        self.kinds = kinds
        for k in set(kinds):
            self.register_buffer(f"mask_{k}", full_mask(k, T, H, W))
        self.ln = nn.LayerNorm(d)
        self.text_head, self.image_head = nn.Linear(d, text_vocab), nn.Linear(d, image_vocab)

    def embed(self, text, text_len, image):
        """text: (B, T) ids (anything beyond text_len is replaced by padding embeddings); image: (B, L <= H*W)."""
        B = text.shape[0]
        pos = torch.arange(self.T)
        is_pad = pos[None, :] >= text_len[:, None]
        te = torch.where(is_pad[..., None], self.pad_emb.expand(B, -1, -1), self.text_emb(text)) + self.text_pos
        L = image.shape[1]
        idx = torch.arange(self.H * self.W)
        img_pos = self.row_emb(idx // self.W) + self.col_emb(idx % self.W)          # H*W, d
        prev = torch.cat([self.boi.expand(B, 1, -1), self.image_emb(image[:, :-1]) if L > 1 else
                          self.boi.new_zeros(B, 0, self.boi.shape[0])], 1)          # shifted: [boi, z_1, ..]
        ie = prev + img_pos[:L]
        return te, ie

    def forward(self, text, text_len, image):
        """Returns (text logits predicting text[1:], image logits predicting every image token)."""
        te, ie = self.embed(text, text_len, image)
        x = torch.cat([te, ie], 1)
        n = x.shape[1]
        for blk, k in zip(self.blocks, self.kinds):
            x = blk(x, getattr(self, f"mask_{k}")[:n, :n])
        x = self.ln(x)
        return self.text_head(x[:, :self.T - 1]), self.image_head(x[:, self.T:])

    def loss(self, text, text_len, image, w_text=1 / 8, w_image=7 / 8):
        """Section 2.2: cross-entropies normalised by their token counts, weighted 1/8 (text) and 7/8 (image).
        Padding positions are not predicted."""
        lt, li = self(text, text_len, image)
        tgt = text[:, 1:]
        valid = torch.arange(1, self.T)[None, :] < text_len[:, None]
        ce_text = F.cross_entropy(lt.reshape(-1, self.text_vocab), tgt.reshape(-1), reduction="none")
        ce_text = (ce_text * valid.flatten()).sum() / valid.sum().clamp(min=1)
        ce_image = F.cross_entropy(li.reshape(-1, self.image_vocab), image.reshape(-1))
        return w_text * ce_text + w_image * ce_image, ce_text.item(), ce_image.item()

    @torch.no_grad()
    def generate(self, text, text_len, prefix=None, temperature=1.0, top_k=None):
        """Sample the H*W image tokens one by one given the caption (and optionally the first image tokens, for
        Section 3.3's image-to-image completion)."""
        B, N = text.shape[0], self.H * self.W
        image = torch.zeros(B, N, dtype=torch.long)
        start = 0
        if prefix is not None:
            image[:, :prefix.shape[1]] = prefix
            start = prefix.shape[1]
        for t in range(start, N):
            _, li = self(text, text_len, image[:, :t + 1])
            logits = li[:, t] / max(temperature, 1e-6)
            if top_k:
                v, _ = logits.topk(top_k, -1)
                logits = logits.masked_fill(logits < v[:, -1:], -float("inf"))
            image[:, t] = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
        return image


def rerank(candidates, scores, k):
    """Section 2.6: keep the top-k of N candidates under a (contrastive) image-text score."""
    order = torch.as_tensor(scores).argsort(descending=True)[:k]
    return [candidates[i] for i in order.tolist()]


def transformer_params(d, layers, text_vocab=16384, image_vocab=8192, T=256, H=32, W=32):
    """Approximate parameter count: 12 d^2 per layer (attention 4 d^2 + MLP 8 d^2) + embeddings and heads."""
    return 12 * d * d * layers + d * (2 * (text_vocab + image_vocab) + 2 * T + H + W)


# ---------------------------------------------------------------------------------------------------- scale tricks
def powersgd_compression_rate(r, d_model):
    """Section 2.5: rate = 1 - 5 r / (8 d_model)."""
    return 1 - 5 * r / (8 * d_model)


class PowerSGD:
    """Rank-r gradient compression with error feedback (Vogels et al. 2019), with the paper's simplification of a
    FIXED random Q (Section 2.5): P = orth(M Q), M_hat = P (M^T P)^T; the residual is carried to the next step."""

    def __init__(self, shape, r, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.Q = torch.randn(shape[1], r, generator=g)
        self.error = torch.zeros(shape)

    def compress(self, grad):
        M = grad + self.error
        P, _ = torch.linalg.qr(M @ self.Q)                                          # orthonormal columns
        Qn = M.T @ P
        approx = P @ Qn.T
        self.error = M - approx
        return approx, P.numel() + Qn.numel()


def fp16_underflow_fraction(grads):
    """Fraction of nonzero values that become exactly 0 when cast to float16 (smallest subnormal ~6e-8)."""
    nz = grads != 0
    return ((grads.half() == 0) & nz).float().sum().item() / max(nz.sum().item(), 1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
