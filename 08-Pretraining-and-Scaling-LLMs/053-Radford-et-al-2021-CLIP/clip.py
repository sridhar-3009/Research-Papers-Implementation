"""Learning Transferable Visual Models From Natural Language Supervision (CLIP; Radford, Kim, Hallacy, Ramesh, Goh,
Agarwal, Sastry, Askell, Mishkin, Clark, Krueger & Sutskever, ICML 2021).

  Data: WIT, 400M (image, text) pairs from the internet (500k queries, up to 20k pairs each).
  Model: an image encoder (modified ResNet with attention pooling, or ViT) and a text encoder (GPT-2-style
    Transformer, 63M, 12 x 512, 8 heads, BPE 49,152, context 76; the feature is the top layer at [EOS]), each followed
    by a LINEAR projection into a shared embedding space.
  Objective (Figure 3): for a batch of N pairs, logits = (L2-normalised I_e)(L2-normalised T_e)^T * exp(t); symmetric
    cross-entropy with the diagonal as targets (InfoNCE / multi-class N-pair loss). The temperature is learned as a
    log-parameterised scalar, initialised to 1/0.07, and clipped so the logit scale never exceeds 100.
    Batch 32,768; 32 epochs; AdamW; cosine schedule.
  Zero-shot classification: embed 'A photo of a {label}.' for every class; the text encoder acts as a hypernetwork
    producing the weights of a linear classifier (no bias, normalised weights). Prompt ensembling averages 80
    templates IN EMBEDDING SPACE (+3.5% on ImageNet; with prompt engineering ~+5%).
  Evaluation helpers: linear probes (logistic regression on frozen features), and 'effective robustness' (accuracy
    under distribution shift above the logit-linear trend of ImageNet-trained models).
  Figure 2: predicting the exact caption (Transformer LM) < predicting a bag of words (3x faster) < contrastive
    (another 4x faster) in zero-shot ImageNet accuracy per unit of compute.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------- encoders
class TextTransformer(nn.Module):
    """GPT-2-style causal Transformer; the representation is the final LN'd state at the [EOS] token."""

    def __init__(self, vocab, d=512, layers=12, heads=8, ctx=76):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Parameter(torch.randn(ctx, d) * 0.01)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, ids, eos_pos):
        T = ids.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
        h = self.ln(self.blocks(self.tok(ids) + self.pos[:T], mask=causal))
        return h[torch.arange(len(ids)), eos_pos]


class CBOW(nn.Module):
    """Continuous bag of words: the mean word embedding (the cheap text encoder of the paper's Figure 2 study)."""

    def __init__(self, vocab, d=64, pad=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=pad)
        self.out_dim, self.pad = d, pad

    def forward(self, ids, eos_pos=None):
        m = (ids != self.pad).float()
        return (self.emb(ids) * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


class AttentionPool(nn.Module):
    """The modified ResNet's replacement for global average pooling: one multi-head attention layer whose query is
    the average-pooled feature (Section 2.4)."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        seq = x.flatten(2).transpose(1, 2)                                         # B, HW, C
        q = seq.mean(1, keepdim=True)
        return self.attn(q, torch.cat([q, seq], 1), torch.cat([q, seq], 1), need_weights=False)[0][:, 0]


class SmallConvNet(nn.Module):
    def __init__(self, c_in=3, width=32, attn_pool=False):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, width, 3, 2, 1), nn.ReLU(), nn.Conv2d(width, 2 * width, 3, 2, 1),
                                 nn.ReLU())
        self.pool = AttentionPool(2 * width) if attn_pool else None
        self.out_dim = 2 * width

    def forward(self, x):
        f = self.net(x)
        return self.pool(f) if self.pool is not None else f.mean((2, 3))


# ---------------------------------------------------------------------------------------------------- CLIP
class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, d_embed=64, init_temp=0.07, max_scale=100.0):
        super().__init__()
        self.image_encoder, self.text_encoder = image_encoder, text_encoder
        self.W_i = nn.Linear(image_encoder.out_dim, d_embed, bias=False)            # linear projections only
        self.W_t = nn.Linear(text_encoder.out_dim, d_embed, bias=False)
        self.t = nn.Parameter(torch.tensor(math.log(1 / init_temp)))                 # log-parameterised scale
        self.max_scale = max_scale

    def logit_scale(self):
        return self.t.exp().clamp(max=self.max_scale)

    def encode_image(self, x):
        return F.normalize(self.W_i(self.image_encoder(x)), dim=-1)

    def encode_text(self, ids, eos_pos=None):
        return F.normalize(self.W_t(self.text_encoder(ids, eos_pos)), dim=-1)

    def forward(self, x, ids, eos_pos=None):
        """Figure 3: the N x N matrix of scaled cosine similarities between images and texts."""
        return self.encode_image(x) @ self.encode_text(ids, eos_pos).T * self.logit_scale()


def clip_loss(logits):
    """Symmetric cross-entropy: rows (image -> text) and columns (text -> image), the diagonal is correct."""
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ---------------------------------------------------------------------------------------------------- zero-shot
@torch.no_grad()
def zero_shot_weights(model, encode_fn, classnames, templates):
    """For each class, embed every template (e.g. 'a photo of a {}.'), average the NORMALISED embeddings, and
    renormalise: the prompt ensemble lives in embedding space, so it costs one vector per class."""
    W = []
    for name in classnames:
        e = model.encode_text(*encode_fn([t.format(name) for t in templates]))
        W.append(F.normalize(e.mean(0), dim=0))
    return torch.stack(W)


@torch.no_grad()
def zero_shot_predict(model, images, W):
    return (model.encode_image(images) @ W.T).argmax(-1)


def linear_probe(train_feats, train_y, test_feats, C=1.0, steps=500):
    """Logistic regression on frozen features (the paper uses scikit-learn's L-BFGS with tuned L2)."""
    K = int(train_y.max()) + 1
    W = torch.zeros(train_feats.shape[1], K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(train_feats @ W + b, train_y) + (W ** 2).sum() / (2 * C * len(train_y))
        loss.backward()
        return loss
    opt.step(closure)
    return (test_feats @ W + b).argmax(-1).detach()


# ---------------------------------------------------------------------------------------------------- robustness
def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def effective_robustness(baseline_pairs, model_id_acc, model_ood_acc):
    """Section 3.3: fit logit(OOD) = w logit(ID) + c over ImageNet-trained baselines; a model's effective robustness is
    its OOD accuracy minus the fit's prediction at its ID accuracy."""
    xs = torch.tensor([logit(i) for i, _ in baseline_pairs]); ys = torch.tensor([logit(o) for _, o in baseline_pairs])
    A = torch.stack([xs, torch.ones_like(xs)], 1)
    w, c = torch.linalg.lstsq(A, ys[:, None]).solution.flatten().tolist()
    predicted = 1 / (1 + math.exp(-(w * logit(model_id_acc) + c)))
    return model_ood_acc - predicted, predicted


# ---------------------------------------------------------------------------------------------------- Figure 2 baselines
def bag_of_words_loss(image_features, head, word_ids, vocab, pad=0):
    """Predictive baseline: from the image, predict the caption's bag of words (softmax over the vocabulary, target
    = the normalised word-count distribution)."""
    target = torch.zeros(len(word_ids), vocab, device=word_ids.device)
    target.scatter_add_(1, word_ids, (word_ids != pad).float())
    target = target / target.sum(1, keepdim=True).clamp(min=1)
    return -(target * F.log_softmax(head(image_features), -1)).sum(1).mean()


def count_params(m):
    return sum(p.numel() for p in m.parameters())
