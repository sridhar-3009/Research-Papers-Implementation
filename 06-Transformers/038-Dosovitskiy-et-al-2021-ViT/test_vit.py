"""Tests for the Vision Transformer (Dosovitskiy et al. 2021). A few seconds.

Run with:  python3 -m pytest -q
"""

import torch
import torch.nn.functional as F

from vit import (ViT, corrupt_patches, count_params, embedding_filters_pca, mean_attention_distance,
                 mean_colour_targets, patchify, position_similarity, unpatchify)

torch.manual_seed(0)


def test_patchify_is_a_reshape_of_the_image():
    img = torch.randn(2, 3, 8, 12)
    x = patchify(img, 4)
    assert x.shape == (2, 6, 48)                                                   # N = HW / P^2 = 6, P^2 C = 48
    assert torch.equal(x[0, 4].reshape(4, 4, 3).permute(2, 0, 1), img[0, :, 4:8, 4:8])  # row 1, col 1
    assert torch.equal(unpatchify(x, 4, 3, 8, 12), img)


def test_model_sizes_match_table_1():
    with torch.device("meta"):
        assert abs(count_params(ViT()) / 1e6 - 86) < 1.5                           # ViT-B/16
        assert abs(count_params(ViT(D=1024, layers=24, heads=16, mlp=4096)) / 1e6 - 307) < 4      # ViT-L/16
        assert abs(count_params(ViT(patch=14, D=1280, layers=32, heads=16, mlp=5120)) / 1e6 - 632) < 2  # ViT-H/14


def test_tokens_and_equations():
    m = ViT(16, 4, 3, D=32, layers=1, heads=4, mlp=64, classes=5).eval()
    img = torch.randn(2, 3, 16, 16)
    z0 = m.tokens(img)
    assert z0.shape == (2, 17, 32)                                                 # [class] + 16 patches
    assert torch.allclose(z0[:, 0], (m.cls + m.pos[:, :1])[0].expand(2, -1))
    b = m.blocks[0]
    h = b.ln1(z0)
    zp = b.attn(h, h, h, need_weights=False)[0] + z0                               # Eq. 2
    z1 = b.mlp(b.ln2(zp)) + zp                                                     # Eq. 3
    assert torch.allclose(m(img, return_tokens=True), m.ln(z1), atol=1e-5)          # Eq. 4 (before the head)
    assert torch.equal(m(img), torch.zeros(2, 5))                                  # zero-initialised head


def test_higher_resolution_fine_tuning():
    m = ViT(16, 4, 3, D=32, layers=1, heads=4, mlp=64, classes=5)
    old = m.pos.detach().clone()
    m.resize_positions(4)                                                          # same grid: unchanged
    assert torch.allclose(m.pos, old, atol=1e-5)
    m.resize_positions(8)                                                          # 32 x 32 images now
    assert m.pos.shape == (1, 65, 32) and torch.equal(m.pos[:, 0], old[:, 0])
    assert m(torch.randn(1, 3, 32, 32)).shape == (1, 5)
    m.new_head(7)
    assert torch.equal(m(torch.randn(1, 3, 32, 32)), torch.zeros(1, 7))


def test_hybrid_uses_cnn_feature_map_as_1x1_patches():
    m = ViT(32, 1, 3, D=32, layers=1, heads=4, mlp=64, classes=5, hybrid_downsample=4)
    assert m.tokens(torch.randn(2, 3, 32, 32)).shape == (2, 1 + 8 * 8, 32)


def test_analysis_tools():
    m = ViT(16, 4, 3, D=32, layers=2, heads=4, mlp=64, classes=5)
    S = position_similarity(m)
    assert S.shape == (16, 16) and torch.allclose(S.diag(), torch.ones(16))
    comps, s = embedding_filters_pca(m, k=5)
    assert comps.shape == (5, 48) and torch.all(s[:-1] >= s[1:])
    d = mean_attention_distance(m, torch.randn(4, 3, 16, 16))
    max_dist = (2 * 12 ** 2) ** 0.5
    assert d.shape == (2, 4) and (d >= 0).all() and (d <= max_dist).all()


def test_masked_patch_prediction_targets_and_corruption():
    img = torch.zeros(1, 3, 8, 8)
    img[0, 0, :4, :4] = 1.0                                                        # a pure red patch
    t = mean_colour_targets(img, 4)
    assert t[0, 0].item() == (7 * 8 + 0) * 8 + 0 and t[0, 1].item() == 0          # 3 bits per channel, 512 classes
    x = torch.randn(200, 64, 10)
    out, chosen, masked = corrupt_patches(x, generator=torch.Generator().manual_seed(0))
    assert abs(chosen.float().mean().item() - 0.5) < 0.02
    assert abs(masked[chosen].float().mean().item() - 0.8) < 0.02
    assert torch.equal(out[~chosen], x[~chosen]) and (out[masked] == 0).all()


def test_vit_learns_a_simple_global_task():
    torch.manual_seed(0)
    m = ViT(16, 4, 3, D=32, layers=1, heads=4, mlp=64, classes=2, head="mlp")
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(150):
        y = torch.randint(0, 2, (64,))
        x = torch.rand(64, 3, 16, 16) * 0.5 + 0.5 * y[:, None, None, None]        # dark vs bright images
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1
