import math

import torch
import torch.nn.functional as F

from clip import (CBOW, CLIP, AttentionPool, SmallConvNet, TextTransformer, bag_of_words_loss, clip_loss,
                  effective_robustness, linear_probe, zero_shot_predict, zero_shot_weights)


def small_clip():
    torch.manual_seed(0)
    return CLIP(SmallConvNet(3, 8), CBOW(20, 16), d_embed=16)


def test_logits_are_scaled_cosines_and_temperature_is_clipped():
    m = small_clip()
    x, ids = torch.randn(4, 3, 8, 8), torch.randint(1, 20, (4, 5))
    lg = m(x, ids)
    I, T = m.encode_image(x), m.encode_text(ids)
    assert torch.allclose(I.norm(dim=-1), torch.ones(4)) and torch.allclose(lg, I @ T.T / 0.07, atol=1e-4)
    with torch.no_grad():
        m.t.fill_(10.0)                                                            # exp(10) >> 100
    assert m.logit_scale().item() == 100.0


def test_symmetric_loss_matches_the_pseudocode():
    torch.manual_seed(0)
    lg = torch.randn(5, 5)
    labels = torch.arange(5)
    manual = (F.cross_entropy(lg, labels) + F.cross_entropy(lg.T, labels)) / 2
    assert torch.allclose(clip_loss(lg), manual)
    assert abs(clip_loss(torch.zeros(8, 8)).item() - math.log(8)) < 1e-6              # chance level: log N
    assert clip_loss(torch.eye(8) * 50) < 1e-6                                          # perfect matching


def test_zero_shot_prompt_ensemble_is_mean_of_normalised_embeddings():
    m = small_clip()
    vocab = {w: i + 1 for i, w in enumerate("a photo of the cat dog big small".split())}
    enc = lambda texts: (torch.tensor([[vocab[w] for w in t.split()] + [0] * (5 - len(t.split())) for t in texts]),)
    W = zero_shot_weights(m, enc, ["cat", "dog"], ["a photo of the {}", "a big {}", "a small {}"])
    e = m.encode_text(*enc(["a photo of the cat", "a big cat", "a small cat"]))
    assert torch.allclose(W[0], F.normalize(e.mean(0), dim=0), atol=1e-6) and torch.allclose(W.norm(dim=-1), torch.ones(2))
    pred = zero_shot_predict(m, torch.randn(3, 3, 8, 8), W)
    assert pred.shape == (3,) and pred.max() <= 1


def test_text_transformer_uses_eos_and_is_causal():
    torch.manual_seed(0)
    t = TextTransformer(30, d=32, layers=2, heads=4, ctx=8).eval()
    ids = torch.randint(1, 30, (2, 8))
    eos = torch.tensor([3, 5])
    f = t(ids, eos)
    ids2 = ids.clone(); ids2[0, 6] = (ids[0, 6] + 1) % 30                            # change a token AFTER eos
    assert torch.allclose(t(ids2, eos)[0], f[0], atol=1e-5)
    assert f.shape == (2, 32)


def test_attention_pool_and_encoders():
    torch.manual_seed(0)
    assert AttentionPool(8, 2)(torch.randn(2, 8, 4, 4)).shape == (2, 8)
    assert SmallConvNet(3, 8, attn_pool=True)(torch.randn(2, 3, 16, 16)).shape == (2, 16)
    c = CBOW(10, 4)
    ids = torch.tensor([[1, 2, 0, 0]])
    assert torch.allclose(c(ids), (c.emb.weight[1] + c.emb.weight[2]) / 2)        # padding ignored


def test_linear_probe_and_effective_robustness():
    torch.manual_seed(0)
    X = torch.randn(200, 5); y = (X[:, 0] > 0).long()
    assert (linear_probe(X, y, X) == y).float().mean() > 0.95
    base = [(p, p ** 2) for p in (0.5, 0.6, 0.7, 0.8)]                              # OOD falls faster than ID
    er, pred = effective_robustness(base, 0.7, 0.6)
    assert abs(pred - 0.49) < 0.03 and er > 0.08                                     # 0.6 beats the trend's ~0.49


def test_bag_of_words_loss():
    head = torch.nn.Linear(4, 6)
    f = torch.randn(2, 4)
    ids = torch.tensor([[1, 2, 2, 0], [3, 0, 0, 0]])
    lp = F.log_softmax(head(f), -1)
    manual = -((lp[0, 1] + 2 * lp[0, 2]) / 3 + lp[1, 3]) / 2
    assert torch.allclose(bag_of_words_loss(f, head, ids, 6), manual, atol=1e-6)


def test_training_learns_to_match_pairs():
    torch.manual_seed(0)
    m = CLIP(SmallConvNet(1, 8), CBOW(11, 16), d_embed=16)
    protos = torch.randn(10, 1, 8, 8)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(150):
        y = torch.randperm(10)
        x = protos[y] + 0.1 * torch.randn(10, 1, 8, 8)
        loss = clip_loss(m(x, (y + 1)[:, None]))
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.3                                                         # chance: log 10 = 2.3
