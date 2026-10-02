import math

import torch
import torch.nn.functional as F

from dalle import (DALLE, DVAE, PowerSGD, codebook_perplexity, cosine_schedule, dvae_loss, fp16_underflow_fraction,
                   full_mask, image_mask, kl_to_uniform, layer_kinds, logit_laplace_log_prob, phi, phi_inv,
                   powersgd_compression_rate, rerank, transformer_params)


def test_logit_laplace_is_a_density_on_the_unit_interval():
    mu, log_b = torch.tensor(0.3), torch.tensor(math.log(0.4))
    x = torch.linspace(1e-6, 1 - 1e-6, 400001)
    total = torch.trapezoid(logit_laplace_log_prob(x, mu, log_b).exp(), x).item()
    assert abs(total - 1) < 1e-3
    # it is the density of sigmoid(L) with L ~ Laplace(mu, b): check one point by change of variables
    lap = torch.distributions.Laplace(mu, log_b.exp())
    v = torch.tensor(0.7)
    expect = lap.log_prob(torch.logit(v)) - torch.log(v * (1 - v))
    assert torch.allclose(logit_laplace_log_prob(v, mu, log_b), expect, atol=1e-6)


def test_phi_mapping():
    x = torch.tensor([0.0, 127.5, 255.0])
    assert torch.allclose(phi(x), torch.tensor([0.1, 0.5, 0.9]))
    assert torch.allclose(phi_inv(phi(x)), x)


def test_kl_to_uniform_and_perplexity():
    K = 8
    assert torch.allclose(kl_to_uniform(torch.zeros(1, K, 1, 1)), torch.zeros(1, 1, 1), atol=1e-6)
    onehot = torch.full((1, K, 1, 1), -1e4); onehot[0, 3] = 0
    assert torch.allclose(kl_to_uniform(onehot), torch.tensor(math.log(K)), atol=1e-4)
    assert abs(codebook_perplexity(torch.arange(K), K) - K) < 1e-6
    assert abs(codebook_perplexity(torch.zeros(10, dtype=torch.long), K) - 1) < 1e-6


def test_gumbel_softmax_hardens_as_tau_falls():
    torch.manual_seed(0)
    logits = torch.tensor([1.0, 0.0, -1.0]).expand(20000, 3)
    soft = F.gumbel_softmax(logits, tau=1.0, dim=1)
    cold = F.gumbel_softmax(logits, tau=1 / 16, dim=1)
    assert cold.max(1).values.mean() > 0.95 > soft.max(1).values.mean()
    freq = torch.bincount(cold.argmax(1), minlength=3).float() / 20000             # samples follow softmax(logits)
    assert torch.allclose(freq, F.softmax(logits[0], 0), atol=0.02)


def test_dvae_shapes_and_loss():
    torch.manual_seed(0)
    d = DVAE(K=16, width=16, down=4)
    x = phi(torch.randint(0, 256, (2, 3, 16, 16)).float())
    assert d.tokens(x).shape == (2, 4, 4)                                          # 16x16 -> 4x4 tokens
    rec, kl, soft = d.elbo_terms(x, tau=1.0)
    assert rec.shape == kl.shape == (2,) and torch.allclose(soft.sum(1), torch.ones(2, 4, 4), atol=1e-5)
    loss = dvae_loss(d, x, 1.0, 6.6)
    loss.backward()
    assert torch.isfinite(loss)
    assert d.reconstruct(x).shape == x.shape
    assert abs(cosine_schedule(0, 100, 1.0, 1 / 16) - 1) < 1e-9 and abs(cosine_schedule(100, 100, 1.0, 1 / 16) - 1 / 16) < 1e-9


def test_attention_masks_match_figure_11():
    H = W = 4
    row, col, conv = image_mask("row", H, W), image_mask("column", H, W), image_mask("conv", H, W)
    i = 9                                                                         # row 2, column 1
    assert row[i].nonzero().flatten().tolist() == [5, 6, 7, 8, 9]                 # previous W tokens + itself
    assert col[i].nonzero().flatten().tolist() == [1, 5, 9]                       # same column, rows above
    assert conv[i].nonzero().flatten().tolist() == [4, 5, 6, 8, 9]                # causal 3x3 neighbourhood
    for m in (row, col, conv):
        assert not torch.any(torch.triu(m, 1))                                    # never the future
    full = full_mask("row", 3, H, W)
    assert torch.all(full[3:, :3])                                                # every image token sees all text
    assert torch.equal(full[:3, :3], torch.tril(torch.ones(3, 3, dtype=torch.bool))) and not torch.any(full[:3, 3:])
    assert layer_kinds(5) == ["row", "column", "row", "row", "conv"]              # 'row, column, row, row', ..., conv


def test_transformer_is_causal_and_uses_padding_tokens():
    torch.manual_seed(0)
    m = DALLE(20, 8, 4, 2, 2, d=32, layers=3, heads=4).eval()
    text, img = torch.randint(0, 20, (1, 4)), torch.randint(0, 8, (1, 4))
    _, li = m(text, torch.tensor([4]), img)
    img2 = img.clone(); img2[0, 3] = (img[0, 3] + 1) % 8
    _, li2 = m(text, torch.tensor([4]), img2)
    assert torch.allclose(li[:, :4], li2[:, :4], atol=1e-6)                        # logits never see their target
    _, a = m(text, torch.tensor([2]), img)
    text2 = text.clone(); text2[0, 2:] = (text[0, 2:] + 1) % 20
    _, b = m(text2, torch.tensor([2]), img)
    assert torch.allclose(a, b, atol=1e-6)                                        # tokens past the caption ignored
    loss, ct, ci = m.loss(text, torch.tensor([4]), img)
    assert abs(loss.item() - (ct / 8 + 7 * ci / 8)) < 1e-4
    assert m.generate(text, torch.tensor([4])).shape == (1, 4)


def test_scale_numbers_from_the_paper():
    assert abs(transformer_params(3968, 64) / 1e9 - 12.1) < 0.3                   # 'a 12-billion parameter' model
    assert (256 * 256 * 3) // (32 * 32) == 192                                    # context reduced 192x
    rates = [powersgd_compression_rate(r, d) for r, d in ((512, 1920), (640, 2688), (896, 3968))]
    assert [round(r, 2) for r in rates] == [0.83, 0.85, 0.86]                       # Table 1


def test_powersgd_error_feedback_and_underflow():
    torch.manual_seed(0)
    comp, plain = PowerSGD((64, 64), r=4), PowerSGD((64, 64), r=4)
    total_true, total_sent, total_plain = torch.zeros(64, 64), torch.zeros(64, 64), torch.zeros(64, 64)
    for _ in range(200):
        g = torch.randn(64, 64) * 0.1 + torch.outer(torch.randn(64), torch.randn(64)) * 0.05
        approx, n = comp.compress(g)
        plain.error.zero_()                                                       # no error feedback
        total_true += g; total_sent += approx; total_plain += plain.compress(g)[0]
    assert n == 2 * 64 * 4                                                        # 512 numbers instead of 4096
    assert torch.allclose(total_true - total_sent, comp.error, atol=1e-3)         # telescoping: only the last residual
    assert (total_true - total_sent).norm() < 0.5 * (total_true - total_plain).norm()   # is ever missing
    assert fp16_underflow_fraction(torch.tensor([1e-9, 1e-3, 1e-10, 0.0])) == 2 / 3
    assert rerank(["a", "b", "c"], [0.1, 0.9, 0.5], 2) == ["b", "c"]
