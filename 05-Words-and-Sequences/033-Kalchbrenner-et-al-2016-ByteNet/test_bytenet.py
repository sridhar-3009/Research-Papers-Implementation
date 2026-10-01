"""Tests for ByteNet (Kalchbrenner et al. 2016). A few seconds.

Run with:  python3 -m pytest -q
"""

import torch
import torch.nn.functional as F

from bytenet import (EOS, PAD, MU, ByteNet, ByteNetLM, Conv1d, dilations, receptive_field, saliency, unfold_length)

torch.manual_seed(0)


def grad_support(f, x, pos):
    """Input positions whose value affects output position `pos`."""
    x = x.clone().requires_grad_(True)
    f(x)[0, :, pos].sum().backward()
    return (x.grad[0].abs().sum(0) > 0).nonzero()[:, 0].tolist()


def test_masked_convolution_is_causal_and_unmasked_is_centred():
    c = Conv1d(2, 3, k=3, dilation=2, causal=True)
    assert grad_support(c, torch.randn(1, 2, 20), 10) == [6, 8, 10]                # i - 2r, i - r, i
    u = Conv1d(2, 3, k=3, dilation=2, causal=False)
    assert grad_support(u, torch.randn(1, 2, 20), 10) == [8, 10, 12]               # centred


def test_dilation_schedule_and_receptive_field():
    assert dilations(10) == [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
    assert receptive_field(3, dilations(5)) == 63 and receptive_field(3, [1] * 5) == 11
    assert receptive_field(3, dilations(30)) == 373                                # the paper's LM: it states 315


def test_receptive_field_equals_the_gradient_support():
    m = ByteNetLM(10, d=8, n_blocks=5, k=3, block="relu", dropout=0).eval()
    e = m.emb(torch.randint(0, 10, (1, 100))).detach().transpose(1, 2)
    support = grad_support(lambda x: m.stack(x), e, 80)
    assert support == list(range(80 - m.receptive_field + 1, 81))                  # exactly 63 positions, none ahead


def test_mu_block_formula():
    mu = MU(4, 3, 1, causal=True)
    h = torch.randn(2, 4, 7)
    g1, g2, g3, u = mu.conv(h).chunk(4, 1)
    want = torch.sigmoid(g1) * torch.tanh(torch.sigmoid(g2) * h + torch.sigmoid(g3) * torch.tanh(u))
    assert torch.allclose(mu(h), want)


def test_the_full_decoder_cannot_see_the_future_but_the_encoder_sees_everything():
    m = ByteNet(10, 10, d=8, n_blocks=4, a=1.2).eval()
    src = torch.randint(3, 10, (1, 8))
    tgt_in = torch.randint(3, 10, (1, 9))
    out = m(src, tgt_in)
    tgt2 = tgt_in.clone(); tgt2[0, 6] = (tgt2[0, 6] + 1 - 3) % 7 + 3                # change target position 6
    out2 = m(src, tgt2)
    assert torch.allclose(out[0, :6], out2[0, :6], atol=1e-6) and not torch.allclose(out[0, 6:], out2[0, 6:])
    src2 = src.clone(); src2[0, -1] = (src2[0, -1] + 1 - 3) % 7 + 3                 # change the LAST source char
    assert not torch.allclose(m.encode(src)[..., 0], m.encode(src2)[..., 0])       # position 0 sees it


def test_dynamic_unfolding_lengths():
    m = ByteNet(10, 10, d=8, n_blocks=2, a=1.2, b=0)
    assert unfold_length(50) == 60 and m.encode(torch.randint(3, 10, (2, 50))).shape[-1] == 60
    out = m(torch.randint(3, 10, (2, 5)), torch.randint(3, 10, (2, 12)))           # decoding past |t^| = 6 works
    assert out.shape == (2, 12, 10)


def test_log_prob_ignores_padding():
    m = ByteNet(10, 10, d=8, n_blocks=2).eval()
    src = torch.randint(3, 10, (1, 5))
    tgt = torch.tensor([[4, 5, EOS]])
    a = m.log_prob(src, tgt)
    b = m.log_prob(src, torch.tensor([[4, 5, EOS, PAD, PAD]]))
    assert torch.allclose(a, b, atol=1e-5)


def test_saliency_shapes_and_causality():
    m = ByteNet(10, 10, d=8, n_blocks=3, a=1.0, b=1).eval()
    S, T = saliency(m, torch.randint(3, 10, (1, 6)), torch.randint(3, 10, (1, 7)))
    assert S.shape == (7, 6) and T.shape == (7, 7)
    assert torch.all(torch.triu(T, 1) == 0)                                         # output i ignores inputs > i


def test_dilation_lets_a_shallow_stack_learn_a_long_lag():
    torch.manual_seed(0)
    V, T, L = 8, 24, 10
    m = ByteNetLM(V, d=32, n_blocks=3, k=3, block="relu", dropout=0)               # rates 1, 2, 4: sees 15
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(150):
        x = torch.randint(3, V, (16, T))
        for t in range(L, T):
            x[:, t] = x[:, t - L]
        loss = F.cross_entropy(m(x[:, :-1])[:, L:].reshape(-1, V), x[:, L + 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1
    assert receptive_field(3, [1, 1, 1]) < L                                        # without dilation it couldn't
