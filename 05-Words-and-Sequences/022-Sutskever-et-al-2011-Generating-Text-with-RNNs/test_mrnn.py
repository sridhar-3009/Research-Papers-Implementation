"""Tests: each checks a statement or tool of Sutskever, Martens & Hinton (2011). A few seconds.

Run with:  python3 -m pytest -q
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mrnn import (MRNN, CharRNN, HessianFree, TensorRNN, bits_per_char, count_params, debag, nll, one_hot, sample,
                  sparse_init_)

torch.manual_seed(0)


# ---- sizes stated in the paper ----

def test_big_mrnn_has_about_4_9_million_parameters():
    with torch.device("meta"):
        m = MRNN(vocab=86, hidden=1500, factors=1500)                 # 'which have 4,900,000 parameters'
    assert round(count_params(m) / 1e5) == 49


def test_rnn_500_has_slightly_more_parameters_than_mrnn_350():
    # Section 3.2: 'an RNN with 500 hidden units and an MRNN with 350 hidden units and 350 factors
    # (so the RNN has slightly more parameters)'
    rnn, mrnn = count_params(CharRNN(86, 500)), count_params(MRNN(86, 350, 350))
    assert rnn > mrnn and (rnn - mrnn) / rnn < 0.01


# ---- Eqs. (5)-(10) ----

def test_mrnn_step_uses_the_character_specific_matrix():
    m = MRNN(5, 4, 6)
    h = torch.randn(1, 4)
    for c in range(5):
        x = one_hot(torch.tensor([c]), 5)
        expected = torch.tanh(h @ m.effective_matrix(c).T + m.W_hx(x))
        assert torch.allclose(m.step(x, h), expected, atol=1e-6)


def test_mrnn_is_a_factored_tensor_rnn():
    m = MRNN(5, 4, 6)
    t = TensorRNN(5, 4)
    with torch.no_grad():
        t.W_hh.copy_(torch.stack([m.effective_matrix(c) for c in range(5)]))
        t.W_hx.load_state_dict(m.W_hx.state_dict()); t.W_oh.load_state_dict(m.W_oh.state_dict())
    xs = one_hot(torch.randint(0, 5, (7, 3)), 5)
    assert torch.allclose(m(xs)[0], t(xs)[0], atol=1e-5)


def test_each_character_matrix_has_rank_at_most_F():
    m = MRNN(5, 10, 3)
    assert torch.linalg.matrix_rank(m.effective_matrix(2)).item() <= 3


def test_sparse_init_gives_15_connections_per_unit():
    W = torch.empty(40, 100)
    sparse_init_(W, k=15)
    assert ((W != 0).sum(1) == 15).all()


# ---- Section 4: probabilities ----

def test_untrained_uniform_model_costs_log2_M_bits():
    m = CharRNN(8, 4)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    ids = torch.randint(0, 8, (20, 3))
    assert bits_per_char(m, ids) == pytest.approx(3.0)              # log2(8)


def test_sampling():
    m = MRNN(6, 5, 5)
    out = sample(m, [1, 2, 3], 20, generator=torch.Generator().manual_seed(0))
    assert len(out) == 20 and all(0 <= c < 6 for c in out)


class Bigram(nn.Module):
    """A fake 'language model' that strongly prefers the transitions of one fixed text."""

    def __init__(self, text, alphabet):
        super().__init__()
        self.M = len(alphabet)
        table = torch.full((self.M, self.M), -5.0)
        for a, b in zip(text, text[1:]):
            table[alphabet.index(a), alphabet.index(b)] = 5.0
        self.table = nn.Parameter(table)

    def forward(self, xs, h=None):
        return xs @ self.table, h


def test_debagging_finds_the_order_the_model_prefers():
    # In "ab cd ef" a space is only ever followed by c or e, and f by nothing, so only one order of
    # the bag {cd, ab, ef} uses nothing but known letter pairs. (A bigram model could NOT order
    # "the cat sat": every pair in "sat the cat" also occurs in "the cat sat". Real debagging needs
    # long contexts, which is the paper's point.)
    alphabet = sorted(set("ab cd ef"))
    model = Bigram("ab cd ef", alphabet)
    encode = lambda s: [alphabet.index(c) for c in s]
    best, _ = debag(model, ["cd", "ab", "ef"], encode)
    assert list(best) == ["ab", "cd", "ef"]


# ---- Hessian-free ----

def explicit_gauss_newton(model, ids):
    """G = sum_i J_i^T (diag(p_i) - p_i p_i^T) J_i / n, built from the full Jacobian (tiny models only)."""
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for p in model.parameters()]
    flat0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def logits_of(flat):
        prm, i = {}, 0
        for n, s in zip(names, shapes):
            k = math.prod(s)
            prm[n] = flat[i:i + k].view(s)
            i += k
        return torch.func.functional_call(model, prm, (one_hot(ids[:-1], model.M),))[0].reshape(-1, model.M)

    J = torch.autograd.functional.jacobian(logits_of, flat0)                     # (n, M, P)
    p = F.softmax(logits_of(flat0), -1)
    G = torch.zeros(len(flat0), len(flat0))
    for i in range(p.shape[0]):
        H = torch.diag(p[i]) - torch.outer(p[i], p[i])
        G += J[i].T @ H @ J[i]
    return G / p.shape[0]


def test_gauss_newton_vector_product_is_exact():
    torch.manual_seed(1)
    m = MRNN(3, 2, 2)
    ids = torch.randint(0, 3, (4, 2))
    hf = HessianFree(m)
    G = explicit_gauss_newton(m, ids)
    v = torch.randn(G.shape[0])
    assert torch.allclose(hf.gauss_newton_product(ids, v), G @ v, atol=1e-5)
    assert torch.linalg.eigvalsh(G).min() > -1e-6                                 # G is positive semi-definite


def test_conjugate_gradient_solves_spd_systems():
    A = torch.randn(10, 10); A = A @ A.T + torch.eye(10)
    b = torch.randn(10)
    hf = HessianFree(MRNN(2, 2, 2), cg_iters=50)
    x = hf.conjugate_gradient(lambda v: A @ v, b, torch.zeros(10))
    assert torch.allclose(A @ x, b, atol=1e-4)


def test_hessian_free_steps_reduce_the_loss():
    torch.manual_seed(2)
    m = MRNN(4, 8, 8)
    ids = torch.tensor([[0, 1, 2, 3] * 5] * 4).T                                  # a repeating pattern (T=20, N=4)
    hf = HessianFree(m, lam=1.0, cg_iters=20)
    first = nll(m, ids)[0].item() / nll(m, ids)[1]
    for _ in range(5):
        hf.step(ids, ids)
    last = nll(m, ids)[0].item() / nll(m, ids)[1]
    assert last < 0.5 * first
