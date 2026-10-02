import math

import torch
import torch.nn.functional as F

from infogan import (LatentSpec, MLPDiscriminatorQ, MLPGenerator, MNISTDiscriminatorQ, MNISTGenerator,
                     cluster_accuracy, infogan_step, lemma_5_1_sides, mi_lower_bound, mi_terms,
                     mutual_information_discrete, q_out_dim, split_q)


def test_latent_spec_matches_mnist_setup():
    spec = LatentSpec(62, (10,), 2)
    assert spec.dim == 74                                                         # Appendix C.1
    z, cat, cont = spec.sample(500)
    v = spec.pack(z, cat, cont)
    assert v.shape == (500, 74) and torch.all(v[:, 62:72].sum(1) == 1)
    assert cont.min() >= -1 and cont.max() <= 1 and z.abs().max() <= 1
    assert abs(spec.entropy() - (math.log(10) + 2 * math.log(2))) < 1e-9
    assert q_out_dim(spec) == 14


def test_lemma_5_1():
    torch.manual_seed(0)
    Px = torch.rand(4); Px /= Px.sum()
    Pyx = torch.rand(4, 3); Pyx /= Pyx.sum(1, keepdim=True)
    f = torch.randn(4, 3)
    lhs, rhs = lemma_5_1_sides(Px, Pyx, f)
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_variational_bound_on_mutual_information():
    """For a discrete channel c -> x: E[log Q(c|x)] + H(c) <= I(c; x), with equality iff Q = P(c|x)."""
    torch.manual_seed(0)
    Pc = torch.full((3,), 1 / 3)
    Pxc = torch.rand(3, 5); Pxc /= Pxc.sum(1, keepdim=True)
    joint = Pc[:, None] * Pxc
    I = mutual_information_discrete(joint)
    Hc = math.log(3)
    posterior = joint / joint.sum(0, keepdim=True)                                 # P(c|x)
    tight = (joint * posterior.log()).sum() + Hc
    assert torch.allclose(tight, I, atol=1e-6)
    for _ in range(20):
        Q = torch.rand(3, 5); Q /= Q.sum(0, keepdim=True)
        assert (joint * Q.log()).sum() + Hc <= I + 1e-6
    deterministic = torch.eye(3) / 3                                               # x reveals c exactly
    assert torch.allclose(mutual_information_discrete(deterministic), torch.tensor(Hc), atol=1e-6)


def test_l_i_extremes_for_categorical_codes():
    spec = LatentSpec(4, (10,), 0)
    cat = torch.randint(0, 10, (64, 1))
    perfect = F.one_hot(cat[:, 0], 10).float() * 50
    assert abs(mi_lower_bound(perfect, cat, torch.zeros(64, 0), spec).item() - math.log(10)) < 1e-6
    uniform = torch.zeros(64, 10)
    assert abs(mi_lower_bound(uniform, cat, torch.zeros(64, 0), spec).item()) < 1e-6   # Q learns nothing: 0


def test_continuous_q_is_a_gaussian_log_likelihood():
    spec = LatentSpec(4, (), 2)
    q = torch.tensor([[0.1, -0.3, math.log(0.5), math.log(2.0)]])
    cont = torch.tensor([[0.4, 0.5]])
    _, lq = mi_terms(q, torch.zeros(1, 0).long(), cont, spec)
    ref = torch.distributions.Normal(torch.tensor([0.1, -0.3]), torch.tensor([0.5, 2.0])).log_prob(cont).sum()
    assert torch.allclose(lq, ref, atol=1e-6)
    logits, mu, ls = split_q(q, spec)
    assert logits == [] and mu.shape == (1, 2) and ls.shape == (1, 2)


def test_mnist_networks_match_table_1():
    spec = LatentSpec(62, (10,), 2)
    G, DQ = MNISTGenerator(spec.dim), MNISTDiscriminatorQ(q_out_dim(spec))
    x = G(spec.pack(*spec.sample(4)))
    assert x.shape == (4, 1, 28, 28) and x.min() >= 0 and x.max() <= 1
    d, q = DQ(x)
    assert d.shape == (4,) and q.shape == (4, 14)


def test_cluster_accuracy_is_permutation_invariant():
    labels = torch.randint(0, 5, (200,))
    perm = torch.tensor([3, 0, 4, 1, 2])
    assert cluster_accuracy(perm[labels], labels, 5) == 1.0
    noisy = perm[labels].clone(); noisy[:20] = (noisy[:20] + 1) % 5
    assert abs(cluster_accuracy(noisy, labels, 5) - 0.9) < 0.02


def test_infogan_uses_the_code_and_gan_does_not():
    def data(n):
        k = torch.randint(0, 4, (n,))
        centres = torch.tensor([[2., 2], [-2, 2], [-2, -2], [2, -2]])
        return centres[k] + 0.1 * torch.randn(n, 2)
    spec = LatentSpec(4, (4,), 0)
    res = {}
    for info in (True, False):
        torch.manual_seed(0)
        G, DQ = MLPGenerator(spec.dim, hidden=64), MLPDiscriminatorQ(2, q_out_dim(spec), hidden=64)
        og = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999))
        od = torch.optim.Adam(DQ.parameters(), 1e-3, betas=(0.5, 0.999))
        for _ in range(400):
            _, _, li = infogan_step(G, DQ, og, od, data(128), spec, info=info)
        res[info] = li
    assert res[True] > 1.2 and res[False] < 0.7                                    # H(c) = log 4 = 1.386
