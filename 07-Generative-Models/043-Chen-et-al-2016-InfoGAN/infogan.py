"""InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
(Chen, Duan, Houthooft, Schulman, Sutskever & Abbeel, NIPS 2016).

  generator input = incompressible noise z + latent code c = (c_1, ..., c_L), P(c) = prod_i P(c_i)
  goal: high mutual information I(c; G(z, c)) = H(c) - H(c | G(z, c))                                (Eq. 2)
  min_G max_D V_I(D, G) = V(D, G) - lambda I(c; G(z, c))                                              (Eq. 3)
  variational lower bound (Eqs. 4-5), with an auxiliary network Q(c|x):
      I(c; G(z, c)) >= L_I(G, Q) = E_{c~P(c), x~G(z,c)}[log Q(c|x)] + H(c)       (tight when Q = P(c|x))
  min_{G,Q} max_D V_InfoGAN = V(D, G) - lambda L_I(G, Q)                                              (Eq. 6)
  - Q shares the discriminator's body and adds one small head (Section 6, Appendix C);
  - categorical codes: softmax Q; continuous codes: factored Gaussian Q (mean, exp-parameterised std);
  - lambda = 1 for discrete codes, smaller for continuous codes (differential entropy is on another scale).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------- latent codes
class LatentSpec:
    """z_dim noise ~ U(-1, 1) (or N(0, 1)), categorical codes of sizes `cats` (uniform), `n_cont` codes ~ U(-1, 1)."""

    def __init__(self, z_dim=62, cats=(10,), n_cont=2, noise="uniform"):
        self.z_dim, self.cats, self.n_cont, self.noise = z_dim, tuple(cats), n_cont, noise
        self.dim = z_dim + sum(cats) + n_cont                                       # MNIST: 62 + 10 + 2 = 74

    def sample(self, n):
        z = torch.rand(n, self.z_dim) * 2 - 1 if self.noise == "uniform" else torch.randn(n, self.z_dim)
        cat = torch.stack([torch.randint(0, k, (n,)) for k in self.cats], 1) if self.cats else torch.zeros(n, 0).long()
        cont = torch.rand(n, self.n_cont) * 2 - 1
        return z, cat, cont

    def pack(self, z, cat, cont):
        onehots = [F.one_hot(cat[:, i], k).float() for i, k in enumerate(self.cats)]
        return torch.cat([z, *onehots, cont], 1)

    def entropy(self):
        """H(c): log K per categorical code, log 2 per U(-1, 1) code (differential entropy)."""
        return sum(math.log(k) for k in self.cats) + self.n_cont * math.log(2)


# ---------------------------------------------------------------------------------------------------- Q and L_I
def split_q(q_out, spec):
    """Q head output -> (list of categorical logits, continuous mean, continuous log-std)."""
    logits, i = [], 0
    for k in spec.cats:
        logits.append(q_out[:, i:i + k]); i += k
    mu, log_std = q_out[:, i:i + spec.n_cont], q_out[:, i + spec.n_cont:i + 2 * spec.n_cont]
    return logits, mu, log_std


def q_out_dim(spec):
    return sum(spec.cats) + 2 * spec.n_cont


def mi_terms(q_out, cat, cont, spec):
    """E[log Q(c|x)] split into the categorical part and the continuous part (per-sample means).
    L_I = these + H(c) (Eq. 5)."""
    logits, mu, log_std = split_q(q_out, spec)
    lq_cat = sum(-F.cross_entropy(l, cat[:, i]) for i, l in enumerate(logits)) if logits else torch.tensor(0.0)
    if spec.n_cont:
        lq_cont = (-0.5 * math.log(2 * math.pi) - log_std - 0.5 * ((cont - mu) / log_std.exp()) ** 2).sum(1).mean()
    else:
        lq_cont = torch.tensor(0.0)
    return lq_cat, lq_cont


def mi_lower_bound(q_out, cat, cont, spec):
    lq_cat, lq_cont = mi_terms(q_out, cat, cont, spec)
    return lq_cat + lq_cont + spec.entropy()


# ---------------------------------------------------------------------------------------------------- networks
class MNISTGenerator(nn.Module):
    """Appendix C.1, Table 1: FC 1024 -> FC 7x7x128 -> upconv 64 -> upconv 1 (sigmoid output)."""

    def __init__(self, in_dim=74):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
                                nn.Linear(1024, 7 * 7 * 128), nn.BatchNorm1d(7 * 7 * 128), nn.ReLU())
        self.up = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, v):
        return self.up(self.fc(v).view(-1, 128, 7, 7))


class MNISTDiscriminatorQ(nn.Module):
    """Table 1: a shared body (conv 64 -> conv 128 -> FC 1024, leaky ReLU 0.1) with a D head (1 logit) and a
    Q head (FC 128 -> BN -> lReLU -> FC to the code parameters)."""

    def __init__(self, q_dim=14):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.1),
                                  nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.Flatten(),
                                  nn.Linear(128 * 7 * 7, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.1))
        self.d = nn.Linear(1024, 1)
        self.q = nn.Sequential(nn.Linear(1024, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Linear(128, q_dim))

    def forward(self, x):
        h = self.body(x)
        return self.d(h).squeeze(-1), self.q(h)


class MLPGenerator(nn.Module):
    def __init__(self, in_dim, out_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, v):
        return self.net(v)


class MLPDiscriminatorQ(nn.Module):
    def __init__(self, in_dim=2, q_dim=6, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.LeakyReLU(0.1), nn.Linear(hidden, hidden),
                                  nn.LeakyReLU(0.1))
        self.d, self.q = nn.Linear(hidden, 1), nn.Linear(hidden, q_dim)

    def forward(self, x):
        h = self.body(x)
        return self.d(h).squeeze(-1), self.q(h)


# ---------------------------------------------------------------------------------------------------- training
def infogan_step(G, DQ, opt_g, opt_d, real, spec, lam_cat=1.0, lam_cont=0.1, info=True):
    """One InfoGAN iteration (Eq. 6).
      D/Q step: maximise the GAN objective for D and lambda * L_I for Q (shared body).
      G step:   non-saturating GAN loss - lambda * L_I.
    info=False gives the paper's baseline: a regular GAN whose Q is trained (to MEASURE L_I) but G ignores it.
    Returns (d_loss, g_loss, L_I estimate)."""
    n = len(real)
    z, cat, cont = spec.sample(n)
    fake = G(spec.pack(z, cat, cont))
    d_real, _ = DQ(real)
    d_fake, q_fake = DQ(fake.detach())
    lq_cat, lq_cont = mi_terms(q_fake, cat, cont, spec)
    loss_d = -(F.logsigmoid(d_real).mean() + F.logsigmoid(-d_fake).mean()) - (lam_cat * lq_cat + lam_cont * lq_cont)
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()

    d_fake, q_fake = DQ(fake)
    lq_cat, lq_cont = mi_terms(q_fake, cat, cont, spec)
    loss_g = -F.logsigmoid(d_fake).mean()
    if info:
        loss_g = loss_g - (lam_cat * lq_cat + lam_cont * lq_cont)
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
    return loss_d.item(), loss_g.item(), (lq_cat + lq_cont).item() + spec.entropy()


# ---------------------------------------------------------------------------------------------------- evaluation
def cluster_accuracy(pred, labels, k=None):
    """'Matching each category in c1 to a digit type' (Section 7.2): the best one-to-one assignment (Hungarian)."""
    from scipy.optimize import linear_sum_assignment
    k = k or int(max(pred.max(), labels.max()) + 1)
    M = torch.zeros(k, k)
    for p, l in zip(pred.tolist(), labels.tolist()):
        M[p, l] += 1
    rows, cols = linear_sum_assignment(-M.numpy())
    return M[rows, cols].sum().item() / len(pred)


def mutual_information_discrete(joint):
    """Exact I(C; X) = H(C) - H(C|X) from a joint probability table P(c, x)."""
    pc, px = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
    m = joint > 0
    return (joint[m] * (joint[m] / (pc @ px)[m]).log()).sum()


def lemma_5_1_sides(Px, Py_given_x, f):
    """Both sides of Lemma 5.1 for discrete X, Y: E_{x, y|x}[f(x, y)] and E_{x, y|x, x'|y}[f(x', y)]."""
    joint = Px[:, None] * Py_given_x                                                # P(x, y)
    lhs = (joint * f).sum()
    Px_given_y = joint / joint.sum(0, keepdim=True)                                  # P(x'|y)
    rhs = (joint.sum(0) * (Px_given_y * f).sum(0)).sum()                             # sum_y P(y) E_{x'|y} f(x', y)
    return lhs, rhs


def count_params(m):
    return sum(p.numel() for p in m.parameters())
