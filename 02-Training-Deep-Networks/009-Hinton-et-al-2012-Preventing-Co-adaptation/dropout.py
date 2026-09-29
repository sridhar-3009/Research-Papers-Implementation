"""Dropout as introduced by Hinton, Srivastava, Krizhevsky, Sutskever & Salakhutdinov (2012).

  Training: on every presentation of every case, each hidden unit is dropped
            (output set to 0) with probability 0.5, and each input pixel with 0.2.
  Max-norm: the incoming weight vector of each hidden unit has squared length <= l
            (l = 15); if an update breaks this, the vector is scaled back down.
  Testing:  the "mean network" - every unit present, outgoing weights multiplied by the
            probability the unit was kept (halved for hidden units).
  Appendix A.1: init N(0, 0.01^2); minibatches of 100; learning rate 10 * 0.998^epoch
            applied as (1 - momentum) * lr; momentum rising linearly 0.5 -> 0.99.

Written with explicit masks (no nn.Dropout) so every step is visible.
"""

import numpy as np
import torch


class DropoutNet(torch.nn.Module):
    """sizes e.g. [784, 800, 800, 10]; logistic hidden units; softmax output.
    keep_input / keep_hidden = probability a unit is KEPT (0.8 / 0.5 in the paper)."""

    def __init__(self, sizes, keep_input=1.0, keep_hidden=1.0, init_std=0.01, act="logistic", seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.W = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(a, b, generator=gen) * init_std) for a, b in zip(sizes[:-1], sizes[1:])])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(b)) for b in sizes[1:]])
        self.keep = [keep_input] + [keep_hidden] * (len(sizes) - 2)   # keep prob for each layer's INPUT
        self.act = torch.sigmoid if act == "logistic" else torch.relu

    def forward(self, x, mode="mean", masks=None):
        """mode="train": sample a thinned network (fresh masks for every case).
        mode="mean":  the mean network - no dropout, weights scaled by keep prob.
        masks: optional list of fixed 0/1 masks, one per layer input (for exact
        enumeration of sub-networks)."""
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            p = self.keep[i]
            if masks is not None:
                h = h * masks[i]
                Wi = W
            elif mode == "train" and p < 1:
                h = h * (torch.rand_like(h) < p).float()        # drop each unit independently
                Wi = W
            else:
                Wi = W * p                                     # mean network: scale outgoing weights
            h = h @ Wi + b
            if i < len(self.W) - 1:
                h = self.act(h)
        return h                                               # logits


@torch.no_grad()
def max_norm_(net, max_sq_len=15.0):
    """Rescale every hidden unit's incoming weight vector whose squared length
    exceeds l, so that it has squared length exactly l (Appendix A.1)."""
    for W in net.W[:-1]:                                       # hidden layers only
        sq = (W ** 2).sum(0)
        scale = torch.where(sq > max_sq_len, torch.sqrt(max_sq_len / sq), torch.ones_like(sq))
        W.mul_(scale)


def train(net, X, y, epochs, dropout=True, max_norm=15.0, lr0=10.0, lr_decay=0.998, mom_start=0.5,
          mom_end=0.99, mom_epochs=500, batch=100, Xte=None, yte=None, seed=0, device="cpu"):
    """The training procedure of Appendix A.1:
        dw_t = p_t dw_{t-1} - (1 - p_t) eps_t <grad>
        eps_t = eps_0 f^t,   p_t = momentum, rising linearly from 0.5 to 0.99 over T epochs.
    (Pass lr_decay / mom_epochs rescaled for runs much shorter than the paper's 3000 epochs.)
    Returns test-error history (one entry per epoch) if a test set is given."""
    gen = torch.Generator().manual_seed(seed)
    vel = [torch.zeros_like(p) for p in net.parameters()]
    history = []
    for epoch in range(epochs):
        eps = lr0 * lr_decay ** epoch
        mom = mom_start + (mom_end - mom_start) * min(1.0, epoch / mom_epochs)
        perm = torch.randperm(len(X), generator=gen).to(X.device)
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            logits = net(X[idx], mode="train" if dropout else "mean")
            loss = torch.nn.functional.cross_entropy(logits, y[idx])      # average over the minibatch
            net.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p, v in zip(net.parameters(), vel):
                    v.mul_(mom).sub_((1 - mom) * eps * p.grad)
                    p.add_(v)
            if max_norm:
                max_norm_(net, max_norm)
        if Xte is not None:
            history.append(test_errors(net, Xte, yte))
    return history


@torch.no_grad()
def test_errors(net, X, y, mode="mean"):
    """Number of misclassified test cases (the paper counts errors out of 10,000)."""
    return int((net(X, mode=mode).argmax(1) != y).sum())


@torch.no_grad()
def mc_average_errors(net, X, y, k, seed=0):
    """Average the predicted probabilities of k randomly sampled dropout networks."""
    torch.manual_seed(seed)
    probs = sum(torch.softmax(net(X, mode="train"), 1) for _ in range(k)) / k
    return int((probs.argmax(1) != y).sum())
