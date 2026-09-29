"""Adam and the optimizers it is compared with - Kingma & Ba (2015) - written from scratch.

Every optimizer has the same interface:
    opt = Adam(alpha=0.001)
    params = opt.step(params, grads)     # lists of arrays -> updated list of arrays
They work on NumPy arrays and on PyTorch tensors (only +, *, /, ** and abs are used,
plus a max for AdaMax).

  Adam        Algorithm 1 (with bias correction, Section 3)
  AdaMax      Algorithm 2 (Section 7.1): the infinity-norm variant
  AdaGrad     Duchi et al. 2011 (Section 5)
  RMSProp     Tieleman & Hinton 2012 (Section 5), optionally with momentum
  AdaDelta    Zeiler 2012 (used in Figure 2)
  SGDNesterov SGD with Nesterov momentum (Sutskever et al. 2013, Paper 008)
  TemporalAverage   Section 7.2: exponential moving average of the parameters
"""

import numpy as np


def _max(a, b):
    """Element-wise maximum for NumPy arrays or torch tensors."""
    if isinstance(a, np.ndarray):
        return np.maximum(a, b)
    import torch
    return torch.maximum(a, b)


def _zeros_like(x):
    return x * 0


class Adam:
    """Algorithm 1.

        m_t = beta1 m_{t-1} + (1 - beta1) g_t            first moment (mean of g)
        v_t = beta2 v_{t-1} + (1 - beta2) g_t^2          second raw moment (mean of g^2)
        m^_t = m_t / (1 - beta1^t)                       bias correction (Section 3)
        v^_t = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - alpha m^_t / (sqrt(v^_t) + eps)

    bias_correction=False turns Adam into "RMSProp with momentum" (Section 6.4).
    decay_sqrt_t=True uses alpha_t = alpha / sqrt(t) (the logistic-regression
    experiments and the convergence theorem, Section 4)."""

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, bias_correction=True, decay_sqrt_t=False):
        self.alpha, self.beta1, self.beta2, self.eps = alpha, beta1, beta2, eps
        self.bias_correction, self.decay_sqrt_t = bias_correction, decay_sqrt_t
        self.t, self.m, self.v = 0, None, None

    def step(self, params, grads):
        if self.m is None:
            self.m = [_zeros_like(p) for p in params]                 # m_0 = 0
            self.v = [_zeros_like(p) for p in params]                 # v_0 = 0
        self.t += 1
        alpha = self.alpha / np.sqrt(self.t) if self.decay_sqrt_t else self.alpha
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            if self.bias_correction:
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            else:
                m_hat, v_hat = self.m[i], self.v[i]
            out.append(p - alpha * m_hat / (v_hat ** 0.5 + self.eps))
        return out


class AdaMax:
    """Algorithm 2: replace the L2-based v_t by an exponentially weighted infinity norm
        u_t = max(beta2 u_{t-1}, |g_t|)
        theta_t = theta_{t-1} - (alpha / (1 - beta1^t)) m_t / u_t
    No bias correction is needed for u_t, and every step satisfies |step| <= alpha."""

    def __init__(self, alpha=0.002, beta1=0.9, beta2=0.999, eps=1e-12):
        self.alpha, self.beta1, self.beta2, self.eps = alpha, beta1, beta2, eps
        self.t, self.m, self.u = 0, None, None

    def step(self, params, grads):
        if self.m is None:
            self.m = [_zeros_like(p) for p in params]
            self.u = [_zeros_like(p) for p in params]
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.u[i] = _max(self.beta2 * self.u[i], abs(g))
            out.append(p - (self.alpha / (1 - self.beta1 ** self.t)) * self.m[i] / (self.u[i] + self.eps))
        return out


class AdaGrad:
    """theta_{t+1} = theta_t - alpha g_t / sqrt(sum_{i<=t} g_i^2)   (Section 5)."""

    def __init__(self, alpha=0.01, eps=1e-8):
        self.alpha, self.eps, self.s = alpha, eps, None

    def step(self, params, grads):
        if self.s is None:
            self.s = [_zeros_like(p) for p in params]
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.s[i] = self.s[i] + g * g
            out.append(p - self.alpha * g / (self.s[i] ** 0.5 + self.eps))
        return out


class RMSProp:
    """Divide the gradient by a running RMS of recent gradients (no bias correction).
    momentum > 0 gives the 'RMSProp with momentum' of Graves (2013) (Section 5)."""

    def __init__(self, alpha=0.001, rho=0.9, momentum=0.0, eps=1e-8):
        self.alpha, self.rho, self.momentum, self.eps = alpha, rho, momentum, eps
        self.r, self.buf = None, None

    def step(self, params, grads):
        if self.r is None:
            self.r = [_zeros_like(p) for p in params]
            self.buf = [_zeros_like(p) for p in params]
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.r[i] = self.rho * self.r[i] + (1 - self.rho) * g * g
            self.buf[i] = self.momentum * self.buf[i] + g / (self.r[i] ** 0.5 + self.eps)
            out.append(p - self.alpha * self.buf[i])
        return out


class AdaDelta:
    """Zeiler (2012): step = -sqrt(E[dx^2] + eps) / sqrt(E[g^2] + eps) * g. No learning rate."""

    def __init__(self, rho=0.95, eps=1e-6):
        self.rho, self.eps, self.eg, self.ex = rho, eps, None, None

    def step(self, params, grads):
        if self.eg is None:
            self.eg = [_zeros_like(p) for p in params]
            self.ex = [_zeros_like(p) for p in params]
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.eg[i] = self.rho * self.eg[i] + (1 - self.rho) * g * g
            dx = -((self.ex[i] + self.eps) ** 0.5) / ((self.eg[i] + self.eps) ** 0.5) * g
            self.ex[i] = self.rho * self.ex[i] + (1 - self.rho) * dx * dx
            out.append(p + dx)
        return out


class SGDNesterov:
    """SGD with Nesterov momentum in the common 'look-ahead folded in' form:
        v <- mu v - lr g;   theta <- theta + mu v - lr g
    (equivalent to Eqs. 3-4 of Paper 008, written in terms of the gradient at theta)."""

    def __init__(self, lr=0.01, mu=0.9):
        self.lr, self.mu, self.v = lr, mu, None

    def step(self, params, grads):
        if self.v is None:
            self.v = [_zeros_like(p) for p in params]
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i] = self.mu * self.v[i] - self.lr * g
            out.append(p + self.mu * self.v[i] - self.lr * g)
        return out


class TemporalAverage:
    """Section 7.2: theta_bar_t = beta2 theta_bar_{t-1} + (1 - beta2) theta_t, theta_bar_0 = 0,
    with the same bias correction as Adam: theta_hat_t = theta_bar_t / (1 - beta2^t).
    Use theta_hat (not the noisy last iterate) for evaluation."""

    def __init__(self, beta=0.999):
        self.beta, self.t, self.avg = beta, 0, None

    def update(self, params):
        if self.avg is None:
            self.avg = [_zeros_like(p) for p in params]
        self.t += 1
        self.avg = [self.beta * a + (1 - self.beta) * p for a, p in zip(self.avg, params)]
        return [a / (1 - self.beta ** self.t) for a in self.avg]


OPTIMIZERS = {"Adam": Adam, "AdaMax": AdaMax, "AdaGrad": AdaGrad, "RMSProp": RMSProp,
              "AdaDelta": AdaDelta, "SGDNesterov": SGDNesterov}
