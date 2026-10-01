"""The problems of Chapters 3-4 of Sutskever's thesis, generated exactly as described (Section 4.4).

Pathological long-term dependency problems (inputs (T', N, d), targets, mask, loss kind):
  addition, multiplication, xor   two marked values among noise: target f(u_I, u_J) at the end
  temporal_order                  2 special symbols among 6; 4 classes at the end
  temporal_order_3bit             3 special symbols; 8 classes
  random_permutation              symbols 1..100; first == last (in {1, 2}); predict the next symbol
  memorization                    5 random bits, T blanks, a trigger, then reproduce the bits

  bouncing_balls                  Chapter 3 / 4.5: synthetic videos of balls bouncing in a box
  success                         Hochreiter & Schmidhuber's criterion: < 1% of test sequences wrong
"""

import numpy as np
import torch


def _i(rng, a, b):
    """i[a, b]: a uniform random integer in [a, b] (inclusive)."""
    return int(rng.integers(a, b + 1))


# ---------------------------------------------------------------------------
# 4.4.1 addition, multiplication, xor
# ---------------------------------------------------------------------------

def marked_pairs(T, N, rng, op):
    """Each sequence: length T' ~ i[T, 11T/10]; I ~ i[1, T'/10], J ~ i[T'/10, T'/2]; inputs (u_t, marker_t)
    with u_t ~ U[0,1] (or random bits for xor); the target at T' is (u_I + u_J)/2, u_I u_J or u_I xor u_J.
    Sequences of different lengths are right-aligned (padded at the start with zeros) so that every
    target is at the last step."""
    lens = [_i(rng, T, 11 * T // 10) for _ in range(N)]
    L = max(lens)
    X = np.zeros((L, N, 2))
    Y = np.zeros((L, N, 1))
    M = np.zeros((L, N))
    for n, Tn in enumerate(lens):
        u = rng.integers(0, 2, Tn).astype(float) if op == "xor" else rng.random(Tn)
        I = _i(rng, 1, max(1, Tn // 10)) - 1
        J = _i(rng, Tn // 10, Tn // 2) - 1
        if J == I:
            J += 1
        off = L - Tn
        X[off:, n, 0] = u
        X[off + I, n, 1] = X[off + J, n, 1] = 1.0
        Y[-1, n, 0] = {"add": (u[I] + u[J]) / 2, "mul": u[I] * u[J], "xor": float(int(u[I]) ^ int(u[J]))}[op]
        M[-1, n] = 1
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), torch.tensor(M, dtype=torch.float32), "mse"


def addition(T, N, rng):
    return marked_pairs(T, N, rng, "add")


def multiplication(T, N, rng):
    return marked_pairs(T, N, rng, "mul")


def xor(T, N, rng):
    return marked_pairs(T, N, rng, "xor")


# ---------------------------------------------------------------------------
# 4.4.2-4.4.3 temporal order
# ---------------------------------------------------------------------------

def _symbols(T, N, n_sym, positions, rng):
    X = np.zeros((T, N, n_sym))
    cls = np.zeros(N, dtype=np.int64)
    for n in range(N):
        seq = rng.integers(2, n_sym, T)                       # irrelevant symbols 3..6 -> ids 2..5
        code = 0
        for lo, hi in positions:
            s = int(rng.integers(0, 2))                        # special symbol 1 or 2 -> id 0 or 1
            seq[_i(rng, lo, hi)] = s
            code = 2 * code + s
        X[np.arange(T), n, seq] = 1
        cls[n] = code
    Y = np.zeros((T, N), dtype=np.int64)
    Y[-1] = cls
    M = np.zeros((T, N)); M[-1] = 1
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y), torch.tensor(M, dtype=torch.float32), "ce"


def temporal_order(T, N, rng):
    """I ~ i[T/10, 2T/10], J ~ i[5T/10, 6T/10]; target: the ordered pair (a, b), 4 classes."""
    return _symbols(T, N, 6, [(T // 10, 2 * T // 10), (5 * T // 10, 6 * T // 10)], rng)


def temporal_order_3bit(T, N, rng):
    """Three special symbols in [T/10, 2T/10], [3T/10, 4T/10], [6T/10, 7T/10]; 8 classes."""
    return _symbols(T, N, 6, [(T // 10, 2 * T // 10), (3 * T // 10, 4 * T // 10), (6 * T // 10, 7 * T // 10)], rng)


# ---------------------------------------------------------------------------
# 4.4.4 random permutation, 4.4.5 memorization
# ---------------------------------------------------------------------------

def random_permutation(T, N, rng):
    """Symbols 1..100 (ids 0..99). First and last symbol are equal and in {1, 2}; the rest from 3..100.
    Target at every step = the next input; only the last one is predictable (and it is what counts)."""
    seq = rng.integers(2, 100, (T, N))
    first = rng.integers(0, 2, N)
    seq[0], seq[-1] = first, first
    X = np.zeros((T, N, 100))
    X[np.arange(T)[:, None], np.arange(N)[None, :], seq] = 1
    Y = np.zeros((T, N), dtype=np.int64)
    Y[:-1] = seq[1:]
    M = np.ones((T, N)); M[-1] = 0
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y), torch.tensor(M, dtype=torch.float32), "ce"


def memorization(T, N, rng, n_bits=5, n_values=2):
    """5 random bits (or 10 symbols from 5 values for the '20-bit' variant), then T blank inputs; at
    step T + 5 a trigger appears; the last n_bits targets are the original symbols. Symbols:
    0..n_values-1 = data, n_values = blank (also the target before the end), n_values+1 = trigger."""
    L = T + 2 * n_bits
    K = n_values + 2
    data = rng.integers(0, n_values, (n_bits, N))
    seq = np.full((L, N), n_values)
    seq[:n_bits] = data
    seq[T + n_bits - 1] = n_values + 1                       # trigger in the n_bits-th step before the end
    X = np.zeros((L, N, K))
    X[np.arange(L)[:, None], np.arange(N)[None, :], seq] = 1
    Y = np.full((L, N), n_values, dtype=np.int64)
    Y[-n_bits:] = data
    M = np.ones((L, N))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y), torch.tensor(M, dtype=torch.float32), "ce"


# ---------------------------------------------------------------------------
# success criterion
# ---------------------------------------------------------------------------

@torch.no_grad()
def error_rate(model, data):
    """Fraction of sequences 'misclassified': for continuous targets, |error| > 0.04 at the end;
    for symbols, any wrong prediction among the steps that decide success (the masked ones; for the
    random permutation problem only the last predictable step)."""
    X, Y, M, kind = data
    o, _ = model(X)
    if kind == "mse":
        wrong = ((o[-1, :, 0] - Y[-1, :, 0]).abs() > 0.04)
    else:
        pred = o.argmax(-1)
        if M.sum(0).max() == len(M) - 1:                       # random permutation: judge the last target only
            wrong = pred[-2] != Y[-2]
        else:
            wrong = ((pred != Y) & (M > 0)).any(0)
    return wrong.float().mean().item()


# ---------------------------------------------------------------------------
# Bouncing balls (Chapters 3 and 4)
# ---------------------------------------------------------------------------

def bouncing_balls(T, res=30, n_balls=3, radius=None, rng=None):
    """A (T, res*res) video of n_balls balls moving at constant speed inside the unit square,
    bouncing elastically off the walls and each other. Pixel = clipped sum of Gaussian blobs."""
    rng = rng or np.random.default_rng()
    radius = radius or 0.12
    pos = rng.uniform(radius, 1 - radius, (n_balls, 2))
    for _ in range(100):                                       # avoid initial overlaps
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(n_balls) * 10
        if d.min() > 2 * radius:
            break
        pos = rng.uniform(radius, 1 - radius, (n_balls, 2))
    ang = rng.uniform(0, 2 * np.pi, n_balls)
    vel = 0.03 * np.stack([np.cos(ang), np.sin(ang)], 1)
    grid = (np.arange(res) + 0.5) / res
    gx, gy = np.meshgrid(grid, grid)
    frames = []
    for _ in range(T):
        img = np.zeros((res, res))
        for p in pos:
            img += np.exp(-((gx - p[0]) ** 2 + (gy - p[1]) ** 2) / (radius ** 2))
        frames.append(np.clip(img, 0, 1).ravel())
        pos = pos + vel
        for k in range(2):                                     # walls
            low, high = pos[:, k] < radius, pos[:, k] > 1 - radius
            vel[low | high, k] *= -1
            pos[low, k] = 2 * radius - pos[low, k]
            pos[high, k] = 2 * (1 - radius) - pos[high, k]
        for i in range(n_balls):                               # ball-ball: swap the velocity components along the line
            for j in range(i + 1, n_balls):
                dvec = pos[i] - pos[j]
                dist = np.linalg.norm(dvec)
                if 0 < dist < 2 * radius and (vel[i] - vel[j]) @ dvec < 0:
                    nrm = dvec / dist
                    vi, vj = vel[i] @ nrm, vel[j] @ nrm
                    vel[i] += (vj - vi) * nrm
                    vel[j] += (vi - vj) * nrm
    return np.array(frames)


PROBLEMS = {"addition": addition, "multiplication": multiplication, "xor": xor, "temporal order": temporal_order,
            "3-bit temporal order": temporal_order_3bit, "random permutation": random_permutation,
            "5-bit memorization": memorization,
            "20-bit memorization": lambda T, N, rng: memorization(T, N, rng, n_bits=10, n_values=5)}
