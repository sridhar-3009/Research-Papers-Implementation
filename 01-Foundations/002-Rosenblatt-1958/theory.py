"""Rosenblatt's probability theory of A-unit activity, Equations (1)-(3).

An A-unit (association cell) has x excitatory and y inhibitory connections
("origin points") to random points of the retina. A stimulus lights up a
proportion R of the retina. The unit fires if

    e - i >= theta

where e = number of its excitatory points that are lit, i = inhibitory ones.

  Pa = probability that a random A-unit fires for a stimulus of size R    Eq. (1)
  Pc = probability that an A-unit that fired for stimulus S1 also fires
       for a second stimulus S2                                         Eq. (2)
  Pc_min = (1 - L)^x (1 - G)^y                                          Eq. (3)

These are for a very large retina, where every connection lands on a lit
point independently with probability R.
"""

from functools import lru_cache
from math import comb


def _binom(n, k, p):
    """Probability of exactly k successes in n tries, each with probability p."""
    return comb(n, k) * p ** k * (1 - p) ** (n - k)


def Pa(R, x, y, theta):
    """Eq. (1): expected proportion of A-units activated by a stimulus of size R.

    e ~ Binomial(x, R) lit excitatory points, i ~ Binomial(y, R) lit inhibitory
    points; sum the probability of every (e, i) with e - i >= theta.
    """
    return sum(_binom(x, e, R) * _binom(y, i, R)
               for e in range(x + 1) for i in range(y + 1) if e - i >= theta)


def Pc(R, L, G, x, y, theta):
    """Eq. (2): P(A-unit responds to S2 | it responded to S1).

    Going from S1 to S2:
      L = proportion of S1's lit points that are NOT lit in S2 ("lost"),
      G = proportion of the points left dark by S1 that ARE lit in S2 ("gained").
    Of the unit's e lit excitatory points, l_e ~ Bin(e, L) go dark; of its x - e
    dark ones, g_e ~ Bin(x - e, G) light up (same for inhibitory: l_i, g_i).
    The unit fires for S2 if  e - i - l_e + l_i + g_e - g_i >= theta.
    """
    both = 0.0
    for e in range(x + 1):
        for i in range(y + 1):
            if e - i < theta:
                continue                             # didn't fire for S1
            p_s1 = _binom(x, e, R) * _binom(y, i, R)
            p_s2 = 0.0
            for le in range(e + 1):
                for li in range(i + 1):
                    for ge in range(x - e + 1):
                        for gi in range(y - i + 1):
                            if e - i - le + li + ge - gi >= theta:
                                p_s2 += (_binom(e, le, L) * _binom(i, li, L)
                                         * _binom(x - e, ge, G) * _binom(y - i, gi, G))
            both += p_s1 * p_s2
    pa = Pa(R, x, y, theta)
    return both / pa if pa > 0 else 0.0


def Pc_min(L, G, x, y):
    """Eq. (3): the smallest Pc can get (reached at high thresholds, where the unit
    only keeps firing if NOTHING about its connections changes)."""
    return (1 - L) ** x * (1 - G) ** y


# ---- Converting "how the two stimuli relate" into L and G ----

def overlap_to_LG(R, C):
    """Two stimuli of the same size R that share a fraction C of their lit points.

    C = 0: completely separate (no points in common), C = 1: identical.
      L = 1 - C                  (the part of S1 not shared is lost)
      G = R(1 - C) / (1 - R)     (S2's new points, as a share of S1's dark area)
    """
    return 1 - C, R * (1 - C) / (1 - R)


# ---- Rosenblatt's Eq. (4): the shape of every learning curve ----

def learning_curve(n, c1, c2, c3, c4, pa, Ne):
    """Eq. (4): P = P(N_ar > 0) * Phi(Z),  Z = (c1 n + c2) / sqrt(c3 n + c4 n^2).

    n = number of stimuli learned per response. P(N_ar > 0) = 1 - (1 - Pa)^Ne is
    the chance that at least one of the Ne effective units responds at all.
    The paper's formulas for c1..c4 are too damaged in the scan to rebuild, so
    this function is here to show the SHAPE of the law; simulate.py measures real
    curves instead.
    """
    from math import erf, sqrt
    Z = (c1 * n + c2) / sqrt(c3 * n + c4 * n * n)
    phi = 0.5 * (1 + erf(Z / sqrt(2)))
    return (1 - (1 - pa) ** Ne) * phi
