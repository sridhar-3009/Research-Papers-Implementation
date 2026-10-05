import numpy as np

import supcon as S


def test_gradients_match_finite_differences_and_jensen():
    rng = np.random.default_rng(1)
    Z = S.normalize(rng.standard_normal((8, 4)))[0]
    lab = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    for kind in ("out", "in"):
        L, dZ = S.contrastive_loss(Z, lab, 0.3, kind)
        i, j = 3, 2
        Zp, Zm = Z.copy(), Z.copy()
        Zp[i, j] += 1e-6; Zm[i, j] -= 1e-6
        num = (S.contrastive_loss(Zp, lab, 0.3, kind)[0] - S.contrastive_loss(Zm, lab, 0.3, kind)[0]) / 2e-6
        assert abs(num - dZ[i, j]) < 1e-6
    assert S.contrastive_loss(Z, lab, 0.3, "in")[0] <= S.contrastive_loss(Z, lab, 0.3, "out")[0] + 1e-12


def test_single_positive_reduces_to_simclr_form():
    rng = np.random.default_rng(2)
    Z = S.normalize(rng.standard_normal((6, 3)))[0]
    ids = np.array([0, 1, 2, 0, 1, 2])                                # each sample has exactly one positive
    Lo, _ = S.contrastive_loss(Z, ids, 0.5, "out")
    Li, _ = S.contrastive_loss(Z, ids, 0.5, "in")
    assert abs(Lo - Li) < 1e-12                                       # with |P(i)| = 1 Eq. 2 = Eq. 3 = Eq. 1


def test_normalize_backward_and_hard_positive():
    rng = np.random.default_rng(3)
    U = rng.standard_normal((1, 5))
    Z, n = S.normalize(U)
    g = rng.standard_normal((1, 5))
    du = S.normalize_backward(Z, n, g)
    e = 1e-6
    num = np.array([(S.normalize(U + e * np.eye(5)[k])[0] - S.normalize(U - e * np.eye(5)[k])[0]) @ g.T / (2 * e)
                    for k in range(5)]).ravel()
    assert np.allclose(du.ravel(), num, atol=1e-6)
    h = S.hard_positive_gradient([0.0, 0.95])
    assert h[0.0] > 20 * h[0.95]


def test_supcon_training_learns():
    Xtr, ytr, Xte, yte = S.load_digits_split()
    p, hist = S.train_contrastive(Xtr[:400], ytr[:400], "out", epochs=15, lr=3e-3)
    assert hist[-1] < hist[0]
    W, b = S.linear_probe(p, Xtr[:400], ytr[:400], epochs=200)
    assert S.probe_accuracy(p, W, b, Xte, yte) > 0.8
