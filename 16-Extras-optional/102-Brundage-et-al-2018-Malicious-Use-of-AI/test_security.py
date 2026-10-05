import numpy as np

import security as S


def test_cheaper_tailoring_expands_threats_and_defence_restores():
    hi, lo = S.attack_economics(20), S.attack_economics(0.2)
    assert lo["tailored"] > 10 * hi["tailored"] and lo["expected harm"] > 5 * hi["expected harm"]
    f = S.defence_needed(0.2, 20)
    assert 0 < f < 1
    assert abs(S.attack_economics(0.2, defence_factor=f)["expected harm"] - hi["expected harm"]) / hi["expected harm"] < 0.02
    assert S.capable_actors(1.0, [0.5, 2.0, 3.0, 0.1]) == 0.5


def test_fgsm_increases_loss_and_respects_bounds():
    Xtr, ytr, Xte, yte = S.load_digits()
    W, b = S.train_softmax(Xtr[:400], ytr[:400], epochs=100)
    Xa = S.fgsm(W, b, Xte, yte, 0.1)
    assert Xa.min() >= 0 and Xa.max() <= 1 and np.abs(Xa - Xte).max() <= 0.1 + 1e-12
    assert S.accuracy(W, b, Xa, yte) < S.accuracy(W, b, Xte, yte) - 0.1


def test_poisoning():
    rng = np.random.default_rng(0)
    y = np.array([7, 7, 1, 2, 7, 3])
    yt = S.poison_labels(y, 1.0, rng, targeted=True)
    assert list(yt) == [1, 1, 1, 2, 1, 3]
    yr = S.poison_labels(np.arange(10), 0.5, rng)
    assert (yr != np.arange(10)).sum() == 5                            # flipped labels always change
    Xtr, ytr, Xte, yte = S.load_digits()
    W, b = S.train_softmax(Xtr, S.poison_labels(ytr, 1.0, rng, targeted=True), epochs=100)
    assert ((Xte[yte == 7] @ W + b).argmax(1) == 1).mean() > 0.9
