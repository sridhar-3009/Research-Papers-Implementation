import numpy as np

import debt as D


def test_logreg_learns_and_prediction_bias_is_small():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 2))
    y = (rng.random(3000) < D.sigmoid(2 * X[:, 0] - X[:, 1])).astype(float)
    w = D.fit_logreg(X, y)
    assert w[0] > 1.0 and w[1] < -0.4
    assert abs(D.prediction_bias(w, X, y)) < 0.01                   # calibrated on its own training data


def test_cace_removing_a_feature_moves_the_others():
    _, out = D.cace_experiment()
    assert out["remove feature x0"][0] > 0.3                        # x1 absorbs x0's credit
    assert out["change L2 from 0.001 to 0.05"][0] > 0.1


def test_legacy_feature_cleanup_hurts_only_the_model_that_used_it():
    Xtr, ytr, _ = D.product_number_world(n=6000, n_products=40)
    Xte, yte, _ = D.product_number_world(n=6000, n_products=40, sample_seed=1)
    Xc, yc, _ = D.product_number_world(n=6000, n_products=40, sample_seed=1, old_populated=False)
    P = 40
    both = D.fit_logreg(Xtr, ytr, lr=5.0)
    new = D.fit_logreg(Xtr[:, P:], ytr, lr=5.0)
    assert D.log_loss(both, Xc, yc) > D.log_loss(both, Xte, yte) + 0.01
    assert np.isclose(D.log_loss(new, Xc[:, P:], yc), D.log_loss(new, Xte[:, P:], yte))


def test_cascade_threshold_and_correlation_failures():
    cc = D.correction_cascade()
    assert cc["A with a v2"] < cc["A with a v1"]                    # a improves ...
    assert cc["A' with a v2 + OLD correction"] > cc["A' with a v1 + correction"]   # ... and A' gets worse
    td = D.threshold_drift()
    assert td["v2 with v1's fixed threshold"] < 0.85 < td["v2 precision (re-learned)"]
    cb = D.correlation_break()
    assert cb["accuracy after decoupling"] < cb["causal-only model after"] - 0.05
