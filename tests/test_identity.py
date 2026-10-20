import numpy as np

from lsgen.eval.identity import gate, identity_score, make_metrics, pair_zscore, precision_at


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_pair_zscore_standardises_off_diagonal_and_ignores_scale():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(10, 10))
    z = pair_zscore(s)
    off = ~np.eye(10, dtype=bool)
    assert np.isclose(z[off].mean(), 0) and np.isclose(z[off].std(), 1) and np.allclose(pair_zscore(3 * s + 2), z)


def test_identity_score_is_zero_when_both_encoders_agree():
    x = unit(np.random.default_rng(0).normal(size=(12, 6)))
    assert np.allclose(identity_score(x, x), 0)


def test_gate_keeps_only_the_k_nearest_neighbours():
    sim = np.array([[1, .9, .1, .5], [.9, 1, .2, .3], [.1, .2, 1, .8], [.5, .3, .8, 1]])
    g = gate(np.ones((4, 4)), sim, k=1)
    assert np.array_equal(np.isfinite(g), np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], bool))


def test_precision_at_excludes_self():
    fine = np.array([0, 0, 1, 1])
    assert precision_at(np.equal.outer(fine, fine).astype(float), fine, n=1) == 1.0


def test_identity_recovers_subclass_hidden_behind_appearance():
    """DINO = subclass + appearance, DreamSim = appearance only: subtracting the look-alike part should beat DINO alone."""
    rng, n = np.random.default_rng(0), 120
    fine, look = rng.integers(0, 4, n), rng.integers(0, 4, n)
    sub, app = 3 * rng.normal(size=(4, 16)), 4 * rng.normal(size=(4, 16))
    dino = unit(sub[fine] + app[look] + 0.5 * rng.normal(size=(n, 16)))
    dreamsim = unit(app[look] + 0.5 * rng.normal(size=(n, 16)))
    m = make_metrics(dino, dreamsim, fine, k=30, n=15)  # beyond the ~7 images sharing subclass and look, which DINO already ranks first
    assert m["gain_auroc"] > 0.05 and m["gain_p15_gated"] > 0.1 and 0.2 < m["p15_chance"] < 0.3
