import numpy as np

from lsgen.eval.representation import deltas_per_make, knn_accuracy


def hierarchy(seed=0, d=32, n=40):
    """3 makes (2, 2, 1 subclasses): make centres far apart, subclass centres nearer, small noise."""
    rng = np.random.default_rng(seed)
    centres = {c: 3 * rng.normal(size=d) for c in range(3)}
    x, make, fine, f = [], [], [], 0
    for c, k in enumerate([2, 2, 1]):
        for _ in range(k):
            sub = centres[c] + 1.0 * rng.normal(size=d)
            x.append(sub + 0.3 * rng.normal(size=(n, d)))
            make += [c] * n
            fine += [f] * n
            f += 1
    return np.concatenate(x), np.array(make), np.array(fine)


def test_deltas_ordering_and_single_class_make():
    x, make, fine = hierarchy()
    r = deltas_per_make(x, make, fine).set_index("make_id")
    assert (r.loc[[0, 1], "d0"] < r.loc[[0, 1], "d1"]).all() and (r.d1.dropna() < r.loc[[0, 1], "d2"]).all()
    assert np.isnan(r.loc[2, "d1"]) and r.loc[2, "K_c"] == 1 and r.n.sum() == len(x)


def test_deltas_known_value():
    x = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)  # two identical pairs, orthogonal
    r = deltas_per_make(x, np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])).set_index("make_id")
    assert np.allclose(r.d0, 0) and np.allclose(r.d2, 2) and r.d1.isna().all()


def test_knn_accuracy():
    x, make, fine = hierarchy()
    idx = np.arange(len(x))
    assert knn_accuracy(x[idx % 2 == 0], make[idx % 2 == 0], x[idx % 2 == 1], make[idx % 2 == 1]) == 1.0
    assert knn_accuracy(x[idx % 2 == 0], fine[idx % 2 == 0], x[idx % 2 == 1], fine[idx % 2 == 1]) > 0.95
