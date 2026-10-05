import numpy as np

from lsgen.eval.representation import (cluster_space, deltas_per_make, distance_matrix, hungarian_acc, khat_error,
                                       knn_accuracy, knn_correct, l2norm, participation_ratio, probe_correct, r2_metric,
                                       r3_metric, r4_metric, r5_make, recall1_correct, select_k, trace_cov)


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


def blobs(k, n=60, d=20, sep=10.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([sep * np.eye(d)[i] + rng.normal(size=(n, d)) for i in range(k)]), np.repeat(np.arange(k), n)


def test_deltas_ordering_and_single_class_make():
    x, make, fine = hierarchy()
    r = deltas_per_make(x, make, fine).set_index("make_id")
    assert (r.loc[[0, 1], "d0"] < r.loc[[0, 1], "d1"]).all() and (r.d1.dropna() < r.loc[[0, 1], "d2"]).all()
    assert np.isnan(r.loc[2, "d1"]) and r.loc[2, "K_c"] == 1 and r.n.sum() == len(x)


def test_deltas_known_value_and_resampled_rows_exclude_self():
    x = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)  # two identical pairs, orthogonal
    make = fine = np.array([0, 0, 1, 1])
    r = deltas_per_make(x, make, fine).set_index("make_id")
    assert np.allclose(r.d0, 0) and np.allclose(r.d2, 2) and r.d1.isna().all()
    m = r3_metric(distance_matrix(x), make, fine)([0, 0, 1])  # repeated rows must not pair an image with its own copy
    assert m["d0"] == 0 and m["d2"] == 2 and np.isnan(m["eq1"]) and np.isnan(m["d0/d1"])


def test_r3_indicators_and_ratios():
    x, make, fine = hierarchy()
    m = r3_metric(distance_matrix(x), make, fine)(np.flatnonzero(make == 0))
    assert m["eq1"] == 1.0 and m["eq2"] == 1.0 and m["d0/d1"] < 1 and m["d1/d2"] < 1


def test_knn_probe_and_recall1():
    x, make, fine = hierarchy()
    odd, even = np.arange(len(x)) % 2 == 1, np.arange(len(x)) % 2 == 0
    assert knn_accuracy(x[even], make[even], x[odd], make[odd]) == 1.0
    assert knn_correct(x[even], fine[even], x[odd], fine[odd]).mean() > 0.95
    assert probe_correct(x[even], fine[even], x[odd], fine[odd]).mean() > 0.95
    m = make == 0
    assert recall1_correct(x[m], fine[m]).mean() > 0.95
    assert recall1_correct(np.array([[1., 0], [1, 0.01], [0, 1], [0.01, 1]]), np.array([0, 0, 1, 1])).all()


def test_hungarian_acc():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert hungarian_acc(y, np.array([2, 2, 0, 0, 1, 1])) == 1.0  # permuted ids
    assert hungarian_acc(y, np.zeros(6, int)) == 1 / 3
    assert np.isclose(hungarian_acc(y, np.array([0, 0, 1, 1, 2, 3])), 5 / 6)  # more clusters than classes


def test_select_k_recovers_blobs_and_single_cluster():
    x, _ = blobs(3)
    assert select_k(l2norm(x - x.mean(0))) == 3
    one = np.random.default_rng(0).normal(size=(300, 50))
    assert khat_error(one) == 0  # single Gaussian: gap-statistic fallback returns K = 1


def test_r2_metric_on_separated_subclasses():
    x, y = blobs(3)
    make = np.zeros(len(x), int)
    z = cluster_space(x, make)
    m = r2_metric(z, make, y, {0: 3})(np.arange(len(x)))
    assert m["acc"] == 1 and m["ari"] == 1 and "khat" not in m
    m = r2_metric(z, make, y, {0: 99}, label_free=True)(np.arange(len(x)))  # the oracle K is not used
    assert m["nmi"] > 0.99 and m["khat"] == 3


def test_r4_variation_ratio_and_effective_rank():
    rng = np.random.default_rng(0)
    centres = 5 * np.eye(8)[:3]
    x = np.concatenate([c + 0.2 * rng.normal(size=(100, 8)) for c in centres])
    fine = np.repeat(np.arange(3), 100)
    m = r4_metric(x, fine)(np.arange(300))
    assert m["var_ratio"] < 0.2 and 6 < m["pr_subclass"] <= 8  # isotropic 8-d noise within subclasses
    assert np.isclose(participation_ratio(np.outer(rng.normal(size=50), np.ones(4))), 1)  # rank one
    same = np.repeat(centres, 100, axis=0)
    assert r4_metric(same, fine)(np.arange(300))["var_ratio"] == 0 and trace_cov(same[:100]) == 0


def test_r5_stability():
    x, _ = blobs(3)
    z = cluster_space(x, np.zeros(len(x), int))
    r = r5_make(z, 3, n_boot=5)
    assert len(r["ari_seeds"]) == 3 and np.allclose(r["ari_seeds"], 1) and np.allclose(r["ari_boot"], 1)
    flat = np.random.default_rng(0).normal(size=(200, 20))
    assert np.mean(r5_make(flat, 2, n_boot=5)["ari_boot"]) < 0.9  # no structure -> unstable
