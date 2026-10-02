import numpy as np

from lsgen.eval.synthesis import (ClassFidelity, choose_tau, frechet_distance, g3_metric, g6_make, kid, make_correct,
                                  prdc, prevalence_tv, rare_classes, smoothed_kl, tv_distance)

rng = np.random.default_rng(0)


def gauss(mu, n, d=8, s=1.0):
    return np.asarray(mu) + s * rng.normal(size=(n, d))


def test_frechet_distance_known():
    x = rng.normal(size=(20000, 4))
    assert frechet_distance(x, rng.normal(size=(20000, 4))) < 0.05
    assert abs(frechet_distance(x, x + np.array([3.0, 0, 0, 0])) - 9.0) < 1e-6  # pure mean shift: |mu|^2
    assert abs(frechet_distance(x, 2 * x) - 4.0) < 0.2  # pure scale: d * (2 - 1)^2


def test_kid_unbiased_and_sensitive():
    same = np.array([kid(gauss(0, 40), gauss(0, 40)) for _ in range(300)])
    assert abs(same.mean()) < 4 * same.std() / np.sqrt(len(same))  # unbiased: mean 0 within sampling error, even at n=40
    assert kid(gauss(0, 40), gauss(1.0, 40)) > 0.1
    assert np.isnan(kid(gauss(0, 1), gauss(0, 40)))
    big = kid(gauss(0, 3000), gauss(0, 3000), n_subsets=5, subset_size=500)  # subset path
    assert abs(big) < 0.01


def test_prdc_same_vs_disjoint_and_mode_dropping():
    real = gauss(0, 1000, 2)
    p = prdc(real, gauss(0, 1000, 2))
    assert all(0.85 < p[k] < 1.2 for k in p) and p["coverage"] > 0.9
    far = prdc(real, gauss(100, 1000, 2))
    assert far["precision"] == far["recall"] == far["coverage"] == far["density"] == 0
    half = np.concatenate([gauss(-6, 500, 2, 0.3), gauss(6, 500, 2, 0.3)])  # real has two modes, fake only one
    p = prdc(half, gauss(-6, 500, 2, 0.3))
    assert p["precision"] > 0.9 and 0.4 < p["coverage"] < 0.6
    assert np.isnan(prdc(real[:5], real)["precision"])


def test_g2_make_correct_tv():
    probs = np.array([[.2, .8], [.6, .4], [.4, .6]])
    assert make_correct(probs, np.array([1, 0, 0])).tolist() == [True, True, False]
    assert tv_distance([.5, .5], [1, 0]) == 0.5
    assert prevalence_tv(np.array([0, 0, 1, 1]), np.array([30, 10])) == 0.25


def test_g3_tv_and_smoothed_kl():
    make_of_fine = np.array([0, 0, 0, 1])
    probs = np.zeros((10, 4))
    probs[:, 0] = 1  # evaluator always says class 0 for make 0
    test_counts = np.array([5, 5, 0, 7])
    m = g3_metric(probs, np.zeros(10, int), make_of_fine, test_counts)(np.arange(10))
    assert np.isclose(m["tv"], 0.5) and m["kl"] > 0
    probs[:, 0], probs[:5, 0], probs[5:, 1] = 0, 1, 1  # half / half = the test histogram
    m = g3_metric(probs, np.zeros(10, int), make_of_fine, test_counts)(np.arange(10))
    assert m["tv"] == 0 and m["kl"] < 0.1
    assert smoothed_kl([3, 3], [3, 3]) == 0 and smoothed_kl([5, 5], [10, 0]) > smoothed_kl([5, 5], [6, 4])


def test_rare_and_tau():
    assert rare_classes(np.arange(1, 9)).tolist() == [True, True] + [False] * 6
    probs = np.array([[.95, .05]] * 10 + [[.6, .4]] * 10)
    y = np.array([0] * 10 + [0] * 5 + [1] * 5)  # confident ones always right, the .6 ones half right
    assert choose_tau(probs, y, 0.9) == 0.95 and choose_tau(probs, y, 0.5) == 0.6


def make_fidelity(n_rare_samples):
    """4 fine classes (0,1 in make 0; 2,3 in make 1); class 1 is rare. Generated = same law as real."""
    mu = np.eye(8)[:4] * 2
    test = np.concatenate([gauss(m, 40) for m in mu])
    train = np.concatenate([gauss(m, 40) for m in mu])
    fine = np.repeat(np.arange(4), 40)
    n_gen = [20, n_rare_samples]
    gen = np.concatenate([gauss(mu[0], 20), gauss(mu[1], n_rare_samples)])
    probs = np.eye(4)[np.repeat([0, 1], n_gen)]
    cf = ClassFidelity(gen, probs, np.zeros(len(gen), int), np.array([0, 0, 1, 1]), 0.9, test, fine, train, fine,
                       np.array([False, True, False, False]))
    return cf, np.arange(len(gen))


def test_class_fidelity_coverage_of_rare_classes():
    cf, idx = make_fidelity(2)
    t = cf.table(idx)
    assert t.covered.tolist() == [True, False] and t.n.tolist() == [20, 2]
    m = cf(idx)
    assert m["frac_covered"] == 0.5 and m["frac_rare_covered"] == 0 and m["frac_common_covered"] == 1
    cf, idx = make_fidelity(30)
    m = cf(idx)
    assert m["frac_rare_covered"] == 1 and m["coverage"] > 0.5 and abs(m["kid"]) < 0.6
    assert abs(m["coverage_ratio"] - 1) < 0.5  # indistinguishable from the real ceiling


def test_class_fidelity_ignores_low_confidence_and_wrong_make():
    cf, idx = make_fidelity(30)
    cf.conf = np.full_like(cf.conf, 0.5)
    assert cf(idx)["frac_covered"] == 0
    cf, idx = make_fidelity(30)
    cf.pred = np.full_like(cf.pred, 3)  # predicted classes belong to make 1, samples requested make 0
    assert cf(idx)["frac_covered"] == 0


def test_g6_matching_purity_and_unmatched():
    real = [gauss(np.eye(8)[0] * 2, 40), gauss(np.eye(8)[1] * 2, 40)]
    gen = np.concatenate([gauss(np.eye(8)[1] * 2, 20), gauss(np.eye(8)[0] * 2, 20)])  # group 0 imitates class 1
    group = np.repeat([0, 1], 20)
    probs = np.eye(2)[np.repeat([1, 0], 20)]
    m = g6_make(gen, group, probs, real)
    assert m["matched_kid"] < kid(gen[:20], real[0]) / 2   # matched pair far cheaper than the wrong one
    assert m["purity"] == 1 and m["entropy"] == 0 and m["n_unmatched_real"] == 0
    one = g6_make(gen[:20], group[:20], probs[:20], real)  # a single group cannot cover two subclasses
    assert one["n_unmatched_real"] == 1 and one["n_groups"] == 1
    mixed = g6_make(gen, np.tile([0, 1], 20), probs, real)  # groups mix both subclasses
    assert mixed["purity"] == 0.5 and np.isclose(mixed["entropy"], np.log(2))
