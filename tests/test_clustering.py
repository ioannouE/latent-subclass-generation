import numpy as np

from lsgen.eval.clustering import diffusion_fit, fitter, knn_graph, leiden_fit
from lsgen.eval.representation import cluster_space, hungarian_acc, l2norm, r2_metric, r5_make
from tests.test_representation import blobs


def test_knn_graph_symmetric_and_bounded_with_duplicates():
    x, _ = blobs(2, n=30)
    a = knn_graph(np.concatenate([x, x[:5]]), 10)  # duplicated rows: distance 0, weight 1
    assert (abs(a - a.T) > 1e-12).nnz == 0 and a.diagonal().max() == 0 and a.data.min() > 0 and a.data.max() <= 1


def test_leiden_oracle_k_and_label_free_purity():
    x, y = blobs(3)
    z = cluster_space(x, np.zeros(len(x), int))
    assert hungarian_acc(y, leiden_fit(z, 3)) == 1  # disconnected blobs: bisection reaches exactly 3 communities
    c = leiden_fit(z)
    assert c.max() + 1 >= 3 and all(len(np.unique(y[c == i])) == 1 for i in np.unique(c))  # never merges blobs
    assert (leiden_fit(z, seed=1) == leiden_fit(z, seed=1)).all()


def test_diffusion_eigengap_recovers_blobs():
    x, y = blobs(3)
    z = cluster_space(x, np.zeros(len(x), int))
    assert hungarian_acc(y, diffusion_fit(z)) == 1 and hungarian_acc(y, diffusion_fit(z, 3)) == 1


def test_r2_and_r5_with_graph_fits():
    x, y = blobs(3)
    make = np.zeros(len(x), int)
    z = cluster_space(x, make)
    m = r2_metric(z, make, y, {0: 99}, label_free=True, fit=fitter("diffusion"))(np.arange(len(x)))  # the oracle K is not used
    assert m["acc"] == 1 and m["khat"] == 3
    r = r5_make(z, 3, n_boot=3, fit=fitter("leiden"))
    assert np.allclose(r["ari_seeds"], 1) and np.allclose(r["ari_boot"], 1)


def test_full_embedding_space_and_graph_fit_on_it():
    x, y = blobs(3)
    z = cluster_space(x, np.zeros(len(x), int), None)  # no PCA
    assert z.shape == x.shape and np.allclose(z, l2norm(x))
    assert hungarian_acc(y, diffusion_fit(z, 3)) == 1 and hungarian_acc(y, leiden_fit(z, 3)) == 1
