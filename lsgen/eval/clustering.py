"""Graph-based alternatives to k-means for R2 / R5 (docs/EXPERIMENT_CLUSTERING.md): Leiden community detection (Traag et al. 2019) and
diffusion-map spectral clustering (Coifman & Lafon 2006), both on the same kNN graph of the R2 feature space. fit(z, k, seed) returns cluster
labels: k clusters (the oracle K), or the method's own label-free number of clusters if k is None."""
import random
from functools import partial

import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix, triu
from sklearn.neighbors import NearestNeighbors

from lsgen.eval.representation import K_MAX, cluster


def knn_graph(z, n_neighbors=15):
    """Symmetric kNN affinity with self-tuning Gaussian weights exp(-d_ij^2 / (s_i s_j)), s_i = distance from i to its n_neighbors-th
    neighbour (Zelnik-Manor & Perona 2004). Sparse (n, n)."""
    d, j = NearestNeighbors(n_neighbors=n_neighbors).fit(z).kneighbors()  # no query: a point is not its own neighbour
    s = np.maximum(d[:, -1], 1e-12)
    w = np.exp(-d ** 2 / (s[:, None] * s[j]))
    a = csr_matrix((w.ravel(), (np.repeat(np.arange(len(z)), n_neighbors), j.ravel())), shape=(len(z), len(z)))
    return a.maximum(a.T)


def leiden(a, resolution, seed=0):
    """Leiden partition of the weighted graph a maximising modularity at the given resolution (igraph's implementation)."""
    e = triu(a, 1).tocoo()
    g = ig.Graph(n=a.shape[0], edges=np.column_stack([e.row, e.col]).tolist())
    g.es["weight"] = e.data.tolist()
    ig.set_random_number_generator(random.Random(seed))
    return np.array(g.community_leiden("modularity", weights="weight", resolution=resolution, n_iterations=-1).membership)


def leiden_fit(z, k=None, seed=0, n_neighbors=15, resolution=1.0, steps=30):
    """Label-free: Leiden at the fixed resolution (1 = standard modularity). Oracle K: bisection of log10 resolution in [-3, 3] for the
    partition with k communities (the closest one found if no resolution gives exactly k)."""
    a = knn_graph(z, n_neighbors)
    if k is None:
        return leiden(a, resolution, seed)
    lo, hi, best = -3.0, 3.0, None
    for _ in range(steps):
        c = leiden(a, 10 ** ((lo + hi) / 2), seed)
        if best is None or abs(c.max() + 1 - k) < abs(best.max() + 1 - k):
            best = c
        if c.max() + 1 == k:
            break
        lo, hi = ((lo + hi) / 2, hi) if c.max() + 1 < k else (lo, (lo + hi) / 2)
    return best


def diffusion_fit(z, k=None, seed=0, n_neighbors=15, t=1):
    """Diffusion-map spectral clustering: right eigenvectors psi_i of the random walk P = D^-1 A on the kNN graph scaled by lambda_i^t
    (diffusion coordinates), k-means with k clusters on the first k - 1 non-trivial ones. Label-free k: the largest eigengap
    lambda_k - lambda_{k+1} over k = 2..K_MAX (von Luxburg 2007), the search range of the k-means K-hat."""
    a = knn_graph(z, n_neighbors).toarray()
    d = a.sum(1)
    lam, v = np.linalg.eigh(a / np.sqrt(np.outer(d, d)))  # symmetric conjugate of P: same eigenvalues, deterministic
    lam, psi = lam[::-1], v[:, ::-1] / np.sqrt(d)[:, None]
    if k is None:
        k_max = min(K_MAX, len(z) - 2)
        k = 2 + int(np.argmax(lam[1:k_max] - lam[2:k_max + 1]))
    return cluster(psi[:, 1:k] * lam[1:k] ** t, k, seed)


FITS = {"leiden": leiden_fit, "diffusion": diffusion_fit}


def fitter(algorithm, **params):
    """fit(z, k, seed) of `algorithm` with the parameters of a config's `clustering` entry."""
    return partial(FITS[algorithm], **params)
