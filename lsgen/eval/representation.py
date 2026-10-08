"""Representation metrics R1-R5 (docs/METRICS.md). Embeddings are L2-normalised first.

Per-image quantities (kNN / probe / recall@1 correctness) are bootstrapped with stats.bootstrap_table; the others are
`metric(idx)` closures for stats.bootstrap, where idx are global image indices of one make.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

K_MAX = 25      # K-hat search range
SIL_MIN = 0.10  # best silhouette below this -> K-hat falls back to the gap statistic (fixed, not tuned)


def l2norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---- R1 and within-make probes: per-image correctness vectors
def knn_correct(train_x, train_y, test_x, test_y, k=20):
    return KNeighborsClassifier(k).fit(l2norm(train_x), train_y).predict(l2norm(test_x)) == np.asarray(test_y)


def knn_accuracy(train_x, train_y, test_x, test_y, k=20):
    return knn_correct(train_x, train_y, test_x, test_y, k).mean()


def probe_correct(train_x, train_y, test_x, test_y, C=10.0):
    """Multinomial logistic-regression probe (fixed C), fit on train embeddings."""
    return LogisticRegression(C=C, max_iter=500).fit(l2norm(train_x), train_y).predict(l2norm(test_x)) == np.asarray(test_y)


def recall1_correct(x, fine):
    """Within-make Recall@1 (MaskCon protocol): is the nearest other image of the same make of the same fine class?
    x: embeddings of one make."""
    s = l2norm(x) @ l2norm(x).T
    np.fill_diagonal(s, -np.inf)
    return fine[s.argmax(1)] == fine


def recall_at_k(x, labels, ks=(1, 2, 5, 10), chunk=1000):
    """Global Recall@K as in the MaskCon paper: fraction of images with at least one same-label image among their K nearest
    other images (cosine). {k: recall}."""
    x, labels, hits = l2norm(x), np.asarray(labels), {k: 0 for k in ks}
    for a in range(0, len(x), chunk):
        s = x[a:a + chunk] @ x.T
        s[np.arange(len(s)), np.arange(a, a + len(s))] = -np.inf
        top = np.argsort(-s, axis=1)[:, :max(ks)]
        same = labels[top] == labels[a:a + chunk, None]
        for k in ks:
            hits[k] += same[:, :k].any(1).sum()
    return {k: hits[k] / len(x) for k in ks}


# ---- R2 / R5: clustering
def cluster_space(x, make, n=50, seed=0):
    """Rows of each make projected on that make's own top-n PCs (L2-normalised input); clustering runs here."""
    z = np.zeros((len(x), n), np.float32)
    for c in np.unique(make):
        m = make == c
        p = PCA(min(n, m.sum() - 1, x.shape[1]), random_state=seed).fit_transform(l2norm(x[m]))
        z[np.ix_(m, np.arange(p.shape[1]))] = p
    return z


def cluster(z, k, seed=0, n_init=10):
    return KMeans(k, n_init=n_init, random_state=seed).fit_predict(z)


def hungarian_acc(y, c):
    """Clustering accuracy under the best one-to-one matching of clusters to classes."""
    y, c = np.unique(y, return_inverse=True)[1], np.unique(c, return_inverse=True)[1]
    m = np.zeros((y.max() + 1, c.max() + 1), int)
    np.add.at(m, (y, c), 1)
    r, col = linear_sum_assignment(-m)
    return m[r, col].sum() / len(y)


def gap_k(z, k_max=K_MAX, seed=0, n_ref=10):
    """Gap statistic (Tibshirani et al.), uniform reference in the data's bounding box; smallest K with
    gap(K) >= gap(K+1) - s(K+1). Can return 1."""
    rng, ks = np.random.default_rng(seed), range(1, min(k_max, len(z) - 1) + 1)
    logw = lambda a, k: np.log(KMeans(k, n_init=3, random_state=seed).fit(a).inertia_)  # noqa: E731
    ref = np.array([[logw(rng.uniform(z.min(0), z.max(0), z.shape), k) for k in ks] for _ in range(n_ref)])
    gap, s = ref.mean(0) - np.array([logw(z, k) for k in ks]), ref.std(0) * np.sqrt(1 + 1 / n_ref)
    return next((i + 1 for i in range(len(gap) - 1) if gap[i] >= gap[i + 1] - s[i + 1]), len(gap))


def select_k(z, k_max=K_MAX, seed=0):
    """Label-free K-hat: silhouette sweep over K = 2..k_max (silhouette is undefined at K = 1); the gap statistic, which can
    return 1, decides if the best silhouette < SIL_MIN, i.e. when there is no evidence of more than one cluster."""
    ks = list(range(2, min(k_max, len(z) - 1) + 1))
    sil = [silhouette_score(z, cluster(z, k, seed, n_init=3)) for k in ks]
    return ks[int(np.argmax(sil))] if max(sil) >= SIL_MIN else gap_k(z, k_max, seed)


def r2_metric(z, make, fine, k_of_make, seed=0, label_free=False):
    """Per make (K_c >= 2): ACC / NMI / ARI of k-means with the oracle K = K_c ("uses K"), or with the label-free K-hat
    (also returned as "khat"). K-hat is unreliable on bootstrap resamples: bootstrap it with `subsample`."""
    def metric(idx):
        zz, y = z[idx], fine[idx]
        k = select_k(zz, seed=seed) if label_free else k_of_make[make[idx[0]]]
        c = cluster(zz, k, seed)
        return {"acc": hungarian_acc(y, c), "nmi": nmi(y, c), "ari": ari(y, c)} | ({"khat": k} if label_free else {})
    return metric


def r5_make(z, k, seed=0, n_seeds=3, n_boot=20):
    """Stability of oracle-K k-means on one make: ARI between runs with different seeds, and between the full-data
    clustering and clusterings fitted on bootstrap resamples (then applied to all points). Lists of ARI values."""
    labs = [cluster(z, k, seed + s) for s in range(n_seeds)]
    rng = np.random.default_rng(seed)
    boot = [ari(labs[0], KMeans(k, n_init=10, random_state=seed + b).fit(z[rng.integers(0, len(z), len(z))]).predict(z))
            for b in range(n_boot)]
    return {"ari_seeds": [ari(a, b) for a, b in itertools.combinations(labs, 2)], "ari_boot": boot}


def khat_error(z, k_c=1):
    """K-hat minus the true number of classes (use on K_c = 1 makes: an over-fragmenting method has error > 0)."""
    return select_k(z) - k_c


# ---- R3: hierarchy geometry
def distance_matrix(x):
    """Squared distances between L2-normalised embeddings, NaN diagonal (self-pairs excluded)."""
    x = l2norm(x.astype(np.float32))
    d = 2 - 2 * x @ x.T
    np.fill_diagonal(d, np.nan)
    return d


def r3_metric(d, make, fine):
    """d0 same fine class, d1 same make other fine class, d2 other make (mean squared distance), their ratios and the
    indicators of Eq. (1) d0 < d1 and Eq. (2) d1 < d2, for the make of the given rows (rows may repeat)."""
    def metric(rows):
        a, c = d[rows], make[rows[0]]
        same_make = np.broadcast_to((make == c)[None, :], a.shape)
        same_fine = fine[rows][:, None] == fine[None, :]
        pick = lambda m: np.nanmean(a[m]) if np.isfinite(a[m]).any() else np.nan  # noqa: E731
        d0, d1, d2 = pick(same_fine), pick(same_make & ~same_fine), pick(~same_make)
        ok = np.isfinite(d1)
        return {"d0": d0, "d1": d1, "d2": d2, "d0/d1": d0 / d1, "d1/d2": d1 / d2,
                "eq1": float(d0 < d1) if ok else np.nan, "eq2": float(d1 < d2) if ok else np.nan}
    return metric


def deltas_per_make(x, make, fine):
    f = r3_metric(distance_matrix(x), make, fine)
    rows = []
    for c in np.unique(make):
        r = np.flatnonzero(make == c)
        rows.append({"make_id": c, "n": len(r), "K_c": len(np.unique(fine[r])), **{k: v for k, v in f(r).items() if k in ("d0", "d1", "d2")}})
    return pd.DataFrame(rows)


# ---- R4: within-subclass variation
def trace_cov(x):
    return np.var(x, axis=0, ddof=1).sum()


def participation_ratio(x):
    """Effective rank (sum lambda)^2 / sum lambda^2 of the covariance spectrum; NaN if all points coincide."""
    s = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2
    return s.sum() ** 2 / (s ** 2).sum() if s.sum() > 0 else np.nan


def r4_metric(x, fine, min_n=3):
    """var_ratio: within-subclass trace-cov (size-weighted over subclasses) / trace-cov of the whole make;
    pr_subclass: mean participation ratio of the subclasses (degenerate ones are ignored). x must be L2-normalised."""
    def metric(rows):
        xs, ys = x[rows], fine[rows]
        subs = [xs[ys == s] for s in np.unique(ys) if (ys == s).sum() >= min_n]
        w, pr = np.array([len(s) for s in subs]), np.array([participation_ratio(s) for s in subs])
        return {"var_ratio": (w * [trace_cov(s) for s in subs]).sum() / w.sum() / trace_cov(xs),
                "pr_subclass": pr[~np.isnan(pr)].mean() if (~np.isnan(pr)).any() else np.nan}
    return metric
