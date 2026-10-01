"""Representation metrics on embeddings (docs/PLAN.md R1, R3). Embeddings are L2-normalised first."""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def l2norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def knn_accuracy(train_x, train_y, test_x, test_y, k=20):
    """R1: kNN accuracy (cosine geometry, majority vote)."""
    return KNeighborsClassifier(k).fit(l2norm(train_x), train_y).score(l2norm(test_x), test_y)


def deltas_per_make(x, make, fine):
    """R3: per make, mean squared distances between L2-normalised embeddings, self-pairs excluded:
    d0 same fine class, d1 same make but different fine class (NaN if the make has one class), d2 different make."""
    x = l2norm(x.astype(np.float32))
    d = 2 - 2 * x @ x.T
    np.fill_diagonal(d, np.nan)
    rows = []
    for c in np.unique(make):
        m = make == c
        a = d[m]
        same_make = np.broadcast_to((make == c)[None, :], a.shape)
        same_fine = fine[m][:, None] == fine[None, :]
        pick = lambda mask: np.nanmean(a[mask]) if np.isfinite(a[mask]).any() else np.nan  # noqa: E731
        rows.append({"make_id": c, "n": int(m.sum()), "K_c": len(np.unique(fine[m])),
                     "d0": pick(same_fine), "d1": pick(same_make & ~same_fine), "d2": pick(~same_make)})
    return pd.DataFrame(rows)
