"""Use case A metrics G1-G6 (docs/METRICS.md). All functions take embeddings / evaluator probabilities (not images).

Subclass-level sets are small (about 40 test images per class): use KID (unbiased) and coverage, never FID there.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances


# ---- G1: realism on feature vectors (Inception pool3 for the headline FID/KID, DINOv2-L for FD_DINOv2)
def frechet_distance(x, y):
    mu = x.mean(0) - y.mean(0)
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvals(s1 @ s2).real, 0, None)).sum()
    return float(mu @ mu + np.trace(s1) + np.trace(s2) - 2 * tr_sqrt)


def _mmd2(x, y, degree=3):
    """Unbiased MMD^2 with the KID polynomial kernel (x.y / d + 1)^3."""
    d, m, n = x.shape[1], len(x), len(y)
    kxx, kyy, kxy = ((a @ b.T / d + 1) ** degree for a, b in ((x, x), (y, y), (x, y)))
    return (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean()


def kid(x, y, n_subsets=100, subset_size=1000, seed=0):
    """Kernel Inception Distance (unbiased; can be slightly negative). Sets up to `subset_size` are compared whole,
    larger ones by averaging over random subsets. NaN if either set has fewer than 2 points."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if min(len(x), len(y)) < 2:
        return np.nan
    if max(len(x), len(y)) <= subset_size:
        return float(_mmd2(x, y))
    rng, m = np.random.default_rng(seed), min(subset_size, len(x), len(y))
    return float(np.mean([_mmd2(x[rng.choice(len(x), m, replace=False)], y[rng.choice(len(y), m, replace=False)])
                          for _ in range(n_subsets)]))


def _kth_radius(x, k):
    return np.partition(euclidean_distances(x, x), k, axis=1)[:, k]  # column 0 is the point itself


def prdc(real, fake, k=5):
    """Precision / recall (Kynkaanniemi 2019) and density / coverage (Naeem 2020) from k-NN balls."""
    real, fake = np.asarray(real, np.float32), np.asarray(fake, np.float32)
    if min(len(real), len(fake)) <= k:
        return dict.fromkeys(("precision", "recall", "density", "coverage"), np.nan)
    rr, rf, d = _kth_radius(real, k), _kth_radius(fake, k), euclidean_distances(real, fake)
    inside = d <= rr[:, None]  # fake sample j inside the ball of real sample i
    return {"precision": inside.any(0).mean(), "recall": (d <= rf[None, :]).any(1).mean(),
            "density": inside.sum(0).mean() / k, "coverage": (d.min(1) <= rr).mean()}


# ---- G2 / G3: make and subclass prevalence
def make_correct(make_probs, requested_make):
    """G2: does the make evaluator predict the requested make?"""
    return make_probs.argmax(1) == requested_make


def tv_distance(p, q):
    return 0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum()


def prevalence_tv(pred_make, test_make_counts):
    """G2: TV between the make histogram of the samples (evaluator prediction) and of test."""
    p = np.bincount(pred_make, minlength=len(test_make_counts))
    return tv_distance(p / p.sum(), test_make_counts / test_make_counts.sum())


def smoothed_kl(q_counts, p_counts):
    """KL(q || p) of add-one-smoothed histograms: q = real (test), p = generated; penalises missing subclasses."""
    q, p = (np.asarray(c, float) + 1 for c in (q_counts, p_counts))
    q, p = q / q.sum(), p / p.sum()
    return float((q * np.log(q / p)).sum())


def g3_metric(fine_probs, requested_make, make_of_fine, test_fine_counts):
    """metric(idx) over the samples requested for one make: TV and smoothed KL between the evaluator's fine-label
    histogram (restricted to that make's classes, so O2 is isolated from O1) and the test histogram."""
    def metric(idx):
        classes = np.flatnonzero(make_of_fine == requested_make[idx[0]])
        p = np.bincount(fine_probs[idx][:, classes].argmax(1), minlength=len(classes))
        q = test_fine_counts[classes]
        return {"tv": tv_distance(p / p.sum(), q / q.sum()), "kl": smoothed_kl(q, p)}
    return metric


# ---- G4 / G5: rare subclasses and within-subclass fidelity
def rare_classes(train_fine_counts):
    """Bottom quartile of fine classes by train frequency (boolean mask)."""
    return train_fine_counts <= np.quantile(train_fine_counts, 0.25)


def choose_tau(val_probs, val_y, target=0.9):
    """Smallest confidence threshold at which the (calibrated) evaluator's val accuracy is >= target."""
    conf, correct = val_probs.max(1), val_probs.argmax(1) == val_y
    for t in np.sort(np.unique(conf)):
        if correct[conf >= t].mean() >= target:
            return float(t)
    return 1.0


class ClassFidelity:
    """G4 + G5 for one generated set. Samples requested for a make are assigned to the fine class the evaluator
    predicts with confidence >= tau (and that belongs to the requested make). A class is covered if it gets >= min_n
    samples; covered classes get KID and coverage against test_g, plus the real ceiling (train_g subsample of the same
    size against test_g). Call with the sample indices of one requested make (stats.bootstrap metric)."""

    def __init__(self, gen, probs, requested_make, make_of_fine, tau, test, test_fine, train, train_fine, rare,
                 min_n=5, k=3, seed=0):
        self.gen, self.pred, self.conf, self.req, self.mk = gen, probs.argmax(1), probs.max(1), requested_make, make_of_fine
        self.tau, self.rare, self.min_n, self.k, self.seed = tau, rare, min_n, k, seed
        self.test = {g: test[test_fine == g] for g in range(len(rare))}
        self.train = {g: train[train_fine == g] for g in range(len(rare))}

    def table(self, idx):
        cols = ["fine_id", "rare", "n", "covered", "kid", "coverage", "kid_ceiling", "coverage_ceiling"]
        c, rows = self.req[idx[0]], []
        ok = (self.conf[idx] >= self.tau) & (self.mk[self.pred[idx]] == c)
        for g in np.flatnonzero(self.mk == c):
            x = self.gen[idx[ok & (self.pred[idx] == g)]]
            r = {"fine_id": g, "rare": bool(self.rare[g]), "n": len(x), "covered": len(x) >= self.min_n}
            if r["covered"]:
                rng = np.random.default_rng([self.seed, g, len(x)])
                ref = self.train[g][rng.choice(len(self.train[g]), min(len(x), len(self.train[g])), replace=False)]
                r |= {"kid": kid(x, self.test[g]), "coverage": prdc(self.test[g], x, self.k)["coverage"],
                      "kid_ceiling": kid(ref, self.test[g]), "coverage_ceiling": prdc(self.test[g], ref, self.k)["coverage"]}
            rows.append(r)
        t = pd.DataFrame(rows).reindex(columns=cols)
        t["covered"] = t.covered.astype(bool)
        return t

    def __call__(self, idx):
        t = self.table(idx)
        cv = t[t.covered]
        return {"frac_covered": t.covered.mean(), "frac_rare_covered": t.covered[t.rare].mean(),
                "frac_common_covered": t.covered[~t.rare].mean(), "kid": cv.kid.mean(),
                "kid_ratio": cv.kid.mean() / cv.kid_ceiling.mean(), "coverage": cv.coverage.mean(),
                "coverage_ratio": cv.coverage.mean() / cv.coverage_ceiling.mean()}


# ---- G6: alignment of generated groups with real subclasses (methods that output a group id)
def g6_make(gen, group, probs, real):
    """gen (n, d) samples requested for one make with group ids; probs (n, K) evaluator probabilities over this make's
    K classes; real: list of K arrays of test embeddings. Groups with < 2 samples are dropped (KID undefined)."""
    gids = [g for g in np.unique(group) if (group == g).sum() >= 2]
    cost = np.array([[kid(gen[group == g], r) for r in real] for g in gids])
    ri, ci = linear_sum_assignment(cost)
    hist = [np.bincount(probs[group == g].argmax(1), minlength=len(real)) / (group == g).sum() for g in gids]
    return {"matched_kid": cost[ri, ci].mean(), "purity": np.mean([h.max() for h in hist]),
            "entropy": np.mean([-(h[h > 0] * np.log(h[h > 0])).sum() for h in hist]),
            "n_groups": len(gids), "n_unmatched_real": len(real) - len(ri)}


def g6_metric(gen, group, fine_probs, requested_make, make_of_fine, test, test_fine):
    def metric(idx):
        classes = np.flatnonzero(make_of_fine == requested_make[idx[0]])
        return g6_make(gen[idx], group[idx], fine_probs[idx][:, classes], [test[test_fine == g] for g in classes])
    return metric
