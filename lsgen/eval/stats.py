"""Bootstrap CIs: images are resampled within each coarse class (1000 resamples), macro = mean over classes.
`subsample` replaces the resampling by draws without replacement of that fraction of the images, for metrics that break
on duplicated images (K-hat: duplicates form tight pairs and inflate the silhouette-selected K)."""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _boots(metric, idx, names, n_boot, seed, subsample):
    rng = np.random.default_rng(seed)
    draw = (lambda: rng.choice(idx, len(idx))) if subsample is None else \
        (lambda: rng.choice(idx, max(2, round(subsample * len(idx))), replace=False))
    out = [metric(draw()) for _ in range(n_boot)]
    return np.array([[o[n] for n in names] for o in out])  # (n_boot, n_metrics)


def bootstrap(metric, groups, n_boot=1000, seed=0, n_jobs=1, subsample=None):
    """metric(idx) -> {name: float} for image indices idx; groups: {class: index array}.
    Returns a DataFrame (group, metric, mean, lo, hi, n); the point estimate uses the original images, lo/hi are the
    2.5/97.5 percentiles over resamples, group "macro" averages the classes (NaN classes ignored) inside every resample."""
    keys = list(groups)
    idxs = [np.asarray(groups[g]) for g in keys]
    point = [metric(i) for i in idxs]
    names = list(point[0])
    pt = np.array([[p[n] for n in names] for p in point], float)  # (classes, metrics)
    seeds = np.random.SeedSequence(seed).spawn(len(keys))
    boots = np.stack(Parallel(n_jobs)(delayed(_boots)(metric, i, names, n_boot, s, subsample) for i, s in zip(idxs, seeds)), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        macro, (mlo, mhi) = np.nanmean(pt, 0), np.nanpercentile(np.nanmean(boots, 1), [2.5, 97.5], axis=0)
    rows = [dict(group=g, metric=n, mean=pt[i, j], lo=lo[i, j], hi=hi[i, j], n=len(idxs[i]))
            for i, g in enumerate(keys) for j, n in enumerate(names)]
    return pd.DataFrame(rows + [dict(group="macro", metric=n, mean=macro[j], lo=mlo[j], hi=mhi[j], n=len(keys))
                                for j, n in enumerate(names)])


def bootstrap_table(df, by, cols, n_boot=1000, seed=0, n_jobs=1):
    """Per-class mean of per-image (or per-episode) columns `cols` of df, grouped by df[by], with CIs."""
    v = df[cols].to_numpy(float)

    def metric(idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return dict(zip(cols, np.nanmean(v[idx], 0)))

    return bootstrap(metric, {g: np.flatnonzero(df[by] == g) for g in df[by].unique()}, n_boot, seed, n_jobs)


def point_table(metric, groups):
    """The table of `bootstrap` without CIs (lo / hi NaN), for metrics that no resampling scheme gives valid CIs for."""
    points = {g: metric(np.asarray(i)) for g, i in groups.items()}
    names = list(next(iter(points.values())))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        macro = [dict(group="macro", metric=n, mean=np.nanmean([p[n] for p in points.values()]), lo=np.nan, hi=np.nan, n=len(groups)) for n in names]
    rows = [dict(group=g, metric=n, mean=p[n], lo=np.nan, hi=np.nan, n=len(groups[g])) for g, p in points.items() for n in names]
    return pd.DataFrame(rows + macro)


def without_ci(table, metrics):
    """Copy of a bootstrap table with the CIs of `metrics` removed."""
    table = table.copy()
    table.loc[table.metric.isin(metrics), ["lo", "hi"]] = np.nan
    return table
