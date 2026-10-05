import numpy as np
import pandas as pd

from lsgen.eval.stats import bootstrap, bootstrap_table


def test_bootstrap_macro_ci_and_determinism():
    v = np.r_[np.random.default_rng(0).normal(0, 1, 200), np.random.default_rng(1).normal(5, 1, 200)]
    groups = {"a": np.arange(200), "b": np.arange(200, 400)}
    r = bootstrap(lambda i: {"m": v[i].mean()}, groups, 300, 0)
    a, b, macro = (r[r.group == g].iloc[0] for g in ("a", "b", "macro"))
    assert np.isclose(a["mean"], v[:200].mean()) and np.isclose(macro["mean"], (a["mean"] + b["mean"]) / 2)
    assert a.lo < a["mean"] < a.hi and 0.2 < a.hi - a.lo < 0.36  # 2 * 1.96 / sqrt(200) = 0.277
    assert macro.hi - macro.lo < a.hi - a.lo  # averaging two classes tightens the interval
    assert r.equals(bootstrap(lambda i: {"m": v[i].mean()}, groups, 300, 0))


def test_subsample_draws_distinct_images():
    seen = []
    bootstrap(lambda i: seen.append(i) or {"m": 0.0}, {"a": np.arange(100)}, 5, 0, subsample=0.6)
    assert all(len(i) == 60 and len(set(i)) == 60 for i in seen[1:])  # first call is the point estimate


def test_bootstrap_table_ignores_nan_and_unequal_classes():
    df = pd.DataFrame({"g": ["x"] * 4 + ["y"] * 2, "v": [1, 1, np.nan, 1, 3, 3.0]})
    r = bootstrap_table(df, "g", ["v"], 50, 0).set_index("group")
    assert r.loc["x", "mean"] == 1 and r.loc["y", "mean"] == 3 and r.loc["macro", "mean"] == 2
    assert (r.loc["y", ["lo", "hi"]] == 3).all()
