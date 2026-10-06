import numpy as np
import pandas as pd

from lsgen.eval.validation import read_metric, verdict, worse_than

nan = np.nan


def test_worse_than_with_and_without_ci():
    assert worse_than(0.9, 0.8, 1.0, 0.2, 0.1, 0.3, "lower")  # intervals apart
    assert not worse_than(0.25, 0.15, 0.35, 0.2, 0.1, 0.3, "lower")  # overlap: within noise
    assert worse_than(0.2, 0.15, 0.25, 0.9, 0.8, 1.0, "higher")  # lower than the floor of a higher-is-better metric
    assert not worse_than(0.95, 0.9, 1.0, 0.9, 0.85, 0.95, "higher")
    assert worse_than(1.0, nan, nan, 0.1, nan, nan, "lower") and not worse_than(0.11, nan, nan, 0.1, nan, nan, "lower")
    assert not worse_than(0.0, nan, nan, 0.1, nan, nan, "lower")  # better than the floor is not a failure


def test_verdict():
    assert [verdict(True, True), verdict(False, True), verdict(True, False), verdict(False, False)] == ["ok", "MISSED", "also", "same"]


def test_read_metric(tmp_path):
    pd.DataFrame({"group": [1, "macro"], "metric": ["tv", "tv"], "mean": [0.1, 0.2], "lo": [0, 0.1], "hi": [1, 0.3]}).to_csv(tmp_path / "g3.csv", index=False)
    pd.DataFrame({"fid": [3.0], "kid": [0.1]}).to_csv(tmp_path / "g1.csv", index=False)
    assert read_metric(tmp_path, "g3", "tv") == (0.2, 0.1, 0.3)
    value, lo, hi = read_metric(tmp_path, "g1", "fid")
    assert value == 3.0 and np.isnan(lo) and np.isnan(hi)
