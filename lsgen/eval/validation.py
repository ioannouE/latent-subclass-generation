"""Metric validation (docs/PLAN.md A5): did each control move the metrics it must fail, relative to the real-vs-real
floor (C2)? Verdicts: "ok" (expected and worse than C2), "MISSED" (expected, not worse), "also" (not expected, but worse),
"same" (not expected, not worse)."""
import numpy as np
import pandas as pd


def worse_than(value, lo, hi, base, base_lo, base_hi, better, margin=0.25):
    """Is the control clearly worse than the floor? With CIs: the intervals must not overlap on the bad side; without:
    worse by more than margin * |floor|."""
    sign = 1 if better == "lower" else -1  # sign * value larger = worse
    if np.isnan([lo, hi, base_lo, base_hi]).any():
        return bool(sign * (value - base) > margin * max(abs(base), 1e-3))
    return bool(lo > base_hi if sign == 1 else hi < base_lo)


def verdict(is_worse, expected):
    return ("ok" if is_worse else "MISSED") if expected else ("also" if is_worse else "same")


def read_metric(run_dir, table, name, b_set=None):
    """(value, lo, hi) of one metric from a result directory: macro row of a long table, or the single row of a wide one."""
    t = pd.read_csv(f"{run_dir}/{table}.csv")
    if "metric" in t:
        t = t[(t.group == "macro") & (t.metric == name)]
        if b_set is not None:
            t = t[t.set == b_set]
        return tuple(float(t[c].iloc[0]) for c in ("mean", "lo", "hi"))
    return float(t[name].iloc[0]), np.nan, np.nan


def headline_table(run_dir, cfg, case):
    """Headline metrics of one result directory: group (what it evaluates), metric, value, CI, and whether a control validates it.
    Use case B has one block per episode set."""
    validated = {m for per_control in cfg["expected"][case].values() for m in per_control}
    sets = [None] if case == "A" else sorted(pd.read_csv(f"{run_dir}/by_set.csv").set.unique())
    rows = []
    for b_set in sets:
        for group, names in cfg["headline"][case].items():
            for name in names:
                table, column, _ = cfg["metrics"][case][name]
                value, lo, hi = read_metric(run_dir, table, column, b_set)
                rows.append({"evaluates": group, "metric": name, "set": b_set, "value": value, "lo": lo, "hi": hi, "validated": name in validated})
    return pd.DataFrame(rows)
