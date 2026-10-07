import json

import numpy as np
import pandas as pd

from lsgen.eval.report import write_results


def test_rerunning_one_table_keeps_the_others_and_their_provenance(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    for s in ("train", "val", "test"):
        (splits / f"{s}.txt").write_text("a\nb\n")
    (splits / "exclude.txt").write_text("b\n")
    out = tmp_path / "out"
    write_results(out, {"r1": pd.DataFrame({"v": [1]}), "r4": pd.DataFrame({"v": [2]})}, {"x": 1}, 0, "hash", splits)
    first = json.loads((out / "summary.json").read_text())
    assert first["excluded_conflicting_images"] == {"train": 1, "val": 1, "test": 1}
    first["table_git_hash"]["r1"] = "old"  # pretend r1 came from another commit
    (out / "summary.json").write_text(json.dumps(first))
    write_results(out, {"r4": pd.DataFrame({"v": [3]})}, {"x": 1}, 0, "hash", splits)
    s = json.loads((out / "summary.json").read_text())
    assert pd.read_csv(out / "r1.csv").v[0] == 1 and pd.read_csv(out / "r4.csv").v[0] == 3
    assert set(s["tables"]) == {"r1", "r4"} and s["table_git_hash"]["r1"] == "old" and s["table_git_hash"]["r4"] == s["git_hash"]


def _fake_run(root, acc, d01):
    """Result directory of a method on 3 makes (make 2 has K_c = 1): long tables as written by scripts/eval_representation.py."""
    root.mkdir()
    long = lambda metrics: pd.DataFrame([dict(group=g, metric=m, mean=v[i], lo=v[i] - .01, hi=v[i] + .01, n=10)
                                         for m, v in metrics.items() for i, g in enumerate(["0", "1", "2"][:len(v)])] +
                                        [dict(group="macro", metric=m, mean=sum(v) / len(v), lo=0, hi=1, n=3) for m, v in metrics.items()])
    nan = float("nan")
    for table, metrics in {"r1": {"knn_make": [.9, .8, .7], "probe_make": [.9, .8, .7]}, "r2_within_make": {"recall1": [.9, .8], "probe_fine": [.9, .8]},
                           "r2_oracle": {"acc": acc, "nmi": acc, "ari": acc}, "r2_khat": {"acc": acc, "nmi": acc, "ari": acc, "khat": [2, 3]},
                           "r3": {m: d01 if m == "d0/d1" else [.5, .5, .5] for m in ("d0", "d1", "d2", "d0/d1", "d1/d2", "eq1", "eq2")},
                           "r4": {"var_ratio": acc, "pr_subclass": acc}}.items():
        long(metrics).to_csv(root / f"{table}.csv", index=False)
    pd.DataFrame({"make_id": ["0", "1", "2", "macro"], "khat_error": [nan, nan, 1, 1], "ari_seeds": [.9, .8, nan, .85], "ari_boot": [.9, .8, nan, .85],
                  "ari_boot_lo": [nan, nan, nan, .8], "ari_boot_hi": [nan, nan, nan, .9]}).to_csv(root / "r5.csv", index=False)


def test_macro_row_per_make_table_and_worst_makes(tmp_path):
    from lsgen.eval.report import macro_row, per_make_table, worst_makes
    _fake_run(tmp_path / "enc", acc=[.4, .9], d01=[.7, .2, .5])
    row = macro_row(tmp_path / "enc")
    assert np.isclose(row["acc_oracle"], .65) and row["acc_oracle_lo"] == 0 and row["khat_error"] == 1 and row["ari_boot_hi"] == .9
    pm = per_make_table(tmp_path / "enc")
    assert pm.index.tolist() == [0, 1, 2] and pm.n_test.tolist() == [10, 10, 10]
    assert np.isnan(pm.loc[2, "acc_oracle"]) and pm.loc[0, "acc_oracle"] == .4 and pm.loc[2, "khat_error"] == 1  # K_c = 1 make: no ACC, has K-hat error
    assert worst_makes(pm, "acc_oracle", 1).index.tolist() == [0]  # lowest ACC; the NaN make is not a candidate
    assert worst_makes(pm, "d0/d1", 2, lowest=False).index.tolist() == [0, 2]  # highest ratio = worst
