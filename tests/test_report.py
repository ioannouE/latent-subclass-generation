import json

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
