import numpy as np
import pandas as pd

from lsgen.data.splits import load_split, make_splits, save_splits


def fake_meta(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for split, n0 in (("train", 1), ("test", 1)):
        k = 0
        for c in range(30):
            for _ in range(int(rng.integers(24, 69))):
                k += n0
                rows.append((f"{split}_{k:05d}", split, c))
    return pd.DataFrame(rows, columns=["id", "official_split", "fine_id"])


def test_splits_partition_stratified_deterministic(tmp_path):
    meta = fake_meta()
    s = make_splits(meta, 0.10, 0)
    tr, te = set(meta.id[meta.official_split == "train"]), set(meta.id[meta.official_split == "test"])
    assert set(s["train"]) | set(s["val"]) == tr and not set(s["train"]) & set(s["val"])
    assert set(s["test"]) == te
    assert set(s["test_A"]) | set(s["test_B"]) == te and not set(s["test_A"]) & set(s["test_B"])
    assert abs(len(s["val"]) / len(tr) - 0.10) < 0.005
    assert abs(len(s["test_A"]) - len(s["test_B"])) <= 1
    fine = meta.set_index("id").fine_id
    for name in ("val", "test_A", "test_B"):
        assert fine[s[name]].nunique() == 30  # every class present
    # val class proportions track train proportions
    p_tr = fine[list(tr)].value_counts(normalize=True).sort_index()
    p_val = fine[s["val"]].value_counts(normalize=True).sort_index()
    assert (p_tr - p_val).abs().max() < 0.01
    assert make_splits(meta, 0.10, 0) == s
    assert make_splits(meta, 0.10, 1)["val"] != s["val"]
    save_splits(s, tmp_path)
    assert all(load_split(tmp_path, k) == v for k, v in s.items())
