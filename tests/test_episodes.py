import importlib.util
import json

import numpy as np

import pandas as pd

from lsgen.data.episodes import build_mixed, build_single, flat, near_dup_index
from lsgen.data.episodes import pools as split_pools
from lsgen.utils import REPO_ROOT


def build_episodes_script():
    spec = importlib.util.spec_from_file_location("build_episodes", REPO_ROOT / "scripts" / "build_episodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_episode_file_is_valid_json(tmp_path):
    mod = build_episodes_script()
    tp = pools()
    eps = build_mixed(tp, MAKE_OF, 10, [0.5], 1, 10, 2, 0, {})[0]
    header = {"family": "mixed", "skipped": [(10, (0, 1), 0.8, "x")], "pairs_per_make": {0: 2}}
    mod.write_episodes(tmp_path / "e.json", header, eps)
    d = json.loads((tmp_path / "e.json").read_text())
    assert d["family"] == "mixed" and d["episodes"] == json.loads(json.dumps(eps))


def pools(n_classes=6, sizes=(12, 20, 30, 40, 25, 18), prefix="test"):
    return {g: sorted(f"{prefix}_{g:02d}_{i:03d}" for i in range(sizes[g])) for g in range(n_classes)}


MAKE_OF = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}


def test_single_test_episodes_disjoint_and_eligible():
    tp = pools()
    eps, elig, skipped = build_single(tp, tp, MAKE_OF, [1, 5, 10], 5, 10, 0, "single_test", {})
    # eligible iff size >= k + 10
    assert elig == {1: 6, 5: 5, 10: 4}  # sizes 12, 20, 30, 40, 25, 18
    assert [(k, g) for k, g, _ in skipped] == [(5, 0), (10, 0), (10, 5)]
    for e in eps:
        g = e["fine_ids"][0]
        S, T = set(e["support"]), set(e["target"])
        assert len(S) == e["k"] and not S & T and S | T == set(tp[g])
        assert e["make_id"] == MAKE_OF[g] and e["support_source"] == "test"
    assert len(eps) == 5 * (6 + 5 + 4)
    assert len({e["episode_id"] for e in eps}) == len(eps)
    reps = [tuple(e["support"]) for e in eps if e["k"] == 5 and e["fine_ids"] == [3]]
    assert len(set(reps)) == 5  # repetitions differ


def test_single_deterministic_and_seed_dependent():
    tp = pools()
    a = build_single(tp, tp, MAKE_OF, [5], 5, 10, 0, "single_test", {})[0]
    b = build_single(tp, tp, MAKE_OF, [5], 5, 10, 0, "single_test", {})[0]
    c = build_single(tp, tp, MAKE_OF, [5], 5, 10, 1, "single_test", {})[0]
    assert json.dumps(a) == json.dumps(b) and json.dumps(a) != json.dumps(c)


def test_single_train_source():
    tp, trp = pools(), pools(sizes=(3, 20, 30, 40, 25, 18), prefix="train")
    eps, elig, skipped = build_single(trp, tp, MAKE_OF, [5], 2, 10, 0, "single_train", {})
    assert elig == {5: 5} and skipped[0][1] == 0  # class 0 has only 3 train images
    for e in eps:
        g = e["fine_ids"][0]
        assert all(i.startswith("train_") for i in e["support"]) and e["target"] == tp[g]
        assert e["support_source"] == "train"


def test_near_duplicates_avoided_between_support_and_target():
    tp = pools()
    # every image of class 3 except the last two has a near-duplicate inside class 3
    ids = tp[3]
    nd = near_dup_index([(ids[i], ids[i + 1]) for i in range(0, 36, 2)])
    eps = build_single(tp, tp, MAKE_OF, [1], 5, 10, 0, "single_test", nd)[0]
    for e in [e for e in eps if e["fine_ids"] == [3]]:
        assert e["n_near_dup_S_T"] == 0
        (s,) = e["support"]
        assert not nd.get(s, set()) & set(e["target"])
    # impossible to avoid: all pairs connected -> kept and recorded
    nd_all = near_dup_index([(a, b) for a in ids for b in ids if a < b])
    e = [e for e in build_single(tp, tp, MAKE_OF, [1], 1, 10, 0, "single_test", nd_all, max_redraws=3)[0]
         if e["fine_ids"] == [3]][0]
    assert e["n_near_dup_S_T"] == len(ids) - 1 and e["redraws"] == 3


def test_mixed_ratios_and_pairs():
    tp = pools()
    eps, pairs_used, skipped = build_mixed(tp, MAKE_OF, 10, [0.2, 0.5, 0.8], 2, 10, 2, 0, {})
    assert pairs_used == {0: 2, 1: 1}  # make 0 has 3 pairs, capped to 2; make 2 has one class
    for e in eps:
        a, b = e["fine_ids"]
        assert MAKE_OF[a] == MAKE_OF[b] == e["make_id"]
        Sa, Sb = e["support_by_class"][str(a)], e["support_by_class"][str(b)]
        assert len(Sa) == round(e["ratio"] * 10) and len(Sa) + len(Sb) == 10
        for g, S in ((a, Sa), (b, Sb)):
            T = e["target_by_class"][str(g)]
            assert set(S) <= set(tp[g]) and not set(S) & set(T) and set(S) | set(T) == set(tp[g])
        assert len(flat(e, "support")) == 10
    # every subclass keeps >= min_target targets; class 0 (12 images) can only give <= 2 supports
    assert all(len(ids) >= 10 for e in eps for ids in e["target_by_class"].values())
    assert all(len(e["support_by_class"]["0"]) <= 2 for e in eps if 0 in e["fine_ids"])
    assert skipped and all(isinstance(x[3], str) for x in skipped)
    assert np.isclose(sorted({e["ratio"] for e in eps}), [0.2, 0.5, 0.8]).all()


def test_pools_follow_the_scheme_and_drop_its_conflicts():
    # the same images under two schemes: the labels of the given scheme group them, its exclusion list removes images
    meta = pd.DataFrame({"split": ["test", "test", "test", "train", "test"], "fine_id": [3, 3, 7, 3, 7]},
                        index=pd.Index(["test_a", "test_b", "test_c", "train_d", "test_e"], name="id"))
    assert split_pools(meta, "test", set()) == {3: ["test_a", "test_b"], 7: ["test_c", "test_e"]}
    assert split_pools(meta, "test", {"test_b", "test_c"}) == {3: ["test_a"], 7: ["test_e"]}
    assert split_pools(meta, "train", {"test_b"}) == {3: ["train_d"]}
    tp = split_pools(meta.assign(fine_id=[0] * 5), "test", {"test_b"})
    eps = build_single(tp, tp, {0: 0}, [1], 2, 1, 0, "single_test", {})[0]
    assert eps and all("test_b" not in e["support"] + e["target"] for e in eps)


def test_near_dup_pairs_threshold(tmp_path):
    mod = build_episodes_script()
    pd.DataFrame({"id_a": ["a", "b", "c"], "id_b": ["x", "y", "z"], "sscd_sim": [0.9, 0.7, 0.6],
                  "exact_duplicate": [False, False, True]}).to_csv(tmp_path / "all.csv", index=False)
    pd.DataFrame({"test_id": ["t"], "train_id": ["u"], "sscd_sim": [0.3]}).to_csv(tmp_path / "cross.csv", index=False)
    assert mod.near_dup_pairs(tmp_path, ["all.csv"], 0.75) == [("a", "x"), ("c", "z")]  # below threshold kept only if exact
    assert mod.near_dup_pairs(tmp_path, ["all.csv", "cross.csv"]) == [("a", "x"), ("b", "y"), ("c", "z"), ("t", "u")]
