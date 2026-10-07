"""Result files: every table is a CSV, and summary.json next to them records provenance and the data exclusion."""
import json
from pathlib import Path

import pandas as pd

from lsgen.data.splits import load_split
from lsgen.utils import git_hash, provenance, write_json


def exclusion_note(splits_dir, split_names=("train", "val", "test")):
    """Images of each split removed because they are conflicting (data/splits/exclude.txt, docs/DATA.md)."""
    removed = {s: len(load_split(splits_dir, s, exclude=False)) - len(load_split(splits_dir, s)) for s in split_names}
    return {"excluded_conflicting_images": removed,
            "note": "conflicting images (same photo under different fine labels) are left out of every split"}


def write_results(out_dir, tables, cfg, seed, dataset_hash, splits_dir, **extra):
    """tables: {name: DataFrame} -> out_dir/<name>.csv and out_dir/summary.json (git hash, config, seed, manifest hash)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    prior = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {}
    git = {**dict.fromkeys(prior.get("tables", {}), prior.get("git_hash")), **prior.get("table_git_hash", {})}  # tables kept from earlier runs
    write_json(out_dir / "summary.json", {**provenance(cfg, seed, dataset_hash), **exclusion_note(splits_dir),
                                          "tables": {**prior.get("tables", {}), **{n: len(t) for n, t in tables.items()}},
                                          "table_git_hash": {**git, **dict.fromkeys(tables, git_hash())}, **extra})


# ---- Milestone 6: assembling the representation results (reports/repr/<encoder>_<crop>/*.csv)
MACRO = {  # result table -> {metric in the table: column in the summary}
    "r1": {"knn_make": "knn_make", "probe_make": "probe_make"},
    "r2_within_make": {"recall1": "recall1", "probe_fine": "probe_fine"},
    "r2_oracle": {"acc": "acc_oracle", "nmi": "nmi_oracle", "ari": "ari_oracle"},
    "r2_khat": {"acc": "acc_khat", "nmi": "nmi_khat", "ari": "ari_khat", "khat": "khat"},
    "r3": {"d0": "d0", "d1": "d1", "d2": "d2", "d0/d1": "d0/d1", "d1/d2": "d1/d2", "eq1": "eq1", "eq2": "eq2"},
    "r4": {"var_ratio": "var_ratio", "pr_subclass": "pr_subclass"}}
R5 = ("ari_seeds", "ari_boot", "khat_error")


def macro_row(run_dir):
    """One row of macro means with 95% CIs (`<metric>`, `<metric>_lo`, `<metric>_hi`) from a result directory."""
    row = {}
    for table, metrics in MACRO.items():
        t = pd.read_csv(Path(run_dir) / f"{table}.csv")
        m = t[t.group == "macro"].set_index("metric")
        for src, dst in metrics.items():
            row |= {dst: m.at[src, "mean"], f"{dst}_lo": m.at[src, "lo"], f"{dst}_hi": m.at[src, "hi"]}
    r5 = pd.read_csv(Path(run_dir) / "r5.csv").query("make_id == 'macro'").iloc[0]
    return row | {c: r5[c] for c in R5} | {"ari_boot_lo": r5.ari_boot_lo, "ari_boot_hi": r5.ari_boot_hi}


def per_make_table(run_dir):
    """One row per make (index make_id) with the point estimates of every metric; NaN where a make has no value."""
    cols = {}
    for table, metrics in MACRO.items():
        t = pd.read_csv(Path(run_dir) / f"{table}.csv")
        t = t[t.group != "macro"].assign(group=lambda d: d.group.astype(int))
        for src, dst in metrics.items():
            cols[dst] = t[t.metric == src].set_index("group")["mean"]
        if table == "r1":
            cols["n_test"] = t[t.metric == "knn_make"].set_index("group")["n"]
    r5 = pd.read_csv(Path(run_dir) / "r5.csv").query("make_id != 'macro'").assign(make_id=lambda d: d.make_id.astype(int)).set_index("make_id")
    return pd.DataFrame(cols | {c: r5[c] for c in R5}).rename_axis("make_id")


def worst_makes(per_make, column, n=10, lowest=True):
    """The n makes with the lowest (or highest) value of `column`, NaN rows (undefined for the make) left out."""
    d = per_make.dropna(subset=[column])
    return d.nsmallest(n, column) if lowest else d.nlargest(n, column)
