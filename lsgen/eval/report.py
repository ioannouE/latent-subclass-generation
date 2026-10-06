"""Result files: every table is a CSV, and summary.json next to them records provenance and the data exclusion."""
import json
from pathlib import Path

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
