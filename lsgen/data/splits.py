"""Frozen splits. Fine labels are used here ONLY to stratify (documented in docs/DATA.md).

train  = official train minus val        val    = 10% of official train, stratified by fine_id
test   = official test (never for tuning) test_A / test_B = stratified halves of test
"""
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

SPLITS = ("train", "val", "test", "test_A", "test_B")


def _stratified(ids, labels, test_size, seed):
    rest, held = next(StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(ids, labels))
    return sorted(ids[i] for i in rest), sorted(ids[i] for i in held)


def make_splits(meta, val_frac=0.10, seed=0):
    """meta: DataFrame with id, official_split, fine_id. Returns {split: sorted list of ids}."""
    tr = meta[meta.official_split == "train"].sort_values("id")
    te = meta[meta.official_split == "test"].sort_values("id")
    train, val = _stratified(tr.id.tolist(), tr.fine_id.to_numpy(), val_frac, seed)
    test_A, test_B = _stratified(te.id.tolist(), te.fine_id.to_numpy(), 0.5, seed)
    return {"train": train, "val": val, "test": sorted(te.id), "test_A": test_A, "test_B": test_B}


def save_splits(splits, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")


def load_split(out_dir, name, exclude_eval=False):
    """exclude_eval=True drops the label-conflict test images (eval_exclude.txt): use it for every
    test-side evaluation reference."""
    ids = Path(out_dir, f"{name}.txt").read_text().split()
    if exclude_eval:
        drop = set(Path(out_dir, "eval_exclude.txt").read_text().split())
        ids = [i for i in ids if i not in drop]
    return ids
