"""Fine-label oracle of the generation ladder (upper bound, labelled "uses fine labels"): writes the train images' fine classes in the
per-image cluster-file format that scripts/train_ldm.py reads (`conditioning.labels`), so the oracle differs from the cluster rungs only in
its label file. This script, not the training code, reads the fine labels. CPU, seconds:
    python scripts/oracle_labels.py --config configs/gen_oracle_maskcon_new.yaml
Output: <derived_root>/gen_labels[_<labels>]/oracle.csv (id, split, make_id, cluster = fine class).
"""
import argparse
from pathlib import Path

from lsgen.data.labels import load_labels, suffix
from lsgen.data.splits import load_split
from lsgen.utils import REPO_ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_oracle_maskcon_new.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    derived = Path(dcfg["derived_root"])
    meta = load_labels(derived, dcfg["data_root"], cfg["labels"])
    ids = load_split(Path(dcfg["data_root"]) / "splits", "train", labels=cfg["labels"])
    out = derived / f"gen_labels{suffix(cfg['labels'], dcfg)}" / "oracle.csv"
    out.parent.mkdir(exist_ok=True)
    meta.loc[ids, ["make_id", "fine_id"]].rename(columns={"fine_id": "cluster"}).rename_axis("id").reset_index().assign(split="train")[
        ["id", "split", "make_id", "cluster"]].to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
