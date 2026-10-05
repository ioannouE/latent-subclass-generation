"""Representation metrics R1-R5 (docs/METRICS.md) for cached embeddings. CPU, hours at n_boot=1000:
    sbatch slurm/eval_representation.sh
Output: <reports_root>/repr/<encoder>_<crop>/*.csv + summary.json. Fit/reference = train, scored on test.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lsgen.eval.report import write_results
from lsgen.eval.representation import (cluster_space, distance_matrix, khat_error, knn_correct, l2norm, probe_correct,
                                       r2_metric, r3_metric, r4_metric, r5_make, recall1_correct)
from lsgen.eval.stats import bootstrap, bootstrap_table
from lsgen.features.extract import read_csv
from lsgen.utils import REPO_ROOT, load_config, setup_logging

log = logging.getLogger("eval_representation")


def load(derived, enc, split, meta):
    files, x = read_csv(derived / "features" / enc / f"{split}.csv")
    return x, meta.loc[[f[:-4] for f in files]]


def r5_tables(z, groups, k_of_make):
    """R5 per make: mean ARI between seeds / bootstrap fits (K_c >= 2) and K-hat error (K_c = 1). The macro CI of the
    ARI columns is the 2.5/97.5 percentile of the make-averaged ARI of the i-th bootstrap fit."""
    rows, runs = [], []
    for c, idx in groups.items():
        if k_of_make[c] >= 2:
            r = r5_make(z[idx], k_of_make[c])
            rows.append({"make_id": c, "ari_seeds": np.mean(r["ari_seeds"]), "ari_boot": np.mean(r["ari_boot"])})
            runs.append(r["ari_boot"])
        else:
            rows.append({"make_id": c, "khat_error": khat_error(z[idx])})
    t = pd.DataFrame(rows)
    macro = {"make_id": "macro", **t.drop(columns="make_id").mean().to_dict()}
    lo, hi = np.percentile(np.mean(runs, 0), [2.5, 97.5])
    return pd.concat([t, pd.DataFrame([macro | {"ari_boot_lo": lo, "ari_boot_hi": hi}])], ignore_index=True)


def repr_tables(xtr, mtr, xte, mte, cfg):
    make, fine = mte.make_id.to_numpy(), mte.fine_id.to_numpy()
    groups = {c: np.flatnonzero(make == c) for c in np.unique(make)}
    k_of_make = {c: len(np.unique(fine[i])) for c, i in groups.items()}
    multi = {c: i for c, i in groups.items() if k_of_make[c] >= 2}
    nb, nj, seed = cfg["n_boot"], cfg["n_jobs"], cfg["seed"]
    log.info("%d makes, %d with K_c >= 2, %d with K_c = 1", len(groups), len(multi), len(groups) - len(multi))

    r1 = pd.DataFrame({"make_id": make, "knn_make": knn_correct(xtr, mtr.make_id, xte, make, cfg["knn_k"]),
                       "probe_make": probe_correct(xtr, mtr.make_id, xte, make)})
    recall1, probe_fine = np.full(len(make), np.nan), np.full(len(make), np.nan)
    for c, idx in multi.items():
        tr = (mtr.make_id == c).to_numpy()
        recall1[idx] = recall1_correct(xte[idx], fine[idx])
        probe_fine[idx] = probe_correct(xtr[tr], mtr.fine_id[tr], xte[idx], fine[idx])
    within = pd.DataFrame({"make_id": make, "recall1": recall1, "probe_fine": probe_fine})[np.isfinite(recall1)]

    z = cluster_space(xte, make, seed=seed)
    return {
        "r1": bootstrap_table(r1, "make_id", ["knn_make", "probe_make"], nb, seed, nj),
        "r2_within_make": bootstrap_table(within, "make_id", ["recall1", "probe_fine"], nb, seed, nj),
        "r2_oracle": bootstrap(r2_metric(z, make, fine, k_of_make, seed), multi, nb, seed, nj),
        "r2_khat": bootstrap(r2_metric(z, make, fine, k_of_make, seed, label_free=True), multi, nb, seed, nj, cfg["khat_subsample"]),
        "r3": bootstrap(r3_metric(distance_matrix(xte), make, fine), groups, nb, seed, nj),
        "r4": bootstrap(r4_metric(l2norm(xte), fine), multi, nb, seed, nj),
        "r5": r5_tables(z, groups, k_of_make)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_representation.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    meta = pd.read_parquet(derived / "metadata.parquet", columns=["id", "make_id", "fine_id"]).set_index("id")
    for enc in cfg["encoders"]:
        for size in cfg["crop_sizes"]:
            name = f"{enc}_{size}"
            (xtr, mtr), (xte, mte) = load(derived, name, "train", meta), load(derived, name, "test", meta)
            log.info("%s: train %d, test %d", name, len(xtr), len(xte))
            write_results(Path(dcfg["reports_root"]) / "repr" / name, repr_tables(xtr, mtr, xte, mte, cfg), cfg, cfg["seed"],
                          dataset_hash, splits_dir, encoder=name, n_train=len(xtr), n_test=len(xte))


if __name__ == "__main__":
    main()
