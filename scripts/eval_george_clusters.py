"""Scores GEORGE's own clusters (george_clusters/george_s<seed>.csv, written by scripts/train_george.py) against the hidden subclasses:
ACC (Hungarian) / NMI / ARI per make with K_c >= 2, macro mean with bootstrap CIs, as R2 but for the clusters GEORGE chose (its own K).
GEORGE clusters only the train images, so this is scored on train; for a like-for-like reference, k-means on the frozen DINOv3-B train
embeddings is scored the same way, with the oracle K and with the label-free K-hat (r2_metric). CPU:
    sbatch slurm/eval_george_clusters.sh
Output: <reports_root>/repr/george_clusters_s<seed>/{george,reference_oracle,reference_khat}.csv + summary.json.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from lsgen.data.labels import load_labels
from lsgen.eval.report import write_results
from lsgen.eval.representation import cluster_space, hungarian_acc, r2_metric
from lsgen.eval.stats import bootstrap
from lsgen.features.extract import read_csv
from lsgen.utils import REPO_ROOT, load_config, setup_logging

log = logging.getLogger("eval_george_clusters")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_george_clusters.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    meta = load_labels(derived, dcfg["data_root"])[["make_id", "fine_id"]]
    nb, nj, seed = cfg["n_boot"], cfg["n_jobs"], cfg["seed"]

    clusters = pd.read_csv(derived / "george_clusters" / f"george_s{seed}.csv", index_col="id")
    fine, make, assign = meta.fine_id[clusters.index].to_numpy(), clusters.make_id.to_numpy(), clusters.cluster.to_numpy()
    groups = {c: np.flatnonzero(make == c) for c in np.unique(make)}
    k_of_make = {c: len(np.unique(fine[i])) for c, i in groups.items()}
    multi = {c: i for c, i in groups.items() if k_of_make[c] >= 2}
    log.info("%d train images, %d makes, %d with K_c >= 2", len(make), len(groups), len(multi))
    george = bootstrap(lambda i: {"acc": hungarian_acc(fine[i], assign[i]), "nmi": normalized_mutual_info_score(fine[i], assign[i]),
                                  "ari": adjusted_rand_score(fine[i], assign[i])}, multi, nb, seed, nj)
    george = pd.concat([george, pd.DataFrame([dict(group="macro", metric="k_george", mean=np.mean([len(np.unique(assign[i])) for i in multi.values()]),
                                                   lo=np.nan, hi=np.nan, n=len(multi))])], ignore_index=True)

    files, x = read_csv(derived / "features" / cfg["reference"] / "train.csv")
    x = pd.DataFrame(x, index=[f[:-4] for f in files]).loc[clusters.index].to_numpy()
    z = cluster_space(x, make, seed=seed)
    tables = {"george": george,
              "reference_oracle": bootstrap(r2_metric(z, make, fine, k_of_make, seed), multi, nb, seed, nj),
              "reference_khat": bootstrap(r2_metric(z, make, fine, k_of_make, seed, label_free=True), multi, nb, seed, nj, cfg["subsample"])}
    write_results(Path(dcfg["reports_root"]) / "repr" / f"george_clusters_s{seed}", tables, cfg, seed, dataset_hash, splits_dir,
                  reference=cfg["reference"], split="train", n=len(make))
    for name, t in tables.items():
        log.info("%s\n%s", name, t[t.group == "macro"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
