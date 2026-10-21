"""Scores geosub's own prototype clusters (geosub_clusters/<run>.csv, written by scripts/train_geosub.py) against the hidden subclasses,
on one split (`split` in the config: val to choose loss weights, test only for the final report). Per coarse class with K_c >= 2:
ACC (Hungarian) / NMI / ARI of the prototype assignment, i.e. with geosub's own label-free K-hat (prototypes active on train), and
flip_ari = ARI between the assignments of the crops and of their mirror images; macro mean with bootstrap CIs. Per coarse class (all):
K-hat, K_c (both on train) and K-hat error = K-hat - K_c (point estimates). The embedding itself is scored by scripts/eval_representation.py (R1-R5).
Like-for-like reference (`reference` in the config, as scripts/eval_george_clusters.py): k-means on the frozen embedding of the same images of the
same split, R2's pipeline with the oracle K and with the label-free K-hat (r2_metric). CPU:
    sbatch slurm/eval_geosub.sh
Output: <reports_root>/<report_dir>/<run>_clusters_<split>/{clusters,khat}.csv and <reference>_clusters_<split>/{reference_oracle,reference_khat}.csv,
each with summary.json.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lsgen.data.labels import load_labels
from lsgen.data.splits import load_split
from lsgen.eval.report import write_results
from lsgen.eval.representation import assignment_metric, cluster_space, r2_metric
from lsgen.eval.stats import bootstrap
from lsgen.features.extract import read_csv
from lsgen.utils import REPO_ROOT, load_config, setup_logging

log = logging.getLogger("eval_geosub")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_geosub_maskcon.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    meta = load_labels(derived, dcfg["data_root"], cfg["labels"])[["make_id", "fine_id"]]
    nb, nj, seed, split = cfg["n_boot"], cfg["n_jobs"], cfg["seed"], cfg["split"]

    for run in cfg["runs"]:
        clusters = pd.read_csv(derived / "geosub_clusters" / f"{run}.csv", index_col="id")
        train = meta.loc[clusters.index[clusters.split == "train"]]
        k_train = train.groupby("make_id").fine_id.nunique()  # K-hat is estimated on train, so K_c is counted there
        clusters = clusters[clusters.split == split]
        fine, make = meta.fine_id[clusters.index].to_numpy(), meta.make_id[clusters.index].to_numpy()
        groups = {c: np.flatnonzero(make == c) for c in np.unique(make)}
        k_of_make = {c: len(np.unique(fine[i])) for c, i in groups.items()}
        multi = {c: i for c, i in groups.items() if k_of_make[c] >= 2}
        khat = torch.load(derived / "repr_models" / f"{run}.pt", map_location="cpu")["active"].sum(1).numpy()
        k = pd.DataFrame({"make_id": k_train.index, "khat": khat[k_train.index], "k_c": k_train.to_numpy()})
        k["khat_error"] = k.khat - k.k_c
        k = pd.concat([k, pd.DataFrame([{"make_id": "macro", **k.drop(columns="make_id").mean().to_dict(),
                                         "abs_khat_error": k.khat_error.abs().mean()}])], ignore_index=True)
        log.info("%s %s: %d images, %d coarse classes, %d with K_c >= 2", run, split, len(make), len(groups), len(multi))
        tables = {"clusters": bootstrap(assignment_metric(fine, clusters.cluster.to_numpy(), clusters.cluster_flip.to_numpy()),
                                        multi, nb, seed, nj),
                  "khat": k}
        write_results(Path(dcfg["reports_root"]) / cfg["report_dir"] / f"{run}_clusters_{split}", tables, cfg, seed, dataset_hash,
                      splits_dir, run=run, split=split, n=len(make))
        log.info("%s\n%s\n%s", run, tables["clusters"].query("group == 'macro'").round(3).to_string(index=False),
                 k.tail(1).round(2).to_string(index=False))

    ids = load_split(splits_dir, split, labels=cfg["labels"])  # the images the runs' clusters cover
    fine, make = meta.fine_id[ids].to_numpy(), meta.make_id[ids].to_numpy()
    groups = {c: np.flatnonzero(make == c) for c in np.unique(make)}
    k_of_make = {c: len(np.unique(fine[i])) for c, i in groups.items()}
    multi = {c: i for c, i in groups.items() if k_of_make[c] >= 2}
    files, x = read_csv(derived / "features" / cfg["reference"] / f"{split}.csv")
    z = cluster_space(pd.DataFrame(x, index=[f[:-4] for f in files]).loc[ids].to_numpy(), make, seed=seed)
    tables = {"reference_oracle": bootstrap(r2_metric(z, make, fine, k_of_make, seed), multi, nb, seed, nj),
              "reference_khat": bootstrap(r2_metric(z, make, fine, k_of_make, seed, label_free=True), multi, nb, seed, nj, cfg["subsample"])}
    write_results(Path(dcfg["reports_root"]) / cfg["report_dir"] / f"{cfg['reference']}_clusters_{split}", tables, cfg, seed, dataset_hash,
                  splits_dir, reference=cfg["reference"], split=split, n=len(make))
    for name, t in tables.items():
        log.info("%s\n%s", name, t.query("group == 'macro'").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
