"""Pre-check of the DINO-minus-DreamSim identity score before any training (docs/EXPERIMENT_IDSCORE.md). CPU only, reads cached
frozen embeddings:
    sbatch slurm/identity_check.sh
Output: <reports_root>/identity_check/{table.csv, summary.json}; table rows are per make and "macro", CIs from 63.2% subsamples of each make.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lsgen.data.labels import load_labels
from lsgen.data.splits import load_split
from lsgen.eval.identity import make_metrics
from lsgen.eval.report import write_results
from lsgen.eval.representation import l2norm
from lsgen.eval.stats import bootstrap
from lsgen.features.extract import read_csv
from lsgen.utils import REPO_ROOT, load_config, setup_logging

log = logging.getLogger("identity_check")


def load_embeddings(derived, encoder, cfg, ids):
    files, x = read_csv(derived / "features" / f"{encoder}_{cfg['crop_size']}" / f"{cfg['split']}.csv")
    return l2norm(pd.DataFrame(x, index=[f[:-4] for f in files]).loc[ids].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/identity_check.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    ids = load_split(splits_dir, cfg["split"])
    labels = load_labels(derived, dcfg["data_root"]).loc[ids]
    make, fine = labels.make_id.to_numpy(), labels.fine_id.to_numpy()
    dino, dreamsim = (load_embeddings(derived, cfg[e], cfg, ids) for e in ("dino", "dreamsim"))
    groups = {c: np.flatnonzero(make == c) for c in np.unique(make) if len(np.unique(fine[make == c])) >= 2}  # makes with K_c >= 2
    log.info("%d %s images after excluding conflicting ones, %d makes with K_c >= 2", len(ids), cfg["split"], len(groups))

    table = bootstrap(lambda i: make_metrics(dino[i], dreamsim[i], fine[i], cfg["gate_k"], cfg["top_n"]), groups,
                      cfg["n_boot"], cfg["seed"], cfg["n_jobs"], cfg["subsample"])
    macro = table[table.group == "macro"].set_index("metric")
    gain = macro.loc[f"gain_p{cfg['top_n']}_gated"]
    passed = bool(gain["mean"] >= cfg["decision"]["min_gain"] and gain["lo"] > 0)
    verdict = "go: the gated identity score finds same-subclass neighbours better than DINO alone" if passed else \
        "stop: the gated identity score does not beat DINO alone by the fixed margin"
    write_results(Path(dcfg["reports_root"]) / "identity_check", {"table": table}, cfg, cfg["seed"], dataset_hash, splits_dir,
                  decision=cfg["decision"], verdict=verdict)
    log.info("\n%s\nverdict: %s", macro[["mean", "lo", "hi"]].round(3).to_string(), verdict)


if __name__ == "__main__":
    main()
