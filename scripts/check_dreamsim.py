"""DreamSim checks (configs/features_dreamsim.yaml). Needs a GPU and the weights cache; submit with
    STAGE=demo sbatch slurm/check_dreamsim.sh        the package demo prints the published distances (0.4453, 0.2756)
    STAGE=features sbatch slurm/check_dreamsim.sh    after extraction: image order, counts (test = n_test), unit norm of the cached embeddings
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from dreamsim import dreamsim
from PIL import Image

from lsgen.data.splits import load_split
from lsgen.features.extract import read_csv
from lsgen.utils import REPO_ROOT, load_config, setup_logging

log = logging.getLogger("check_dreamsim")


def demo(cfg, weights):
    model, preprocess = dreamsim(pretrained=True, device="cuda", cache_dir=str(weights))
    images = REPO_ROOT / cfg["demo"]["images"]
    ref, a, b = (preprocess(Image.open(images / f"{n}.png")).cuda() for n in ("ref_1", "img_a_1", "img_b_1"))
    got = [model(ref, a).item(), model(ref, b).item()]
    log.info("distances %s, expected %s", np.round(got, 4), cfg["demo"]["distances"])
    assert np.allclose(got, cfg["demo"]["distances"], atol=cfg["demo"]["tol"]), "DreamSim demo distances differ from the published ones"


def features(cfg, derived, splits_dir):
    ids = {s: load_split(splits_dir, s) for s in cfg["splits"]}
    assert len(ids["test"]) == cfg["n_test"], f"test has {len(ids['test'])} images, expected {cfg['n_test']}"
    for name in cfg["encoders"]:
        for size in cfg["crop_sizes"]:
            for split, split_ids in ids.items():
                files, x = read_csv(derived / "features" / f"{name}_{size}" / f"{split}.csv")
                assert files == [f"{i}.png" for i in split_ids], f"{name} {split}: image order differs from the split"
                assert np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-3), f"{name} {split}: not unit norm"
            log.info("%s %d: %s", name, size, {s: len(v) for s, v in ids.items()} | {"dim": x.shape[1]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["demo", "features"])
    ap.add_argument("--config", default="configs/features_dreamsim.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    derived = Path(dcfg["derived_root"])
    if args.stage == "demo":
        demo(cfg, derived / "weights" / "dreamsim")
    else:
        features(cfg, derived, Path(dcfg["data_root"]) / "splits")


if __name__ == "__main__":
    main()
