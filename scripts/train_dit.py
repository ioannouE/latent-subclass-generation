"""Train the official DiT (external/DiT, facebookresearch/DiT) on the train split, unchanged: this script only writes the training data in
DiT's format and launches its train.py. The data is an ImageFolder of the `<crop>_<dit.image_size>` crops (symlinks; conflicting images
left out by load_split), one folder per conditioning label (lsgen.methods.gen.ldm.image_labels: the coarse class, or with `conditioning`
one label per (coarse class, cluster) pair), named so that ImageFolder's class index is the label. DiT applies its own centre crop (a no-op
on the square crops), horizontal flip and label dropout (p = 0.1). Training runs whole epochs up to the first one that reaches `dit.steps`;
DiT saves a checkpoint every `dit.ckpt_every` steps. GPU:
    CONFIG=configs/gen_dit_pilot_maskcon_new.yaml sbatch slurm/train_dit.sh
One run per seed (`seeds`, or `seed` alone for the pilot: run `dit.name`).
Output: <derived_root>/generation[_<labels>]/<run>/ with imagefolder/, labels.json (label table and train counts, for sampling) and DiT's
results/<index>-<model>/{log.txt, checkpoints/<step>.pt}; the checkpoint paths are printed to stdout.
"""
import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from lsgen.data.labels import load_labels, suffix
from lsgen.data.splits import load_split
from lsgen.methods.gen.ldm import image_labels, runs, write_imagefolder
from lsgen.utils import REPO_ROOT, git_hash, load_config, setup_logging

log = logging.getLogger("train_dit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_dit_pilot_maskcon_new.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    D = cfg["dit"]
    derived, sfx = Path(dcfg["derived_root"]), suffix(cfg["labels"], dcfg)
    meta = load_labels(derived, dcfg["data_root"], cfg["labels"])[["make_id"]]
    ids = load_split(Path(dcfg["data_root"]) / "splits", "train", labels=cfg["labels"])
    coarse = meta.make_id.loc[ids].to_numpy()
    images = derived / f"{cfg['crop']}_{D['image_size']}"

    for seed, run in runs(cfg, "dit"):
        out = derived / f"generation{sfx}" / run
        label, pairs = image_labels(cfg, derived, ids, coarse, seed)
        write_imagefolder(out / "imagefolder", images, ids, label)
        (out / "labels.json").write_text(json.dumps({"seed": seed, "pairs": pairs.tolist(), "train_counts": np.bincount(label).tolist(),
                                                      "config": cfg["_config_path"], "git_hash": git_hash()}))
        epochs = math.ceil(D["steps"] / (len(ids) // D["global_batch_size"]))
        log.info("%s: %d train images, %d labels, %d epochs", run, len(ids), len(pairs), epochs)
        subprocess.run(["torchrun", "--standalone", f"--nproc_per_node={D['gpus']}", str(REPO_ROOT / D["repo"] / "train.py"),
                        "--data-path", str(out / "imagefolder"), "--results-dir", str(out / "results"), "--model", D["model"],
                        "--image-size", str(D["image_size"]), "--num-classes", str(len(pairs)), "--epochs", str(epochs),
                        "--global-batch-size", str(D["global_batch_size"]), "--global-seed", str(seed), "--vae", D["vae"],
                        "--num-workers", str(D["num_workers"]), "--ckpt-every", str(D["ckpt_every"])], check=True, stdout=sys.stderr)  # stdout: checkpoint paths only
        latest = sorted((out / "results").glob("*-*"))[-1]
        for ckpt in sorted((latest / "checkpoints").glob("*.pt")):
            print(ckpt, flush=True)


if __name__ == "__main__":
    main()
