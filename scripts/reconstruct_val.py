"""G-0c autoencoder choice: encode and decode the val images with each candidate (configs/gen_pilot_*.yaml `autoencoders`) and write them
as a generated set scored against val (method.json reference_split: val) by scripts/eval_generation.py: G1 is the reconstruction FID / KID,
and E1 (samples.csv carries the image's own fine class; evaluation only) is the fine evaluator's accuracy on reconstructions next to its
accuracy on the real val images, i.e. how much subclass detail the autoencoder keeps. PSNR is recorded in method.json. GPU:
    sbatch slurm/autoencoders.sh
Output: <derived_root>/generation[_<labels>]/recon_<name>/ (paths printed to stdout; logging goes to stderr). Test is not used.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lsgen.data.datasets import CarsCoarseDataset
from lsgen.data.labels import load_labels, suffix
from lsgen.data.splits import load_split
from lsgen.methods.gen.ldm import load_latents
from lsgen.utils import REPO_ROOT, git_hash, load_config, setup_logging

log = logging.getLogger("reconstruct_val")


def to_uint8(x):
    return (x * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_pilot_maskcon_new.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    device = torch.device("cuda")
    derived, sfx = Path(dcfg["derived_root"]), suffix(cfg["labels"], dcfg)
    meta = load_labels(derived, dcfg["data_root"], cfg["labels"])
    ids = load_split(Path(dcfg["data_root"]) / "splits", "val", labels=cfg["labels"])
    ds = CarsCoarseDataset(meta.reset_index(), derived, ids, cfg["crop"], cfg["image_size"], augment="none")
    dl = torch.utils.data.DataLoader(ds, cfg["sampling"]["batch_size"], num_workers=cfg["num_workers"])

    for name, spec in cfg["autoencoders"].items():
        ae = load_latents(name, spec, cfg["image_size"], derived / f"autoencoders{sfx}").to(device)
        out = derived / f"generation{sfx}" / f"recon_{name}"
        out.mkdir(parents=True, exist_ok=True)
        se, i = [], 0
        for x, _ in dl:
            real, rec = to_uint8(x), to_uint8(ae.decode(ae.encode(x.to(device))))
            se += list(((rec.astype(float) - real) ** 2).mean((1, 2, 3)))
            for im in rec:
                Image.fromarray(im).save(out / f"{i:06d}.png")
                i += 1
        psnr = float(np.mean(10 * np.log10(255 ** 2 / np.maximum(se, 1e-10))))
        meta.loc[ids, ["make_id", "fine_id"]].assign(filename=[f"{j:06d}.png" for j in range(len(ids))])[
            ["filename", "make_id", "fine_id"]].to_csv(out / "samples.csv", index=False)
        (out / "method.json").write_text(json.dumps({"name": f"recon_{name}", "uses": spec["uses"], "reference_split": "val", "autoencoder": spec,
                                                     "psnr": psnr, "git_hash": git_hash(), "seed": cfg["seed"]}, indent=1))
        log.info("%s: %d val reconstructions, PSNR %.2f dB", name, len(ids), psnr)
        print(out)
        del ae
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
