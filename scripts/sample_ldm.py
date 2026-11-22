"""Sample use-case-A sets from a checkpoint of scripts/train_ldm.py: conditioning labels in train prevalence (prevalence-matching: coarse
classes, or (coarse, cluster) pairs, so clusters in proportion to their train size within each class), DDIM with each
number of steps in `sampling.ddim_steps`, classifier-free guidance `sampling.guidance`. The initial noise is the same for every set (same
seed: the run's), so sets differ only by checkpoint and sampler. Sets are scored against `sampling.reference_split` (default val: the
G-0c pilot; test only for the final runs at the fixed budget). GPU:
    sbatch slurm/generate.sh            (G-0c pilot: slurm/gen_pilot.sh)
Output: <derived_root>/generation[_<labels>]/<run>/step<step>_ddim<steps>/ with samples.csv + method.json (paths printed to stdout), <run>
being the checkpoint's directory. With cluster conditioning, samples.csv also has the requested `cluster` and `group` (the label: G6).
"""
import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from lsgen.data.labels import suffix
from lsgen.methods.gen.ldm import build_unet, load_latents, prevalence_labels, sample, schedulers
from lsgen.utils import REPO_ROOT, git_hash, load_config, setup_logging

log = logging.getLogger("sample_ldm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_pilot_maskcon_new.yaml")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    device, s, L = torch.device("cuda"), cfg["sampling"], cfg["ldm"]
    derived, sfx = Path(dcfg["derived_root"]), suffix(cfg["labels"], dcfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    unet = build_unet(ck["arch"], ck["latent_shape"][0], ck["n_classes"])
    unet.load_state_dict(ck["ema"])
    unet = unet.to(device).eval()
    _, ddim = schedulers(L["scheduler"])
    spec = cfg["autoencoders"][ck["autoencoder"]]
    ae = load_latents(ck["autoencoder"], spec, cfg["image_size"], derived / f"autoencoders{sfx}").to(device)
    run, seed = Path(args.ckpt).parent.name, ck.get("seed", cfg["seed"])
    labels = torch.from_numpy(prevalence_labels(ck["train_counts"], s["n"], seed)).to(device)
    cond = cfg.get("conditioning", {"kind": "coarse"})
    requested = pd.DataFrame({"filename": [f"{i:06d}.png" for i in range(s["n"])], "make_id": ck["pairs"][labels.cpu().numpy(), 0]})
    if cond["kind"] != "coarse":
        requested["cluster"], requested["group"] = ck["pairs"][labels.cpu().numpy(), 1], labels.cpu().numpy()

    for steps in s["ddim_steps"]:
        name = f"step{ck['step']:06d}_ddim{steps}"
        out = derived / f"generation{sfx}" / run / name
        out.mkdir(parents=True, exist_ok=True)
        g, t0 = torch.Generator(device).manual_seed(seed), time.time()
        for b in range(0, s["n"], s["batch_size"]):
            y = labels[b:b + s["batch_size"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = sample(unet, ddim, y, ck["latent_shape"], steps, s["guidance"], ck["n_classes"], g)
            x = ae.decode(z.float())
            for i, im in enumerate((x * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()):
                Image.fromarray(im).save(out / f"{b + i:06d}.png")
        seconds = time.time() - t0
        requested.to_csv(out / "samples.csv", index=False)
        (out / "method.json").write_text(json.dumps({
            "name": f"{run}_{name}", "uses": spec["uses"] + cond.get("uses", []), "reference_split": s.get("reference_split", "val"), "conditioning": cond,
            "n_labels": ck["n_classes"], "autoencoder": ck["autoencoder"], "step": ck["step"], "train_hours": ck["train_hours"], "ddim_steps": steps,
            "guidance": s["guidance"], "sampling_seconds": seconds, "git_hash": git_hash(), "seed": seed}, indent=1))
        log.info("%s: %d samples in %.0f s", name, s["n"], seconds)
        print(out)


if __name__ == "__main__":
    main()
