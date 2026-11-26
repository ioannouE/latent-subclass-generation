"""Train the controlled latent diffusion generator (lsgen/methods/gen/ldm.py) on the train split (`ldm` in configs/gen_*.yaml). The
conditioning label is the coarse class, or, with `conditioning`, one label per (coarse class, cluster) pair (condition_labels) read from a
per-image cluster file (`conditioning.labels`, under derived_root, `{seed}` = the run's seed): label-free clusters (scripts/train_ksub.py,
<derived_root>/ksub_clusters/) or, for the labelled oracle only, the fine classes (scripts/oracle_labels.py). The train images and their
horizontal flips are encoded once with the frozen autoencoder and kept in GPU memory. EMA weights are saved at each step in `checkpoints`,
with the training time so far. GPU:
    sbatch slurm/generate.sh            (G-0c pilot: slurm/gen_pilot.sh)
One run per seed (lsgen.methods.gen.ldm.runs).
Output: <derived_root>/generation[_<labels>]/<run>/step_<step>.pt (paths printed to stdout) and <reports_root>/generation[_<labels>]/<run>_train.json.
"""
import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from lsgen.data.datasets import CarsCoarseDataset
from lsgen.data.labels import load_labels, suffix
from lsgen.data.splits import load_split
from lsgen.methods.gen.ldm import build_unet, diffusion_loss, drop_labels, ema_update, image_labels, load_latents, runs, schedulers
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, write_json

log = logging.getLogger("train_ldm")


@torch.no_grad()
def encode_train(ae, meta, derived, ids, cfg, device):
    """Latents of the images in `ids` order, followed by the latents of their flips."""
    ds = CarsCoarseDataset(meta, derived, ids, cfg["crop"], cfg["image_size"], augment="none")
    z, z_flip = [], []
    for x, _ in torch.utils.data.DataLoader(ds, cfg["sampling"]["batch_size"], num_workers=cfg["num_workers"]):
        x = x.to(device)
        z.append(ae.encode(x))
        z_flip.append(ae.encode(x.flip(-1)))
    return torch.cat(z + z_flip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_pilot_maskcon_new.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("diffusion training needs a GPU: sbatch slurm/generate.sh")
    device, L = torch.device("cuda"), cfg["ldm"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    derived, sfx = Path(dcfg["derived_root"]), suffix(cfg["labels"], dcfg)
    meta = load_labels(derived, dcfg["data_root"], cfg["labels"])[["make_id"]]
    ids = load_split(Path(dcfg["data_root"]) / "splits", "train", labels=cfg["labels"])
    coarse = meta.make_id.loc[ids].to_numpy()

    ae = load_latents(L["autoencoder"], cfg["autoencoders"][L["autoencoder"]], cfg["image_size"], derived / f"autoencoders{sfx}").to(device)
    z = encode_train(ae, meta.reset_index(), derived, ids, cfg, device)
    del ae
    torch.cuda.empty_cache()
    log.info("%d train images (+ flips), latents %s, std %.3f", len(ids), tuple(z.shape[1:]), z.std().item())

    for seed, run in runs(cfg):
        label, pairs = image_labels(cfg, derived, ids, coarse, seed)
        log.info("seed %d: %d conditioning labels (%s per coarse class)", seed, len(pairs), np.bincount(pairs[:, 0]).tolist())
        y = torch.from_numpy(np.tile(label, 2)).to(device)  # images, then their flips (encode_train)
        history = train(L, seed, derived / f"generation{sfx}" / run, z, y, pairs, np.bincount(label), device)
        write_json(Path(dcfg["reports_root"]) / f"generation{sfx}" / f"{run}_train.json", {**provenance(cfg, seed), "history": history})


def train(L, seed, out, z, y, pairs, counts, device):
    """Train one seed; save EMA checkpoints in `out`, print their paths, return the training log. pairs: (coarse, cluster) of each label;
    counts: train images per label, for prevalence-matching sampling."""
    n_classes = len(pairs)
    torch.manual_seed(seed)
    unet = build_unet(L["arch"], z.shape[1], n_classes).to(device)
    ema = copy.deepcopy(unet).requires_grad_(False)
    ddpm, _ = schedulers(L["scheduler"])
    opt = torch.optim.AdamW(unet.parameters(), L["lr"], weight_decay=0.0)
    log.info("seed %d: UNet %.1fM parameters", seed, sum(p.numel() for p in unet.parameters()) / 1e6)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device).manual_seed(seed)
    history, train_s, t0 = [], 0.0, time.time()
    for step in range(1, L["steps"] + 1):
        idx = torch.randint(len(z), (L["batch_size"],), generator=g, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = diffusion_loss(unet, ddpm, z[idx], drop_labels(y[idx], L["p_uncond"], n_classes, g), g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        ema_update(ema, unet, step, L["ema_decay"])
        if step % L["log_every"] == 0:
            torch.cuda.synchronize()
            train_s += time.time() - t0
            row = {"step": step, "loss": loss.item(), "img_per_s": L["log_every"] * L["batch_size"] / (time.time() - t0), "train_hours": train_s / 3600}
            history.append(row)
            log.info(json.dumps(row))
            t0 = time.time()
        if step in L["checkpoints"]:
            torch.save({"ema": ema.state_dict(), "step": step, "seed": seed, "arch": L["arch"], "latent_shape": tuple(z.shape[1:]),
                        "n_classes": n_classes, "pairs": pairs, "train_counts": counts, "train_hours": train_s / 3600, "autoencoder": L["autoencoder"]},
                       out / f"step_{step:06d}.pt")
            print(out / f"step_{step:06d}.pt", flush=True)
            t0 = time.time()  # saving is not training time
    return history


if __name__ == "__main__":
    main()
