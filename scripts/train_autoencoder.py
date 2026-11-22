"""G-0c: train the own autoencoder (MONAI AutoencoderKL; L1 + LPIPS + KL, patch adversarial loss from `adv_start`) on the train split,
the candidate against the frozen SD-VAE. Images only: no labels are read. GPU:
    sbatch slurm/autoencoders.sh
The latent scale (1 / std of train latents) is stored with the weights. Val L1 / LPIPS are logged for information, not for selection.
Output: <derived_root>/autoencoders[_<labels>]/own/model.pt and <reports_root>/autoencoder[_<labels>]_own.json (the `own` candidate of
`autoencoders` in the config; suffix as in lsgen.data.labels.suffix, since the train split depends on the scheme's exclusion list).
"""
import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator

from lsgen.data.datasets import CarsCoarseDataset
from lsgen.data.labels import suffix
from lsgen.data.splits import load_split
from lsgen.methods.gen.ldm import build_autoencoder, kl_loss
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, write_json

log = logging.getLogger("train_autoencoder")


def loader(meta, derived, ids, cfg, train):
    ds = CarsCoarseDataset(meta, derived, ids, cfg["crop"], cfg["image_size"], augment="hflip" if train else "none")
    return torch.utils.data.DataLoader(ds, cfg["own_ae"]["batch_size"], shuffle=train, drop_last=train, num_workers=cfg["num_workers"],
                                       generator=torch.Generator().manual_seed(cfg["seed"]))


def forever(dl):
    while True:
        yield from dl


@torch.no_grad()
def evaluate(ae, perceptual, dl, device):
    ae.eval()
    l1 = lp = n = 0
    for x, _ in dl:
        x = x.to(device) * 2 - 1
        rec = ae(x)[0]
        l1 += (rec - x).abs().mean((1, 2, 3)).sum().item()
        lp += perceptual(rec, x).item() * len(x)
        n += len(x)
    ae.train()
    return l1 / n, lp / n


@torch.no_grad()
def latent_std(ae, dl, device, n_batches=50):
    return torch.cat([ae.encode(x.to(device) * 2 - 1)[0].flatten() for x, _ in itertools.islice(dl, n_batches)]).std().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_pilot_maskcon_new.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("autoencoder training needs a GPU: sbatch slurm/autoencoders.sh")
    device, a = torch.device("cuda"), cfg["own_ae"]
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    meta = pd.read_parquet(derived / "metadata.parquet", columns=["id", "make_id"])  # the dataset needs a label column; it is not used
    dl = {s: loader(meta, derived, load_split(splits_dir, s, labels=cfg["labels"]), cfg, s == "train") for s in ("train", "val")}

    ae = build_autoencoder(a["arch"]).to(device)
    disc = PatchDiscriminator(spatial_dims=2, in_channels=3, **a["disc"]).to(device)
    perceptual = PerceptualLoss(spatial_dims=2, network_type=a["perceptual_net"]).to(device)
    adversarial = PatchAdversarialLoss(criterion="least_squares")
    opt_g = torch.optim.Adam(ae.parameters(), a["lr"])
    opt_d = torch.optim.Adam(disc.parameters(), a["lr"])

    history, batches, t0 = [], forever(dl["train"]), time.time()
    for step in range(1, a["steps"] + 1):
        x = next(batches)[0].to(device) * 2 - 1
        rec, mu, sigma = ae(x)
        terms = {"l1": (rec - x).abs().mean(), "lpips": perceptual(rec, x), "kl": kl_loss(mu, sigma)}
        loss = terms["l1"] + a["perceptual_weight"] * terms["lpips"] + a["kl_weight"] * terms["kl"]
        use_adv = step > a["adv_start"]
        if use_adv:
            terms["g_adv"] = adversarial(disc(rec)[-1], target_is_real=True, for_discriminator=False)
            loss = loss + a["adv_weight"] * terms["g_adv"]
        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        opt_g.step()
        if use_adv:
            d_loss = 0.5 * (adversarial(disc(rec.detach())[-1], target_is_real=False, for_discriminator=True)
                            + adversarial(disc(x)[-1], target_is_real=True, for_discriminator=True))
            opt_d.zero_grad(set_to_none=True)
            (a["adv_weight"] * d_loss).backward()
            opt_d.step()
            terms["d"] = d_loss
        if step % a["log_every"] == 0 or step == a["steps"]:
            val_l1, val_lpips = evaluate(ae, perceptual, dl["val"], device)
            row = {"step": step, **{k: v.item() for k, v in terms.items()}, "val_l1": val_l1, "val_lpips": val_lpips,
                   "img_per_s": step * len(x) / (time.time() - t0)}
            history.append(row)
            log.info(json.dumps(row))

    ae.eval()
    scale = 1 / latent_std(ae, dl["train"], device)
    sfx = suffix(cfg["labels"], dcfg)
    out = derived / f"autoencoders{sfx}" / "own"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ae.state_dict(), "arch": a["arch"], "scale": scale}, out / "model.pt")
    write_json(Path(dcfg["reports_root"]) / f"autoencoder{sfx}_own.json",
               {**provenance(cfg, cfg["seed"]), "scale": scale, "gpu_hours": (time.time() - t0) / 3600, "history": history})
    log.info("saved %s (latent scale %.4f)", out / "model.pt", scale)


if __name__ == "__main__":
    main()
