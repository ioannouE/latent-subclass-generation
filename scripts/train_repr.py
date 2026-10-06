"""Step 2: SupCon / SimCLR / SupCon+SimCLR / MaskCon / FALCON fine-tuning of the last blocks of a frozen-recipe backbone
(configs/repr_baselines.yaml). Coarse (make) labels only; FALCON additionally needs the number of fine classes K. Needs a GPU:
    sbatch slurm/train_repr.sh
Output per method and seed: <derived_root>/features/<method>_s<seed>_<crop size>/{train,val,test}.csv (+ .json), the same format as the
frozen encoders, so scripts/eval_representation.py scores them by name; and <derived_root>/repr_models/<method>_s<seed>.pt.
"""
import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T

from lsgen.data.datasets import CarsTwoViews
from lsgen.data.splits import load_split
from lsgen.eval.representation import knn_accuracy
from lsgen.features.encoders import Embedder
from lsgen.features.extract import embed_images, read_csv, write_csv
from lsgen.methods.repr.falcon import Falcon, coarse_neighbors
from lsgen.methods.repr.losses import Contrastive
from lsgen.methods.repr.maskcon import MaskCon
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, sha256_file, write_json

log = logging.getLogger("train_repr")


def build_backbone(cfg, device):
    """Embedder (CLS output, as in the frozen-encoder tables) with only the last blocks and the final norm trainable."""
    model = Embedder(cfg["backbone"]).to(device)
    trainable = list(model.model.blocks[-cfg["trainable_blocks"]:].parameters()) + list(model.model.norm.parameters())
    for p in trainable:
        p.requires_grad_(True)
    return model, trainable


def augmentation(model, cfg):
    a = cfg["aug"]
    cfg_m = model.model.pretrained_cfg
    return T.Compose([
        T.RandomResizedCrop(model.input_size, scale=(a["scale_min"], 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(a["color_jitter"], a["color_jitter"], a["color_jitter"], a["color_jitter"] / 4)], a["jitter_p"]),
        T.RandomGrayscale(a["grayscale_p"]), T.ToTensor(), T.Normalize(cfg_m["mean"], cfg_m["std"])])


def lr_factor(step, warmup, total):
    return (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def embed_split(model, derived, crop, ids, cfg, device):
    return embed_images(model, [derived / crop / f"{i}.png" for i in ids], model.transform, device,
                        cfg["batch_size_embed"], cfg["num_workers"])


def build_objective(kind, params, net, cfg, num_coarse, device):
    if kind == "contrastive":
        return Contrastive(net, params["supcon"], params["simclr"], cfg["temperature"])
    if kind == "maskcon":
        return MaskCon(net, cfg["head"][1], device=device, **params)
    return Falcon(net, num_coarse=num_coarse, device=device, **params)


def neighbor_table(cfg, derived, ids, make, k):
    """(N, k) same-make nearest neighbours of each train image in the frozen features of the training backbone."""
    files, x = read_csv(derived / "features" / f"{cfg['backbone']}_{cfg['crop'].split('_')[-1]}" / "train.csv")
    x = pd.DataFrame(x, index=[f[:-4] for f in files]).loc[ids].to_numpy()
    return torch.from_numpy(coarse_neighbors(x, make, k))


def train(cfg, kind, params, seed, ids, meta, derived, device, val=None):
    """Fine-tune and return the Embedder. val = (ids, make labels) for the monitoring kNN only."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    params = {k: v for k, v in params.items() if k != "kind"}
    model, trainable = build_backbone(cfg, device)
    dim = model(torch.zeros(1, 3, model.input_size, model.input_size, device=device)).shape[1]
    hidden, out = cfg["head"]
    head = torch.nn.Linear(dim, params["num_fine"]) if kind == "falcon" else \
        torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, out))
    net = torch.nn.Sequential(model, head.to(device))
    variant, size = cfg["crop"].split("_")
    make_of = meta.set_index("id").make_id
    make = make_of[ids].to_numpy()
    weak = model.transform if kind != "contrastive" else None  # MaskCon key view, FALCON EMA view and neighbours: no augmentation
    neighbors = neighbor_table(cfg, derived, ids, make, params.pop("neighbor_pool")) if kind == "falcon" else None
    ds = CarsTwoViews(meta, derived, ids, augmentation(model, cfg), variant, int(size), weak, neighbors,
                      params.pop("n_neighbors", 5))
    dl = torch.utils.data.DataLoader(ds, cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["num_workers"],
                                     persistent_workers=True)
    objective = build_objective(kind, params, net, cfg, int(make.max()) + 1, device)
    opt = torch.optim.AdamW([{"params": trainable, "lr": cfg["lr_backbone"]}, {"params": head.parameters(), "lr": cfg["lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    total, warmup = cfg["epochs"] * len(dl), cfg["warmup_epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warmup, total))
    scaler = torch.amp.GradScaler()
    for epoch in range(cfg["epochs"]):
        losses = []
        objective.start_epoch(epoch + 1)
        for i, batch in enumerate(dl):
            loss = objective.loss([b.to(device) for b in batch])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            objective.end_step(i)
            losses.append(loss.item())
        msg = f"epoch {epoch + 1}/{cfg['epochs']} loss {np.mean(losses):.4f}"
        if val is not None and ((epoch + 1) % cfg["eval_every"] == 0 or epoch + 1 == cfg["epochs"]):
            xtr, xva = (embed_split(model, derived, cfg["crop"], i, cfg, device) for i in (ids, val[0]))
            msg += f" | val kNN@20 make acc {knn_accuracy(xtr, make_of[ids], xva, val[1]):.4f}"
        log.info(msg)
    return model, float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/repr_baselines.yaml")
    ap.add_argument("--methods", nargs="*", help="subset of the methods in the config (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("training needs a GPU: sbatch slurm/train_repr.sh")
    device = torch.device("cuda")
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    meta = pd.read_parquet(derived / "metadata.parquet", columns=["id", "make_id"])
    make_of = meta.set_index("id").make_id
    ids = {s: load_split(splits_dir, s) for s in ("train", "val", "test")}
    log.info("images per split after excluding conflicting ones: %s", {s: len(v) for s, v in ids.items()})
    size = cfg["crop"].split("_")[-1]

    for name, params in cfg["methods"].items():
        if args.methods and name not in args.methods:
            continue
        for seed in cfg["seeds"]:
            run = f"{name}_s{seed}"
            log.info("%s: %s, seed %d", run, params, seed)
            model, final_loss = train(cfg, params["kind"], params, seed, ids["train"], meta, derived, device,
                                      val=(ids["val"], make_of[ids["val"]].to_numpy()))
            (derived / "repr_models").mkdir(exist_ok=True)
            torch.save({n: p for n, p in model.named_parameters() if p.requires_grad}, derived / "repr_models" / f"{run}.pt")  # trained weights only
            out = derived / "features" / f"{run}_{size}"
            out.mkdir(parents=True, exist_ok=True)
            for split, split_ids in ids.items():
                emb = embed_split(model, derived, cfg["crop"], split_ids, cfg, device)
                write_csv(out / f"{split}.csv", [f"{i}.png" for i in split_ids], emb)
                write_json(out / f"{split}.json", {
                    **provenance(cfg, seed, dataset_hash), "encoder": run, "family": model.family, "backbone": cfg["backbone"],
                    "method": params, "final_train_loss": final_loss, "crop": cfg["crop"], "split": split,
                    "n": len(split_ids), "dim": emb.shape[1], "labels": "coarse (make) only",
                    "exclude_sha256": sha256_file(splits_dir / "exclude.txt")})
                log.info("%s %s: %s", run, split, emb.shape)


if __name__ == "__main__":
    main()
