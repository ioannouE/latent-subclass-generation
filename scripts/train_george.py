"""Step 2 baseline: GEORGE (Sohoni et al., NeurIPS 2020; third_party/hidden-stratification) on coarse (make) labels only, same backbone
recipe as configs/repr_baselines.yaml (configs/repr_george.yaml). Three stages, as in the paper:
  1. ERM: fine-tune the last blocks of the backbone to predict the make; take its train-set activations (CLS embedding).
  2. Per make: UMAP, then a GMM whose number of components is chosen by silhouette score (GEORGE's UMAPReducer / AutoKMixtureModel).
  3. Fine-tune a fresh backbone to predict the make with group DRO over the (make, cluster) groups (GEORGE's LossComputer).
Needs a GPU: sbatch slurm/train_george.sh
Output per seed: <derived_root>/features/george<tag>_s<seed>_<size> (GDRO model, primary) and george_erm<tag>_s<seed>_<size> (stage-1 model), in the
format of the frozen encoders, so scripts/eval_representation.py scores them by name (configs/eval_repr_george.yaml); and
<derived_root>/george_clusters/george<tag>_s<seed>.csv with the cluster of every train image. `tag` (config, default empty) keeps runs on
another label scheme (`labels`) from overwriting the default ones.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import sklearn.utils
from sklearn.utils._chunking import get_chunk_n_rows

sklearn.utils.get_chunk_n_rows = get_chunk_n_rows  # GEORGE imports it from its old location
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "hidden-stratification"))
from stratification.classification.losses.loss_computer import LossComputer  # noqa: E402
from stratification.cluster.models.cluster import AutoKMixtureModel  # noqa: E402
from stratification.cluster.models.reduction import UMAPReducer  # noqa: E402

from lsgen.data.datasets import CarsCoarseDataset  # noqa: E402
from lsgen.data.labels import load_labels  # noqa: E402
from lsgen.data.splits import load_split  # noqa: E402
from lsgen.eval.representation import knn_accuracy  # noqa: E402
from lsgen.features.extract import write_csv  # noqa: E402
from lsgen.methods.repr.common import forward_fp16  # noqa: E402
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, sha256_file, write_json  # noqa: E402
from train_repr import augmentation, build_backbone, embed_split, lr_factor  # noqa: E402

log = logging.getLogger("train_george")


class AugmentedImages(torch.utils.data.Dataset):
    """One augmented view of each image, with its make_id and its row index (to look up the cluster)."""

    def __init__(self, base, transform):
        self.base, self.transform = base, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, make_id = self.base.items[idx]
        return self.transform(Image.open(path).convert("RGB")), make_id, idx


def fit(cfg, ids, meta, derived, make, group_of, device, seed, val):
    """Fine-tune backbone + linear head on the make. group_of=None: ERM; else group DRO over the groups group_of[row]."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, trainable = build_backbone(cfg, device)
    head = torch.nn.Linear(model(torch.zeros(1, 3, model.input_size, model.input_size, device=device)).shape[1], int(make.max()) + 1).to(device)
    net = torch.nn.Sequential(model, head)
    variant, size = cfg["crop"].split("_")
    ds = AugmentedImages(CarsCoarseDataset(meta, derived, ids, variant, int(size), augment="none"), augmentation(model, cfg))
    dl = torch.utils.data.DataLoader(ds, cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["num_workers"],
                                     persistent_workers=True)
    if group_of is None:
        loss_fn = lambda logits, y, g: torch.nn.functional.cross_entropy(logits, y)  # noqa: E731
    else:
        counts = torch.bincount(torch.from_numpy(group_of)).float()
        dro = LossComputer(torch.nn.CrossEntropyLoss(reduction="none"), True, len(counts), counts, cfg["george"]["robust_lr"],
                           size_adjustments=[cfg["george"]["size_adjustment"]] * len(counts))
        group_of = torch.from_numpy(group_of).to(device)
        loss_fn = lambda logits, y, g: dro(logits, y, group_of[g])[0]  # noqa: E731
    opt = torch.optim.AdamW([{"params": trainable, "lr": cfg["lr_backbone"]}, {"params": head.parameters(), "lr": cfg["lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    total, warmup = cfg["epochs"] * len(dl), cfg["warmup_epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warmup, total))
    scaler = torch.amp.GradScaler()
    for epoch in range(cfg["epochs"]):
        losses = []
        for x, y, g in dl:
            x, y, g = x.to(device), y.to(device), g.to(device)
            loss = loss_fn(forward_fp16(net, x), y, g)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        msg = f"epoch {epoch + 1}/{cfg['epochs']} loss {np.mean(losses):.4f}"
        if (epoch + 1) % cfg["eval_every"] == 0 or epoch + 1 == cfg["epochs"]:
            xtr, xva = (embed_split(model, derived, cfg["crop"], i, cfg, device) for i in (ids, val[0]))
            msg += f" | val kNN@20 make acc {knn_accuracy(xtr, make, xva, val[1]):.4f}"
        log.info(msg)
    return model, float(np.mean(losses))


def george_clusters(activations, make, cfg, seed):
    """GEORGE stage 2: per make, UMAP then GMM with K by silhouette. Returns globally unique cluster ids (N,) and K per make."""
    g, assign, floor, ks = cfg["george"], np.zeros(len(make), int), 0, {}
    for m in np.unique(make):
        rows = np.flatnonzero(make == m)
        x = UMAPReducer(n_components=g["umap_components"], random_state=seed).fit(activations[rows]).transform(activations[rows])
        k_max = min(g["max_k"], len(rows) - 1)
        gmm = AutoKMixtureModel("gmm", k_max, seed=seed, search=True).fit(x)
        assign[rows], ks[int(m)] = floor + gmm.predict(x), gmm.n_clusters
        floor += gmm.n_clusters
        log.info("make %d: %d images, K = %d", m, len(rows), gmm.n_clusters)
    return assign, ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/repr_george.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("training needs a GPU: sbatch slurm/train_george.sh")
    device = torch.device("cuda")
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    meta = load_labels(derived, dcfg["data_root"], cfg.get("labels", "make"))[["make_id"]].reset_index()  # coarse labels only
    make_of = meta.set_index("id").make_id
    ids = {s: load_split(splits_dir, s) for s in ("train", "val", "test")}
    log.info("images per split after excluding conflicting ones: %s", {s: len(v) for s, v in ids.items()})
    make, val = make_of[ids["train"]].to_numpy(), (ids["val"], make_of[ids["val"]].to_numpy())
    size = cfg["crop"].split("_")[-1]

    tag = cfg.get("tag", "")
    for seed in cfg["seeds"]:
        erm, erm_loss = fit(cfg, ids["train"], meta, derived, make, None, device, seed, val)
        assign, ks = george_clusters(embed_split(erm, derived, cfg["crop"], ids["train"], cfg, device), make, cfg, seed)
        (derived / "george_clusters").mkdir(exist_ok=True)
        pd.DataFrame({"id": ids["train"], "make_id": make, "cluster": assign}).to_csv(derived / "george_clusters" / f"george{tag}_s{seed}.csv", index=False)
        gdro, gdro_loss = fit(cfg, ids["train"], meta, derived, make, assign, device, seed, val)
        for run, model, loss in ((f"george_erm{tag}_s{seed}", erm, erm_loss), (f"george{tag}_s{seed}", gdro, gdro_loss)):
            for split, split_ids in ids.items():
                emb = embed_split(model, derived, cfg["crop"], split_ids, cfg, device)
                out = derived / "features" / f"{run}_{size}"
                out.mkdir(parents=True, exist_ok=True)
                write_csv(out / f"{split}.csv", [f"{i}.png" for i in split_ids], emb)
                write_json(out / f"{split}.json", {
                    **provenance(cfg, seed, dataset_hash), "encoder": run, "family": model.family, "backbone": cfg["backbone"],
                    "method": cfg["george"], "k_per_make": ks, "final_train_loss": loss, "crop": cfg["crop"], "split": split,
                    "n": len(split_ids), "dim": emb.shape[1], "labels": f"coarse only ({cfg.get('labels', 'make')} scheme)",
                    "exclude_sha256": sha256_file(splits_dir / "exclude.txt")})
                log.info("%s %s: %s", run, split, emb.shape)


if __name__ == "__main__":
    main()
