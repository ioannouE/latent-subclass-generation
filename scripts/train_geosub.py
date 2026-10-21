"""geosub (lsgen/methods/geosub): residual head z = norm(f + g(f)) and subprototypes per coarse class on frozen backbone features f,
trained with L_geo (distil the frozen similarities), L_supcon (coarse classes), L_sub (augmentation-consistent prototype assignment)
and L_var (within-class variance floor). Coarse labels only (configs/geosub_maskcon.yaml: `labels` picks the scheme). Needs a GPU:
    sbatch slurm/train_geosub.sh
The backbone is frozen, so its features are cached once (skipped when present) and every run trains on them in minutes:
    <derived_root>/features/<backbone>_<size>/{train,val,test}.csv        clean crops (scripts/extract_features.py, must exist)
    <derived_root>/features/<backbone>_<size>_views/view<v>.csv          `views` random augmentations of the train crops
    <derived_root>/features/<backbone>_<size>_flip/{train,val,test}.csv  mirror images (as scripts/gate0_nuisance.py)
Output per method and seed (run = geosub_<method><tag>_s<seed>):
    <derived_root>/features/<run>_<size>/{train,val,test}.csv (+ .json)  z of the clean crops, scored by scripts/eval_representation.py
    <derived_root>/geosub_clusters/<run>.csv   per image: split, coarse class, prototype assignment of the crop and of its mirror image
                                               (among the prototypes active on train), scored by scripts/eval_geosub.py
    <derived_root>/repr_models/<run>.pt        head and prototypes
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as TF

from lsgen.data.labels import load_labels, scheme
from lsgen.data.splits import load_split
from lsgen.eval.representation import knn_accuracy, l2norm
from lsgen.features.encoders import Embedder
from lsgen.features.extract import embed_images, read_csv, write_csv
from lsgen.methods.geosub.losses import geosub_loss
from lsgen.methods.geosub.model import GeoSub, active_prototypes, init_prototypes
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, sha256_file, write_json
from train_repr import augmentation, lr_factor

log = logging.getLogger("train_geosub")


def load_features(path, ids):
    """(N, D) float32 tensor of L2-normalised features, rows in the order of ids."""
    files, x = read_csv(path)
    return torch.from_numpy(l2norm(pd.DataFrame(x, index=[f[:-4] for f in files]).loc[ids].to_numpy()).astype(np.float32))


def cache_features(cfg, derived, ids, device, manifest):
    """Embeds the augmented train views and the mirror images of every split with the frozen backbone, if not cached yet.
    manifest: fields recorded in the .json next to every cached file (provenance, exclusion list)."""
    name = f"{cfg['backbone']}_{cfg['crop'].split('_')[-1]}"
    if not (derived / "features" / name / "train.csv").exists():
        raise FileNotFoundError(f"frozen features {name} missing: run slurm/extract_features.sh first")
    todo = {derived / "features" / f"{name}_views" / f"view{v}.csv": ("train", v) for v in range(cfg["views"])}
    todo |= {derived / "features" / f"{name}_flip" / f"{s}.csv": (s, None) for s in ids}
    todo = {p: t for p, t in todo.items() if not p.exists()}
    if not todo:
        return
    model = Embedder(cfg["backbone"]).to(device)
    aug = augmentation(model, cfg)
    flipped = lambda im: model.transform(TF.hflip(im))  # noqa: E731
    for path, (split, v) in todo.items():
        if v is not None:
            torch.manual_seed(v)  # reproducible views (this also seeds the DataLoader workers)
        emb = embed_images(model, [derived / cfg["crop"] / f"{i}.png" for i in ids[split]], flipped if v is None else aug, device,
                           cfg["batch_size_embed"], cfg["num_workers"])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, [f"{i}.png" for i in ids[split]], emb)
        write_json(path.with_suffix(".json"), {**manifest, "encoder": cfg["backbone"], "crop": cfg["crop"], "split": split,
                                              "view": "mirror" if v is None else v, "n": len(ids[split]), "dim": emb.shape[1]})
        log.info("cached %s: %s", path, emb.shape)


def train(cfg, weights, seed, views, f_train, coarse, device, val=None):
    """Train the head and prototypes on cached features. views: (V, N, D) augmented train features, f_train (N, D) clean ones,
    coarse (N,). Each image gets two different random views per step. val = (features, coarse labels) for the monitoring kNN only."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n_coarse, n_proto = int(coarse.max()) + 1, cfg["n_proto"]
    model = GeoSub(views.shape[2], cfg["hidden"], init_prototypes(f_train, coarse, n_coarse, n_proto, gen)).to(device)
    views, f_train, coarse = views.to(device), f_train.to(device), coarse.to(device)
    opt = torch.optim.AdamW(model.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = len(coarse) // cfg["batch_size"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg["warmup_epochs"] * steps, cfg["epochs"] * steps))
    args = {k: cfg[k] for k in ("tau_geo", "tau_student", "tau_teacher", "usage", "temperature")}
    for epoch in range(cfg["epochs"]):
        perm, logs = torch.randperm(len(coarse), device=device), []
        for i in range(steps):
            idx = perm[i * cfg["batch_size"]:(i + 1) * cfg["batch_size"]]
            v1 = torch.randint(len(views), (len(idx),), device=device)
            v2 = (v1 + torch.randint(1, len(views), (len(idx),), device=device)) % len(views)
            loss, terms = geosub_loss(model, views[v1, idx], views[v2, idx], coarse[idx], weights, **args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            logs.append({"loss": loss.item(), **{k: v.item() for k, v in terms.items()}})
        with torch.no_grad():
            active = active_prototypes(model.assign(model(f_train), coarse), coarse, n_coarse, n_proto, cfg["min_count"])
        msg = " ".join(f"{k} {np.mean([t[k] for t in logs]):.4f}" for k in logs[0])
        msg = f"epoch {epoch + 1}/{cfg['epochs']} {msg} | K-hat/class {active.sum(1).float().mean():.2f} (max {int(active.sum(1).max())})"
        if val is not None and ((epoch + 1) % cfg["eval_every"] == 0 or epoch + 1 == cfg["epochs"]):
            with torch.no_grad():
                xtr, xva = model(f_train).cpu().numpy(), model(val[0].to(device)).cpu().numpy()
            msg += f" | val kNN@20 coarse acc {knn_accuracy(xtr, coarse.cpu().numpy(), xva, val[1]):.4f}"
        log.info(msg)
    return model.eval(), active, float(np.mean([t["loss"] for t in logs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/geosub_maskcon.yaml")
    ap.add_argument("--methods", nargs="*", help="subset of the methods in the config (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("training needs a GPU: sbatch slurm/train_geosub.sh")
    device = torch.device("cuda")
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    coarse_of = load_labels(derived, dcfg["data_root"], cfg["labels"]).make_id  # coarse labels only
    ids = {s: load_split(splits_dir, s, labels=cfg["labels"]) for s in ("train", "val", "test")}
    log.info("images per split after excluding conflicting ones: %s", {s: len(v) for s, v in ids.items()})
    exclude_sha = sha256_file(splits_dir / scheme(cfg["labels"])["exclude"])
    cache_features(cfg, derived, ids, device, {**provenance(cfg, None, dataset_hash), "exclude_sha256": exclude_sha})

    size = cfg["crop"].split("_")[-1]
    name = f"{cfg['backbone']}_{size}"
    feats = {s: load_features(derived / "features" / name / f"{s}.csv", ids[s]) for s in ids}
    flips = {s: load_features(derived / "features" / f"{name}_flip" / f"{s}.csv", ids[s]) for s in ids}
    views = torch.stack([load_features(derived / "features" / f"{name}_views" / f"view{v}.csv", ids["train"]) for v in range(cfg["views"])])
    coarse = {s: torch.from_numpy(coarse_of[ids[s]].to_numpy()) for s in ids}

    for method, weights in cfg["methods"].items():
        if args.methods and method not in args.methods:
            continue
        for seed in cfg["seeds"]:
            run = f"geosub_{method}{cfg.get('tag', '')}_s{seed}"
            log.info("%s: weights %s, seed %d", run, weights, seed)
            model, active, final_loss = train(cfg, weights, seed, views, feats["train"], coarse["train"], device,
                                              val=(feats["val"], coarse["val"].numpy()))
            (derived / "repr_models").mkdir(exist_ok=True)
            torch.save({**model.state_dict(), "active": active.cpu()}, derived / "repr_models" / f"{run}.pt")
            clusters = []
            for split, split_ids in ids.items():
                with torch.no_grad():
                    z, z_flip, c = model(feats[split].to(device)), model(flips[split].to(device)), coarse[split].to(device)
                    clusters.append(pd.DataFrame({"id": split_ids, "split": split, "make_id": c.cpu().numpy(),
                                                  "cluster": model.assign(z, c, active).cpu().numpy(),
                                                  "cluster_flip": model.assign(z_flip, c, active).cpu().numpy()}))
                out = derived / "features" / f"{run}_{size}"
                out.mkdir(parents=True, exist_ok=True)
                write_csv(out / f"{split}.csv", [f"{i}.png" for i in split_ids], z.cpu().numpy())
                write_json(out / f"{split}.json", {
                    **provenance(cfg, seed, dataset_hash), "encoder": run, "family": "geosub", "backbone": cfg["backbone"],
                    "method": weights, "final_train_loss": final_loss, "crop": cfg["crop"], "split": split, "n": len(split_ids),
                    "dim": z.shape[1], "khat": active.sum(1).tolist(), "labels": f"coarse only ({cfg['labels']} scheme)",
                    "exclude_sha256": exclude_sha})
                log.info("%s %s: %s", run, split, tuple(z.shape))
            (derived / "geosub_clusters").mkdir(exist_ok=True)
            pd.concat(clusters).to_csv(derived / "geosub_clusters" / f"{run}.csv", index=False)


if __name__ == "__main__":
    main()
