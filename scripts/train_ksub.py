"""ksub (lsgen/methods/repr/ksub): known-K subclass baseline on cached frozen features f. Residual head z = norm(f + g(f)) (geosub's) and
K_c non-learnable prototypes per coarse class (k-means on frozen train features, then EMA); L = L_coarse (SupCon) + lambda_sub L_sub
(soft within-class contrastive) + lambda_var L_var (within-subclass variance floor). Coarse labels plus K_c per class (coarse_classes):
every run "uses K". Settings in configs/ksub_*.yaml; a run with no loss term ("frozen") is the k-means initialisation itself. Needs a GPU:
    CONFIG=configs/ksub_cifar100.yaml sbatch slurm/train_ksub.sh
Input: <derived_root>/features/<backbone>_<size>/{train,val,test}.csv (scripts/extract_features.py). No augmentation: f is the clean crop.
Output per method and seed (run = ksub_<method><tag>_s<seed>):
    <derived_root>/features/<run>_<size>/{train,val,test}.csv (+ .json)  z, scored by scripts/eval_representation.py (not for "frozen": z = f)
    <derived_root>/ksub_clusters/<run>.csv   per image: split, coarse class, final assignment argmax_g r_ig (scripts/summarize_ksub.py)
    <derived_root>/repr_models/<run>.pt      head and prototypes
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lsgen.data.labels import coarse_classes, load_labels, scheme
from lsgen.data.splits import load_split
from lsgen.eval.representation import knn_accuracy
from lsgen.features.extract import write_csv
from lsgen.methods.repr.ksub.losses import ksub_loss
from lsgen.methods.repr.ksub.model import ema_update, head, init_prototypes, soft_assign
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, sha256_file, write_json
from train_geosub import load_features
from train_repr import lr_factor

log = logging.getLogger("train_ksub")


def class_batch(rows_of, n_classes, per_class, gen):
    """Indices of per_class random images (without replacement) from each of n_classes random coarse classes; rows_of[c]: rows of class c."""
    classes = torch.randperm(len(rows_of), generator=gen)[:n_classes]
    return torch.cat([rows_of[c][torch.randperm(len(rows_of[c]), generator=gen)[:per_class]] for c in classes])


def train(cfg, weights, seed, f, coarse, k_c, device, val=None):
    """Head and prototypes from cached train features f (N, D), coarse (N,) and k_c (n_coarse,). No weights: no training (k-means only).
    val = (features, coarse labels) for the monitoring kNN only."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    mu = init_prototypes(f, coarse, k_c.tolist(), seed).to(device)
    net = head(f.shape[1], cfg["hidden"]).to(device)
    if not weights:
        return net.eval(), mu, float("nan")
    rows_of = [torch.nonzero(coarse == c).squeeze(1) for c in range(len(k_c))]
    f, coarse, k_c = f.to(device), coarse.to(device), k_c.to(device)
    opt = torch.optim.AdamW(net.g.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = len(coarse) // (cfg["classes_per_batch"] * cfg["per_class"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg["warmup_epochs"] * steps, cfg["epochs"] * steps))
    for epoch in range(cfg["epochs"]):
        logs = []
        for _ in range(steps):
            idx = class_batch(rows_of, cfg["classes_per_batch"], cfg["per_class"], gen).to(device)
            z, c = net(f[idx]), coarse[idx]
            r = soft_assign(z.detach(), c, mu, k_c, cfg["tau_a"])  # constant: no gradient through r or mu
            loss, terms = ksub_loss(z, f[idx], c, r, weights, cfg["tau"], cfg["temperature"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema_update(mu, z.detach(), c, r, cfg["momentum"])
            logs.append({"loss": loss.item(), **{k: v.item() for k, v in terms.items()}})
        msg = f"epoch {epoch + 1}/{cfg['epochs']} " + " ".join(f"{k} {np.mean([t[k] for t in logs]):.4f}" for k in logs[0])
        if val is not None and ((epoch + 1) % cfg["eval_every"] == 0 or epoch + 1 == cfg["epochs"]):
            with torch.no_grad():
                xtr, xva = net(f).cpu().numpy(), net(val[0].to(device)).cpu().numpy()
            msg += f" | val kNN@20 coarse acc {knn_accuracy(xtr, coarse.cpu().numpy(), xva, val[1]):.4f}"
        log.info(msg)
    return net.eval(), mu, float(np.mean([t["loss"] for t in logs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ksub_cifar100.yaml")
    ap.add_argument("--methods", nargs="*", help="subset of the methods in the config (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("training needs a GPU: sbatch slurm/train_ksub.sh")
    device = torch.device("cuda")
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    coarse_of = load_labels(derived, dcfg["data_root"], cfg["labels"]).make_id  # coarse labels only
    k_c = torch.tensor(coarse_classes(dcfg["data_root"], cfg["labels"]).K_c.to_numpy())  # uses K: the only fine-label information
    ids = {s: load_split(splits_dir, s, labels=cfg["labels"]) for s in ("train", "val", "test")}
    log.info("images per split after excluding conflicting ones: %s; K_c %s", {s: len(v) for s, v in ids.items()}, k_c.tolist())
    exclude_sha = sha256_file(splits_dir / scheme(cfg["labels"])["exclude"])

    size = cfg["crop"].split("_")[-1]
    feats = {s: load_features(derived / "features" / f"{cfg['backbone']}_{size}" / f"{s}.csv", ids[s]) for s in ids}
    coarse = {s: torch.from_numpy(coarse_of[ids[s]].to_numpy()) for s in ids}

    for method, weights in cfg["methods"].items():
        if args.methods and method not in args.methods:
            continue
        for seed in cfg["seeds"]:
            run = f"ksub_{method}{cfg.get('tag', '')}_s{seed}"
            log.info("%s: weights %s, seed %d", run, weights, seed)
            net, mu, final_loss = train(cfg, weights, seed, feats["train"], coarse["train"], k_c, device,
                                        val=(feats["val"], coarse["val"].numpy()))
            (derived / "repr_models").mkdir(exist_ok=True)
            torch.save({"head": net.state_dict(), "prototypes": mu.cpu(), "k_c": k_c}, derived / "repr_models" / f"{run}.pt")
            clusters = []
            for split, split_ids in ids.items():
                with torch.no_grad():
                    z, c = net(feats[split].to(device)), coarse[split].to(device)
                    clusters.append(pd.DataFrame({"id": split_ids, "split": split, "make_id": c.cpu().numpy(),
                                                  "cluster": soft_assign(z, c, mu, k_c.to(device)).argmax(1).cpu().numpy()}))
                if not weights:
                    continue  # frozen: z = f, scored as the backbone itself
                out = derived / "features" / f"{run}_{size}"
                out.mkdir(parents=True, exist_ok=True)
                write_csv(out / f"{split}.csv", [f"{i}.png" for i in split_ids], z.cpu().numpy())
                write_json(out / f"{split}.json", {
                    **provenance(cfg, seed, dataset_hash), "encoder": run, "family": "ksub", "backbone": cfg["backbone"],
                    "method": weights, "final_train_loss": final_loss, "crop": cfg["crop"], "split": split, "n": len(split_ids),
                    "dim": z.shape[1], "labels": f"coarse ({cfg['labels']} scheme) + K_c per class: uses K", "exclude_sha256": exclude_sha})
                log.info("%s %s: %s", run, split, tuple(z.shape))
            (derived / "ksub_clusters").mkdir(exist_ok=True)
            pd.concat(clusters).to_csv(derived / "ksub_clusters" / f"{run}.csv", index=False)


if __name__ == "__main__":
    main()
