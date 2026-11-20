"""ksub (lsgen/methods/repr/ksub): known-K subclass baseline on cached frozen features f. Residual head z = norm(f + g(f)) (geosub's) and
K_c non-learnable prototypes per coarse class (k-means on frozen train features, then EMA); L = L_coarse (SupCon) + lambda_sub L_sub
(soft within-class contrastive) + lambda_var L_var (within-subclass variance floor). Coarse labels plus K_c per class: the true K_c
(coarse_classes, `k: oracle`, the default: "uses K"), or a label-free K-hat estimated per seed on the frozen train features (`k: khat`, estimate_k
with `k_max`; fixed for the whole run). Prototypes (initial and re-clustered) come from k-means, or from the algorithm of a `clustering`
entry (e.g. diffusion maps, lsgen/eval/clustering.py). A method may set `var_floor` (L_var floor, default 1), and its own `momentum`,
`recluster_every` and `tau_a` (default: the config's). Settings in configs/ksub_*.yaml; a run with no loss term ("frozen") is the k-means initialisation itself. Needs a GPU:
    CONFIG=configs/ksub_cifar100.yaml sbatch slurm/train_ksub.sh
Input: <derived_root>/features/<backbone>_<size>/{train,val,test}.csv (scripts/extract_features.py). No augmentation: f is the clean crop.
Output per method and seed (run = ksub_<method><tag>_s<seed>):
    <derived_root>/features/<run>_<size>/{train,val,test}.csv (+ .json)  z, scored by scripts/eval_representation.py (not for "frozen": z = f)
    <derived_root>/ksub_clusters/<run>.csv   per image: split, coarse class, final assignment argmax_g r_ig (scripts/summarize_ksub.py)
    <derived_root>/repr_models/<run>.pt      head, prototypes and the K_c used
    <derived_root>/ksub_ari/<run>.csv        with ari_every: per epoch and coarse class, train ARI of the current assignments vs fresh / initial k-means,
                                             and the mean assignment entropy / log K_c (all before that epoch's refresh)
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
from lsgen.eval.clustering import fitter
from lsgen.eval.representation import knn_accuracy
from lsgen.features.extract import write_csv
from lsgen.methods.repr.ksub.losses import ksub_loss
from lsgen.methods.repr.ksub.model import assignment_ari, assignment_entropy, ema_update, estimate_k, head, init_prototypes, soft_assign
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
    val = (features, coarse labels) for the monitoring kNN only. Also returns the per-class assignment ARI rows (ari_every; else empty)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    fit = fitter(**cfg["clustering"]) if "clustering" in cfg else None  # None: k-means
    mu = init_prototypes(f, coarse, k_c.tolist(), seed, fit).to(device)
    mu0 = mu.clone()  # initial k-means, for the optional assignment ARI (ari_every)
    net = head(f.shape[1], cfg["hidden"]).to(device)
    if not weights:
        return net.eval(), mu, float("nan"), []
    momentum, refresh = weights.get("momentum", cfg["momentum"]), weights.get("recluster_every", cfg.get("recluster_every"))
    tau_a = weights.get("tau_a", cfg["tau_a"])
    ari_rows = []
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
            r = soft_assign(z.detach(), c, mu, k_c, tau_a)  # constant: no gradient through r or mu
            loss, terms = ksub_loss(z, f[idx], c, r, weights, cfg["tau"], cfg["temperature"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema_update(mu, z.detach(), c, r, momentum)
            logs.append({"loss": loss.item(), **{k: v.item() for k, v in terms.items()}})
        msg = f"epoch {epoch + 1}/{cfg['epochs']} " + " ".join(f"{k} {np.mean([t[k] for t in logs]):.4f}" for k in logs[0])
        if val is not None and ((epoch + 1) % cfg["eval_every"] == 0 or epoch + 1 == cfg["epochs"]):
            with torch.no_grad():
                xtr, xva = net(f).cpu().numpy(), net(val[0].to(device)).cpu().numpy()
            msg += f" | val kNN@20 coarse acc {knn_accuracy(xtr, coarse.cpu().numpy(), xva, val[1]):.4f}"
        if cfg.get("ari_every") and (epoch + 1) % cfg["ari_every"] == 0:  # diagnostic only, on train: nothing feeds back into training
            with torch.no_grad():
                z_all = net(f)
                ari_fresh, ari_init = assignment_ari(z_all, f, coarse, mu, mu0, k_c, seed)  # before this epoch's refresh, if any
                entropy = assignment_entropy(z_all, coarse, mu, k_c, tau_a)
            ari_rows += [{"epoch": epoch + 1, "make_id": k, "ari_fresh": a, "ari_init": b, "entropy": h}
                         for k, (a, b, h) in enumerate(zip(ari_fresh, ari_init, entropy))]
            msg += (f" | ARI current vs fresh k-means on z {np.nanmean(ari_fresh):.4f}, vs initial k-means on f {np.nanmean(ari_init):.4f}"
                    f", entropy / log K_c {np.nanmean(entropy):.4f}")
        log.info(msg)
        if refresh and (epoch + 1) % refresh == 0:  # variant: prototypes reset to a fresh clustering of z
            with torch.no_grad():
                mu = init_prototypes(net(f).cpu(), coarse.cpu(), k_c.tolist(), seed, fit).to(device)
    return net.eval(), mu, float(np.mean([t["loss"] for t in logs])), ari_rows


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
    classes = coarse_classes(dcfg["data_root"], cfg["labels"])
    ids = {s: load_split(splits_dir, s, labels=cfg["labels"]) for s in ("train", "val", "test")}
    log.info("images per split after excluding conflicting ones: %s", {s: len(v) for s, v in ids.items()})
    exclude_sha = sha256_file(splits_dir / scheme(cfg["labels"])["exclude"])

    size = cfg["crop"].split("_")[-1]
    feats = {s: load_features(derived / "features" / f"{cfg['backbone']}_{size}" / f"{s}.csv", ids[s]) for s in ids}
    coarse = {s: torch.from_numpy(coarse_of[ids[s]].to_numpy()) for s in ids}
    if cfg.get("k", "oracle") == "oracle":  # uses K: the only fine-label information
        k_of_seed = {seed: torch.tensor(classes.K_c.to_numpy()) for seed in cfg["seeds"]}
        k_note = "K_c per class: uses K"
    else:  # label-free: coarse labels only
        k_of_seed = {seed: estimate_k(feats["train"], coarse["train"], len(classes), cfg["k_max"], seed) for seed in cfg["seeds"]}
        k_note = f"label-free K-hat per class (estimate_k on frozen train features, k_max {cfg['k_max']})"
    for seed, k in k_of_seed.items():
        log.info("seed %d: K_c %s", seed, k.tolist())

    for method, weights in cfg["methods"].items():
        if args.methods and method not in args.methods:
            continue
        for seed in cfg["seeds"]:
            run, k_c = f"ksub_{method}{cfg.get('tag', '')}_s{seed}", k_of_seed[seed]
            log.info("%s: weights %s, seed %d", run, weights, seed)
            net, mu, final_loss, ari_rows = train(cfg, weights, seed, feats["train"], coarse["train"], k_c, device,
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
                    "dim": z.shape[1], "labels": f"coarse ({cfg['labels']} scheme) + {k_note}", "k_c": k_c.tolist(), "exclude_sha256": exclude_sha})
                log.info("%s %s: %s", run, split, tuple(z.shape))
            (derived / "ksub_clusters").mkdir(exist_ok=True)
            pd.concat(clusters).to_csv(derived / "ksub_clusters" / f"{run}.csv", index=False)
            if ari_rows:
                (derived / "ksub_ari").mkdir(exist_ok=True)
                pd.DataFrame(ari_rows).to_csv(derived / "ksub_ari" / f"{run}.csv", index=False)


if __name__ == "__main__":
    main()
