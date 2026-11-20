"""Does geosub's training learn anything beyond its initialisation? For each trained run (scripts/train_geosub.py) and split, compares the
trained model with its own starting point, rebuilt from the run's recorded config and seed: the initial prototypes on the frozen features,
pruned by the same min_count rule (no training). Diagnostic only (train and val; never test, never used to choose). Per run, split:
    acc_trained / acc_init       Hungarian ACC per coarse class, macro: trained model vs initial prototypes on frozen f
    acc_protos_on_f              trained prototypes applied to frozen f (what the prototypes learned, without the head)
    changed, ari_init_trained    share of images whose prototype changed during training, and ARI between the two partitions
    khat_trained / khat_init     prototypes active on train
    proto_cos                    mean cosine between each active prototype and its initial position
    drift                        1 - mean cos(z, f)
    knn_purity_z / knn_purity_f  share of the k nearest neighbours (same coarse class) with the same fine class, in z and in frozen f
Plus t-SNE of one coarse class (frozen f vs z; coloured by fine class and by prototype) for the runs in `tsne`. CPU:
    sbatch slurm/diagnose_geosub.sh
Output: <reports_root>/<report_dir>/geosub_diagnostics/{runs.csv,summary.csv,summary.json,tsne_<class>_<run>.png}
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from lsgen.data.labels import coarse_classes, load_labels
from lsgen.data.splits import load_split
from lsgen.eval.representation import hungarian_acc
from lsgen.methods.geosub.model import GeoSub, active_prototypes, init_prototypes
from lsgen.methods.repr.ksub.model import init_prototypes as kmeans_prototypes
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, write_json
from train_geosub import load_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("diagnose_geosub")
SPLITS = ("train", "val")


def assign(protos, active, x, coarse):
    """(N,) nearest active prototype of the image's class (cosine)."""
    cos = torch.einsum("bd,bpd->bp", x, F.normalize(protos[coarse], dim=2))
    return cos.masked_fill(~active[coarse], float("-inf")).argmax(1).numpy()


def per_class(fn, coarse, *xs):
    """Mean over coarse classes of fn(x[class], ...)."""
    return float(np.mean([fn(*(x[coarse == c] for x in xs)) for c in np.unique(coarse)]))


def knn_purity(x, coarse, fine, k):
    """Mean share of each image's k nearest neighbours (same coarse class, itself excluded) that have its fine class."""
    out = []
    for c in np.unique(coarse):
        xc, yc = x[coarse == c], fine[coarse == c]
        s = xc @ xc.T
        np.fill_diagonal(s, -np.inf)
        out.append((yc[np.argsort(-s, 1)[:, :k]] == yc[:, None]).mean(1))
    return float(np.concatenate(out).mean())


def initial_prototypes(rcfg, f, coarse, k, seed):
    """The run's starting prototypes, as in scripts/train_geosub.py: k-means centroids or random train features of each class."""
    if rcfg.get("proto_init", "random") == "kmeans":
        return kmeans_prototypes(f, coarse, k, seed)
    return init_prototypes(f, coarse, len(k), max(k), torch.Generator().manual_seed(seed))


def tsne_plot(path, title, f, z, fine, proto, seed):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    for i, (name, x) in enumerate((("frozen f", f), ("geosub z", z))):
        e = TSNE(2, init="pca", metric="cosine", random_state=seed).fit_transform(x)
        for j, (colour, label) in enumerate(((fine, "fine class"), (proto, "prototype"))):
            ax[i, j].scatter(e[:, 0], e[:, 1], c=pd.factorize(colour)[0] % 20, cmap="tab20", s=5)
            ax[i, j].set(title=f"{title}: {name}, coloured by {label}", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/diagnose_geosub_maskcon.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    torch.set_num_threads(cfg["n_jobs"])
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    out = Path(dcfg["reports_root"]) / cfg["report_dir"] / "geosub_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    meta = load_labels(derived, dcfg["data_root"], cfg["labels"])
    ids = {s: load_split(splits_dir, s, labels=cfg["labels"]) for s in SPLITS}
    f = {s: load_features(derived / "features" / cfg["backbone"] / f"{s}.csv", ids[s]) for s in SPLITS}
    coarse = {s: torch.from_numpy(meta.make_id[ids[s]].to_numpy()) for s in SPLITS}
    fine = {s: meta.fine_id[ids[s]].to_numpy() for s in SPLITS}
    n_coarse, size = len(coarse_classes(dcfg["data_root"], cfg["labels"])), cfg["backbone"].split("_")[-1]
    k_c = coarse_classes(dcfg["data_root"], cfg["labels"]).K_c.tolist()
    purity_f = {s: knn_purity(f[s].numpy(), coarse[s].numpy(), fine[s], cfg["knn_k"]) for s in SPLITS}

    rows, inits = [], {}
    for name, run in cfg["runs"].items():
        for seed in cfg["seeds"]:
            info = json.loads((derived / "features" / f"{run}_s{seed}_{size}" / "train.json").read_text())
            rcfg, known_k = info["config"], info["method"].get("known_k", False)
            k = k_c if known_k else [rcfg["n_proto"]] * n_coarse
            key = (rcfg.get("proto_init", "random"), tuple(k), seed)
            if key not in inits:
                inits[key] = initial_prototypes(rcfg, f["train"], coarse["train"], k, seed)
            p0 = inits[key]
            state = torch.load(derived / "repr_models" / f"{run}_s{seed}.pt", map_location="cpu")
            model = GeoSub(p0.shape[2], rcfg["hidden"], state["prototypes"], n_valid=k)
            # runs saved before the `valid` buffer existed used all P prototypes of every class
            model.load_state_dict({"valid": model.valid, **{key_: v for key_, v in state.items() if key_ != "active"}})
            active = state["active"]
            a0_train = assign(p0, model.valid, f["train"], coarse["train"])
            active0 = model.valid if known_k else active_prototypes(torch.from_numpy(a0_train), coarse["train"], n_coarse, p0.shape[1],
                                                                    rcfg["min_count"])
            proto_cos = F.cosine_similarity(F.normalize(state["prototypes"], dim=2), F.normalize(p0, dim=2), dim=2)[active].mean()
            for s in SPLITS:
                with torch.no_grad():
                    z = model(f[s])
                c = coarse[s].numpy()
                a, a0, a_f = assign(state["prototypes"], active, z, coarse[s]), assign(p0, active0, f[s], coarse[s]), \
                    assign(state["prototypes"], active, f[s], coarse[s])
                rows.append({"method": name, "seed": seed, "split": s,
                             "acc_trained": per_class(hungarian_acc, c, fine[s], a), "acc_init": per_class(hungarian_acc, c, fine[s], a0),
                             "acc_protos_on_f": per_class(hungarian_acc, c, fine[s], a_f),
                             "changed": float((a != a0).mean()), "ari_init_trained": per_class(adjusted_rand_score, c, a0, a),
                             "khat_trained": active.sum(1).float().mean().item(), "khat_init": active0.sum(1).float().mean().item(),
                             "proto_cos": proto_cos.item(), "drift": 1 - (z * f[s]).sum(1).mean().item(),
                             "knn_purity_z": knn_purity(z.numpy(), c, fine[s], cfg["knn_k"]), "knn_purity_f": purity_f[s]})
                if s == "train" and seed == cfg["seeds"][0]:
                    for cls in cfg["tsne"].get(name, []):
                        g = meta.coarse_name[ids[s]].to_numpy() == cls
                        tsne_plot(out / f"tsne_{cls}_{name}.png", f"{cls}, {name} s{seed}", f[s].numpy()[g], z.numpy()[g], fine[s][g],
                                  a[g], seed)
            log.info("%s s%d: %s", name, seed, {k_: round(v, 3) for k_, v in rows[-2].items() if isinstance(v, float)})

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    summary = runs.drop(columns="seed").groupby(["split", "method"], sort=False).agg(["mean", "std"]).round(4)
    summary.to_csv(out / "summary.csv")
    write_json(out / "summary.json", {**provenance(cfg), "exclude": "conflicting images dropped by load_split"})
    log.info("\n%s", summary.xs("mean", axis=1, level=1).to_string())


if __name__ == "__main__":
    main()
