"""Summary of the ksub runs on test (scripts/train_ksub.py, then R1-R5 by scripts/eval_representation.py with configs/eval_repr_ksub_*.yaml):
one table run x {kNN coarse, ACC oracle K, NMI, d0/d1, d1/d2, var ratio, ARI boot, ACC own}, mean ± std over seeds, and a per-class table.
ACC own = Hungarian accuracy of the model's own final assignments (argmax_g r_ig) per class with K_c >= 2, macro mean. A method with no loss
term ("frozen") is the backbone itself: its R1-R5 do not depend on the seed (std 0), its own assignment is the nearest k-means centroid. CPU, seconds:
    python scripts/summarize_ksub.py --config configs/ksub_cifar100.yaml [--methods frozen sub_only sub_var]
Output: <reports_root>/<report_dir>/{summary.md, summary.csv, per_class.csv} (per_class: means over seeds).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from lsgen.data.labels import coarse_classes, load_labels
from lsgen.eval.report import exclusion_note, macro_row, per_make_table
from lsgen.eval.representation import hungarian_acc
from lsgen.utils import REPO_ROOT, git_hash, load_config

COLUMNS = {"knn_make": "kNN coarse", "acc_oracle": "ACC oracle K", "nmi_oracle": "NMI oracle K", "d0/d1": "d0/d1", "d1/d2": "d1/d2",
           "var_ratio": "var ratio", "ari_boot": "ARI boot", "acc_own": "ACC own"}


def own_accuracy(clusters, fine):
    """Series (index make_id): Hungarian ACC of the test assignments against the hidden subclasses, classes with K_c >= 2 on test only."""
    t = clusters[clusters.split == "test"].assign(fine=lambda d: fine.loc[d.id].to_numpy())
    return t.groupby("make_id").filter(lambda d: d.fine.nunique() >= 2).groupby("make_id")[["fine", "cluster"]].apply(lambda d: hungarian_acc(d.fine, d.cluster))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ksub_cifar100.yaml")
    ap.add_argument("--methods", nargs="*", help="subset of the methods in the config (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["methods"] = {m: w for m, w in cfg["methods"].items() if not args.methods or m in args.methods}
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    reports, size = Path(dcfg["reports_root"]) / cfg["report_dir"], cfg["crop"].split("_")[-1]
    fine = load_labels(derived, dcfg["data_root"], cfg["labels"]).fine_id
    rows, per_class, git = [], [], {"train": set(), "eval": set()}
    for method, weights in cfg["methods"].items():
        for seed in cfg["seeds"]:
            run = f"ksub_{method}{cfg.get('tag', '')}_s{seed}"
            repr_dir = reports / (f"{run}_{size}" if weights else f"{cfg['backbone']}_{size}")  # no loss term: z = f
            own = own_accuracy(pd.read_csv(derived / "ksub_clusters" / f"{run}.csv"), fine)
            macro = macro_row(repr_dir)
            rows.append({"method": method, "seed": seed, **{k: macro[k] for k in COLUMNS if k in macro}, "acc_own": own.mean()})
            per_class.append(per_make_table(repr_dir).assign(acc_own=own, method=method, seed=seed))
            git["eval"].add(json.loads((repr_dir / "summary.json").read_text())["git_hash"])
            if weights:
                git["train"].add(json.loads((derived / "features" / f"{run}_{size}" / "test.json").read_text())["git_hash"])

    runs = pd.DataFrame(rows)
    runs.to_csv(reports / "summary.csv", index=False)
    stats = runs.groupby("method", sort=False)[list(COLUMNS)].agg(["mean", "std"])
    table = pd.DataFrame({name: [f"{stats.at[m, (k, 'mean')]:.3f} ± {stats.at[m, (k, 'std')]:.3f}" for m in stats.index]
                          for k, name in COLUMNS.items()}, index=stats.index).rename_axis("run")

    names = coarse_classes(dcfg["data_root"], cfg["labels"])
    per_class = pd.concat(per_class).reset_index().groupby(["method", "make_id"], sort=False).mean(numeric_only=True).drop(columns="seed")
    per_class.join(names, on="make_id").to_csv(reports / "per_class.csv")
    wide = per_class[["acc_oracle", "acc_own"]].unstack("method")
    wide.columns = [f"{m} {'ACC oracle K' if k == 'acc_oracle' else 'ACC own'}" for k, m in wide.columns]
    wide = names.join(per_class.xs(next(iter(cfg["methods"])), level="method")[["n_test"]]).join(wide)

    removed = exclusion_note(splits_dir, labels=cfg["labels"])["excluded_conflicting_images"]
    manifest = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    (reports / "summary.md").write_text(f"""\
# ksub (known-K subclass baseline): {cfg['labels']}, frozen {cfg['backbone']}, test

**Uses K**: every run is given the true number of subclasses K_c per coarse class (fine labels are otherwise used for evaluation only).
Provenance: summary written at git `{git_hash()}` from `{cfg['_config_path']}`; training git {sorted(git['train'])}, R1-R5 git {sorted(git['eval'])};
seeds {cfg['seeds']}; dataset manifest `{manifest}`. **{sum(removed.values())} conflicting images ({removed}) are excluded from every split**
(the `{cfg['labels']}` exclusion list, configs/labels.yaml). Per-run R1-R5 (with CIs): `<run>_{size}/`; per seed: `summary.csv`.

Mean ± std over seeds of macro means over coarse classes. kNN coarse = R1 kNN@20; ACC / NMI oracle K = R2 k-means with K_c on z (per-class
PCA-50); d0/d1, d1/d2 = R3; var ratio = R4; ARI boot = R5; ACC own = Hungarian ACC of the model's own assignment argmax_g r_ig (frozen:
nearest k-means centroid of the train features). Classes with K_c = 1 are left out of the clustering columns.

{table.to_markdown()}

## Per class (mean over seeds; all metrics: `per_class.csv`)

{wide.round(3).reset_index(drop=True).to_markdown(index=False)}
""")


if __name__ == "__main__":
    main()
