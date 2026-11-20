"""Per coarse class and per subclass comparison of ksub runs on test (after scripts/summarize_ksub.py for each run's config).
Per coarse class: the per_class.csv metrics of each run (means over seeds). Per subclass: the share of its test images in the cluster
Hungarian-matched to it (0 if it gets no cluster: fewer clusters than subclasses), and whether its largest cluster is also the largest cluster
of another subclass (merged), from each run's own assignment (<derived_root>/ksub_clusters/<run>.csv), mean over seeds. CPU, seconds:
    python scripts/compare_ksub_classes.py --config configs/ksub_compare_maskcon_new.yaml
Output: <reports_root>/<out_dir>/{classes.csv, subclasses.csv, summary.md}.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from lsgen.data.labels import load_labels
from lsgen.eval.report import exclusion_note
from lsgen.utils import REPO_ROOT, git_hash, load_config

METRICS = {"acc_oracle": "ACC oracle K", "acc_own": "ACC own", "k": "K", "recall1": "R@1 within", "d1/d2": "d1/d2", "ari_boot": "ARI boot"}


def subclass_recall(fine, cluster):
    """DataFrame per subclass: n, matched (share in its Hungarian-matched cluster, 0 if unmatched), merged (its largest cluster is also the
    largest cluster of another subclass)."""
    y, yi = np.unique(fine, return_inverse=True)
    _, ci = np.unique(cluster, return_inverse=True)
    m = np.zeros((len(y), ci.max() + 1), int)
    np.add.at(m, (yi, ci), 1)
    n, matched = m.sum(1), np.zeros(len(y))
    r, c = linear_sum_assignment(-m)
    matched[r] = m[r, c] / n[r]
    top = m.argmax(1)
    return pd.DataFrame({"fine_id": y, "n": n, "matched": matched, "merged": [(top == t).sum() > 1 for t in top]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ksub_compare_maskcon_new.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    derived, out = Path(dcfg["derived_root"]), Path(dcfg["reports_root"]) / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(derived, dcfg["data_root"], cfg["labels"])
    classes, subclasses = [], []
    for name, (config, method) in cfg["runs"].items():
        rcfg = load_config(REPO_ROOT / config)
        pc = pd.read_csv(Path(dcfg["reports_root"]) / rcfg["report_dir"] / "per_class.csv").query("method == @method")
        classes.append(pc.set_index(["make_id", "coarse_name", "K_c"])[list(METRICS)].assign(run=name))
        for seed in rcfg["seeds"]:
            t = pd.read_csv(derived / "ksub_clusters" / f"ksub_{method}{rcfg.get('tag', '')}_s{seed}.csv").query("split == 'test'")
            t["fine_id"] = labels.fine_id.loc[t.id].to_numpy()
            for c, d in t.groupby("make_id"):
                subclasses.append(subclass_recall(d.fine_id, d.cluster).assign(run=name, seed=seed, make_id=c))

    classes = pd.concat(classes).reset_index()
    classes.to_csv(out / "classes.csv", index=False)
    sub = pd.concat(subclasses).groupby(["run", "make_id", "fine_id"], sort=False)[["n", "matched", "merged"]].mean().reset_index()
    sub = sub.join(labels.drop_duplicates("fine_id").set_index("fine_id")[["coarse_name", "fine_name"]], on="fine_id")
    sub.to_csv(out / "subclasses.csv", index=False)

    runs = list(cfg["runs"])
    sections = []
    for key, title in METRICS.items():
        wide = classes.pivot_table(index=["coarse_name", "K_c"], columns="run", values=key, sort=False)[runs]
        sections.append(f"### {title}\n\n{wide.round(3).reset_index().to_markdown(index=False)}")
    counts = sub.assign(failed=sub.matched < 0.5).groupby(["coarse_name", "run"], sort=False).agg(
        subclasses=("matched", "size"), failed=("failed", "sum"), merged=("merged", "sum"), matched=("matched", "mean"))
    failed = counts.failed.unstack("run")[runs].round(1)
    merged = counts.merged.unstack("run")[runs].round(1)
    worst = sub.query("run == @cfg['worst_run']").nsmallest(cfg["n_worst"], "matched")[["coarse_name", "fine_name", "n", "matched", "merged"]]
    removed = exclusion_note(Path(dcfg["data_root"]) / "splits", labels=cfg["labels"])["excluded_conflicting_images"]
    (out / "summary.md").write_text(f"""\
# ksub per class and per subclass: {cfg['labels']}, test

Written at git `{git_hash()}` from `{cfg['_config_path']}`. **{sum(removed.values())} conflicting images ({removed}) are excluded from every
split.** Runs (config, method): {', '.join(f"`{n}` = {c[1]} in `{c[0]}`" for n, c in cfg['runs'].items())}. Means over seeds.
ACC oracle K = k-means with the true K_c on z (evaluation only); ACC own and K = the run's own clustering and its number of clusters;
R@1 within = within-class fine Recall@1. Per subclass: matched = share of its test images in its Hungarian-matched own cluster (0 if
unmatched); failed = matched < 0.5; merged = its largest cluster is also the largest of another subclass (counts are means over seeds).

## Per coarse class

{chr(10).join(sections)}

## Subclasses per coarse class: failed (matched < 0.5)

{failed.reset_index().to_markdown(index=False)}

## Subclasses per coarse class: merged

{merged.reset_index().to_markdown(index=False)}

## {cfg['n_worst']} worst subclasses of `{cfg['worst_run']}` (all subclasses: `subclasses.csv`)

{worst.round(3).to_markdown(index=False)}
""")


if __name__ == "__main__":
    main()
