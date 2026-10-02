"""Analysis of the cached embeddings: kNN accuracy (R1), hierarchy geometry (R3), evaluator classifier curves, and
UMAP / t-SNE of the embedding space. Reference = train, queries = test (descriptive only). CPU:
    sbatch slurm/analyze_features.sh
Output: reports/features.md, reports/features/*.csv|png.
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from lsgen.eval.representation import deltas_per_make, knn_accuracy, l2norm  # noqa: E402
from lsgen.features.extract import read_csv  # noqa: E402
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging  # noqa: E402

log = logging.getLogger("analyze_features")


def load(derived, enc, split, meta):
    files, x = read_csv(derived / "features" / enc / f"{split}.csv")
    return x, meta.loc[[f[:-4] for f in files]]


def project(x, cfg):
    x = l2norm(x)
    z_umap = umap.UMAP(n_neighbors=cfg["umap_neighbors"], metric="cosine", random_state=cfg["seed"]).fit_transform(x)
    xp = PCA(min(50, x.shape[1]), random_state=cfg["seed"]).fit_transform(x)
    z_tsne = TSNE(perplexity=cfg["tsne_perplexity"], init="pca", random_state=cfg["seed"], n_jobs=cfg["n_jobs"]).fit_transform(xp)
    return {"UMAP": z_umap, "t-SNE": z_tsne}


def metrics(cfg, derived, meta):
    rows, r3 = [], []
    for enc in cfg["encoders"]:
        for size in cfg["crop_sizes"]:
            name = f"{enc}_{size}"
            (xtr, mtr), (xte, mte) = load(derived, name, "train", meta), load(derived, name, "test", meta)
            d = deltas_per_make(xte, mte.make_id.to_numpy(), mte.fine_id.to_numpy())
            d.insert(1, "make", d.make_id.map(mte.drop_duplicates("make_id").set_index("make_id").make))
            multi = d[d.K_c >= 2]
            rows.append({"encoder": enc, "crop": size, "dim": xte.shape[1],
                         "knn_make": knn_accuracy(xtr, mtr.make_id, xte, mte.make_id, cfg["knn_k"]),
                         "knn_fine": knn_accuracy(xtr, mtr.fine_id, xte, mte.fine_id, cfg["knn_k"]),
                         "d0": d.d0.mean(), "d1": multi.d1.mean(), "d2": d.d2.mean(),
                         "median_d0/d1": (multi.d0 / multi.d1).median(), "median_d1/d2": (multi.d1 / multi.d2).median(),
                         "%makes_d0<d1": 100 * (multi.d0 < multi.d1).mean(), "%makes_d1<d2": 100 * (multi.d1 < multi.d2).mean()})
            r3.append(d.assign(encoder=enc, crop=size))
            log.info("%s: %s", name, {k: round(float(v), 3) for k, v in rows[-1].items() if k not in ("encoder", "crop", "dim")})
    return pd.DataFrame(rows), pd.concat(r3)


def fig_knn(m, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in zip(axes, ["knn_make", "knn_fine"], ["make (49)", "fine class (196)"]):
        p = m.pivot(index="encoder", columns="crop", values=col).loc[m.encoder.unique()]
        p.plot.bar(ax=ax, rot=20, width=0.8)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=7)
        ax.set(title=f"kNN@20 accuracy, {title}", xlabel="", ylim=(0, 1.08))
        ax.legend(title="crop px")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_r3(r3, path):
    r3 = r3[r3.K_c >= 2].assign(name=lambda d: d.encoder + "_" + d.crop.astype(str), r01=lambda d: d.d0 / d.d1, r12=lambda d: d.d1 / d.d2)
    names = list(dict.fromkeys(r3.name))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes, ["r01", "r12"], ["d0 / d1 (same subclass vs same make)", "d1 / d2 (same make vs other makes)"]):
        ax.boxplot([r3.loc[r3.name == n, col] for n in names], tick_labels=names, showfliers=True)
        ax.axhline(1, color="r", ls="--", lw=1)
        ax.set_title(title, fontsize=10); ax.set_ylabel("ratio per make with K_c >= 2 (<1 desired)")
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_per_make(r3, enc, path):
    d = r3[(r3.encoder + "_" + r3.crop.astype(str) == enc) & (r3.K_c >= 2)].copy()
    d["ratio"] = d.d0 / d.d1
    d = d.sort_values("ratio")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(d.K_c, d.ratio, s=25)
    for _, r in d.iterrows():
        ax.annotate(r.make, (r.K_c, r.ratio), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.axhline(1, color="r", ls="--", lw=1)
    ax.set(xlabel="K_c (fine classes in the make)", ylabel="d0 / d1", title=f"{enc}: subclass separability per make (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_evaluator(reports, path):
    ev = {k: json.loads((reports / f"evaluator_{k}.json").read_text()) for k in ("fine", "make")}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, d in ev.items():
        axes[0].plot([h["val_acc"] for h in d["history"]], label=f"{k} (test {d['test_top1']:.3f})")
    axes[0].set(title="evaluator val accuracy", xlabel="epoch", ylabel="top-1")
    axes[0].legend()
    for ax, a, b, title in [(axes[1], "test_ece_uncalibrated", "test_ece_calibrated", "test ECE"),
                            (axes[2], "test_nll_uncalibrated", "test_nll_calibrated", "test NLL")]:
        x = np.arange(2)
        ax.bar(x - 0.2, [ev[k][a] for k in ev], 0.4, label="before scaling")
        ax.bar(x + 0.2, [ev[k][b] for k in ev], 0.4, label="after scaling")
        ax.set_xticks(x, [f"{k}\nT={ev[k]['temperature']:.2f}" for k in ev])
        ax.set(title=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_global(enc_proj, makes, top, path):
    """rows = encoders, cols = UMAP / t-SNE; the `top` largest makes coloured, the rest grey."""
    keep = makes.make.value_counts().index[:top]
    fig, axes = plt.subplots(len(enc_proj), 2, figsize=(11, 4.6 * len(enc_proj)), squeeze=False)
    for r, (enc, proj) in enumerate(enc_proj.items()):
        for c, (alg, z) in enumerate(proj.items()):
            ax = axes[r, c]
            other = ~makes.make.isin(keep).to_numpy()
            ax.scatter(*z[other].T, s=2, c="lightgrey", label="other makes")
            for i, mk in enumerate(keep):
                s = (makes.make == mk).to_numpy()
                ax.scatter(*z[s].T, s=3, color=plt.cm.tab10(i), label=mk)
            ax.set(title=f"{enc}: {alg} (test, n={len(z)})", xticks=[], yticks=[])
            if r == 0 and c == 1:
                ax.legend(markerscale=4, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_within(enc, rows, path):
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 5 * len(rows)), squeeze=False)
    for r, (make, labels, proj) in enumerate(rows):
        codes = pd.Categorical(labels).codes
        for c, (alg, z) in enumerate(proj.items()):
            ax = axes[r, c]
            ax.scatter(*z.T, s=6, c=codes, cmap="tab20")
            ax.set(title=f"{enc}: {make}, {alg} (test, n={len(z)}, K_c={codes.max() + 1})", xticks=[], yticks=[])
        handles = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab20(i / max(codes.max(), 1)), label=n)
                   for i, n in enumerate(pd.Categorical(labels).categories)]
        axes[r, 1].legend(handles=handles, fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/analysis.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    derived, reports = Path(dcfg["derived_root"]), Path(dcfg["reports_root"])
    out = reports / "features"
    out.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(Path(dcfg["data_root"]) / "hierarchy.csv").set_index("fine_id")
    meta = pd.read_parquet(derived / "metadata.parquet", columns=["id", "make_id", "fine_id"]).set_index("id")
    meta["make"] = h.loc[meta.fine_id, "make"].to_numpy()
    meta["label"] = (h.loc[meta.fine_id, "model"] + " " + h.loc[meta.fine_id, "year"].astype(str)).to_numpy()

    m, r3 = metrics(cfg, derived, meta)
    m.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    r3.to_csv(out / "r3_per_make.csv", index=False, float_format="%.4f")
    fig_knn(m, out / "knn.png")
    fig_r3(r3, out / "r3_ratios.png")
    fig_per_make(r3, cfg["projection_encoders"][0], out / "r3_per_make.png")
    fig_evaluator(reports, out / "evaluator.png")

    enc_proj = {}
    for enc in cfg["projection_encoders"]:
        x, mt = load(derived, enc, "test", meta)
        enc_proj[enc] = project(x, cfg)
        log.info("projected %s", enc)
    fig_global(enc_proj, mt, cfg["top_makes"], out / "projection_global.png")  # mt: same test rows for every encoder

    for enc in cfg["within_make_encoders"]:
        x, mt = load(derived, enc, "test", meta)
        rows = []
        for make in cfg["within_makes"]:
            s = (mt.make == make).to_numpy()
            rows.append((make, mt.label[s].to_numpy(), project(x[s], cfg)))
            log.info("projected %s %s (n=%d)", enc, make, s.sum())
        fig_within(enc, rows, out / f"projection_within_make_{enc}.png")

    prov = provenance(cfg, cfg["seed"], json.loads((Path(dcfg["data_root"]) / "manifests" / "raw_manifest.json").read_text())["manifest_hash"])
    n_ex = len((Path(dcfg["data_root"]) / "splits" / "exclude.txt").read_text().split())
    lines = ["# Feature analysis (Milestone 3)", "", "<!-- generated by scripts/analyze_features.py; do not edit -->",
             f"Provenance: git `{prov['git_hash']}`, seed {cfg['seed']}, dataset manifest `{prov['dataset_manifest_hash'][:16]}`, "
             f"created {prov['created_utc']}", "",
             f"kNN@{cfg['knn_k']} uses train as reference and test as queries; d0/d1/d2 are on test (R3, docs/PLAN.md); "
             f"{n_ex} conflicting images are excluded everywhere. Descriptive only: nothing here is used for tuning.", "",
             "## Representation metrics", "", m.to_markdown(index=False, floatfmt=".3f"), "",
             "![kNN](features/knn.png)", "", "![R3 ratios](features/r3_ratios.png)", "",
             "![R3 per make](features/r3_per_make.png)", "", "## Evaluator classifiers", "", "![evaluator](features/evaluator.png)", "",
             "## Embedding space (test images)", "", "![global](features/projection_global.png)", ""]
    lines += [f"![{e}](features/projection_within_make_{e}.png)\n" for e in cfg["within_make_encoders"]]
    (reports / "features.md").write_text("\n".join(lines))
    log.info("wrote %s", reports / "features.md")


if __name__ == "__main__":
    main()
