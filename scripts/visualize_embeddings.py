"""UMAP and t-SNE of the test embeddings of every encoder and representation method, coloured by coarse class and by subclass,
for each label scheme in configs/visualize.yaml. CPU:
    sbatch slurm/visualize_embeddings.sh
Output: reports/visualisations/<scheme>/<encoder>.png (global panel coloured by coarse class + one subclass panel per coarse class
in `within`, rows UMAP / t-SNE) and overview_{umap,tsne}.png (all encoders, coloured by coarse class). Encoders in a scheme's
`clusterings` get extra rows: the same projections coloured by that clustering's label-free clusters (R2 pipeline, lsgen/eval/clustering.py).
"""
import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lsgen.data.labels import load_labels  # noqa: E402
from lsgen.data.splits import load_split  # noqa: E402
from lsgen.eval.clustering import fitter  # noqa: E402
from lsgen.eval.projection import cached_projection  # noqa: E402
from lsgen.eval.representation import cluster_space  # noqa: E402
from lsgen.features.extract import read_csv  # noqa: E402
from lsgen.utils import REPO_ROOT, load_config, setup_logging  # noqa: E402

log = logging.getLogger("visualize_embeddings")
GREY = "lightgrey"


def coarse_colors(names, top):
    """{coarse class: colour} for the `top` largest classes (tab10); every other class is grey."""
    return {c: plt.cm.tab10(i) for i, c in enumerate(names.value_counts().index[:top])}


def draw_coarse(ax, z, names, colors, title):
    other = ~names.isin(colors).to_numpy()
    ax.scatter(*z[other].T, s=2, c=GREY)
    for c, col in colors.items():
        ax.scatter(*z[(names == c).to_numpy()].T, s=3, color=col)
    ax.set(title=title, xticks=[], yticks=[])


def draw_subclass(ax, z, names, title):
    """One colour per subclass (tab20); returns legend handles."""
    cat = pd.Categorical(names)
    ax.scatter(*z.T, s=6, c=cat.codes, cmap="tab20", vmin=0, vmax=19)
    ax.set(title=title, xticks=[], yticks=[])
    return [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab20(i % 20), label=n) for i, n in enumerate(cat.categories)]


def figure_encoder(enc, scheme, glob, names, within, colors, n, path, method=None):
    """rows = UMAP / t-SNE (first the same coloured by `method`'s clusters, if given); columns = all test images by coarse class, then the
    subclasses inside each `within` coarse class."""
    algs = list(glob)
    rows = ([(alg, True) for alg in algs] if method else []) + [(alg, False) for alg in algs]
    fig, axes = plt.subplots(len(rows), 1 + len(within), figsize=(4.2 * (1 + len(within)), 4.3 * len(rows)), squeeze=False)
    for r, (alg, pred) in enumerate(rows):
        if pred:
            axes[r, 0].axis("off")
        else:
            draw_coarse(axes[r, 0], glob[alg], names, colors, f"{alg}: coarse class")
        for c, (name, (z, sub, clusters)) in enumerate(within.items(), 1):
            if pred:
                draw_subclass(axes[r, c], z[alg], clusters, f"{alg}: {name}, {method} clusters (K={len(np.unique(clusters))})")
                continue
            handles = draw_subclass(axes[r, c], z[alg], sub, f"{alg}: {name} subclasses (n={len(z[alg])})")
            if r == len(rows) - 1:
                axes[r, c].legend(handles=handles, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    axes[len(rows) - len(algs), 0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=col, label=c) for c, col in colors.items()],
                      fontsize=6, loc="upper left", markerscale=0.8)
    fig.suptitle(f"{enc} - {scheme} labels - test, n={n}, conflicting images excluded", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_overview(projections, names, colors, alg, scheme, path, ncols=5):
    encs = list(projections)
    nrows = -(-len(encs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, enc in zip(axes.flat, encs):
        ax.axis("on")
        draw_coarse(ax, projections[enc][alg], names, colors, enc)
        ax.title.set_fontsize(8)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=col, label=c) for c, col in colors.items()],
               loc="lower center", ncol=len(colors), fontsize=8)
    fig.suptitle(f"{alg}, {scheme} labels, coloured by coarse class (test, conflicting images excluded)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/visualize.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    derived, out_root = Path(dcfg["derived_root"]), Path(dcfg["reports_root"]) / "visualisations"
    for scheme, scfg in cfg["schemes"].items():
        test_ids = load_split(Path(dcfg["data_root"]) / "splits", "test", labels=scheme)
        labels = load_labels(derived, dcfg["data_root"], scheme)
        out = out_root / scheme
        out.mkdir(parents=True, exist_ok=True)
        overview, names, colors = {}, None, None
        fits = {enc: (c["algorithm"], fitter(**c)) for enc, c in scfg.get("clusterings", {}).items()}
        for enc in cfg["frozen"] + scfg["methods"]:
            path = derived / "features" / enc / "test.csv"
            if not path.exists():
                log.warning("%s: no embeddings at %s, skipped (run the stage that creates them)", enc, path)
                continue
            files, x = read_csv(path)
            ids = np.array([f[:-4] for f in files])
            keep = np.isin(ids, test_ids)  # cached embeddings may hold images a scheme excludes on top of exclude.txt
            assert keep.sum() == len(test_ids), f"{enc}: test embeddings do not cover the test split"
            x, ids = x[keep], ids[keep]
            lab = labels.loc[ids]
            glob = cached_projection(derived / "projections" / f"{enc}_{'' if keep.all() else scheme + '_'}global.npz", x, cfg)
            names, colors = lab.coarse_name, coarse_colors(lab.coarse_name, scfg["top_coarse"])
            within = {}
            for coarse in scfg["within"]:
                s = (lab.coarse_name == coarse).to_numpy()
                sub = lab.fine_name[s].str.removeprefix(coarse + " ")
                clusters = fits[enc][1](cluster_space(x[s], np.zeros(s.sum()), seed=cfg["seed"]), None, cfg["seed"]) if enc in fits else None
                within[coarse] = (cached_projection(derived / "projections" / f"{enc}_{scheme}_{coarse}.npz", x[s], cfg), sub, clusters)
            figure_encoder(enc, scheme, glob, names, within, colors, len(ids), out / f"{enc}.png", fits.get(enc, (None,))[0])
            overview[enc] = glob
            log.info("%s %s: figure written", scheme, enc)
        for alg, fname in (("UMAP", "overview_umap.png"), ("t-SNE", "overview_tsne.png")):
            figure_overview(overview, names, colors, alg, scheme, out / fname)
        log.info("%s: %d encoders, wrote %s", scheme, len(overview), out)


if __name__ == "__main__":
    main()
