"""Figures for a generated set (use case A; descriptive, not metrics), written next to its scores by scripts/eval_generation.py:
samples.png       samples grouped by requested coarse class;
nearest_real.png  random samples next to their nearest train image in CLIP-L space, captioned with the fine evaluator's prediction for the
                  sample and the true fine class of the neighbour;
embedding.png     UMAP / t-SNE (CLIP-L) of the generated images placed among the real reference images, and of the generated set alone."""
import matplotlib
import numpy as np
from PIL import Image

from lsgen.eval.projection import project

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREY = "#c8c8c8"


def _img(path, size):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC))


def sample_grid(paths, coarse, coarse_names, out, per_class=8, size=96):
    """One row per requested coarse class: its first `per_class` samples."""
    classes = np.unique(coarse)
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.1, len(classes) * 1.15), squeeze=False)
    for r, c in enumerate(classes):
        idx = np.flatnonzero(coarse == c)[:per_class]
        for j, ax in enumerate(axes[r]):
            ax.set_axis_off()
            if j < len(idx):
                ax.imshow(_img(paths[idx[j]], size))
        axes[r, 0].text(-0.08, 0.5, coarse_names[c], transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=7)
    left = min(0.35, 0.03 + 0.009 * max(len(coarse_names[c]) for c in classes))  # room for the row labels
    fig.subplots_adjust(left=left, right=0.99, top=0.99, bottom=0.01, wspace=0.03, hspace=0.05)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def nearest_real_grid(paths, real_paths, nn_idx, sim, sample_captions, real_captions, out, n=24, pairs=4, seed=0, size=128):
    """n random samples (seeded) | nearest real image, `pairs` pairs per row, with captions and the cosine similarity."""
    pick = np.sort(np.random.default_rng(seed).choice(len(paths), min(n, len(paths)), replace=False))
    rows = -(-len(pick) // pairs)
    fig, axes = plt.subplots(rows, 2 * pairs, figsize=(2 * pairs * 1.6, rows * 1.9), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for k, i in enumerate(pick):
        a, b = axes[k // pairs, 2 * (k % pairs)], axes[k // pairs, 2 * (k % pairs) + 1]
        a.imshow(_img(paths[i], size))
        a.set_title(f"gen: {sample_captions[i]}", fontsize=5.5)
        b.imshow(_img(real_paths[nn_idx[i]], size))
        b.set_title(f"real: {real_captions[nn_idx[i]]} ({sim[i]:.2f})", fontsize=5.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def embedding_map(real, real_coarse, gen, gen_coarse, coarse_names, cfg, out, seed=0):
    """Rows UMAP / t-SNE. Columns: real images by coarse class; real (grey) with the generated images on top, by requested coarse class
    (one joint projection); the generated set alone (its own projection). Both sets are subsampled to cfg["max_points"]."""
    rng = np.random.default_rng(seed)
    sub = lambda n: np.sort(rng.choice(n, min(n, cfg["max_points"]), replace=False))  # noqa: E731
    ri, gi = sub(len(real)), sub(len(gen))
    real, real_coarse, gen, gen_coarse = real[ri], real_coarse[ri], gen[gi], gen_coarse[gi]
    pcfg = {**cfg, "seed": seed}
    joint, alone = project(np.concatenate([real, gen]), pcfg), project(gen, pcfg)
    classes = np.unique(np.concatenate([real_coarse, gen_coarse]))
    colour = {c: plt.cm.tab20(k % 20) for k, c in enumerate(classes)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for r, method in enumerate(("UMAP", "t-SNE")):
        zr, zg = joint[method][:len(real)], joint[method][len(real):]
        for c in classes:
            axes[r, 0].scatter(*zr[real_coarse == c].T, s=2, color=colour[c])
            axes[r, 1].scatter(*zg[gen_coarse == c].T, s=3, color=colour[c], zorder=2)
            axes[r, 2].scatter(*alone[method][gen_coarse == c].T, s=3, color=colour[c])
        axes[r, 1].scatter(*zr.T, s=2, color=GREY, zorder=1)
        titles = (f"real reference ({len(real)})", f"generated ({len(gen)}) on real (grey)", f"generated alone ({len(gen)})")
        for ax, t in zip(axes[r], titles):
            ax.set(title=f"{method}: {t}", xticks=[], yticks=[])
    handles = [plt.Line2D([], [], marker="o", ls="", color=colour[c], label=coarse_names[c]) for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(classes), 10), fontsize=8, markerscale=1.5)
    fig.tight_layout(rect=(0, 0.05 if len(classes) <= 10 else 0.08, 1, 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)
