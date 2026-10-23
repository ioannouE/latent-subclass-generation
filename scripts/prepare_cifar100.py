"""CIFAR-100 benchmark, Milestone 1: verify the official files, write PNGs, hierarchy, splits, duplicate audit, metadata, report
(docs/DATA_CIFAR100.md). Run on a GPU node (the `duplicates` stage needs SSCD on a GPU; the others are CPU):
    sbatch slurm/prepare_cifar100.sh
    python scripts/prepare_cifar100.py --config configs/data_cifar100.yaml [--stages raw splits duplicates finalize report]
Each stage reads the previous stages' outputs from disk, so stages can be re-run individually.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from lsgen.data import duplicates as dup
from lsgen.data.cifar100 import FILES_MD5, N_IMAGES, build_hierarchy, load_names, load_official, md5_file
from lsgen.data.crops import png_bytes
from lsgen.data.labels import scheme
from lsgen.data.reports import image_grid
from lsgen.data.splits import load_split, make_splits, save_splits
from lsgen.utils import load_config, manifest_hash, provenance, setup_logging, sha256_bytes, sha256_file, write_json

log = logging.getLogger("prepare_cifar100")
STAGES = ("raw", "splits", "duplicates", "finalize", "report")
SPLITS = ("train", "val", "test")


class Paths:
    def __init__(self, cfg):
        self.raw, self.derived, self.data, self.reports = (Path(cfg[k]) for k in ("raw_root", "derived_root", "data_root", "reports_root"))
        self.images = self.derived / "native_32"
        self.raw_index = self.derived / "raw_index.parquet"
        self.metadata = self.derived / "metadata.parquet"
        self.dup_dir = self.derived / "duplicates"
        self.manifests = self.data / "manifests"
        self.raw_manifest = self.manifests / "raw_manifest.json"
        self.splits = self.data / "splits"
        self.hierarchy = self.data / "hierarchy.csv"
        self.duplicates_all = self.data / "duplicates_all.csv"
        self.exclude = self.splits / scheme(cfg["labels"])["exclude"]


def dataset_hash(P):
    return json.loads(P.raw_manifest.read_text())["manifest_hash"]


def stage_raw(cfg, P):
    """Checks the official files (md5, counts, 500 / 100 images per class), writes hierarchy.csv, the raw manifest, one lossless PNG per
    image (native 32 px, no resizing) and raw_index.parquet; re-encodes a random subset to check the PNGs are deterministic."""
    for name, md5 in FILES_MD5.items():
        if md5_file(P.raw / "cifar-100-python" / name) != md5:
            raise ValueError(f"md5 of {name} differs from the official file")
    fine_names, coarse_names = load_names(P.raw)
    rows, images = [], {}
    for split in ("train", "test"):
        x, fine, coarse = load_official(P.raw, split)
        per_class = np.bincount(fine, minlength=100)
        if len(x) != N_IMAGES[split] or (per_class != N_IMAGES[split] // 100).any():
            raise ValueError(f"{split}: {len(x)} images, per-class counts {sorted(set(per_class))}")
        for k in range(len(x)):
            rows.append({"id": f"{split}_{k:05d}", "official_split": split, "row": k, "class_id": int(fine[k]), "make_id": int(coarse[k]),
                         "pixel_sha256": sha256_bytes(x[k].tobytes())})
        images[split] = x
    idx = pd.DataFrame(rows)
    h = build_hierarchy(idx.class_id, idx.make_id, fine_names, coarse_names)
    P.data.mkdir(parents=True, exist_ok=True)
    h.to_csv(P.hierarchy, index=False)
    files = {name: sha256_file(P.raw / "cifar-100-python" / name) for name in FILES_MD5}
    mh = manifest_hash(files)
    write_json(P.raw_manifest, {**provenance(cfg, dataset_manifest_hash=mh), "manifest_hash": mh, "raw_root": str(P.raw), "files": files})

    P.images.mkdir(parents=True, exist_ok=True)
    shas = {}
    for r in idx.itertuples():
        b = png_bytes(Image.fromarray(images[r.official_split][r.row]))
        (P.images / f"{r.id}.png").write_bytes(b)
        shas[f"native_32/{r.id}.png"] = sha256_bytes(b)
    sub = idx.iloc[np.random.default_rng(cfg["seed"]).choice(len(idx), cfg["determinism_check_n"], replace=False)]
    bad = [r.id for r in sub.itertuples() if sha256_bytes(png_bytes(Image.fromarray(images[r.official_split][r.row]))) != shas[f"native_32/{r.id}.png"]
           or not np.array_equal(np.asarray(Image.open(P.images / f"{r.id}.png")), images[r.official_split][r.row])]
    if bad:
        raise RuntimeError(f"PNGs not deterministic or not lossless: {bad[:10]}")
    write_json(P.derived / "png_shas.json", shas)
    idx.to_parquet(P.raw_index, index=False)
    log.info("raw: %d images (%s), %d superclasses x 5 classes; %d PNGs written, %d re-encoded identically; manifest hash %s",
             len(idx), idx.official_split.value_counts().to_dict(), h.make_id.nunique(), len(shas), len(sub), mh[:16])


def stage_splits(cfg, P):
    idx = pd.read_parquet(P.raw_index, columns=["id", "official_split", "class_id"])
    s = make_splits(idx, cfg["val_frac"], cfg["seed"])
    save_splits(s, P.splits)
    write_json(P.manifests / "splits.json", {**provenance(cfg, cfg["seed"], dataset_hash(P)), "counts": {k: len(v) for k, v in s.items()},
                                             "sha256": {k: sha256_file(P.splits / f"{k}.txt") for k in s}})
    log.info("splits: %s", {k: len(v) for k, v in s.items()})


def stage_duplicates(cfg, P):
    """SSCD embeddings of all 60k PNGs (GPU), then every pair of images (any splits) that is pixel-identical or has SSCD cosine >=
    audit_min_sim, with both labels. pHash is not used (as for maskcon_new: alone it flags different images with the same layout)."""
    d = cfg["duplicates"]
    idx = pd.read_parquet(P.raw_index)
    P.dup_dir.mkdir(parents=True, exist_ok=True)
    emb_file = P.dup_dir / "sscd_embeddings.npy"
    model, wpath, wsha = dup.get_sscd(d["sscd_weights_dir"], d["sscd_url"])
    if wsha != d["sscd_sha256"]:
        raise ValueError(f"SSCD weights sha256 {wsha} != config {d['sscd_sha256']}")
    if not emb_file.exists():
        device = dup.resolve_device(d["device"])
        np.save(emb_file, dup.sscd_embed(model, [str(P.images / f"{i}.png") for i in idx.id], d["sscd_resize"], cfg["num_workers"],
                                         d["torch_threads"], device))
    else:
        log.info("using cached SSCD embeddings %s (delete it to recompute)", emb_file)
    pairs = dup.all_pairs_table(idx.id.tolist(), np.load(emb_file), idx.pixel_sha256.to_numpy(), idx.class_id.to_numpy(), d["audit_min_sim"])
    by_id = idx.set_index("id")
    pairs["superclass_conflict"] = by_id.make_id[pairs.id_a].to_numpy() != by_id.make_id[pairs.id_b].to_numpy()
    pairs["splits"] = by_id.official_split[pairs.id_a].to_numpy() + "-" + by_id.official_split[pairs.id_b].to_numpy()
    pairs.sort_values("sscd_sim", ascending=False).to_csv(P.duplicates_all, index=False, float_format="%.4f")
    write_json(P.manifests / "duplicates.json", {**provenance(cfg, cfg["seed"], dataset_hash(P)), "n_pairs": len(pairs),
               "n_exact": int(pairs.exact_duplicate.sum()), "sscd_weights": str(wpath), "sscd_weights_sha256": wsha,
               "duplicates_all_csv_sha256": sha256_file(P.duplicates_all)})
    log.info("pairs with SSCD >= %.2f or pixel-identical: %d (pixel-identical %d, different class %d)", d["audit_min_sim"], len(pairs),
             pairs.exact_duplicate.sum(), pairs.label_conflict.sum())


def conflict_pairs(pairs, threshold):
    """Pairs that count as the same image (pixel-identical, or SSCD cosine >= threshold)."""
    return pairs[pairs.exact_duplicate | (pairs.sscd_sim >= threshold)]


def stage_finalize(cfg, P):
    """metadata.parquet and the exclusion list: an image is conflicting if it is the same image (pixel-identical or SSCD near-duplicate,
    any splits) as one of a different class. Same-class near-duplicates are marked (`near_duplicate`), not removed."""
    idx = pd.read_parquet(P.raw_index)
    flagged = conflict_pairs(pd.read_csv(P.duplicates_all), cfg["duplicates"]["sscd_threshold"])
    split = {i: s for s in SPLITS for i in load_split(P.splits, s, exclude=False)}
    half = {i: s[-1] for s in ("test_A", "test_B") for i in load_split(P.splits, s, exclude=False)}
    meta = idx.rename(columns={"pixel_sha256": "raw_sha256"}).assign(fine_id=idx.class_id)
    meta["path"] = "native_32/" + meta.id + ".png"
    meta["split"], meta["test_half"] = meta.id.map(split), meta.id.map(half)
    if meta.split.isna().any():
        raise ValueError(f"{meta.split.isna().sum()} images without a split")
    meta["n_near_duplicates"] = meta.id.map(pd.concat([flagged.id_a, flagged.id_b]).value_counts()).fillna(0).astype(int)
    meta["near_duplicate"] = meta.n_near_duplicates > 0
    meta["exclude"] = meta.id.isin(set(flagged.loc[flagged.label_conflict, ["id_a", "id_b"]].to_numpy().ravel()))
    P.exclude.write_text("".join(f"{i}\n" for i in sorted(meta.id[meta.exclude])))
    cols = ["id", "path", "official_split", "row", "split", "test_half", "make_id", "fine_id", "class_id", "raw_sha256",
            "near_duplicate", "n_near_duplicates", "exclude"]
    meta[cols].sort_values("id", ignore_index=True).to_parquet(P.metadata, index=False)
    files = {**json.loads((P.derived / "png_shas.json").read_text()), "metadata.parquet": sha256_file(P.metadata)}
    write_json(P.manifests / "derived_summary.json", {**provenance(cfg, cfg["seed"], dataset_hash(P)), "derived_root": str(P.derived),
               "manifest_hash": manifest_hash(files), "n_files": len(files), "exclude_sha256": sha256_file(P.exclude),
               "excluded_per_split": meta[meta.exclude].split.value_counts().reindex(SPLITS, fill_value=0).to_dict()})
    log.info("conflicting images excluded from all experiments: %d %s", meta.exclude.sum(),
             meta[meta.exclude].split.value_counts().reindex(SPLITS, fill_value=0).to_dict())


def stage_report(cfg, P):
    """reports_root/data_summary.md: counts per split and superclass, the duplicate audit per SSCD band (to fix the threshold) with a grid
    of sampled pairs per band."""
    d = cfg["duplicates"]
    meta, h = pd.read_parquet(P.metadata), pd.read_csv(P.hierarchy)
    pairs = pd.read_csv(P.duplicates_all)
    P.reports.mkdir(parents=True, exist_ok=True)
    prov = provenance(cfg, cfg["seed"], dataset_hash(P))
    counts = pd.DataFrame({s: [(meta.split == s).sum(), ((meta.split == s) & meta.exclude).sum()] for s in SPLITS},
                          index=["images", "excluded"]).T.assign(used=lambda t: t.images - t.excluded)
    sup = meta[~meta.exclude].groupby(["make_id", "split"]).size().unstack().reindex(columns=list(SPLITS))
    sup.insert(0, "superclass", h.drop_duplicates("make_id").set_index("make_id").make)
    sup.insert(1, "classes", h.groupby("make_id").size())

    edges = [d["audit_min_sim"], 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    pairs["band"] = pd.cut(pairs.sscd_sim, edges, right=False)
    bands = pairs.groupby("band", observed=False).agg(pairs=("id_a", "size"), different_class=("label_conflict", "sum"),
                                                      different_superclass=("superclass_conflict", "sum"),
                                                      train_test=("splits", lambda s: (s == "train-test").sum()))
    rng = np.random.default_rng(cfg["seed"])
    grids = []
    for band, g in pairs.groupby("band", observed=True):
        g = g[g.label_conflict]
        pick = g.iloc[rng.choice(len(g), min(16, len(g)), replace=False)] if len(g) else g
        if len(pick):
            name = f"dup_pairs_{band.left:.2f}.png"
            image_grid([P.images / f"{i}.png" for r in pick.itertuples() for i in (r.id_a, r.id_b)], ncol=8, tile=64).save(P.reports / name)
            grids.append(f"- SSCD in {band}: ![]({name})")
    lines = ["<!-- generated by scripts/prepare_cifar100.py; do not edit -->",
             f"Provenance: git `{prov['git_hash']}`, config `{prov['config_path']}`, seed {prov['seed']}, "
             f"dataset manifest `{prov['dataset_manifest_hash'][:16]}`, created {prov['created_utc']}", "",
             "# CIFAR-100 data summary", "",
             f"20 superclasses x 5 classes. Conflicting images (same image as one of a different class: pixel-identical or SSCD cosine >= "
             f"{d['sscd_threshold']}, any splits) are excluded from every experiment: {int(meta.exclude.sum())} images.", "",
             counts.to_markdown(), "", "## Images per superclass (after exclusion)", "", sup.to_markdown(), "",
             f"## Duplicate audit (SSCD disc_mixup on the 32 px images upsampled to {d['sscd_resize']}; pairs in any splits)", "",
             f"Pixel-identical pairs: {int(pairs.exact_duplicate.sum())} (different class: {int((pairs.exact_duplicate & pairs.label_conflict).sum())}).",
             "", bands.to_markdown(), "", "Random different-class pairs per band (each pair = two adjacent tiles):", "", *grids, ""]
    (P.reports / "data_summary.md").write_text("\n".join(lines))
    log.info("report written to %s", P.reports / "data_summary.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_cifar100.yaml")
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    P = Paths(cfg)
    for s in STAGES:
        if s in args.stages:
            log.info("=== stage %s", s)
            globals()[f"stage_{s}"](cfg, P)


if __name__ == "__main__":
    main()
