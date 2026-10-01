"""Milestone 1: verify raw data, build hierarchy, crops, splits, duplicate audit, metadata, reports.

Run on a GPU node via `sbatch slurm/prepare_data.sh` (the `duplicates` stage needs a GPU for SSCD; the others use
CPU workers). Usage:
    python scripts/prepare_data.py --config configs/data.yaml [--stages raw hierarchy crops splits duplicates finalize reports]
                                   [--recompute-embeddings] [--allow-cpu]
Each stage reads the previous stages' outputs from disk, so stages can be re-run individually.
"""
import argparse
import json
import logging
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lsgen.data import duplicates as dup
from lsgen.data.crops import crop_bbox15, crop_full_cc, png_bytes
from lsgen.data.hierarchy import build_hierarchy, kc_per_make, make_model_table
from lsgen.data.reports import data_summary_report, preprocessing_report
from lsgen.data.splits import load_split, make_splits, save_splits
from lsgen.data.stanford_cars import BBOX_COLS, load_annotations, load_class_names, raw_manifest, verify_raw
from lsgen.utils import load_config, manifest_hash, provenance, setup_logging, sha256_bytes, sha256_file, write_json

log = logging.getLogger("prepare_data")
STAGES = ("raw", "hierarchy", "crops", "splits", "duplicates", "finalize", "reports")


class Paths:
    def __init__(self, cfg):
        self.raw = Path(cfg["raw_root"])
        self.derived = Path(cfg["derived_root"])
        self.data = Path(cfg["data_root"])
        self.reports = Path(cfg["reports_root"])
        self.raw_index = self.derived / "raw_index.parquet"
        self.crops_meta = self.derived / "crops_meta.parquet"
        self.metadata = self.derived / "metadata.parquet"
        self.derived_manifest = self.derived / "manifest_derived.json"
        self.dup_dir = self.derived / "duplicates"
        self.weights = self.derived / "weights"
        self.manifests = self.data / "manifests"
        self.raw_manifest = self.manifests / "raw_manifest.json"
        self.splits = self.data / "splits"
        self.hierarchy = self.data / "hierarchy.csv"
        self.duplicates = self.data / "duplicates.csv"
        self.duplicates_within_test = self.data / "duplicates_within_test.csv"
        self.exclude = self.splits / "exclude.txt"


def dataset_hash(P):
    return json.loads(P.raw_manifest.read_text())["manifest_hash"]


# ---------------------------------------------------------------------------- stages
def stage_raw(cfg, P):
    df = verify_raw(load_annotations(P.raw), P.raw, cfg["num_workers"])
    P.derived.mkdir(parents=True, exist_ok=True)
    df.to_parquet(P.raw_index, index=False)
    files = raw_manifest(df, P.raw)
    h = manifest_hash(files)
    write_json(P.raw_manifest, {**provenance(cfg, dataset_manifest_hash=h), "manifest_hash": h,
                                "raw_root": str(P.raw), "n_files": len(files), "files": files})
    log.info("raw: %d images verified; manifest hash %s", len(df), h[:16])


def stage_hierarchy(cfg, P):
    h = build_hierarchy(load_class_names(P.raw))
    P.data.mkdir(parents=True, exist_ok=True)
    h.to_csv(P.hierarchy, index=False)
    make_model_table(h).to_csv(P.data / "make_model.csv", index=False)
    kc = kc_per_make(h).sort_values(["K_c", "make"], ascending=[False, True])
    log.info("49 makes. K_c per make:\n%s", kc.to_string(index=False))
    log.info("makes with K_c = 1 (%d): %s", (kc.K_c == 1).sum(), ", ".join(kc.make[kc.K_c == 1]))
    log.info("make-model classes: %d", h.model_id.nunique())


def _crop_one(args):
    """Worker: crops one raw image into all variants. Returns (meta row, {relpath: sha256})."""
    rec, raw_root, out_root, c = args
    with Image.open(Path(raw_root) / rec["rel_path"]) as im:
        im.load()
        crops, m = crop_bbox15(im, [rec[k] for k in BBOX_COLS], c["sizes"], c["bbox15"]["margin"],
                               c["bbox15"]["min_margin"], c["bbox15"]["pad_color"])
        files = {f"bbox15_{sz}/{rec['id']}.png": png_bytes(img) for sz, img in crops.items()}
        files.update({f"full_cc_{sz}/{rec['id']}.png": png_bytes(crop_full_cc(im, sz)) for sz in c["sizes"]})
    shas = {}
    for rel, b in files.items():
        if out_root is not None:
            (Path(out_root) / rel).write_bytes(b)
        shas[rel] = sha256_bytes(b)
    row = {"id": rec["id"], "side": m["side"], "pad_l": m["pad_l"], "pad_t": m["pad_t"], "pad_r": m["pad_r"],
           "pad_b": m["pad_b"], "pad_fraction": m["pad_fraction"], "margin_used": m["margin_used"],
           "was_grayscale": m["was_grayscale"],
           **{f"crop_{k}": v for k, v in zip(("x0", "y0", "x1", "y1"), m["crop_box"])},
           **{f"scale_{sz}": s for sz, s in m["scale"].items()}}
    return row, shas


def stage_crops(cfg, P):
    c = cfg["crops"]
    idx = pd.read_parquet(P.raw_index)
    for v in ("bbox15", "full_cc"):
        for sz in c["sizes"]:
            (P.derived / f"{v}_{sz}").mkdir(parents=True, exist_ok=True)
    recs = idx[["id", "rel_path"] + BBOX_COLS].to_dict("records")
    rows, shas = [], {}
    with Pool(cfg["num_workers"]) as pool:
        for k, (row, s) in enumerate(pool.imap(_crop_one, [(r, str(P.raw), str(P.derived), c) for r in recs], chunksize=16)):
            rows.append(row)
            shas.update(s)
            if k % 2000 == 0:
                log.info("crops %d/%d", k, len(recs))
    cm = pd.DataFrame(rows)
    cm.to_parquet(P.crops_meta, index=False)

    # determinism: re-crop a random subset in memory and compare bytes with what was written
    rng = np.random.default_rng(cfg["seed"])
    sub = [recs[i] for i in rng.choice(len(recs), c["determinism_check_n"], replace=False)]
    with Pool(cfg["num_workers"]) as pool:
        again = pool.map(_crop_one, [(r, str(P.raw), None, c) for r in sub])
    mismatches = [rel for _, s in again for rel, h in s.items() if shas[rel] != h]
    on_disk = [rel for _, s in again for rel in s if sha256_file(P.derived / rel) != shas[rel]]
    det = (f"{len(sub)} images x {2 * len(c['sizes'])} files re-generated: "
           f"{len(mismatches)} byte mismatches, {len(on_disk)} on-disk mismatches")
    write_json(P.derived / "crops_determinism.json", {"result": det, "mismatches": mismatches + on_disk})
    if mismatches or on_disk:
        raise RuntimeError("non-deterministic crops: " + det)
    write_json(P.derived / "crops_shas.json", shas)
    log.info("crops written: %d files; determinism: %s", len(shas), det)
    _preprocessing_report(cfg, P)


def _preprocessing_report(cfg, P):
    c = cfg["crops"]
    idx = pd.read_parquet(P.raw_index)
    meta = pd.read_parquet(P.crops_meta).merge(idx[["id", "orig_W", "orig_H", "bbox_clipped"]], on="id")
    # hard sanity check: padding only where the side unavoidably exceeds an image dimension
    avoidable = (meta.pad_fraction > 0) & (meta.side <= np.minimum(meta.orig_W, meta.orig_H))
    if avoidable.any():
        raise RuntimeError(f"{avoidable.sum()} images padded although the window fits: {meta.id[avoidable].tolist()[:10]}")
    det = json.loads((P.derived / "crops_determinism.json").read_text())["result"]
    ok, share = preprocessing_report(meta, P.derived, P.reports, provenance(cfg, cfg["seed"], dataset_hash(P)),
                                     c["pad_gate_fraction"], c["pad_gate_share"], det, cfg["seed"], c["pad_gate_action"])
    if ok:
        log.info("pad gate passed: %.2f%% of images have pad_fraction > %s", 100 * share, c["pad_gate_fraction"])
    elif c["pad_gate_action"] == "stop":
        log.error("STOP: %.2f%% of images have pad_fraction > %s (gate %.0f%%). See reports/preprocessing.md",
                  100 * share, c["pad_gate_fraction"], 100 * c["pad_gate_share"])
        sys.exit(2)
    else:
        log.warning("pad gate exceeded (%.2f%% of images have pad_fraction > %s); accepted by decision, see docs/DATA.md",
                    100 * share, c["pad_gate_fraction"])


def stage_splits(cfg, P):
    idx = pd.read_parquet(P.raw_index, columns=["id", "official_split", "fine_id"])
    s = make_splits(idx, cfg["val_frac"], cfg["seed"])
    save_splits(s, P.splits)
    write_json(P.manifests / "splits.json", {**provenance(cfg, cfg["seed"], dataset_hash(P)),
                                             "counts": {k: len(v) for k, v in s.items()},
                                             "sha256": {k: sha256_file(P.splits / f"{k}.txt") for k in s}})
    log.info("splits: %s", {k: len(v) for k, v in s.items()})


def stage_duplicates(cfg, P):
    d = cfg["duplicates"]
    idx = pd.read_parquet(P.raw_index, columns=["id", "official_split", "rel_path", "sha256", "fine_id"])
    P.dup_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(P.raw / p) for p in idx.rel_path]
    ids = idx.id.tolist()

    ph_file = P.dup_dir / "phash_raw.npy"
    if not ph_file.exists():
        log.info("pHash on %d raw images", len(paths))
        np.save(ph_file, dup.phash_all(paths, cfg["num_workers"]))
    ph = np.load(ph_file)

    model, wpath, wsha = dup.get_sscd(P.weights, d["sscd_url"])
    if d.get("sscd_sha256") and d["sscd_sha256"] != wsha:
        raise ValueError(f"SSCD weights sha256 {wsha} != config {d['sscd_sha256']}")
    emb_file, emb_info = P.dup_dir / "sscd_raw_embeddings.npy", P.dup_dir / "sscd_raw_embeddings.json"
    if cfg["_recompute_embeddings"] or not emb_file.exists():
        device = dup.resolve_device(d["device"], cfg["_allow_cpu"])
        log.info("SSCD embeddings on %s for %d raw images", device, len(paths))
        np.save(emb_file, dup.sscd_embed(model, paths, d["sscd_resize"], min(8, cfg["num_workers"]), d["torch_threads"], device))
        write_json(emb_info, {**provenance(cfg, cfg["seed"], dataset_hash(P)), "device": str(device),
                              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                              "torch": torch.__version__, "sscd_weights_sha256": wsha})
    else:
        log.info("using cached SSCD embeddings %s (pass --recompute-embeddings to regenerate)", emb_file)
    emb = np.load(emb_file)
    emb_device = json.loads(emb_info.read_text())["device"] if emb_info.exists() else "cpu (login node, before device logging)"
    pd.DataFrame({"id": ids}).to_csv(P.dup_dir / "embedding_ids.csv", index=False)

    te, tr = (idx.official_split == "test").to_numpy(), (idx.official_split == "train").to_numpy()
    table, nn = dup.duplicate_table(list(np.array(ids)[te]), list(np.array(ids)[tr]), ph[te], ph[tr], emb[te], emb[tr],
                                    d["phash_max_hamming"], d["sscd_threshold"])
    # byte-identical files across splits, and pairs whose official fine labels disagree (label noise)
    by_id = idx.set_index("id")
    table["exact_duplicate"] = by_id.loc[table.test_id, "sha256"].to_numpy() == by_id.loc[table.train_id, "sha256"].to_numpy()
    table["label_conflict"] = by_id.loc[table.test_id, "fine_id"].to_numpy() != by_id.loc[table.train_id, "fine_id"].to_numpy()
    table.to_csv(P.duplicates, index=False, float_format="%.4f")
    nn.to_parquet(P.dup_dir / "sscd_nn_test_to_train.parquet", index=False)
    # within official test: support and target sets of use-case-B episodes are both drawn from test
    wt = dup.within_table(list(np.array(ids)[te]), ph[te], emb[te], d["phash_max_hamming"], d["sscd_threshold"])
    wt["exact_duplicate"] = by_id.loc[wt.id_a, "sha256"].to_numpy() == by_id.loc[wt.id_b, "sha256"].to_numpy()
    wt["label_conflict"] = by_id.loc[wt.id_a, "fine_id"].to_numpy() != by_id.loc[wt.id_b, "fine_id"].to_numpy()
    wt.to_csv(P.duplicates_within_test, index=False, float_format="%.4f")
    write_json(P.manifests / "duplicates.json", {**provenance(cfg, cfg["seed"], dataset_hash(P)),
               "n_pairs": len(table), "n_flag_sscd": int(table.flag_sscd.sum()), "n_flag_phash": int(table.flag_phash.sum()),
               "n_exact_duplicate": int(table.exact_duplicate.sum()),
               "n_exact_duplicate_label_conflict": int((table.exact_duplicate & table.label_conflict).sum()),
               "n_pairs_within_test": len(wt),
               "sscd_weights": str(wpath), "sscd_weights_sha256": wsha, "sscd_embedding_device": emb_device,
               "duplicates_csv_sha256": sha256_file(P.duplicates),
               "duplicates_within_test_csv_sha256": sha256_file(P.duplicates_within_test)})
    log.info("near-duplicate pairs: %d (SSCD %d, pHash %d); byte-identical %d, of which label conflicts %d; "
             "within test: %d pairs", len(table), table.flag_sscd.sum(), table.flag_phash.sum(),
             table.exact_duplicate.sum(), (table.exact_duplicate & table.label_conflict).sum(), len(wt))


def stage_finalize(cfg, P):
    idx = pd.read_parquet(P.raw_index)
    h = pd.read_csv(P.hierarchy)[["fine_id", "make_id", "model_id"]]
    cm = pd.read_parquet(P.crops_meta)
    meta = idx.merge(h, on="fine_id", validate="many_to_one").merge(cm, on="id", validate="one_to_one")
    meta = meta.rename(columns={"orig_mode": "raw_mode", "sha256": "raw_sha256"})
    split = {i: s for s in ("train", "val", "test") for i in load_split(P.splits, s, exclude=False)}
    half = {i: s[-1] for s in ("test_A", "test_B") for i in load_split(P.splits, s, exclude=False)}
    meta["split"] = meta.id.map(split)
    meta["test_half"] = meta.id.map(half)
    if meta.split.isna().any():
        raise ValueError(f"{meta.split.isna().sum()} images without a split")
    dups = pd.read_csv(P.duplicates)
    n_dup = pd.concat([dups.test_id, dups.train_id]).value_counts()
    meta["n_near_duplicates"] = meta.id.map(n_dup).fillna(0).astype(int)
    meta["near_duplicate"] = meta.n_near_duplicates > 0
    wt = pd.read_csv(P.duplicates_within_test)
    meta["n_near_duplicates_within_test"] = meta.id.map(pd.concat([wt.id_a, wt.id_b]).value_counts()).fillna(0).astype(int)
    # Conflicting data (decision 1 Oct 2026): the same image under different official fine labels, i.e. byte-identical
    # files or flagged near-duplicate pairs (train-test, within test) whose labels differ. Left out of ALL experiments.
    in_conflict_pair = set(dups.loc[dups.label_conflict, ["test_id", "train_id"]].to_numpy().ravel()) | \
        set(wt.loc[wt.label_conflict, ["id_a", "id_b"]].to_numpy().ravel())
    meta["exclude"] = (meta.groupby("raw_sha256").fine_id.transform("nunique") > 1) | meta.id.isin(in_conflict_pair)
    P.exclude.write_text("".join(f"{i}\n" for i in sorted(meta.id[meta.exclude])))
    log.info("conflicting images excluded from all experiments: %d (%s)",
             meta.exclude.sum(), meta[meta.exclude].split.value_counts().to_dict())
    meta["path"] = "bbox15_128/" + meta.id + ".png"
    cols = ["id", "path", "rel_path", "official_split", "split", "test_half", "make_id", "fine_id", "model_id",
            *BBOX_COLS, "bbox_clipped", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "side",
            *[c for c in meta.columns if c.startswith("scale_")], "pad_l", "pad_t", "pad_r", "pad_b", "pad_fraction",
            "margin_used", "was_grayscale", "raw_mode", "orig_W", "orig_H", "raw_sha256", "near_duplicate", "n_near_duplicates", "n_near_duplicates_within_test", "exclude"]
    meta = meta[cols].sort_values("id", ignore_index=True)
    meta.to_parquet(P.metadata, index=False)

    files = json.loads((P.derived / "crops_shas.json").read_text())
    files["metadata.parquet"] = sha256_file(P.metadata)
    mh = manifest_hash(files)
    prov = provenance(cfg, cfg["seed"], dataset_hash(P))
    write_json(P.derived_manifest, {**prov, "manifest_hash": mh, "n_files": len(files), "files": files})
    per_dir = {}
    for rel, sha in files.items():
        per_dir.setdefault(rel.split("/")[0] if "/" in rel else rel, {})[rel] = sha
    write_json(P.manifests / "derived_summary.json", {
        **prov, "derived_root": str(P.derived), "manifest_hash": mh, "n_files": len(files),
        "per_dir": {k: {"n_files": len(v), "manifest_hash": manifest_hash(v)} for k, v in sorted(per_dir.items())},
        "determinism": json.loads((P.derived / "crops_determinism.json").read_text())["result"]})
    log.info("metadata: %d rows, %d columns; derived manifest hash %s", len(meta), meta.shape[1], mh[:16])


def stage_reports(cfg, P):
    _preprocessing_report(cfg, P)
    meta = pd.read_parquet(P.metadata)
    h = pd.read_csv(P.hierarchy)
    nn_file = P.dup_dir / "sscd_nn_test_to_train.parquet"
    data_summary_report(meta, h, pd.read_csv(P.duplicates), P.reports, provenance(cfg, cfg["seed"], dataset_hash(P)),
                        pd.read_parquet(nn_file) if nn_file.exists() else None, cfg["duplicates"])
    log.info("reports written to %s", P.reports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--recompute-embeddings", action="store_true", help="regenerate cached SSCD embeddings")
    ap.add_argument("--allow-cpu", action="store_true", help="allow SSCD on CPU when no GPU is visible")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    cfg["_recompute_embeddings"], cfg["_allow_cpu"] = args.recompute_embeddings, args.allow_cpu
    P = Paths(cfg)
    for s in STAGES:
        if s in args.stages:
            log.info("=== stage %s", s)
            globals()[f"stage_{s}"](cfg, P)


if __name__ == "__main__":
    main()
