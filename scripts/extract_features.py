"""Milestone 3: cache frozen-encoder embeddings (CSV: filename, z1..zD) for train/val/test crops. Needs a GPU:
    sbatch slurm/extract_features.sh
Embeddings that are already cached for the same images (and exclusion list) are skipped; delete a folder to force a rerun.
Output: <derived_root>/features/<encoder>_<crop size>/<split>.csv (+ .json manifest).
"""
import argparse
import json
import logging
from pathlib import Path

import torch

from lsgen.data.splits import load_split
from lsgen.features.encoders import Embedder
from lsgen.features.extract import embed_images, write_csv
from lsgen.features.gram import GRAM, GramEmbedder
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, sha256_file, write_json

log = logging.getLogger("extract_features")


def cached(out, split, n, exclude_sha):
    """True if the embeddings of this split were already written for the same images (same count and exclusion list)."""
    manifest = out / f"{split}.json"
    if not (out / f"{split}.csv").exists() or not manifest.exists():
        return False
    m = json.loads(manifest.read_text())
    return m["n"] == n and m["exclude_sha256"] == exclude_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/features.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("feature extraction needs a GPU: sbatch slurm/extract_features.sh")
    device = torch.device("cuda")
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    sscd = derived / "weights" / Path(dcfg["duplicates"]["sscd_url"]).name
    dataset_hash = json.loads((Path(dcfg["data_root"]) / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    ids = {s: load_split(splits_dir, s) for s in cfg["splits"]}
    log.info("images per split after excluding conflicting ones: %s", {s: len(v) for s, v in ids.items()})

    exclude_sha = sha256_file(splits_dir / "exclude.txt")
    weights = {"sscd": sscd, "fine_classifier": derived / "evaluators" / "fine" / "model.pt"}
    for name in cfg["encoders"]:
        sizes = [s for s in cfg["crop_sizes"] if cfg.get("only_crop", {}).get(name, s) == s]
        todo = [(size, split) for size in sizes for split in ids
                if not cached(derived / "features" / f"{name}_{size}", split, len(ids[split]), exclude_sha)]
        if not todo:
            log.info("%s: all embeddings already cached, skipped", name)
            continue
        model = (GramEmbedder if name in GRAM else Embedder)(name, weights.get(name)).to(device)
        for size in sizes:
            out = derived / "features" / f"{name}_{size}"
            out.mkdir(parents=True, exist_ok=True)
            for split, split_ids in ids.items():
                if (size, split) not in todo:
                    continue
                emb = embed_images(model, [derived / f"bbox15_{size}" / f"{i}.png" for i in split_ids], model.transform,
                                   device, cfg["batch_size"], cfg["num_workers"])
                write_csv(out / f"{split}.csv", [f"{i}.png" for i in split_ids], emb)
                write_json(out / f"{split}.json", {
                    **provenance(cfg, None, dataset_hash), "encoder": name, "family": model.family,
                    "timm_model": model.timm_name, "encoder_input_size": model.input_size, "crop": f"bbox15_{size}",
                    "split": split, "n": len(split_ids), "dim": emb.shape[1],
                    "exclude_sha256": exclude_sha})
                log.info("%s %d %s: %s", name, size, split, emb.shape)


if __name__ == "__main__":
    main()
