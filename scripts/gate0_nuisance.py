"""Gate 0 (docs/EXPERIMENT_NHP.md): variance of the top within-make principal components of a frozen embedding that cheap nuisance proxies
explain (colour, shape, background from pixels and the bbox; pose from flip-consistency). Needs a GPU to embed the mirror images:
    sbatch slurm/gate0.sh
Output: <reports_root>/gate0/{per_make.csv, macro.csv, summary.json}; the mirror-image embeddings go to <derived_root>/features/<encoder>_flip/.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from lsgen.data.datasets import car_boxes
from lsgen.data.labels import load_labels
from lsgen.data.splits import load_split
from lsgen.eval.representation import l2norm
from lsgen.features.encoders import Embedder
from lsgen.features.extract import embed_images, read_csv, write_csv
from lsgen.features.nuisance import explained, flip_proxies, pixel_proxies
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, write_json

log = logging.getLogger("gate0")
COLUMNS = {"colour": slice(0, 13), "shape": slice(13, 15), "background": slice(15, 17)}  # columns of pixel_proxies (8 hue bins)
MIN_IMAGES = 30


def load_embeddings(path, ids):
    files, x = read_csv(path)
    return pd.DataFrame(x, index=[f[:-4] for f in files]).loc[ids].to_numpy()


def mirror_embeddings(cfg, derived, ids, name, device):
    """Embeddings of the horizontally flipped crops (cached next to the frozen features)."""
    path = derived / "features" / f"{name}_flip" / f"{cfg['split']}.csv"
    if not path.exists():
        model = Embedder(cfg["backbone"]).to(device)
        flipped = lambda im: model.transform(TF.hflip(im))  # noqa: E731
        emb = embed_images(model, [derived / cfg["crop"] / f"{i}.png" for i in ids], flipped, device, cfg["batch_size_embed"], cfg["num_workers"])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, [f"{i}.png" for i in ids], emb)
    return load_embeddings(path, ids)


def per_make_table(emb, emb_flip, pixel, make, cfg):
    rng, rows = np.random.default_rng(cfg["seed"]), []
    for c in np.unique(make):
        i = np.flatnonzero(make == c)
        if len(i) < MIN_IMAGES:
            log.warning("make %d skipped: %d images", c, len(i))
            continue
        pose = flip_proxies(emb[i], emb_flip[i])
        sets = {"pixel": pixel[i], **{g: pixel[i][:, s] for g, s in COLUMNS.items()}, "pose": pose, "pixel+pose": np.hstack([pixel[i], pose])}
        sets["shuffled"] = sets["pixel+pose"][rng.permutation(len(i))]  # control: proxies of other images of the make
        rows.append({"make_id": c, "n": len(i), **{f"{k}_{m}": v for k, p in sets.items()
                                                    for m, v in zip(("explained", "pc1_r2"), explained(emb[i], p, cfg["n_pcs"], cfg["cv_folds"], cfg["seed"]))}})
    return pd.DataFrame(rows)


def macro_table(table, cfg):
    """Macro mean over makes with a 95% CI from resampling the makes."""
    rng, cols = np.random.default_rng(cfg["seed"]), table.drop(columns=["make_id", "n"])
    boots = np.array([cols.iloc[rng.integers(0, len(cols), len(cols))].mean() for _ in range(cfg["n_boot"])])
    return pd.DataFrame({"mean": cols.mean(), "lo": np.percentile(boots, 2.5, axis=0), "hi": np.percentile(boots, 97.5, axis=0)}).rename_axis("metric").reset_index()


def verdict(macro, decision):
    value = macro.set_index("metric").loc["pixel_explained", "mean"]
    if value < decision["weak_below"]:
        return "weak: nuisance proxies explain little of the top within-make PCs; NHP is not expected to help, stop"
    return "supported" if value >= decision["supported_from"] else "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gate0.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("embedding the mirror images needs a GPU: sbatch slurm/gate0.sh")
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    ids = load_split(splits_dir, cfg["split"])
    make = load_labels(derived, dcfg["data_root"])[["make_id"]].loc[ids].make_id.to_numpy()  # coarse labels only
    size, name = int(cfg["crop"].split("_")[-1]), f"{cfg['backbone']}_{cfg['crop'].split('_')[-1]}"
    log.info("%d %s images after excluding conflicting ones, %d makes", len(ids), cfg["split"], len(np.unique(make)))
    emb = l2norm(load_embeddings(derived / "features" / name / f"{cfg['split']}.csv", ids))
    emb_flip = l2norm(mirror_embeddings(cfg, derived, ids, name, torch.device("cuda")))
    boxes = car_boxes(derived, ids, size)
    pixel = np.stack([pixel_proxies(Image.open(derived / cfg["crop"] / f"{i}.png"), b) for i, b in zip(ids, boxes)])
    table = per_make_table(emb, emb_flip, pixel, make, cfg)
    macro = macro_table(table, cfg)
    out = Path(dcfg["reports_root"]) / "gate0"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_make.csv", index=False)
    macro.to_csv(out / "macro.csv", index=False)
    result = verdict(macro, cfg["decision"])
    write_json(out / "summary.json", {**provenance(cfg, cfg["seed"], dataset_hash), "n_images": len(ids), "n_makes": len(table),
                                      "decision": cfg["decision"], "verdict": result})
    log.info("\n%s\nverdict: %s", macro.round(3).to_string(index=False), result)


if __name__ == "__main__":
    main()
