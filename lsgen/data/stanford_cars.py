"""Raw Stanford Cars (Kaggle mirror + labelled test annotations): loading and verification.

This module is data construction: it reads the official class labels (class_id, make-model-year) so that the
hierarchy, splits, episodes and evaluation can be built. Training datasets live in
lsgen/data/datasets.py and never expose fine labels.
"""
import logging
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import scipy.io as sio
from PIL import Image

from lsgen.utils import sha256_file

log = logging.getLogger(__name__)

IMAGE_DIRS = {"train": "cars_train/cars_train", "test": "cars_test/cars_test"}
ANNOS = {"train": "devkit/cars_train_annos.mat", "test": "devkit/cars_test_annos_withlabels.mat"}
UNLABELLED_TEST_ANNOS = "devkit/cars_test_annos.mat"
META = "devkit/cars_meta.mat"
EXPECTED_COUNTS = {"train": 8144, "test": 8041}
N_CLASSES = 196
BBOX_COLS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


def load_class_names(raw_root):
    names = [str(n) for n in sio.loadmat(Path(raw_root) / META, squeeze_me=True)["class_names"]]
    if len(names) != N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} class names, got {len(names)}")
    return names


def _load_mat_annos(path):
    a = sio.loadmat(path, squeeze_me=True)["annotations"]
    df = pd.DataFrame({c: [int(x) for x in a[c]] for c in BBOX_COLS})
    df["fname"] = [str(x) for x in a["fname"]]
    if "class" in a.dtype.names:
        df["class_id"] = [int(x) - 1 for x in a["class"]]  # devkit classes are 1-indexed
    return df


def load_annotations(raw_root):
    """One row per image: id, official_split, fname, rel_path, devkit bbox (1-indexed, inclusive), class_id."""
    raw_root = Path(raw_root)
    parts = []
    for split in ("train", "test"):
        df = _load_mat_annos(raw_root / ANNOS[split])
        if "class_id" not in df:
            raise ValueError(f"{ANNOS[split]} has no class labels; refusing to build an unlabelled {split} split")
        df.insert(0, "official_split", split)
        df.insert(0, "id", [f"{split}_{Path(f).stem}" for f in df["fname"]])
        df["rel_path"] = [f"{IMAGE_DIRS[split]}/{f}" for f in df["fname"]]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    # The labelled test file (third-party copy) must agree record-by-record with the official unlabelled one.
    unl = _load_mat_annos(raw_root / UNLABELLED_TEST_ANNOS)
    lab = df[df.official_split == "test"].reset_index(drop=True)
    if not (unl[["fname"] + BBOX_COLS].equals(lab[["fname"] + BBOX_COLS])):
        raise ValueError("cars_test_annos_withlabels.mat disagrees with the official cars_test_annos.mat")
    return df


def _inspect_image(args):
    raw_root, rel_path = args
    p = Path(raw_root) / rel_path
    try:
        with Image.open(p) as im:
            mode, (W, H) = im.mode, im.size
            im.load()
        return {"rel_path": rel_path, "orig_W": W, "orig_H": H, "orig_mode": mode, "sha256": sha256_file(p), "error": None}
    except Exception as e:  # recorded and raised by verify_raw, never swallowed
        return {"rel_path": rel_path, "orig_W": -1, "orig_H": -1, "orig_mode": None, "sha256": None, "error": repr(e)}


def verify_raw(df, raw_root, num_workers=8):
    """Check counts, classes, one bbox per image, readability, bbox bounds. Returns df + size/mode/sha256."""
    raw_root = Path(raw_root)
    counts = df.official_split.value_counts().to_dict()
    errors = []
    if counts != EXPECTED_COUNTS:
        errors.append(f"split counts {counts} != {EXPECTED_COUNTS}")
    if len(df) != sum(EXPECTED_COUNTS.values()):
        errors.append(f"total {len(df)} != 16185")
    for split in ("train", "test"):
        n_cls = df.loc[df.official_split == split, "class_id"].nunique()
        if n_cls != N_CLASSES or df.class_id.min() != 0 or df.class_id.max() != N_CLASSES - 1:
            errors.append(f"{split}: {n_cls} classes, class_id range [{df.class_id.min()}, {df.class_id.max()}]")
    if df.id.duplicated().any():
        errors.append(f"{df.id.duplicated().sum()} duplicate image ids (i.e. >1 annotation per image)")
    for split, d in IMAGE_DIRS.items():
        on_disk = {p.name for p in (raw_root / d).iterdir()}
        annotated = set(df.loc[df.official_split == split, "fname"])
        if annotated - on_disk:
            errors.append(f"{split}: {len(annotated - on_disk)} annotated images missing, e.g. {sorted(annotated - on_disk)[:3]}")
        if on_disk - annotated:
            log.warning("%s: %d files on disk without annotation (ignored): %s", split, len(on_disk - annotated), sorted(on_disk - annotated)[:5])
    if errors:
        raise ValueError("raw data verification failed:\n" + "\n".join(errors))

    with Pool(num_workers) as pool:
        info = pd.DataFrame(pool.map(_inspect_image, [(str(raw_root), p) for p in df.rel_path], chunksize=64))
    bad = info[info.error.notna()]
    if len(bad):
        raise ValueError(f"{len(bad)} unreadable images:\n{bad[['rel_path', 'error']].head(20).to_string()}")
    df = df.merge(info.drop(columns="error"), on="rel_path", how="left", validate="one_to_one")

    oob = (df.bbox_x1 < 1) | (df.bbox_y1 < 1) | (df.bbox_x2 > df.orig_W) | (df.bbox_y2 > df.orig_H)
    degenerate = (df.bbox_x2 <= df.bbox_x1) | (df.bbox_y2 <= df.bbox_y1)
    if degenerate.any():
        raise ValueError(f"{degenerate.sum()} degenerate bboxes: {df.loc[degenerate, 'id'].tolist()[:10]}")
    if oob.any():
        log.warning("%d bboxes exceed image bounds; they will be clipped: %s", oob.sum(), df.loc[oob, "id"].tolist()[:10])
    df["bbox_clipped"] = oob
    log.info("raw verified: %s images, %s; modes %s", len(df), counts, df.orig_mode.value_counts().to_dict())
    return df


def raw_manifest(df, raw_root):
    """{relative path: sha256} for every raw image and devkit file."""
    files = dict(zip(df.rel_path, df.sha256))
    for f in sorted((Path(raw_root) / "devkit").iterdir()):
        files[f"devkit/{f.name}"] = sha256_file(f)
    return files
