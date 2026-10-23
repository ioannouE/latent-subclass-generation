"""CIFAR-100 (Krizhevsky 2009, python version): 20 superclasses (coarse, `make_id`) > 100 classes (hidden subclass, `fine_id`), 5 per
superclass. Both labels come with the official files. Image ids are `{official_split}_{row:05d}` (row order of the official file)."""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FILES_MD5 = {"train": "16019d7e3df5f24257cddd939b257f8d", "test": "f0ef6b0ae62326f3e7ffdfab6717acfc",
             "meta": "7973b15100ade9c7d40fb424638fde48"}  # as in torchvision.datasets.CIFAR100
N_IMAGES = {"train": 50000, "test": 10000}
N_COARSE, N_FINE = 20, 100


def md5_file(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_official(raw_root, split):
    """(N, 32, 32, 3) uint8 images, fine labels, coarse labels of one official file ("train" or "test")."""
    with open(Path(raw_root) / "cifar-100-python" / split, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = np.asarray(d["data"], np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(x), np.asarray(d["fine_labels"]), np.asarray(d["coarse_labels"])


def load_names(raw_root):
    """(fine names, coarse names), indexed by the official label ids."""
    with open(Path(raw_root) / "cifar-100-python" / "meta", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d["fine_label_names"], d["coarse_label_names"]


def build_hierarchy(fine, coarse, fine_names, coarse_names):
    """One row per class (class_id = fine_id = official fine label, make_id = official coarse label), with the columns of the Stanford
    Cars data/hierarchy.csv so lsgen.data.labels.load_labels reads it unchanged: `make` = superclass name, `make_model` = class name.
    Raises unless every class lies in exactly one superclass and there are 20 superclasses of 5 classes."""
    h = pd.DataFrame({"class_id": fine, "make_id": coarse}).drop_duplicates().sort_values("class_id", ignore_index=True)
    if h.class_id.duplicated().any():
        raise ValueError(f"classes under two superclasses: {sorted(h.class_id[h.class_id.duplicated()])}")
    h["fine_id"] = h.class_id
    h["class_name"] = h["make_model"] = [fine_names[i] for i in h.class_id]
    h["make"] = [coarse_names[i] for i in h.make_id]
    h["K_c"] = h.groupby("make_id").fine_id.transform("nunique")
    if len(h) != N_FINE or h.make_id.nunique() != N_COARSE or (h.K_c != N_FINE // N_COARSE).any():
        raise ValueError(f"expected {N_COARSE} superclasses x {N_FINE // N_COARSE} classes, got {len(h)} classes, "
                         f"K_c {h.groupby('make_id').K_c.first().to_dict()}")
    return h[["class_id", "class_name", "make", "make_model", "make_id", "fine_id", "K_c"]]
