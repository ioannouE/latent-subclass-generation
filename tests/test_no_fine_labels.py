"""Hard rule 1: training datasets never return fine labels."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import lsgen.data.datasets as datasets_mod
from lsgen.data.datasets import CarsCoarseDataset
from lsgen.utils import REPO_ROOT

FINE_TOKENS = re.compile(r"fine_id|model_id|make_model|class_name|\byear\b|fine_label")
# Modules allowed to touch fine labels: evaluation, and data/split/episode construction.
ALLOWED = {"lsgen/eval", "lsgen/data/stanford_cars.py", "lsgen/data/hierarchy.py", "lsgen/data/splits.py",
           "lsgen/data/episodes.py", "lsgen/data/reports.py", "scripts/prepare_data.py", "scripts/build_episodes.py", "scripts/train_eval_classifier.py", "scripts/analyze_features.py",
           "scripts/eval_representation.py", "scripts/eval_generation.py", "scripts/make_controls.py"}


def test_dataset_returns_only_image_and_make(tmp_path):
    meta = pd.DataFrame({"id": ["train_00001", "train_00002"], "make_id": [3, 7], "fine_id": [11, 42],
                         "model_id": [5, 9], "class_name": ["a", "b"]})
    d = tmp_path / "bbox15_128"
    d.mkdir()
    for i in meta.id:
        Image.fromarray(np.zeros((128, 128, 3), np.uint8)).save(d / f"{i}.png")
    ds = CarsCoarseDataset(meta, tmp_path, list(meta.id), augment="hflip")
    for k in range(len(ds)):
        item = ds[k]
        assert isinstance(item, tuple) and len(item) == 2
        x, y = item
        assert isinstance(x, torch.Tensor) and x.shape == (3, 128, 128)
        assert y == meta.make_id[k]
    assert all(len(it) == 2 for it in ds.items)
    assert not any(isinstance(v, pd.DataFrame) for v in vars(ds).values())


def test_datasets_module_never_mentions_fine_labels():
    src = Path(datasets_mod.__file__).read_text()
    assert not FINE_TOKENS.search(src.split('"""', 2)[2])  # skip the module docstring


def test_fine_labels_only_in_allowed_modules():
    offenders = []
    for p in list((REPO_ROOT / "lsgen").rglob("*.py")) + list((REPO_ROOT / "scripts").rglob("*.py")):
        rel = p.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(tuple(ALLOWED)) or p.name == "datasets.py":
            continue
        if FINE_TOKENS.search(p.read_text()):
            offenders.append(rel)
    assert not offenders, f"fine-label columns referenced outside allowed modules: {offenders}"
