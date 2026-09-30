"""Training datasets. They return (image, make_id) and nothing else: no hidden-subclass labels,
no model/year information. tests/test_no_fine_labels.py enforces this, including a static check
that this module never mentions those columns."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF

ALLOWED_COLUMNS = ("id", "make_id")
AUGMENTATIONS = ("none", "hflip", "finegan_jitter")


class CarsCoarseDataset(torch.utils.data.Dataset):
    """Stored crops from derived_root/{variant}_{size}/{id}.png with coarse (make) labels only.

    augment: "hflip" (default: random horizontal flip), "finegan_jitter" (resize to 152, random
    128 crop, flip; label results as such) or "none" (evaluation)."""

    def __init__(self, metadata, derived_root, ids, variant="bbox15", size=128, augment="hflip"):
        if augment not in AUGMENTATIONS:
            raise ValueError(f"augment must be one of {AUGMENTATIONS}")
        meta = pd.read_parquet(metadata, columns=list(ALLOWED_COLUMNS)) if isinstance(metadata, (str, Path)) \
            else metadata.loc[:, list(ALLOWED_COLUMNS)]
        meta = meta.set_index("id")
        missing = set(ids) - set(meta.index)
        if missing:
            raise KeyError(f"{len(missing)} ids not in metadata, e.g. {sorted(missing)[:3]}")
        self.items = [(str(Path(derived_root) / f"{variant}_{size}" / f"{i}.png"), int(meta.at[i, "make_id"])) for i in ids]
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, make_id = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.augment == "finegan_jitter":
            img = TF.resize(img, [int(self.size * 76 / 64)] * 2, interpolation=TF.InterpolationMode.BILINEAR)
            i, j = (int(v) for v in torch.randint(0, img.size[0] - self.size + 1, (2,)))
            img = TF.crop(img, i, j, self.size, self.size)
        if self.augment != "none" and torch.rand(()) < 0.5:
            img = TF.hflip(img)
        return TF.to_tensor(img), make_id
