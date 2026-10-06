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


class CarsTwoViews(CarsCoarseDataset):
    """Contrastive training: two independent random augmentations (`transform`: PIL -> tensor) of each image, and its
    make_id. Still no hidden-subclass labels.
    transform_2: a different transform for the second view (default: the same as the first).
    neighbors: (N, k) row indices into `ids` of each image's neighbours (FALCON); then (view 1, view 2, `n_neighbors` randomly
    drawn neighbours as (n_neighbors, 3, H, W) under transform_2, make_id) is returned."""

    def __init__(self, metadata, derived_root, ids, transform, variant="bbox15", size=128, transform_2=None, neighbors=None,
                 n_neighbors=5):
        super().__init__(metadata, derived_root, ids, variant, size, augment="none")
        self.transform, self.transform_2 = transform, transform_2 or transform
        self.neighbors, self.n_neighbors = neighbors, n_neighbors

    def __getitem__(self, idx):
        path, make_id = self.items[idx]
        img = Image.open(path).convert("RGB")
        views = (self.transform(img), self.transform_2(img))
        if self.neighbors is None:
            return *views, make_id
        chosen = self.neighbors[idx][torch.randperm(self.neighbors.shape[1])[:self.n_neighbors]]
        near = torch.stack([self.transform_2(Image.open(self.items[j][0]).convert("RGB")) for j in chosen])
        return *views, near, make_id
