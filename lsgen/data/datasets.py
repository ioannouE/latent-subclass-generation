"""Training datasets. They return (image, make_id) and nothing else: no hidden-subclass labels,
no model/year information. tests/test_no_fine_labels.py enforces this, including a static check
that this module never mentions those columns."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from lsgen.data.crops import PAD_COLOR, bbox_in_crop, devkit_bbox_to_px

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



def car_boxes(derived, ids, size):
    """(N, 4) the car's bbox in the pixels of the stored `size` crops, rows in the order of ids."""
    cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "orig_W", "orig_H", "crop_x0", "crop_y0", "side"]
    m = pd.read_parquet(derived / "metadata.parquet", columns=["id", *cols]).set_index("id").loc[ids]
    return np.array([bbox_in_crop({"crop_box": (r.crop_x0, r.crop_y0), "side": r.side,
                                   "bbox_px": devkit_bbox_to_px((r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2), r.orig_W, r.orig_H)}, size)
                     for r in m.itertuples()])


PERMUTATIONS = list(itertools.permutations(range(3)))  # colour-channel permutations


def replace_background(img, box, scenery):
    """The pixels of `img` inside box (x0, y0, x1, y1), on top of `scenery` (a PIL image of the same size)."""
    out = scenery.copy()
    out.paste(img.crop(box), box[:2])
    return out


class CarsNuisanceViews(CarsCoarseDataset):
    """NHP training: two independently augmented views of each image, returned as (view 1, view 2, parameters (2, 2), make_id).
    parameters[v] = (colour-permutation id, flipped) of view v: what the adversary must not recover from z_sub. Still no
    hidden-subclass labels.
    crop: PIL -> PIL random crop; post: PIL -> tensor (colour jitter, normalisation). boxes: (N, 4) the car's bbox in the pixels
    of the stored crops (x0, y0, x1, y1), rows in the order of `ids`.
    invariance: each view also gets a random colour-channel permutation, and with probability swap_p / mask_p everything
    outside the bbox replaced by the scenery of another image (its own bbox blanked) / by flat grey."""

    def __init__(self, metadata, derived_root, ids, crop, post, boxes, variant="bbox15", size=256, invariance=True, swap_p=0.3, mask_p=0.2):
        super().__init__(metadata, derived_root, ids, variant, size, augment="none")
        self.crop, self.post, self.boxes, self.invariance, self.swap_p, self.mask_p = crop, post, boxes, invariance, swap_p, mask_p

    def _box(self, idx):
        return tuple(int(round(min(max(v, 0), self.size))) for v in self.boxes[idx])

    def _view(self, img, idx):
        if self.invariance:
            u = torch.rand(())
            if u < self.swap_p:
                j = int(torch.randint(len(self), ()))
                donor = Image.open(self.items[j][0]).convert("RGB")
                donor.paste(PAD_COLOR, self._box(j))
                img = replace_background(img, self._box(idx), donor)
            elif u < self.swap_p + self.mask_p:
                img = replace_background(img, self._box(idx), Image.new("RGB", img.size, PAD_COLOR))
        img = self.crop(img)
        flip = bool(torch.rand(()) < 0.5)
        if flip:
            img = TF.hflip(img)
        perm = int(torch.randint(len(PERMUTATIONS), ())) if self.invariance else 0
        img = Image.merge("RGB", [img.getchannel(c) for c in PERMUTATIONS[perm]])
        return self.post(img), (perm, int(flip))

    def __getitem__(self, idx):
        path, make_id = self.items[idx]
        img = Image.open(path).convert("RGB")
        (v1, a1), (v2, a2) = self._view(img, idx), self._view(img, idx)
        return v1, v2, torch.tensor([a1, a2]), make_id
