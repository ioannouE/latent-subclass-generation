import pickle

import numpy as np
import pytest

from lsgen.data.cifar100 import build_hierarchy, load_official

FINE_NAMES = [f"class{i}" for i in range(100)]
COARSE_NAMES = [f"super{i}" for i in range(20)]


def labels(rng):
    """Fine labels in random order, each class repeated, coarse = a fixed random 20 x 5 assignment."""
    of = rng.permutation(np.repeat(np.arange(20), 5))
    fine = rng.permutation(np.repeat(np.arange(100), 3))
    return fine, of[fine]


def test_hierarchy_20_by_5():
    fine, coarse = labels(np.random.default_rng(0))
    h = build_hierarchy(fine, coarse, FINE_NAMES, COARSE_NAMES)
    assert len(h) == 100 and (h.class_id == h.fine_id).all() and (h.K_c == 5).all()
    row = h.set_index("class_id").loc[fine[0]]
    assert row.make_id == coarse[0] and row.make == COARSE_NAMES[coarse[0]] and row.make_model == FINE_NAMES[fine[0]]


def test_hierarchy_rejects_class_in_two_superclasses():
    fine, coarse = labels(np.random.default_rng(0))
    coarse[0] = (coarse[0] + 1) % 20
    with pytest.raises(ValueError, match="two superclasses"):
        build_hierarchy(fine, coarse, FINE_NAMES, COARSE_NAMES)


def test_load_official_layout(tmp_path):
    """Rows of the official file are 1024 R, then 1024 G, then 1024 B values, row-major: image[y, x, c]."""
    img = np.random.default_rng(0).integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
    (tmp_path / "cifar-100-python").mkdir()
    with open(tmp_path / "cifar-100-python" / "train", "wb") as f:
        pickle.dump({"data": img.transpose(0, 3, 1, 2).reshape(2, -1), "fine_labels": [4, 7], "coarse_labels": [1, 2]}, f)
    x, fine, coarse = load_official(tmp_path, "train")
    assert np.array_equal(x, img) and list(fine) == [4, 7] and list(coarse) == [1, 2]
    assert x[0, 0, 1, 0] == img.transpose(0, 3, 1, 2).reshape(2, -1)[0, 1]  # red channel of pixel (y=0, x=1)
