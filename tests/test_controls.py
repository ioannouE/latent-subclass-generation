import numpy as np
from PIL import Image

from lsgen.eval import controls as C

rng = np.random.default_rng(0)
make_of_fine = np.array([0, 0, 0, 1, 1])  # makes 0 (classes 0-2) and 1 (classes 3-4)
train_fine = np.array([0] * 6 + [1] * 3 + [2] * 1 + [3] * 2 + [4] * 2)


def test_fine_weights_normalise_inside_each_make():
    for mode in ("prevalence", "uniform", "majority"):
        w = C.fine_weights(train_fine, make_of_fine, mode)
        assert np.isclose(w[:3].sum(), 1) and np.isclose(w[3:].sum(), 1)
    assert np.allclose(C.fine_weights(train_fine, make_of_fine, "prevalence")[:3], [0.6, 0.3, 0.1])
    assert np.allclose(C.fine_weights(train_fine, make_of_fine, "uniform")[:3], 1 / 3)
    assert C.fine_weights(train_fine, make_of_fine, "majority")[:3].tolist() == [1, 0, 0]


def test_draw_set_keeps_the_requested_make_and_majority_uses_one_class():
    ids_by_fine = {g: np.array([f"img{g}_{j}" for j in range(5)]) for g in range(5)}
    requested = np.array([0] * 200 + [1] * 50)
    got = C.draw_set(requested, make_of_fine, C.fine_weights(train_fine, make_of_fine, "majority"), C.random_image(ids_by_fine), rng)
    classes = np.array([int(i[3]) for i in got])
    assert set(make_of_fine[classes[:200]]) == {0} and set(make_of_fine[classes[200:]]) == {1}
    assert set(classes[:200]) == {0}  # largest class of make 0
    uniform = C.draw_set(requested, make_of_fine, C.fine_weights(train_fine, make_of_fine, "uniform"), C.random_image(ids_by_fine), rng)
    assert set(int(i[3]) for i in uniform[:200]) == {0, 1, 2}


def test_prototype_collapse_repeats_the_medoid():
    x = np.array([[1.0, 0], [0.9, 0.1], [0, 1.0]])
    assert C.medoid(x) == 0 or C.medoid(x) == 1  # the two similar points; never the outlier
    image_of = C.medoid_image({0: np.array(["a", "b", "c"])}, {0: x})
    assert {image_of(0, rng) for _ in range(5)} == {"a" if C.medoid(x) == 0 else "b"}


def test_split_counts_and_support_controls():
    assert C.split_counts(16, [3], 1.0) == [16] and C.split_counts(16, [3, 4], 0.2) == [3, 13]
    assert sorted(set(C.copy_support_ids(["s1", "s2"], 16, rng))) == ["s1", "s2"] and len(C.copy_support_ids(["s1"], 16, rng)) == 16
    feats = np.array([[1.0, 0], [0, 1.0], [0.9, 0.1], [-1.0, 0]])
    assert C.retrieve_nn_ids(np.array([[1.0, 0.05]]), ["a", "b", "c", "d"], feats, 2) == ["a", "c"]
    assert C.collapse_ids([3, 4], [2, 1], {3: "x", 4: "y"}) == ["x", "x", "y"]
    assert len(C.real_ids([3, 4], [2, 1], {3: np.array(["p", "q", "r"]), 4: np.array(["s"])}, rng)) == 3


def test_image_transforms_keep_size_and_change_pixels():
    im = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    assert C.augment(im, rng).size == (128, 128) and C.degrade(im).size == (128, 128)
    assert np.abs(np.asarray(C.degrade(im), float) - np.asarray(im, float)).mean() > 10  # blurred noise differs a lot
