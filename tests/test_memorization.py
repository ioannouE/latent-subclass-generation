import numpy as np
from PIL import Image

from lsgen.eval.memorization import copy_summary, nearest_similarity, nn_grid

rng = np.random.default_rng(0)


def test_nearest_similarity_exact_copy_and_orthogonal():
    train = np.eye(5)
    sim, nn = nearest_similarity(3 * train[[2, 0]], train)  # scale does not matter
    assert np.allclose(sim, 1) and nn.tolist() == [2, 0]
    sim, _ = nearest_similarity(np.array([[1.0, 1, 0, 0, 0]]), train)
    assert np.isclose(sim[0], 1 / np.sqrt(2))
    q = rng.normal(size=(7, 5))
    assert np.allclose(nearest_similarity(q, train, chunk=3)[0], nearest_similarity(q, train)[0])  # chunking is exact


def test_copy_summary_separates_copying_from_the_null():
    train = rng.normal(size=(500, 128))
    null = nearest_similarity(rng.normal(size=(200, 128)), train)[0]
    copies = nearest_similarity(train[:100] + 0.01 * rng.normal(size=(100, 128)), train)[0]
    s = copy_summary(copies, null)
    assert s["frac_gt"] == 1 and s["null_frac_gt"] == 0 and s["sim_mean"] > 0.99 > s["null_sim_mean"]


def test_nn_grid_writes_most_similar_pairs_first(tmp_path):
    paths = []
    for i, v in enumerate((0, 255)):
        Image.fromarray(np.full((8, 8, 3), v, np.uint8)).save(tmp_path / f"{i}.png")
        paths.append(tmp_path / f"{i}.png")
    nn_grid(paths, paths, np.array([0.2, 0.9]), np.array([0, 1]), tmp_path / "g.png", size=8)
    g = np.asarray(Image.open(tmp_path / "g.png"))
    assert g.shape == (8, 64, 3) and g[0, 0, 0] == 255 and g[0, 8, 0] == 255 and g[0, 16, 0] == 0  # sample 1 first, then 0
