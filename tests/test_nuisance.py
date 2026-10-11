import numpy as np
from PIL import Image

from lsgen.features.nuisance import explained, flip_proxies, pixel_proxies


def test_pixel_proxies_read_colour_shape_and_background():
    img = Image.new("RGB", (100, 100), (50, 50, 50))
    img.paste((255, 0, 0), (20, 30, 80, 70))  # a red 60 x 40 box on dark grey
    p = pixel_proxies(img, (20, 30, 80, 70))
    assert p.shape == (17,) and np.allclose(p[:3], [1, 0, 0]) and np.isclose(p[13], np.log(60 / 40)) and np.isclose(p[14], 0.24)
    assert np.isclose(p[15], 50 / 255, atol=1e-3) and p[16] < 1e-3 and np.isclose(p[5:13].sum(), 1)  # hue histogram sums to 1


def test_explained_finds_a_planted_nuisance_and_not_a_shuffled_one():
    rng, n = np.random.default_rng(0), 300
    proxies = rng.normal(size=(n, 4))
    emb = rng.normal(size=(n, 20)) * 0.3
    emb[:, 0] += 3 * proxies[:, 1]  # the top within-make direction is a function of one proxy
    hit, _ = explained(emb, proxies)
    miss, _ = explained(emb, proxies[rng.permutation(n)])
    assert hit > 0.7 and miss < 0.05


def test_flip_proxies_symmetric_images_and_orientation():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(50, 8))
    c, a = flip_proxies(e, e).T  # mirror image identical to the image: cosine of the (unnormalised) embeddings with itself, no asymmetry
    assert np.allclose(c, (e * e).sum(1)) and np.allclose(a, 0)
    direction, side = rng.normal(size=8), rng.choice([-1, 1], 50)
    mirrored = e + np.outer(side, direction)  # the sign of the asymmetry follows which way each car faces
    assert abs(np.corrcoef(flip_proxies(e, mirrored)[:, 1], side)[0, 1]) > 0.95
