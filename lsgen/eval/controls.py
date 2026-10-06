"""Controls (docs/PLAN.md A5, docs/METRICS.md): trivial "generators" with a known defect, scored by the same pipeline as a
real method. Each function returns the ids of the real bbox15 images (or transformed copies of them) that make up the set.
Everything is seeded through the rng that is passed in."""
import io

import numpy as np
from PIL import Image, ImageFilter


# ---- use case A: one image per requested make (the requested makes follow the make prevalence of test)
def fine_weights(train_fine, make_of_fine, mode):
    """Probability of every fine class inside its make: train prevalence, uniform, or only the largest class."""
    counts = np.bincount(train_fine, minlength=len(make_of_fine)).astype(float)
    w = {"prevalence": counts, "uniform": np.ones_like(counts)}.get(mode)
    if w is None:  # majority
        w = np.zeros_like(counts)
        for m in np.unique(make_of_fine):
            classes = np.flatnonzero(make_of_fine == m)
            w[classes[counts[classes].argmax()]] = 1
    for m in np.unique(make_of_fine):
        w[make_of_fine == m] /= w[make_of_fine == m].sum()
    return w


def draw_set(requested, make_of_fine, weights, image_of, rng):
    """For every requested make: a fine class drawn with `weights` inside the make, then image_of(class, rng) -> image id.
    C1/C6: prevalence + random train image. C3: uniform + random train image. C5: majority + random train image.
    C4: prevalence + the class medoid."""
    out = np.empty(len(requested), object)
    for i, m in enumerate(requested):
        classes = np.flatnonzero(make_of_fine == m)
        out[i] = image_of(rng.choice(classes, p=weights[classes]), rng)
    return out


def random_image(ids_by_fine):
    return lambda g, rng: rng.choice(ids_by_fine[g])


def medoid(x):
    """Index of the row with the smallest mean cosine distance to the others."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return int(np.argmax((x @ x.T).sum(1)))


def medoid_image(ids_by_fine, feats_by_fine):
    """image_of for C4: the medoid of every fine class, computed once."""
    fixed = {g: ids_by_fine[g][medoid(feats_by_fine[g])] for g in ids_by_fine}
    return lambda g, rng: fixed[g]


def degrade(im, sigma=2, quality=20):
    """C6: Gaussian blur followed by a JPEG round trip."""
    buf = io.BytesIO()
    im.filter(ImageFilter.GaussianBlur(sigma)).save(buf, "JPEG", quality=quality)
    return Image.open(buf).convert("RGB")


# ---- use case B: N images per episode
def split_counts(n, fine_ids, ratio):
    """Images per subclass of an episode: all n for a single subclass, round(n * ratio) / rest for a mixed pair (a, b)."""
    return [n] if len(fine_ids) == 1 else [round(n * ratio), n - round(n * ratio)]


def copy_support_ids(support, n, rng):
    """C1b: the support images, cycled to n (each is augmented when written; see augment)."""
    return list(rng.permutation([support[i % len(support)] for i in range(n)]))


def retrieve_nn_ids(support_feats, pool_ids, pool_feats, n):
    """C1b: the n pool images closest (cosine) to the mean of the support embeddings."""
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
    return list(np.asarray(pool_ids)[np.argsort(-(unit(pool_feats) @ unit(unit(support_feats).mean(0))))[:n]])


def real_ids(fine_ids, counts, ids_by_fine, rng):
    """C2: other real images of the support's subclass (no overlap with S or T: they come from the other split)."""
    return [i for g, c in zip(fine_ids, counts) for i in rng.choice(ids_by_fine[g], c, replace=c > len(ids_by_fine[g]))]


def collapse_ids(fine_ids, counts, medoids):
    """C4: the medoid of each subclass repeated."""
    return [medoids[g] for g, c in zip(fine_ids, counts) for _ in range(c)]


def augment(im, rng):
    """Mild augmentation of a copied support image: random flip, random crop of 70-100% of the side, brightness x0.85-1.15."""
    w = im.width
    side = round(w * rng.uniform(0.7, 1))
    x0, y0 = rng.integers(0, w - side + 1, 2)
    im = im.crop((x0, y0, x0 + side, y0 + side)).resize((w, w), Image.LANCZOS)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return Image.fromarray(np.clip(np.asarray(im) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8))
