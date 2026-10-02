"""G7: copying. Cosine similarity of every sample to its nearest train image in SSCD space, against the null of
test -> train similarities (the same real-vs-real quantity: how close an unseen real image is to the train set)."""
import numpy as np
from PIL import Image


def nearest_similarity(queries, reference, chunk=1024):
    """Max cosine similarity of each query to the reference set, and the index of that reference row."""
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    r = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    best = [(s.max(1), s.argmax(1)) for s in (q[i:i + chunk] @ r.T for i in range(0, len(q), chunk))]
    return np.concatenate([b[0] for b in best]), np.concatenate([b[1] for b in best])


def copy_summary(sim, null_sim, threshold=0.5):
    """Mean and share above `threshold` of the samples' nearest-train similarity, with the same numbers for the null."""
    return {"sim_mean": sim.mean(), "frac_gt": (sim > threshold).mean(),
            "null_sim_mean": null_sim.mean(), "null_frac_gt": (null_sim > threshold).mean()}


def nn_grid(sample_paths, train_paths, sim, nn_idx, path, top=64, size=128, pairs=4):
    """Image of the `top` most similar (sample | nearest train image) pairs, `pairs` pairs per row."""
    order = np.argsort(-sim)[:top]
    canvas = Image.new("RGB", (2 * pairs * size, -(-len(order) // pairs) * size))
    for n, i in enumerate(order):
        for j, p in enumerate((sample_paths[i], train_paths[nn_idx[i]])):
            with Image.open(p) as im:
                canvas.paste(im.convert("RGB").resize((size, size)), ((n % pairs * 2 + j) * size, n // pairs * size))
    canvas.save(path)
