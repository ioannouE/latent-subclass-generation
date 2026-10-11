"""Gate 0 (docs/EXPERIMENT_NHP.md): how much of the top within-make principal components of a frozen embedding do cheap nuisance proxies
explain? Proxies need no fine labels: colour of the car, its shape in the frame, background brightness (from pixels and the bbox), and
two pose proxies from flip-consistency of the embedding itself."""
import numpy as np
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pixel_proxies(img, box, hue_bins=8):
    """(17,) proxies of a PIL image and the car's bbox (x0, y0, x1, y1) in its pixels: colour = mean RGB, mean saturation / value and a
    saturation-weighted hue histogram of the box (3 + 2 + hue_bins); shape = log aspect ratio and area fraction of the box;
    background = mean and std of the grey value outside the box."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    rgb, hsv = np.asarray(img.convert("RGB"), np.float32), np.asarray(img.convert("RGB").convert("HSV"), np.float32)
    inside = np.zeros(rgb.shape[:2], bool)
    inside[y0:y1, x0:x1] = True
    car, hue_w = rgb[inside], hsv[inside]
    hist = np.bincount((hue_w[:, 0] / 256 * hue_bins).astype(int), weights=hue_w[:, 1] / 255, minlength=hue_bins)
    grey = rgb.mean(2)
    bg = grey[~inside] if (~inside).sum() > 0.01 * inside.size else grey.ravel()
    return np.concatenate([car.mean(0) / 255, hue_w[:, 1:].mean(0) / 255, hist / max(hist.sum(), 1e-6),
                           [np.log((x1 - x0) / (y1 - y0)), (x1 - x0) * (y1 - y0) / inside.size], [bg.mean() / 255, bg.std() / 255]])


def flip_proxies(emb, emb_flip):
    """(n, 2) for the images of ONE make (L2-normalised embeddings of the images and of their mirror images): cosine between the two
    (view symmetry: high for frontal / rear views) and the signed coordinate of x - flip(x) on its top principal direction
    (which way the car faces). Both are functions of the embedding, so they are reported apart from the pixel proxies."""
    diff = emb - emb_flip
    direction = np.linalg.svd(diff - diff.mean(0), full_matrices=False)[2][0]
    return np.stack([(emb * emb_flip).sum(1), diff @ direction], 1)


def within_make_pcs(emb, n_pcs):
    """(scores (n, n_pcs), eigenvalues (n_pcs,)) of the centred embeddings of one make."""
    z = emb - emb.mean(0)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    return (u * s)[:, :n_pcs], s[:n_pcs] ** 2


def cv_r2(scores, proxies, folds=5, seed=0):
    """(n_pcs,) out-of-fold R^2 of a ridge regression (standardised proxies, alpha by internal CV) onto each score column; negative = no signal."""
    pred = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))), proxies, scores,
                             cv=KFold(folds, shuffle=True, random_state=seed))
    return 1 - ((scores - pred) ** 2).sum(0) / ((scores - scores.mean(0)) ** 2).sum(0)


def explained(emb, proxies, n_pcs=5, folds=5, seed=0):
    """Share of the variance of the make's top n_pcs principal components that the proxies explain out of fold
    (eigenvalue-weighted mean of the per-PC R^2), and the R^2 of the first PC."""
    scores, eig = within_make_pcs(emb, n_pcs)
    r2 = cv_r2(scores, proxies, folds, seed)
    return float((eig * r2).sum() / eig.sum()), float(r2[0])
