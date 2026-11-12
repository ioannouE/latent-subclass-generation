"""ksub (known-K subclasses): geosub's residual head z = norm(f + g(f)) (lsgen/methods/geosub/model.py, imported unchanged) and K_c
prototypes per coarse class that are not learnable: k-means on the frozen train features of the class, then an EMA of the embeddings
softly assigned to them. Prototypes are stored zero-padded as (n_coarse, K, d), K = max K_c; prototype g of class c exists iff g < K_c."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from lsgen.methods.geosub.model import GeoSub


def head(dim, hidden):
    """geosub's residual head (last layer zero-initialised: z = f before training), without its learnable prototypes."""
    return GeoSub(dim, hidden, torch.zeros(0, 0, dim))


def init_prototypes(f, coarse, k_c, seed):
    """(n_coarse, K, d): L2-normalised k-means centroids (n_init 10) of the features f (N, d) of each class c, K_c = k_c[c] of them."""
    mu = torch.zeros(len(k_c), max(k_c), f.shape[1])
    for c, k in enumerate(k_c):
        centres = KMeans(k, n_init=10, random_state=seed).fit(f[coarse == c].numpy()).cluster_centers_
        mu[c, :k] = F.normalize(torch.from_numpy(centres).float(), dim=1)
    return mu


def soft_assign(z, coarse, mu, k_c, tau_a=0.05):
    """(B, K) r_i = softmax_g cos(z_i, mu_{c_i, g}) / tau_a over the K_c prototypes of the image's own class; 0 on the padding."""
    cos = torch.einsum("bd,bkd->bk", z, mu[coarse])
    pad = torch.arange(mu.shape[1], device=z.device)[None] >= k_c[coarse][:, None]
    return (cos / tau_a).masked_fill(pad, float("-inf")).softmax(1)


def assignment_ari(z, f, coarse, mu, mu0, k_c, seed):
    """Prototype-staleness diagnostic: mean over classes with K_c > 1 of the ARI between the current assignments (z to the EMA prototypes mu)
    and (a) a fresh k-means on z (init_prototypes, same seed), (b) the initial k-means on f (f to mu0). Returns (ari_fresh, ari_init)."""
    cur = soft_assign(z, coarse, mu, k_c).argmax(1).cpu().numpy()
    fresh_mu = init_prototypes(z.cpu(), coarse.cpu(), k_c.tolist(), seed).to(z.device)
    fresh = soft_assign(z, coarse, fresh_mu, k_c).argmax(1).cpu().numpy()
    init = soft_assign(f, coarse, mu0, k_c).argmax(1).cpu().numpy()
    c = coarse.cpu().numpy()
    classes = [k for k in range(len(k_c)) if k_c[k] > 1]
    return tuple(float(np.mean([adjusted_rand_score(cur[c == k], ref[c == k]) for k in classes])) for ref in (fresh, init))


def group_weights(r, coarse, n_coarse):
    """(B, n_coarse * K) weight of each image in each (class c, prototype g) group, column c * K + g: r_ig in its own class, 0 elsewhere."""
    return (F.one_hot(coarse, n_coarse)[:, :, None] * r[:, None, :]).flatten(1)


@torch.no_grad()
def ema_update(mu, z, coarse, r, momentum=0.99):
    """In place: every prototype with weight in the batch moves towards the r-weighted mean of the batch embeddings, then is re-normalised."""
    a = group_weights(r, coarse, len(mu))
    w, flat = a.sum(0), mu.view(-1, mu.shape[2])
    hit = w > 0
    flat[hit] = F.normalize(momentum * flat[hit] + (1 - momentum) * (a.T[hit] @ z) / w[hit, None], dim=1)
