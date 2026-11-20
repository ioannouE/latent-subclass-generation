"""ksub (known-K subclasses): geosub's residual head z = norm(f + g(f)) (lsgen/methods/geosub/model.py, imported unchanged) and K_c
prototypes per coarse class that are not learnable: k-means on the frozen train features of the class, then an EMA of the embeddings
softly assigned to them. K_c is the true number of subclasses, or a label-free K-hat (estimate_k). Prototypes are stored zero-padded as (n_coarse, K, d), K = max K_c; prototype g of class c exists iff g < K_c."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from lsgen.eval.representation import cluster_space, select_k
from lsgen.methods.geosub.model import GeoSub


def head(dim, hidden):
    """geosub's residual head (last layer zero-initialised: z = f before training), without its learnable prototypes."""
    return GeoSub(dim, hidden, torch.zeros(0, 0, dim))


def init_prototypes(f, coarse, k_c, seed, fit=None):
    """(n_coarse, K, d): L2-normalised centres of K_c = k_c[c] clusters of the features f (N, d) of each class c: k-means centroids (n_init 10),
    or, with fit (fit(z, k, seed) -> labels, e.g. diffusion maps from lsgen/eval/clustering.py, run on the class's PCA-50 as in R2), the mean
    of f over each cluster."""
    mu = torch.zeros(len(k_c), max(k_c), f.shape[1])
    for c, k in enumerate(k_c):
        x = f[coarse == c]
        if fit is None:
            centres = torch.from_numpy(KMeans(k, n_init=10, random_state=seed).fit(x.numpy()).cluster_centers_).float()
        else:
            labels = torch.from_numpy(fit(cluster_space(x.numpy(), np.zeros(len(x)), seed=seed), k, seed)).long()
            centres = torch.zeros(k, f.shape[1]).index_add_(0, labels, x) / torch.bincount(labels, minlength=k).clamp_min(1)[:, None]
        mu[c, :k] = F.normalize(centres, dim=1)
    return mu


def estimate_k(f, coarse, n_coarse, k_max, seed):
    """(n_coarse,) label-free K-hat per class: R2's estimator (select_k: silhouette sweep over 2..k_max, gap-statistic fallback) on the
    class's own PCA-50 of the features f (N, d), as R2's "ACC K-hat" (lsgen/eval/representation.py). Uses coarse labels only."""
    c = coarse.numpy()
    z = cluster_space(f.numpy(), c, seed=seed)
    return torch.tensor([select_k(z[c == k], k_max, seed) for k in range(n_coarse)])


def soft_assign(z, coarse, mu, k_c, tau_a=0.05):
    """(B, K) r_i = softmax_g cos(z_i, mu_{c_i, g}) / tau_a over the K_c prototypes of the image's own class; 0 on the padding."""
    cos = torch.einsum("bd,bkd->bk", z, mu[coarse])
    pad = torch.arange(mu.shape[1], device=z.device)[None] >= k_c[coarse][:, None]
    return (cos / tau_a).masked_fill(pad, float("-inf")).softmax(1)


def assignment_ari(z, f, coarse, mu, mu0, k_c, seed):
    """Prototype-staleness diagnostic: per coarse class, the ARI between the current assignments (z to the EMA prototypes mu) and (a) a fresh
    k-means on z (init_prototypes, same seed), (b) the initial k-means on f (f to mu0). Returns (ari_fresh, ari_init), each (n_coarse,), NaN where K_c = 1."""
    cur = soft_assign(z, coarse, mu, k_c).argmax(1).cpu().numpy()
    fresh_mu = init_prototypes(z.cpu(), coarse.cpu(), k_c.tolist(), seed).to(z.device)
    fresh = soft_assign(z, coarse, fresh_mu, k_c).argmax(1).cpu().numpy()
    init = soft_assign(f, coarse, mu0, k_c).argmax(1).cpu().numpy()
    c = coarse.cpu().numpy()
    return tuple(np.array([adjusted_rand_score(cur[c == k], ref[c == k]) if k_c[k] > 1 else np.nan for k in range(len(k_c))])
                 for ref in (fresh, init))


def assignment_entropy(z, coarse, mu, k_c, tau_a):
    """(n_coarse,) mean over the class's images of the entropy of r_i (soft_assign at tau_a) divided by log K_c: 0 = hard, 1 = uniform; NaN where K_c = 1."""
    r = soft_assign(z, coarse, mu, k_c, tau_a)
    h = -(r * r.clamp_min(1e-12).log()).sum(1)
    return np.array([(h[coarse == k].mean() / np.log(k_c[k].item())).item() if k_c[k] > 1 else np.nan for k in range(len(k_c))])


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
