"""ksub loss terms on one batch: embeddings z (B, d), frozen L2-normalised features f (B, d), coarse labels (B,) and soft assignments
r (B, K) to the prototypes of each image's own class (soft_assign, no gradient). A weight of 0 switches a term off."""
import torch
import torch.nn.functional as F

from lsgen.methods.repr.ksub.model import group_weights
from lsgen.methods.repr.losses import supcon_loss

TERMS = ("coarse", "sub", "var")


def same_class(coarse):
    """(B, B) True for pairs j != i with the same coarse label."""
    return (coarse[:, None] == coarse[None, :]) & ~torch.eye(len(coarse), dtype=torch.bool, device=coarse.device)


def pair_weights(r, coarse):
    """(B, B) w_ij = r_i . r_j for pairs of the same coarse class, 0 for the others (and for i = j)."""
    return (r @ r.T) * same_class(coarse)


def sub_loss(z, coarse, r, tau=0.1):
    """Soft within-class contrastive: mean_i -sum_j w_ij log softmax_j(z_i . z_j / tau) / sum_j w_ij, the softmax over the other images of
    i's class. Same-subclass pairs are pulled together, different subclasses of the class pushed apart; other classes are not involved."""
    cand = same_class(coarse)
    log_p = (z @ z.T / tau).masked_fill(~cand, float("-inf")).log_softmax(1).masked_fill(~cand, 0)
    w = pair_weights(r, coarse)
    keep = w.sum(1) > 0
    return -((w * log_p).sum(1)[keep] / w.sum(1)[keep]).mean() if keep.any() else z.new_zeros(())


def trace_cov(x, a):
    """(G,) trace of the a-weighted covariance of x (B, d) for each column of a (B, G): sum_i a_i ||x_i - mean||^2 / sum_i a_i."""
    w = a.sum(0)
    mean = a.T @ x / w[:, None]
    return a.T @ (x * x).sum(1) / w - (mean * mean).sum(1)


def var_loss(z, f, coarse, r):
    """Within-subclass variance floor: per (class, prototype) group with sum_i r_ig >= 2, relu(1 - tr Cov_r(z) / tr Cov_r(f)) with the
    same images and weights, averaged over groups. Zero while z keeps at least the frozen spread of every soft subclass."""
    a = group_weights(r, coarse, int(coarse.max()) + 1)
    a = a[:, a.sum(0) >= 2]
    if not a.shape[1]:
        return z.new_zeros(())
    return F.relu(1 - trace_cov(z, a) / trace_cov(f, a).clamp_min(1e-8)).mean()


def ksub_loss(z, f, coarse, r, weights, tau=0.1, temperature=0.1):
    """L = sum of w * term over the terms with a non-zero weight (weights: {term: w}); also returns the unweighted terms for logging.
    L_coarse is SupCon on the coarse labels at `temperature`, L_sub uses `tau`."""
    terms = {"coarse": lambda: supcon_loss(z, coarse, temperature),
             "sub": lambda: sub_loss(z, coarse, r, tau),
             "var": lambda: var_loss(z, f, coarse, r)}
    terms = {k: terms[k]() for k in TERMS if weights.get(k, 0)}
    return sum(weights[k] * v for k, v in terms.items()), terms
