"""Geometry-preserving subprototypes (geosub): a residual head on frozen backbone features and P subprototypes per coarse class.

z = normalise(f + g(f)), f the L2-normalised frozen feature. The last layer of g starts at zero, so before training z = f exactly
(the L_geo-only run checks that the head and normalisation lose nothing). Coarse labels only; the trainer sees features and coarse
labels, nothing else, so it works for any label scheme or dataset with cached features."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_prototypes(f, coarse, n_coarse, n_proto, generator):
    """(n_coarse, n_proto, d) initial prototypes: n_proto random features of each class (with replacement if it has fewer)."""
    out = torch.empty(n_coarse, n_proto, f.shape[1])
    for c in range(n_coarse):
        rows = torch.nonzero(coarse == c).squeeze(1)
        out[c] = f[rows[torch.randint(len(rows), (n_proto,), generator=generator)]]
    return out


class GeoSub(nn.Module):
    def __init__(self, dim, hidden, prototypes, n_valid=None):
        """prototypes (n_coarse, P, d); n_valid (n_coarse,): class c uses only its first n_valid[c] prototypes (default: all P)."""
        super().__init__()
        self.g = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.g[2].weight)
        nn.init.zeros_(self.g[2].bias)
        self.prototypes = nn.Parameter(prototypes.clone())
        n_coarse, n_proto = prototypes.shape[:2]
        n_valid = torch.full((n_coarse,), n_proto) if n_valid is None else torch.as_tensor(n_valid)
        self.register_buffer("valid", torch.arange(n_proto)[None] < n_valid[:, None])

    def forward(self, f):
        return F.normalize(f + self.g(f), dim=1)

    def cosine(self, z, coarse):
        """(B, P) cosine similarity of each embedding to the prototypes of its own coarse class; -1e4 for the unused ones (finite, so that
        their softmax weight is exactly 0 and 0 * log p stays 0 in the losses)."""
        cos = torch.einsum("bd,bpd->bp", z, F.normalize(self.prototypes[coarse], dim=2))
        return cos.masked_fill(~self.valid[coarse], -1e4)

    @torch.no_grad()
    def assign(self, z, coarse, active=None):
        """(B,) index of the most similar prototype of the image's class; only among the active ones if active (n_coarse, P) is given."""
        cos = self.cosine(z, coarse)
        return (cos if active is None else cos.masked_fill(~active[coarse], float("-inf"))).argmax(1)


def active_prototypes(assign, coarse, n_coarse, n_proto, min_count):
    """(n_coarse, P) True for prototypes that are the nearest prototype of at least min_count images (label-free K-hat = row sums)."""
    counts = torch.zeros(n_coarse * n_proto, dtype=torch.long, device=assign.device)
    counts.index_add_(0, coarse * n_proto + assign, torch.ones_like(assign))
    return counts.view(n_coarse, n_proto) >= min_count
