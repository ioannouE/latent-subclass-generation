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
    def __init__(self, dim, hidden, prototypes):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.g[2].weight)
        nn.init.zeros_(self.g[2].bias)
        self.prototypes = nn.Parameter(prototypes.clone())

    def forward(self, f):
        return F.normalize(f + self.g(f), dim=1)

    def cosine(self, z, coarse):
        """(B, P) cosine similarity of each embedding to the prototypes of its own coarse class."""
        return torch.einsum("bd,bpd->bp", z, F.normalize(self.prototypes[coarse], dim=2))

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
