"""Contrastive losses on a batch of two views, rows [view 1; view 2] (2B, d), so image i is rows i and i + B."""
import torch
import torch.nn.functional as F

from lsgen.methods.repr.common import forward_fp16


def supcon_loss(z, labels, temperature=0.1):
    """Supervised contrastive loss (Khosla et al. 2020): the positives of an anchor are all other rows with its label.
    With one distinct label per image (both views share it) this is SimCLR's NT-Xent loss."""
    z = F.normalize(z.float(), dim=1)
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = (z @ z.T / temperature).masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    return -(log_prob.masked_fill(~pos, 0).sum(1) / pos.sum(1)).mean()


def contrastive_loss(z, make, w_supcon, w_simclr, temperature=0.1):
    """w_supcon * SupCon on the coarse (make) labels + w_simclr * SimCLR on image identity; a weight of 0 switches it off.
    make: (B,) coarse labels of the B images."""
    instance = torch.arange(len(make), device=z.device)
    loss = z.new_zeros(())
    if w_supcon:
        loss = loss + w_supcon * supcon_loss(z, make.repeat(2), temperature)
    if w_simclr:
        loss = loss + w_simclr * supcon_loss(z, instance.repeat(2), temperature)
    return loss


class Contrastive:
    """SupCon / SimCLR / SupCon+SimCLR objective. Needs batches (view 1, view 2, make)."""

    def __init__(self, net, supcon, simclr, temperature):
        self.net, self.weights, self.temperature = net, (supcon, simclr), temperature

    def loss(self, batch):
        v1, v2, make = batch
        return contrastive_loss(forward_fp16(self.net, torch.cat([v1, v2])), make, *self.weights, self.temperature)

    def start_epoch(self, epoch):
        pass

    def end_step(self, i):
        pass


def sigreg_loss(z, num_slices=1024, num_points=17):
    """SIGReg (Balestriero & LeCun 2025, LeJEPA): Epps-Pulley statistic of random 1-D projections of z (N, d) against N(0, 1),
    averaged over slices. Zero iff every projection is standard normal, i.e. z is isotropic Gaussian."""
    z = z.float()
    dirs = torch.randn(z.shape[1], num_slices, device=z.device)
    proj = z @ (dirs / dirs.norm(dim=0))
    t = torch.linspace(-5, 5, num_points, device=z.device)
    phi = torch.exp(-t ** 2 / 2)  # characteristic function of N(0, 1)
    w = torch.full_like(t, (t[1] - t[0]).item())  # trapezoid weights times the Gaussian window
    w[[0, -1]] /= 2
    arg = proj[:, :, None] * t
    err = (arg.cos().mean(0) - phi) ** 2 + arg.sin().mean(0) ** 2
    return ((err * phi * w).sum(1) * len(z)).mean()


def lejepa_loss(z, lam=0.05, num_slices=1024, num_points=17):
    """(1 - lam) * invariance + lam * SIGReg. z: (V, B, d) embeddings of V views; invariance = mean squared distance of each
    view to the mean over views."""
    invariance = (z.float().mean(0, keepdim=True) - z.float()).square().mean()
    sigreg = torch.stack([sigreg_loss(v, num_slices, num_points) for v in z]).mean()
    return (1 - lam) * invariance + lam * sigreg


class LeJEPA:
    """LeJEPA objective (no labels: the make in the batch is ignored). Needs batches (view 1, view 2, make)."""

    def __init__(self, net, lam, num_slices, num_points):
        self.net, self.args = net, (lam, num_slices, num_points)

    def loss(self, batch):
        v1, v2, _ = batch
        return lejepa_loss(torch.stack(forward_fp16(self.net, torch.cat([v1, v2])).chunk(2)), *self.args)

    def start_epoch(self, epoch):
        pass

    def end_step(self, i):
        pass
