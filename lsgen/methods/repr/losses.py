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
