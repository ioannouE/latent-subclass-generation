"""MaskCon (Feng & Patras, CVPR 2023; github.com/MrChenFeng/MaskCon_CVPR2023, models/model.py, mode "maskcon"): MoCo with soft
targets. For a key k, every queue entry of the same coarse class is a positive with weight exp((<k, q_j> - max_same) / t0);
entries of other coarse classes are negatives. Ported unchanged except that the queue starts as random vectors that
belong to no class (the original warms it up with keys of the first batches) and is smaller than the training set."""
import torch
import torch.nn.functional as F

from lsgen.methods.repr.common import EMA, forward_fp16


def maskcon_loss(q, k, queue, queue_make, make, t=0.1, t0=0.1, w=1.0):
    """q, k: (B, d) L2-normalised query / key embeddings of the same images; queue: (K, d) keys, queue_make: (K,) their
    coarse labels; make: (B,). Target over [own key, queue] = w * soft same-coarse weights + (1 - w) * own key only."""
    with torch.no_grad():
        same = make[:, None] == queue_make[None]
        s = (k @ queue.T / t0).masked_fill(~same, float("-inf"))
        top = s.max(1, keepdim=True).values
        top = top.masked_fill(top.isinf(), 0)  # no same-class entry in the queue: all weights 0
        soft = torch.cat([torch.ones_like(top), (s - top).exp()], 1)  # the own key has weight 1 (the maximum)
        target = w * soft / soft.sum(1, keepdim=True)
        target[:, 0] += 1 - w
    logits = torch.cat([(q * k).sum(1, keepdim=True), q @ queue.T], 1) / t
    return -(F.log_softmax(logits.float(), 1) * target).sum(1).mean()


class MaskCon:
    """Needs batches (query view, key view, make). Query = strong augmentation, key = weak, as in the original."""

    def __init__(self, net, dim, queue, momentum, t, t0, w, device):
        self.net, self.ema, self.t, self.t0, self.w = net, EMA(net, momentum), t, t0, w
        self.queue = F.normalize(torch.randn(queue, dim, device=device), dim=1)
        self.queue_make, self.ptr = torch.full((queue,), -1, device=device), 0

    def loss(self, batch):
        x_q, x_k, make = batch
        self.ema.update()
        q = F.normalize(forward_fp16(self.net, x_q), dim=1)
        k = F.normalize(self.ema(x_k), dim=1)
        loss = maskcon_loss(q, k, self.queue.clone(), self.queue_make.clone(), make, self.t, self.t0, self.w)
        n = len(k)
        assert len(self.queue) % n == 0, "the queue size must be a multiple of the batch size"  # as in the original
        self.queue[self.ptr:self.ptr + n], self.queue_make[self.ptr:self.ptr + n] = k, make
        self.ptr = (self.ptr + n) % len(self.queue)
        return loss

    def start_epoch(self, epoch):
        pass

    def end_step(self, i):
        pass
