"""Pieces shared by the representation objectives: fp16 forward and an exponential-moving-average copy of a network
(MaskCon's momentum key encoder, FALCON's EMA model)."""
import copy

import torch


def forward_fp16(net, x):
    """Network forward in fp16 autocast; the losses are computed outside it, in fp32."""
    with torch.autocast("cuda", dtype=torch.float16):
        return net(x).float()


class EMA:
    def __init__(self, net, decay):
        self.net, self.decay, self.student = copy.deepcopy(net).requires_grad_(False), decay, net

    @torch.no_grad()
    def update(self):
        for p_ema, p in zip(self.net.parameters(), self.student.parameters()):
            p_ema.lerp_(p, 1 - self.decay)

    @torch.no_grad()
    def __call__(self, x):
        return forward_fp16(self.net, x)
