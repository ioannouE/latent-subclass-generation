"""FALCON (Grcic, Gadetsky, Brbic, ICML 2024; github.com/mlbio-epfl/falcon, train.py + main.py). A classifier over K fine classes
whose fine -> coarse assignment M (a binary matrix) is re-estimated during training. Loss = l1 * coarse CE (log of the
probability mass of the coarse parent's fine classes) + l2 * (fine CE on EMA pseudo-labels restricted to the parent's fine classes
+ consistency with the predictions on the image's nearest same-coarse neighbours) + l3 * entropy regulariser on the mean prediction.
Needs the total number of fine classes K ("uses K"). Differences from the original: M is solved exactly with Hungarian matching
instead of Gurobi, and the neighbours come from the (frozen) features of the same pretrained backbone."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from lsgen.methods.repr.common import EMA, forward_fp16

BIG = 1e6  # forces every coarse class to receive its first fine class


def coarse_neighbors(x, make, k):
    """(N, k) indices of each row's k most cosine-similar rows with the same coarse label, itself excluded."""
    x, neighbors = F.normalize(torch.as_tensor(x, dtype=torch.float32), dim=1), np.zeros((len(x), k), dtype=np.int64)
    for c in np.unique(make):
        idx = np.flatnonzero(make == c)
        assert len(idx) > k, f"coarse class {c} has only {len(idx)} images"
        sim = x[idx] @ x[idx].T
        sim.fill_diagonal_(-2)
        neighbors[idx] = idx[sim.topk(k, dim=1).indices.numpy()]
    return neighbors


def assign_fine_to_coarse(cost, reg):
    """M (F, C) minimising -<cost, M> + reg * sum_c n_c^2 / C, with one coarse parent per fine class and n_c >= 1 fine classes
    per coarse class (n_c = column sum of M): the paper's MIP. The size penalty is convex, so the n-th fine class of a coarse class
    has marginal cost reg * (2n + 1) / C and the optimum is a linear assignment of fine classes to (coarse class, slot) pairs."""
    n_fine, n_coarse = cost.shape
    slots = n_fine - n_coarse + 1
    slot_cost = reg * (2 * np.arange(slots) + 1) / n_coarse
    slot_cost[0] -= BIG
    rows, cols = linear_sum_assignment((-cost[:, :, None] + slot_cost).reshape(n_fine, -1))
    m = torch.zeros(n_fine, n_coarse)
    m[rows, cols // slots] = 1
    return m


def coarse_log_prob(logits, m):
    """(N, C) log of the probability mass that the softmax over `logits` (N, F) puts on each coarse class's fine classes (m: (F, C))."""
    in_class = logits[:, :, None].masked_fill(m[None] == 0, float("-inf"))
    return torch.logsumexp(in_class, dim=1) - torch.logsumexp(logits, dim=1, keepdim=True)


class Falcon:
    """Needs batches (strong view, weak view, weak views of the neighbours (N, k, 3, H, W), make)."""

    def __init__(self, net, num_fine, num_coarse, device, lambdas, temperature, beta_reg, solve_every, soft_epochs, ema_decay):
        self.net, self.ema, self.num_coarse, self.lambdas, self.temperature = net, EMA(net, ema_decay), num_coarse, lambdas, temperature
        self.beta_reg0, self.solve_every, self.soft_epochs, self.device = beta_reg, solve_every, soft_epochs, device
        self.m = assign_fine_to_coarse(torch.randn(num_fine, num_coarse).softmax(-1).numpy(), beta_reg).to(device)

    def start_epoch(self, epoch):
        """epoch is 1-based. Soft pseudo-labels (tau 1 -> 0, cosine) and the size penalty are used in the first half, then hard labels."""
        soft = epoch <= self.soft_epochs
        self.tau = 0.5 * (1 + math.cos(math.pi * (epoch - 1) / self.soft_epochs)) if soft else 0.0
        self.beta_reg = self.beta_reg0 if soft else 0.0
        self.preds, self.makes = [], []

    def loss(self, batch):
        x, x_weak, neighbors, make = batch
        n, k = neighbors.shape[:2]
        l1, l2, l3 = self.lambdas
        logits_ema = self.ema(x_weak)
        neighbor_probs = self.ema(neighbors.flatten(0, 1)).softmax(-1).view(n, k, -1)
        self.preds.append(logits_ema.softmax(-1)), self.makes.append(make)

        logits = forward_fp16(self.net, x)
        probs = logits.softmax(-1)
        coarse_ce = F.cross_entropy(coarse_log_prob(logits, self.m), make)
        consistency = -torch.einsum("nc,nkc->nk", probs, neighbor_probs).clamp_min(1e-8).log().mean()
        q_soft = (logits_ema / self.temperature).masked_fill(self.m.T[make] == 0, float("-inf")).softmax(-1)
        pseudo = self.tau * q_soft + (1 - self.tau) * F.one_hot(q_soft.argmax(-1), q_soft.shape[-1]).float()
        fine_ce = -(F.log_softmax(logits, -1) * pseudo).sum(-1).mean()
        avg = probs.mean(0)
        reg = torch.special.entr(avg).sum().neg() + math.log(probs.shape[1])
        return l1 * coarse_ce + l2 * (fine_ce + consistency) + l3 * reg

    def end_step(self, i):
        """Every `solve_every` steps re-estimate M from the EMA predictions seen since the last solve (this epoch)."""
        if i > 0 and i % self.solve_every == 0:
            preds, make = torch.cat(self.preds), torch.cat(self.makes)
            cost = preds.T @ F.one_hot(make, self.num_coarse).float() / len(make)
            self.m = assign_fine_to_coarse(cost.cpu().numpy(), self.beta_reg).to(self.device)
            self.preds, self.makes = [], []
