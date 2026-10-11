"""Nuisance-factored hierarchical prototypes (NHP). Make labels and augmentations only: no hidden-subclass labels.

A small head on the backbone output gives three codes:
  z_make  SupCon on the make labels (R1).
  z_sub   per-make mixture of learnable prototypes. An image is softly assigned to the prototypes of its own make
          (cosine softmax); the assignment is trained by self-distillation between two views (EMA teacher, sharpened targets,
          as DINO), with nuisance-heavy views (colour permutation, strong colour jitter, background swap or mask, flip, crop)
          so that the assignment must be invariant to them. No equipartition constraint: subclass prevalence is part of what
          the benchmark measures. The anti-collapse term is the (weak) entropy of each make's mean assignment.
          A hierarchy hinge on assignment-weighted distances: Delta0 + m1 < Delta1 and Delta1 + m2 < Delta2 (Delta0 pairs
          sharing a prototype, Delta1 same make other prototype, Delta2 other make).
          Optional (flags): hard-concrete gates on the prototypes with an L0 penalty, so a make can end up with few active
          prototypes; a gradient-reversal adversary that predicts the augmentation parameters from z_sub.
  z_nuis  within-subclass variation, kept from collapsing by the VICReg variance and covariance terms. Not part of the
          embedding used for the subclass metrics.
Evaluation embeddings: z_sub (primary) and z_make + z_sub (secondary), each L2-normalised (see Encoder)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lsgen.methods.repr.common import EMA, forward_fp16
from lsgen.methods.repr.losses import supcon_loss

BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1  # hard-concrete distribution (Louizos et al. 2018)
EPS = 1e-6


class _Reverse(torch.autograd.Function):
    """Gradient reversal: identity forward, negated gradient backward."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


class Head(nn.Module):
    """Backbone feature -> [z_make | z_sub | z_nuis] (dims = (make, sub, nuis)). Also owns the prototypes (n_makes, n_proto, d_sub),
    the gate logits (sparse) and the adversary (adv_classes = number of classes of each augmentation parameter)."""

    def __init__(self, in_dim, hidden, dims, n_makes, n_proto, sparse=False, adv_classes=(), gate_init=3.0):
        super().__init__()
        self.dims, self.sparse, self.adv_classes = tuple(dims), sparse, tuple(adv_classes)
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, sum(self.dims)))
        self.prototypes = nn.Parameter(torch.randn(n_makes, n_proto, self.dims[1]))
        self.log_alpha = nn.Parameter(torch.full((n_makes, n_proto), gate_init)) if sparse else None
        self.adversary = nn.Sequential(nn.Linear(self.dims[1], 64), nn.ReLU(), nn.Linear(64, sum(adv_classes))) if adv_classes else None

    def forward(self, h):
        return self.body(h)

    def split(self, codes):
        return codes.split(self.dims, dim=1)

    def logits(self, z_sub, make):
        """(B, P) cosine similarity of each image to the prototypes of its own make."""
        protos = F.normalize(self.prototypes[make], dim=2)
        return torch.einsum("bd,bpd->bp", F.normalize(z_sub.float(), dim=1), protos)

    def log_gate(self, make, sample):
        """(B, P) log of the gates of the image's make: a hard-concrete sample (training) or its deterministic value; 0 without gates."""
        if not self.sparse:
            return 0.0
        a = self.log_alpha[make]
        if sample:
            u = torch.rand_like(a).clamp(EPS, 1 - EPS)
            a = (u.log() - (1 - u).log() + a) / BETA
        return (torch.sigmoid(a) * (ZETA - GAMMA) + GAMMA).clamp(EPS, 1).log()


def assignment_loss(logit_student, logit_teacher):
    """Self-distillation: cross-entropy between the (sharpened, no gradient) teacher assignment of the other view and the student's.
    Both are logits = cosine / temperature + log gate, over the prototypes of the image's make."""
    return -(logit_teacher.detach().softmax(1) * logit_student.log_softmax(1)).sum(1).mean()


def usage_loss(p, make):
    """Anti-collapse: minus the entropy of the mean assignment of each make with >= 2 images in the batch, averaged over images."""
    makes, inv = make.unique(return_inverse=True)
    count = torch.bincount(inv, minlength=len(makes)).float()
    mean = torch.zeros(len(makes), p.shape[1], device=p.device).index_add_(0, inv, p) / count[:, None]
    entropy = -(mean * (mean + EPS).log()).sum(1)
    keep = count >= 2
    return -(entropy * count)[keep].sum() / count[keep].sum().clamp_min(1)


def hierarchy_loss(z, p, make, m1, m2):
    """Hinges Delta0 + m1 < Delta1 and Delta1 + m2 < Delta2 on squared distances of the L2-normalised z. Pairs of the same make are weighted by
    the probability that both images take the same prototype (Delta0) or not (Delta1); Delta2 is the mean over pairs of different makes."""
    z = F.normalize(z.float(), dim=1)
    d = 2 - 2 * z @ z.T
    same_make = make[:, None] == make[None]
    pair = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    agree = p @ p.T
    w = torch.stack([agree * (same_make & pair), (1 - agree) * (same_make & pair), (~same_make).float()])
    if w.sum((1, 2)).min() < EPS:  # no pair of one kind in the batch
        return z.new_zeros(())
    d0, d1, d2 = (w * d).sum((1, 2)) / w.sum((1, 2))
    return F.relu(d0 + m1 - d1) + F.relu(d1 + m2 - d2)


def nuisance_loss(z, gamma=1.0):
    """VICReg variance (std of every dimension >= gamma) and covariance (decorrelated dimensions) terms on z_nuis."""
    z = z.float() - z.float().mean(0)
    var = F.relu(gamma - (z.var(0) + EPS).sqrt()).mean()
    cov = z.T @ z / (len(z) - 1)
    return var + (cov - torch.diag(cov.diagonal())).pow(2).sum() / z.shape[1]


def l0_penalty(log_alpha):
    """Expected number of open gates per make (hard-concrete L0 relaxation), averaged over makes."""
    return torch.sigmoid(log_alpha - BETA * math.log(-GAMMA / ZETA)).sum(1).mean()


def adversary_loss(head, z_sub, targets):
    """Cross-entropy of predicting each augmentation parameter (targets (B, F) class ids) from z_sub through gradient reversal."""
    out = head.adversary(_Reverse.apply(z_sub.float())).split(head.adv_classes, dim=1)
    return sum(F.cross_entropy(o, targets[:, f]) for f, o in enumerate(out))


def usage_share(p, make, n_makes):
    """(n_makes, P) mean assignment of the images of each make (0 for makes without images)."""
    total = torch.zeros(n_makes, p.shape[1], device=p.device).index_add_(0, make, p)
    return total / torch.bincount(make, minlength=n_makes).clamp_min(1)[:, None]


def active_prototypes(share, min_share):
    """(n_makes,) number of prototypes that take at least min_share of their make's mean assignment."""
    return (share >= min_share).sum(1)


class NHP:
    """Needs batches (view 1, view 2, augmentation parameters (B, 2, F), make). Loss weights w_*: 0 switches a term off."""

    def __init__(self, net, tau_student, tau_teacher, ema_decay, margins, temperature, active_min,
                 w_make, w_assign, w_usage, w_hierarchy, w_nuis, w_l0, w_adversary):
        self.net, self.ema, self.head, self.margins = net, EMA(net, ema_decay), net[1], margins
        self.tau_s, self.tau_t, self.temperature, self.active_min = tau_student, tau_teacher, temperature, active_min
        self.weights = {"make": w_make, "assign": w_assign, "usage": w_usage, "hierarchy": w_hierarchy, "nuis": w_nuis,
                        "l0": w_l0 if self.head.sparse else 0.0, "adversary": w_adversary if self.head.adversary is not None else 0.0}
        self.usage = torch.zeros_like(self.head.prototypes[:, :, 0])
        self.start_epoch(0)

    def loss(self, batch):
        v1, v2, params, make = batch
        b, x, m = len(make), torch.cat([v1, v2]), torch.cat([make, make])
        self.ema.update()
        z_make, z_sub, z_nuis = self.head.split(forward_fp16(self.net, x))
        teacher = self.ema.net[1]
        z_teacher = teacher.split(self.ema(x))[1]
        logit_s = self.head.logits(z_sub, m) / self.tau_s + self.head.log_gate(m, sample=True)
        logit_t = (teacher.logits(z_teacher, m) / self.tau_t + teacher.log_gate(m, sample=False)).roll(b, 0)  # the other view's
        p = logit_s.softmax(1)
        terms = {
            "make": lambda: supcon_loss(z_make, m, self.temperature),
            "assign": lambda: assignment_loss(logit_s, logit_t),
            "usage": lambda: usage_loss(p[:b], make),
            "hierarchy": lambda: hierarchy_loss(z_sub[:b], p[:b], make, *self.margins),
            "nuis": lambda: nuisance_loss(z_nuis),
            "l0": lambda: l0_penalty(self.head.log_alpha),
            "adversary": lambda: adversary_loss(self.head, z_sub, params.flatten(0, 1))}  # rows [view 1; view 2]
        self.terms = {k: terms[k]() for k, w in self.weights.items() if w}
        with torch.no_grad():
            self.usage.index_add_(0, make, logit_t[:b].softmax(1))
            self.entropy.append(-(p * (p + EPS).log()).sum(1).mean().item())
            self.log.append({k: v.item() for k, v in self.terms.items()})
        return sum(self.weights[k] * v for k, v in self.terms.items())

    def start_epoch(self, epoch):
        self.usage.zero_()
        self.entropy, self.log = [], []

    def end_step(self, i):
        pass

    def epoch_report(self):
        """Monitoring string for the epoch log: mean loss terms, assignment entropy, active prototypes per make (label-free)."""
        mean = {k: sum(t[k] for t in self.log) / len(self.log) for k in self.log[0]}
        share = self.usage / self.usage.sum(1, keepdim=True).clamp_min(EPS)
        seen = self.usage.sum(1) > 0
        active = active_prototypes(share, self.active_min)[seen].float()
        terms = " ".join(f"{k} {v:.3f}" for k, v in mean.items())
        return f" | {terms} | entropy {sum(self.entropy) / len(self.entropy):.3f} | active prototypes/make {active.mean():.2f} (max {int(active.max())})"


class Encoder(nn.Module):
    """Backbone + head, for embedding: [L2-normalised z_sub | L2-normalised z_make]. `exports` splits that into the
    primary embedding (z_sub) and the secondary one (z_make + z_sub)."""

    def __init__(self, model, head):
        super().__init__()
        self.model, self.head = model, head
        self.transform, self.family, self.input_size = model.transform, model.family, model.input_size

    def forward(self, x):
        z_make, z_sub, _ = self.head.split(self.head(self.model(x)))
        return torch.cat([F.normalize(z_sub.float(), dim=1), F.normalize(z_make.float(), dim=1)], dim=1)

    def exports(self, emb):
        return {"": emb[:, :self.head.dims[1]], "_cat": emb}
