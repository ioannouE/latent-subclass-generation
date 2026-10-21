"""The four geosub loss terms on a batch of two views (f1, f2) of the same images: frozen L2-normalised features (B, d), their
embeddings z1, z2 (B, d) and coarse labels (B,). A weight of 0 switches a term off."""
import torch
import torch.nn.functional as F

from lsgen.methods.repr.losses import supcon_loss
from lsgen.methods.repr.nhp import assignment_loss, usage_loss

TERMS = ("geo", "supcon", "sub", "var")


def geo_loss(z, f, tau=0.1):
    """Relational distillation of the frozen geometry: per anchor, KL(p_f || p_z), p = softmax over the other rows of the batch of
    cosine / tau. Keeps each image's neighbour ranking; zero iff z has the same similarity profile as f."""
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    log_pf = (f @ f.T / tau).masked_fill(self_mask, float("-inf")).log_softmax(1)
    log_pz = (z @ z.T / tau).masked_fill(self_mask, float("-inf")).log_softmax(1)
    return (log_pf.exp() * (log_pf - log_pz)).masked_fill(self_mask, 0).sum(1).mean()


def sub_loss(cos1, cos2, coarse, tau_student=0.1, tau_teacher=0.04, usage=0.1):
    """Augmentation-consistent assignment to the subprototypes of the image's class (cos: (B, P) cosines of either view): each view's
    assignment is trained towards the sharpened, no-gradient assignment of the other view (swapped prediction), plus `usage` times
    minus the entropy of each class's mean assignment against collapse onto one prototype (no equipartition: prevalence is measured)."""
    swapped = assignment_loss(cos1 / tau_student, cos2 / tau_teacher) + assignment_loss(cos2 / tau_student, cos1 / tau_teacher)
    return swapped / 2 + usage * usage_loss((cos1 / tau_student).softmax(1), coarse)


def var_loss(z, f, coarse):
    """Within-class variance floor: per class with >= 2 images in the batch, relu(1 - trace cov(z_c) / trace cov(f_c)), averaged over
    classes. Zero while z keeps at least the frozen within-class spread."""
    out = [F.relu(1 - z[coarse == c].var(0).sum() / f[coarse == c].var(0).sum().clamp_min(1e-8))
           for c in coarse.unique() if (coarse == c).sum() >= 2]
    return torch.stack(out).mean() if out else z.new_zeros(())


def geosub_loss(model, f1, f2, coarse, weights, tau_geo, tau_student, tau_teacher, usage, temperature):
    """Weighted sum of the terms with a non-zero weight (weights: {term: w}); also returns the unweighted terms for logging."""
    z1, z2 = model(f1), model(f2)
    terms = {
        "geo": lambda: (geo_loss(z1, f1, tau_geo) + geo_loss(z2, f2, tau_geo)) / 2,
        "supcon": lambda: supcon_loss(torch.cat([z1, z2]), coarse.repeat(2), temperature),
        "sub": lambda: sub_loss(model.cosine(z1, coarse), model.cosine(z2, coarse), coarse, tau_student, tau_teacher, usage),
        "var": lambda: (var_loss(z1, f1, coarse) + var_loss(z2, f2, coarse)) / 2}
    terms = {k: terms[k]() for k in TERMS if weights.get(k, 0)}
    return sum(weights[k] * v for k, v in terms.items()), terms
