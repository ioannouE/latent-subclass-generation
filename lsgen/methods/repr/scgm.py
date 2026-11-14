"""SCGM-G (Ni et al., ICLR 2022; github.com/nijingchao/SCGM, scgm_g/scgm_resnet.py, train_scgm_g.py, sinkhornknopp.py). A
superclass-conditional Gaussian mixture over K subclasses shared by all coarse classes: learnable subclass means mu_z (K, d) and
coarse-class embeddings mu_y (C, d), p(z|x) = softmax_k cos(x, mu_z) / tau and p(y|z) = softmax_c (mu_y_c . mu_z) (both normalised).
E-step (every `estep_every` epochs, over the whole train set): hard pseudo-subclass q_i = argmax of the Sinkhorn-Knopp equipartition of
p(z|x_i) p(y_i|z) (sharpness lmd). M-step loss = alpha (CE(p(z|x), q) - log p(y|q)) + CE(relu(x) . mu_y, y).
Needs the total number of subclasses K ("uses K"). Their defaults: no self-distillation (beta 1), no angular normalisation, norm_type
"logit". Differences from the original: the encoder is the shared fine-tuning recipe (configs/repr_baselines*.yaml: backbone, projection
head, optimiser, schedule) instead of a ResNet-50 from scratch with SGD."""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lsgen.methods.repr.common import forward_fp16

log = logging.getLogger(__name__)


def glorot(rows, cols):
    """Glorot & Bengio (AISTATS 2010) uniform init, as the original mu_y and mu_z."""
    r = np.sqrt(6.0 / (rows + cols))
    return 2 * r * torch.rand(rows, cols) - r


class ScgmHead(nn.Module):
    """Projection head (the original fc_enc) with the mixture parameters mu_y (C, out) and mu_z (K, out)."""

    def __init__(self, dim, hidden, out, num_coarse, num_fine):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, out))
        self.mu_y, self.mu_z = nn.Parameter(glorot(num_coarse, out)), nn.Parameter(glorot(num_fine, out))

    def forward(self, f):
        return self.mlp(f)


class Encoder(nn.Module):
    """Backbone + SCGM head, for embedding: [SCGM embedding (what the paper evaluates) | backbone CLS (as every other baseline)]."""

    def __init__(self, model, head):
        super().__init__()
        self.model, self.head = model, head
        self.transform, self.family, self.input_size = model.transform, model.family, model.input_size

    def forward(self, x):
        f = self.model(x)
        return torch.cat([self.head(f).float(), f.float()], dim=1)

    def exports(self, emb):
        d = self.head.mu_z.shape[1]
        return {"": emb[:, :d], "_backbone": emb[:, d:]}


def joint_prob(emb, make, mu_y, mu_z, tau):
    """(n, K) p(z|x) p(y|z) for each image's own coarse class y (the original forward_to_prob)."""
    mu_y, mu_z = F.normalize(mu_y, dim=1), F.normalize(mu_z, dim=1)
    p_z_x = (F.normalize(emb, dim=1) @ mu_z.T / tau).softmax(1)
    return p_z_x * (mu_y @ mu_z.T).softmax(0)[make]


def scgm_loss(emb, q, make, mu_y, mu_z, tau, alpha):
    """The original loss with hard pseudo-subclasses q (n,): alpha * (ls1 + ls2) + ls3."""
    x, mu_y_n, mu_z_n = F.normalize(emb, dim=1), F.normalize(mu_y, dim=1), F.normalize(mu_z, dim=1)
    ls1 = F.cross_entropy(x @ mu_z_n.T / tau, q)  # subclass likelihood p(z|x)
    ls2 = -(mu_y_n @ mu_z_n.T).log_softmax(0)[make, q].mean()  # -log p(y|z) of the pseudo-subclass
    ls3 = F.cross_entropy(F.relu(x) @ mu_y.T, make)  # coarse CE, norm_type "logit": unnormalised mu_y
    return alpha * (ls1 + ls2) + ls3


def sinkhorn_knopp(prob, lmd):
    """The original optimize_l_sk: (n, k) equipartitioned assignment of prob ** lmd (rows sum to 1/n, columns to 1/k) and its row argmax."""
    n, k = prob.shape
    prob = prob.astype(np.float64).T ** lmd  # (k, n)
    r, c = np.ones((k, 1)) / k, np.ones((n, 1)) / n
    err, cnt = 1e6, 0
    while err > 1e-1:
        r = (1.0 / k) / (prob @ c)
        c_new = (1.0 / n) / (r.T @ prob).T
        if cnt % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        cnt += 1
    log.info("sinkhorn-knopp: error %.4f after %d steps", err, cnt)
    prob = (prob * np.squeeze(c)).T * np.squeeze(r)  # (n, k)
    return prob, np.nanargmax(prob, axis=1)


class Scgm:
    """Needs batches (view, make, row index into the train set). The E-step re-embeds the whole train set (`dataset`, same
    augmentation and train mode as the original) at epochs 1, 1 + estep_every, ..."""

    def __init__(self, net, num_fine, tau, alpha, lmd, estep_every, dataset, batch_size, num_workers, device):
        self.net, self.head, self.tau, self.alpha, self.lmd, self.estep_every = net, net[1], tau, alpha, lmd, estep_every
        self.loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        self.q, self.device = torch.zeros(len(dataset), dtype=torch.long, device=device), device

    @torch.no_grad()
    def start_epoch(self, epoch):
        if epoch % self.estep_every != 1:
            return
        prob, rows = [], []
        for x, make, idx in self.loader:
            emb = forward_fp16(self.net, x.to(self.device))
            prob.append(joint_prob(emb, make.to(self.device), self.head.mu_y, self.head.mu_z, self.tau).cpu().numpy())
            rows.append(idx.numpy())
        q = np.empty(len(self.q), dtype=np.int64)
        q[np.concatenate(rows)] = sinkhorn_knopp(np.concatenate(prob), self.lmd)[1]
        self.q = torch.from_numpy(q).to(self.device)
        log.info("E-step at epoch %d: %d of %d subclasses used", epoch, len(np.unique(q)), self.head.mu_z.shape[0])

    def loss(self, batch):
        x, make, idx = batch
        return scgm_loss(forward_fp16(self.net, x), self.q[idx], make, self.head.mu_y, self.head.mu_z, self.tau, self.alpha)

    def end_step(self, i):
        pass
