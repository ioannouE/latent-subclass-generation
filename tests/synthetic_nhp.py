"""Synthetic features for NHP (CPU, seconds). Stands in for the backbone output: the backbone is the identity.

x = make signal + subclass signal + colour-like shift + pose-like shift + noise, each in its own block of dimensions. The nuisance
shifts are strong and shared by all subclasses. Positive views redraw the nuisance classes of an image (invariance on) or keep them
(off) and add a little noise."""
import torch

from lsgen.methods.repr.nhp import NHP, Head

BLOCK = 8
N_COLOUR, N_POSE = 4, 3
K_PER_MAKE = (1, 1, 3, 4, 2, 1)  # planted subclasses per make, including single-subclass makes


class Toy:
    def __init__(self, k_per_make=K_PER_MAKE, n_per_sub=120, signal=2.0, nuisance=3.0, noise=0.5, seed=0, structured=True):
        """structured=False: the subclass signal is an independent random vector per image (no clusters): a negative control."""
        g = torch.Generator().manual_seed(seed)
        self.k_per_make, self.nuisance, self.noise = k_per_make, nuisance, noise
        self.make = torch.cat([torch.full((k * n_per_sub,), m) for m, k in enumerate(k_per_make)])
        self.sub = torch.cat([torch.arange(k).repeat_interleave(n_per_sub) for k in k_per_make])
        self.sub_id = self.sub + torch.tensor([0, *k_per_make]).cumsum(0)[self.make]  # global subclass id
        n, rand = len(self.make), lambda *s: torch.randn(*s, generator=g)
        centres = rand(sum(k_per_make), BLOCK) * signal
        sub_signal = centres[self.sub_id] if structured else rand(n, BLOCK) * signal
        make_signal = rand(len(k_per_make), BLOCK)[self.make] * signal * 2
        self.core = torch.cat([make_signal, sub_signal, torch.zeros(n, 2 * BLOCK)], 1) + rand(n, 4 * BLOCK) * noise
        self.colour, self.pose = torch.randint(0, N_COLOUR, (n,), generator=g), torch.randint(0, N_POSE, (n,), generator=g)
        self.colour_shift, self.pose_shift = rand(N_COLOUR, BLOCK) * nuisance, rand(N_POSE, BLOCK) * nuisance

    def __len__(self):
        return len(self.make)

    def features(self, idx, colour, pose, noise=0.0):
        x = self.core[idx].clone()
        x[:, 2 * BLOCK:3 * BLOCK] += self.colour_shift[colour]
        x[:, 3 * BLOCK:] += self.pose_shift[pose]
        return x + torch.randn_like(x) * noise

    def real(self):
        """The observed features of every image (own nuisance classes)."""
        return self.features(torch.arange(len(self)), self.colour, self.pose)

    def views(self, idx, invariance):
        """(view 1, view 2, augmentation parameters (B, 2, 2) = (colour, pose) of each view) of the images idx."""
        out, params = [], []
        for _ in range(2):
            colour = torch.randint(0, N_COLOUR, (len(idx),)) if invariance else self.colour[idx]
            pose = torch.randint(0, N_POSE, (len(idx),)) if invariance else self.pose[idx]
            out.append(self.features(idx, colour, pose, noise=0.1))
            params.append(torch.stack([colour, pose], 1))
        return out[0], out[1], torch.stack(params, 1)


DEFAULTS = dict(tau_student=0.1, tau_teacher=0.04, ema_decay=0.99, margins=(0.1, 0.1), temperature=0.1, active_min=0.05,
                w_make=1.0, w_assign=1.0, w_usage=0.1, w_hierarchy=1.0, w_nuis=0.1, w_l0=0.05, w_adversary=0.0)


def fit(toy, steps=300, batch=256, invariance=True, sparse=False, adversary=False, n_proto=6, dims=(8, 8, 4), lr=3e-3, seed=0,
        hidden=64, **overrides):
    """Train the NHP head on the toy features. Returns the objective (objective.net[1] is the head)."""
    torch.manual_seed(seed)
    head = Head(4 * BLOCK, hidden, dims, len(toy.k_per_make), n_proto, sparse, (N_COLOUR, N_POSE) if adversary else ())
    net = torch.nn.Sequential(torch.nn.Identity(), head)
    objective = NHP(net, **{**DEFAULTS, **overrides})
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        idx = torch.randint(0, len(toy), (batch,))
        loss = objective.loss([*toy.views(idx, invariance), toy.make[idx]])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return objective


@torch.no_grad()
def embed(objective, x):
    """(z_make, z_sub, z_nuis) of features x."""
    head = objective.net[1]
    return head.split(head(x))
