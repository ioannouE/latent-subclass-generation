import torch
import torch.nn.functional as F

from lsgen.data.labels import coarse_classes
from lsgen.methods.repr.ksub.losses import ksub_loss, pair_weights, sub_loss, var_loss
from lsgen.methods.repr.ksub.model import ema_update, head, init_prototypes, soft_assign
from lsgen.methods.repr.losses import supcon_loss
from lsgen.utils import REPO_ROOT

K_C = torch.tensor([2, 3, 1])  # padded to K = 3


def batch(n=24, d=16, seed=0):
    """Features f (n, d), coarse labels (3 classes) and soft assignments r to the k-means prototypes of each image's class."""
    g = torch.Generator().manual_seed(seed)
    f, coarse = F.normalize(torch.randn(n, d, generator=g), dim=1), torch.arange(n) % 3
    return f, coarse, soft_assign(f, coarse, init_prototypes(f, coarse, K_C.tolist(), seed=0), K_C)


def test_untrained_head_gives_z_equal_f():
    f, _, _ = batch()
    net = head(f.shape[1], 32)
    assert torch.equal(net.g(f), torch.zeros_like(f))  # zero-initialised last layer: the residual is exactly 0
    assert torch.equal(net(f), F.normalize(f, dim=1))
    assert torch.allclose(net(f), f, atol=1e-6)  # f is already unit-norm; re-normalising may change the last bit


def test_soft_assignments_live_on_own_class_prototypes():
    _, coarse, r = batch()
    assert torch.allclose(r.sum(1), torch.ones(len(r)))
    assert (r[:, 1:][coarse == 2] == 0).all() and (r[:, 2][coarse == 0] == 0).all()  # padding gets no weight


def test_pair_weights_zero_across_coarse_classes():
    _, coarse, r = batch()
    w = pair_weights(r, coarse)
    assert (w[coarse[:, None] != coarse[None, :]] == 0).all()
    assert (w.diagonal() == 0).all()
    assert (w[(coarse[:, None] == coarse[None, :]) & ~torch.eye(len(r), dtype=torch.bool)] > 0).all()


def test_sub_loss_has_no_cross_class_terms():
    f, coarse, r = batch()
    per_class = [(coarse == c).sum() * sub_loss(f[coarse == c], coarse[coarse == c], r[coarse == c]) for c in range(3)]
    assert torch.allclose(sub_loss(f, coarse, r), sum(per_class) / len(f))  # = the classes computed separately


def test_coarse_only_is_plain_supcon():
    f, coarse, r = batch()
    z = F.normalize(f + 0.1 * torch.randn_like(f), dim=1)
    loss, terms = ksub_loss(z, f, coarse, r, {"coarse": 1.0, "sub": 0.0, "var": 0.0})
    assert list(terms) == ["coarse"]
    assert torch.equal(loss, supcon_loss(z, coarse, 0.1))


def test_var_loss_zero_when_z_is_f():
    f, coarse, r = batch()
    assert var_loss(f, f, coarse, r) == 0
    shrunk = F.normalize(f + 5 * torch.stack([f[coarse == c].mean(0) for c in coarse]), dim=1)  # pulled to class means
    assert var_loss(shrunk, f, coarse, r) > 0.1


def test_ema_update_moves_only_prototypes_in_the_batch():
    f, coarse, r = batch()
    mu = init_prototypes(f, coarse, K_C.tolist(), seed=0)
    before = mu.clone()
    keep = coarse != 2
    ema_update(mu, f[keep], coarse[keep], r[keep], momentum=0.5)
    assert torch.equal(mu[2], before[2])
    assert not torch.equal(mu[0, :2], before[0, :2])
    assert torch.allclose(mu[:2].norm(dim=2)[K_C[:2, None] > torch.arange(3)], torch.ones(5))


def test_k_c_per_scheme():
    body = coarse_classes(REPO_ROOT / "data", "maskcon_new")
    assert body.index.tolist() == list(range(8)) and body.K_c.sum() == 196 and body.K_c.max() == 50  # Sedan
    make = coarse_classes(REPO_ROOT / "data", "make")
    assert len(make) == 49 and make.K_c.sum() == 189
    assert (coarse_classes(REPO_ROOT / "data" / "cifar100", "cifar100").K_c == 5).all()
