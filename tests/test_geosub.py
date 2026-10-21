import numpy as np
import torch
import torch.nn.functional as F

from lsgen.eval.representation import assignment_metric
from lsgen.methods.geosub.losses import geo_loss, geosub_loss, sub_loss, var_loss
from lsgen.methods.geosub.model import GeoSub, active_prototypes, init_prototypes


def features(n=24, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1), torch.arange(n) % 3


def model_for(f, coarse, n_proto=4):
    return GeoSub(f.shape[1], 32, init_prototypes(f, coarse, int(coarse.max()) + 1, n_proto, torch.Generator().manual_seed(0)))


def test_head_starts_as_identity():
    f, coarse = features()
    model = model_for(f, coarse)
    assert torch.allclose(model(f), f, atol=1e-6)  # zero-initialised last layer: z = normalised f
    assert geo_loss(model(f), f) < 1e-6


def test_geo_loss_detects_changed_geometry():
    f, _ = features()
    assert geo_loss(f, f) < 1e-6
    assert geo_loss(f[torch.randperm(len(f))], f) > 0.1  # same points, other neighbours
    z = f.clone().requires_grad_(True)
    geo_loss(F.normalize(z + 0.1 * torch.randn_like(z), dim=1), f).backward()
    assert torch.isfinite(z.grad).all()  # the masked diagonal gives no NaN gradient


def test_init_prototypes_come_from_own_class():
    f, coarse = features()
    protos = init_prototypes(f, coarse, 3, 5, torch.Generator().manual_seed(0))
    for c in range(3):
        own = f[coarse == c]
        assert all((own == p).all(1).any() for p in protos[c])


def test_sub_loss_rewards_consistent_assignments():
    coarse = torch.zeros(4, dtype=torch.long)
    cos = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert sub_loss(cos, cos, coarse, usage=0) < sub_loss(cos, cos.flip(1), coarse, usage=0)
    collapsed = torch.tensor([[1.0, 0.0]] * 4)  # every image on one prototype: the usage term is higher (less negative)
    assert sub_loss(collapsed, collapsed, coarse, usage=1) - sub_loss(collapsed, collapsed, coarse, usage=0) > \
        sub_loss(cos, cos, coarse, usage=1) - sub_loss(cos, cos, coarse, usage=0)


def test_var_loss_penalises_within_class_collapse_only():
    f, coarse = features()
    assert var_loss(f, f, coarse) == 0
    collapsed = F.normalize(torch.stack([f[coarse == c].mean(0) for c in range(3)])[coarse], dim=1)
    assert np.isclose(var_loss(collapsed, f, coarse).item(), 1.0, atol=1e-5)
    assert var_loss(f, f, torch.arange(len(f))) == 0  # no class with two images: nothing to measure


def test_geosub_loss_weights_and_gradients():
    f, coarse = features()
    model = model_for(f, coarse)
    f2 = F.normalize(f + 0.05 * torch.randn_like(f), dim=1)
    loss, terms = geosub_loss(model, f, f2, coarse, {"geo": 1.0, "sub": 1.0}, 0.1, 0.1, 0.04, 0.1, 0.1)
    assert set(terms) == {"geo", "sub"} and torch.isclose(loss, terms["geo"] + terms["sub"])
    loss.backward()
    assert model.prototypes.grad.abs().sum() > 0 and all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_assign_and_active_prototypes():
    model = GeoSub(2, 4, torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]))
    z, coarse = F.normalize(torch.tensor([[1.0, 0.1], [1.0, -0.1], [0.1, 1.0], [-1.0, 0.0]]), dim=1), torch.zeros(4, dtype=torch.long)
    assign = model.assign(z, coarse)
    assert assign.tolist() == [0, 0, 1, 2]
    active = active_prototypes(assign, coarse, 1, 3, min_count=2)
    assert active.tolist() == [[True, False, False]]
    assert model.assign(z, coarse, active).tolist() == [0, 0, 0, 0]  # only among the active prototypes


def test_assignment_metric():
    fine = np.array([0, 0, 1, 1, 2, 2])
    assign = np.array([5, 5, 7, 7, 7, 7])  # two clusters for three classes
    m = assignment_metric(fine, assign, assign)(np.arange(6))
    assert np.isclose(m["acc"], 4 / 6) and m["flip_ari"] == 1.0
    flipped = np.array([5, 7, 5, 7, 5, 7])  # mirror images land elsewhere
    assert assignment_metric(fine, assign, flipped)(np.arange(6))["flip_ari"] < 0.5
