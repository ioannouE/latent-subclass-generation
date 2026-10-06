import itertools

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from lsgen.data.datasets import CarsTwoViews
from lsgen.methods.repr.falcon import Falcon, assign_fine_to_coarse, coarse_log_prob, coarse_neighbors
from lsgen.methods.repr.losses import contrastive_loss, supcon_loss
from lsgen.methods.repr.maskcon import MaskCon, maskcon_loss

E1, E2, E3, E4 = torch.eye(4)
TAU = 0.1


def test_simclr_known_values():
    z = torch.stack([E1, E2, E1, E2])  # both views aligned, other images orthogonal
    expected = -(1 / TAU - np.log(np.exp(1 / TAU) + 2))  # positive at 1/tau; the 2 other rows at 0
    assert np.isclose(supcon_loss(z, torch.tensor([0, 1, 0, 1]), TAU), expected, atol=1e-5)
    collapsed = torch.ones(6, 4)  # everything identical: every row is a tie, loss = log(n - 1)
    assert np.isclose(supcon_loss(collapsed, torch.arange(3).repeat(2), TAU), np.log(5), atol=1e-5)


def test_supcon_pulls_same_make_together():
    make = torch.tensor([0, 0, 1])  # images 0 and 1 share a make
    same = torch.stack([E1, E1, E2, E1, E1, E2])
    split = torch.stack([E1, E2, E3, E1, E2, E3])
    assert contrastive_loss(same, make, 1, 0, TAU) < contrastive_loss(split, make, 1, 0, TAU)
    assert contrastive_loss(same, make, 0, 1, TAU) > contrastive_loss(split, make, 0, 1, TAU)  # SimCLR wants them apart


def test_contrastive_weights_and_gradients():
    z = torch.randn(6, 8, requires_grad=True)
    make = torch.tensor([0, 0, 1])
    a, b = contrastive_loss(z, make, 1, 0, TAU), contrastive_loss(z, make, 0, 1, TAU)
    assert torch.isclose(contrastive_loss(z, make, 1, 1, TAU), a + b) and contrastive_loss(z, make, 0, 0, TAU) == 0
    (a + b).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_two_views_dataset_returns_two_views_and_make_only(tmp_path):
    import pandas as pd
    meta = pd.DataFrame({"id": ["train_00001"], "make_id": [5], "fine_id": [3]})
    (tmp_path / "bbox15_16").mkdir()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(tmp_path / "bbox15_16" / "train_00001.png")
    ds = CarsTwoViews(meta, tmp_path, ["train_00001"], T.Compose([T.RandomHorizontalFlip(0.5), T.ColorJitter(0.5), T.ToTensor()]), size=16)
    v1, v2, make = ds[0]
    assert v1.shape == v2.shape == (3, 16, 16) and make == 5
    assert any(not torch.equal(*[ds[0][i] for i in (0, 1)]) for _ in range(5))  # the views are drawn independently


def test_maskcon_targets():
    k = E1[None]
    queue = torch.stack([E1, E2, E3])
    queue_make = torch.tensor([0, 0, 1])
    q = k.clone()
    logits = torch.tensor([1.0, 1.0, 0.0, 0.0]) / TAU  # [own key, queue]
    log_p = torch.log_softmax(logits, 0)
    # same-coarse entries: E1 (sim 1, weight 1) and E2 (sim 0, weight exp(-1 / t0)); the other make is masked out
    soft = torch.tensor([1.0, 1.0, np.exp(-1 / TAU), 0.0], dtype=torch.float32)
    expected = -(log_p * soft / soft.sum()).sum().float()
    assert torch.isclose(maskcon_loss(q, k, queue, queue_make, torch.tensor([0]), TAU, TAU, 1.0), expected, atol=1e-5)
    # w = 0 is plain MoCo / InfoNCE with the own key as the only positive
    assert torch.isclose(maskcon_loss(q, k, queue, queue_make, torch.tensor([0]), TAU, TAU, 0.0), -log_p[0], atol=1e-5)
    # nothing of the same make in the queue: the target is the own key only
    assert torch.isclose(maskcon_loss(q, k, queue, queue_make, torch.tensor([2]), TAU, TAU, 1.0), -log_p[0], atol=1e-5)


def test_assign_fine_to_coarse_is_optimal():
    rng = np.random.default_rng(0)
    cost, reg = rng.random((6, 3)), 0.5

    def objective(parent):
        m = np.eye(3)[list(parent)]
        return -(cost * m).sum() + reg * (m.sum(0) ** 2).sum() / 3

    best = min((p for p in itertools.product(range(3), repeat=6) if len(set(p)) == 3), key=objective)
    m = assign_fine_to_coarse(cost, reg).numpy()
    assert (m.sum(1) == 1).all() and (m.sum(0) >= 1).all()
    assert np.isclose(objective(m.argmax(1)), objective(best))


def test_coarse_log_prob_and_neighbors():
    logits = torch.randn(4, 6)
    m = torch.eye(3)[[0, 0, 1, 1, 2, 2]]  # fine classes 2c and 2c + 1 belong to coarse class c
    assert torch.allclose(coarse_log_prob(logits, m).exp().sum(1), torch.ones(4), atol=1e-5)
    x = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [-1, 0.0]], dtype=np.float32)
    make = np.array([0, 0, 1, 1, 0])
    assert coarse_neighbors(x, make, 1)[:, 0].tolist() == [1, 0, 3, 2, 1]  # same make only, never itself


def test_two_views_neighbors_and_weak_view(tmp_path):
    import pandas as pd
    ids = [f"train_{i:05d}" for i in range(4)]
    (tmp_path / "bbox15_16").mkdir()
    for i in ids:
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(tmp_path / "bbox15_16" / f"{i}.png")
    meta = pd.DataFrame({"id": ids, "make_id": [0, 0, 1, 1]})
    neighbors = torch.tensor([[1, 2, 3]] * 4)
    ds = CarsTwoViews(meta, tmp_path, ids, T.ToTensor(), size=16, transform_2=T.Compose([T.ToTensor(), lambda x: x * 0]),
                      neighbors=neighbors, n_neighbors=2)
    v1, v2, near, make = ds[0]
    assert v1.shape == (3, 16, 16) and v2.abs().sum() == 0 and near.shape == (2, 3, 16, 16) and make == 0


def _tiny_net(out):
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, out))


def test_maskcon_and_falcon_step():
    torch.manual_seed(0)
    x, make = torch.randn(8, 3, 2, 2), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    mc = MaskCon(_tiny_net(5), 5, queue=16, momentum=0.9, t=0.1, t0=0.1, w=1.0, device="cpu")
    loss = mc.loss((x, x + 0.1, make))
    loss.backward()
    assert torch.isfinite(loss) and mc.queue_make[:8].tolist() == make.tolist() and mc.ptr == 8
    net = _tiny_net(6)
    fc = Falcon(net, num_fine=6, num_coarse=2, device="cpu", lambdas=(0.5, 0.5, 2.0), temperature=0.9, beta_reg=0.1,
                solve_every=1, soft_epochs=2, ema_decay=0.9)
    fc.start_epoch(1)
    loss = fc.loss((x, x + 0.1, torch.randn(8, 3, 3, 2, 2), make))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    fc.end_step(0), fc.end_step(1)  # step 1 re-solves M from the collected predictions
    assert (fc.m.sum(1) == 1).all() and (fc.m.sum(0) >= 1).all()
