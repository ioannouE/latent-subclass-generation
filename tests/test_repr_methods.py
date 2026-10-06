import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from lsgen.data.datasets import CarsTwoViews
from lsgen.methods.repr.losses import contrastive_loss, supcon_loss

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
