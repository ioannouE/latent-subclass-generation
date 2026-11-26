"""Controlled latent diffusion generator (lsgen/methods/gen/ldm.py): prevalence-matching labels, classifier-free guidance, sampling."""
from pathlib import Path

import numpy as np
import torch

from lsgen.methods.gen.ldm import (build_unet, condition_labels, drop_labels, ema_update, guided_eps, kl_loss, prevalence_labels, runs, sample, schedulers,
                                  write_imagefolder)

SCHEDULE = {"num_train_timesteps": 1000, "schedule": "scaled_linear_beta", "beta_start": 0.0015, "beta_end": 0.0195}


def test_prevalence_labels_match_counts_and_are_seeded():
    counts = np.array([700, 200, 99, 1])
    y = prevalence_labels(counts, 1000, seed=0)
    assert len(y) == 1000 and np.array_equal(np.bincount(y, minlength=4), [700, 200, 99, 1])
    y = prevalence_labels(counts, 333, seed=0)
    assert len(y) == 333 and np.abs(np.bincount(y, minlength=4) - 333 * counts / counts.sum()).max() < 1
    assert np.array_equal(y, prevalence_labels(counts, 333, seed=0)) and not np.array_equal(y, prevalence_labels(counts, 333, seed=1))


def test_drop_labels_uses_the_null_label():
    y = torch.arange(10) % 3
    assert torch.equal(drop_labels(y, 0.0, null=3), y)
    assert (drop_labels(y, 1.0, null=3) == 3).all()
    dropped = drop_labels(torch.zeros(20000, dtype=torch.long), 0.1, null=3, generator=torch.Generator().manual_seed(0)) == 3
    assert abs(dropped.float().mean().item() - 0.1) < 0.01


def test_guidance_combines_conditional_and_unconditional():
    unet = lambda z, t, class_labels: z + class_labels[:, None].float()  # noqa: E731  eps = z + label
    z, t, y = torch.zeros(4, 2), torch.zeros(4), torch.tensor([0, 1, 2, 1])
    assert torch.equal(guided_eps(unet, z, t, y, null=5, w=1), y[:, None].float().expand(4, 2))
    assert torch.equal(guided_eps(unet, z, t, y, null=5, w=0), torch.full((4, 2), 5.0))
    assert torch.equal(guided_eps(unet, z, t, y, null=5, w=2), (2 * y[:, None] - 5).float().expand(4, 2))


def test_sampling_is_deterministic_and_shaped():
    arch = {"channels": [32, 32], "attention_levels": [False, True], "num_res_blocks": 1, "num_head_channels": [0, 32], "norm_num_groups": 8}
    torch.manual_seed(0)
    unet = build_unet(arch, latent_channels=4, n_classes=3).eval()
    _, ddim = schedulers(SCHEDULE)
    y = torch.tensor([0, 2])
    runs = [sample(unet, ddim, y, (4, 8, 8), steps=3, w=1.5, null=3, generator=torch.Generator().manual_seed(7)) for _ in range(2)]
    assert runs[0].shape == (2, 4, 8, 8) and torch.isfinite(runs[0]).all() and torch.equal(*runs)


def test_run_names():
    assert runs({"seed": 0, "ldm": {"name": "pilot"}}) == [(0, "pilot")]
    assert runs({"seed": 0, "seeds": [0, 2], "ldm": {"name": "coarse"}}) == [(0, "coarse_s0"), (2, "coarse_s2")]


def test_kl_is_zero_at_the_prior():
    assert kl_loss(torch.zeros(2, 4, 3, 3), torch.ones(2, 4, 3, 3)).item() == 0
    assert kl_loss(torch.ones(2, 4, 3, 3), torch.ones(2, 4, 3, 3)).item() == 0.5 * 36


def test_ema_warmup():
    model, ema = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.ones_(model.weight), torch.nn.init.zeros_(ema.weight)
    ema_update(ema, model, step=0, decay=0.9999)  # decay min(0.9999, 1 / 10) = 0.1: 90% of the way to the model
    assert abs(ema.weight.item() - 0.9) < 1e-6


def test_condition_labels_one_per_coarse_cluster_pair():
    coarse, cluster = np.array([1, 0, 1, 0, 2, 1]), np.array([3, 0, 0, 0, 5, 3])
    label, pairs = condition_labels(coarse, cluster)
    assert np.array_equal(pairs, [[0, 0], [1, 0], [1, 3], [2, 5]])  # cluster ids repeat across classes but stay distinct labels
    assert np.array_equal(pairs[label], np.stack([coarse, cluster], 1))
    label, pairs = condition_labels(coarse, np.zeros_like(coarse))  # coarse-only rung: the labels are the coarse classes
    assert np.array_equal(label, coarse) and np.array_equal(pairs[:, 0], [0, 1, 2])


def test_imagefolder_class_index_is_the_label(tmp_path):
    from PIL import Image
    from torchvision.datasets import ImageFolder
    images = tmp_path / "crops"
    images.mkdir()
    ids, label = [f"im{i}" for i in range(12)], [0, 2, 10, 1, 2, 10, 9, 0, 1, 11, 9, 11]  # labels >= 10 sort after 9 only if padded
    for i in ids:
        Image.new("RGB", (4, 4)).save(images / f"{i}.png")
    write_imagefolder(tmp_path / "folder", images, ids, label)
    ds = ImageFolder(tmp_path / "folder")
    got = {Path(p).stem: y for p, y in ds.samples}
    assert all(got[i] == y for i, y in zip(ids, label))
