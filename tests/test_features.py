import numpy as np
import pandas as pd
import torch
from PIL import Image

from lsgen.eval.classifier import ece, fit_temperature
from lsgen.features.extract import embed_images, write_csv


class MeanColour(torch.nn.Module):
    def forward(self, x):
        return x.mean((2, 3))


def test_embed_images_and_csv_format(tmp_path):
    paths = []
    for k, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        paths.append(tmp_path / f"train_{k:05d}.png")
        Image.new("RGB", (16, 16), colour).save(paths[-1])
    emb = embed_images(MeanColour(), paths, lambda im: torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255,
                       "cpu", batch_size=2, num_workers=0)
    assert np.allclose(emb, np.eye(3))  # row order follows paths
    write_csv(tmp_path / "e.csv", [p.name for p in paths], emb)
    df = pd.read_csv(tmp_path / "e.csv")
    assert list(df.columns) == ["filename", "z1", "z2", "z3"] and df.filename.tolist() == [p.name for p in paths]
    assert np.allclose(df[["z1", "z2", "z3"]].to_numpy(), np.eye(3))


def test_ece_known_value():
    probs = torch.tensor([[0.9, 0.1]] * 10)
    assert abs(ece(probs, torch.zeros(10, dtype=torch.long)) - 0.1) < 1e-6  # always right at 0.9 confidence
    labels = torch.tensor([0] * 9 + [1])
    assert ece(probs, labels) < 1e-6  # 90% right at 0.9 confidence


def test_temperature_scaling_fixes_overconfidence():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 5, (3000,), generator=g)
    pred = torch.where(torch.rand(3000, generator=g) < 0.7, y, torch.randint(0, 5, (3000,), generator=g))
    logits = 8 * torch.nn.functional.one_hot(pred, 5).float()  # ~76% accurate but ~100% confident
    t = fit_temperature(logits, y)
    assert t > 2
    assert ece((logits / t).softmax(1), y) < 0.2 * ece(logits.softmax(1), y)
