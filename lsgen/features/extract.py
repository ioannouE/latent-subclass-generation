"""Embedding extraction and the CSV format shared by real and generated images:
first column `filename`, then one column per latent dimension z1..zD."""
import pandas as pd
import torch
from PIL import Image


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


@torch.no_grad()
def embed_images(model, paths, transform, device, batch_size=128, num_workers=8):
    """(n, D) float32 array, rows in the order of `paths`."""
    dl = torch.utils.data.DataLoader(Images(paths, transform), batch_size=batch_size, num_workers=num_workers)
    return torch.cat([model(x.to(device)).float().cpu() for x in dl]).numpy()


def write_csv(path, filenames, emb):
    df = pd.DataFrame(emb, columns=[f"z{i + 1}" for i in range(emb.shape[1])])
    df.insert(0, "filename", filenames)
    df.to_csv(path, index=False, float_format="%.6g")


def read_csv(path):
    """Inverse of write_csv: (filenames, (n, D) float32 array)."""
    df = pd.read_csv(path)
    return df.filename.tolist(), df.iloc[:, 1:].to_numpy("float32")
