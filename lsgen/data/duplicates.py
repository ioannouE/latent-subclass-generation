"""Near-duplicate audit between official train and test (raw images): pHash + SSCD.
Pairs are marked, never deleted."""
import logging
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image

from lsgen.utils import sha256_file

log = logging.getLogger(__name__)


def _phash(path):
    with Image.open(path) as im:
        bits = imagehash.phash(im.convert("RGB")).hash.flatten()
    return int(np.packbits(bits).view(">u8")[0])


def phash_all(paths, num_workers=8):
    with Pool(num_workers) as pool:
        return np.array(pool.map(_phash, paths, chunksize=64), dtype=np.uint64)


def hamming_pairs(h_query, h_ref, max_dist, chunk=512):
    """All (i, j, dist) with popcount(h_query[i] ^ h_ref[j]) <= max_dist."""
    out = []
    for s in range(0, len(h_query), chunk):
        d = np.bitwise_count(h_query[s:s + chunk, None] ^ h_ref[None, :])
        i, j = np.nonzero(d <= max_dist)
        out.append(np.stack([i + s, j, d[i, j]], 1))
    return np.concatenate(out).astype(np.int64)


def hamming_min(h_query, h_ref, chunk=512):
    return np.concatenate([np.bitwise_count(h_query[s:s + chunk, None] ^ h_ref[None, :]).min(1)
                           for s in range(0, len(h_query), chunk)])


def get_sscd(weights_dir, url):
    path = Path(weights_dir) / Path(url).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading SSCD weights %s -> %s", url, path)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return torch.jit.load(str(path), map_location="cpu").eval(), path, sha256_file(path)


class _SSCDImages(torch.utils.data.Dataset):
    """SSCD "small_288" evaluation transform: short side -> 288 (bilinear), ImageNet normalisation."""

    def __init__(self, paths, resize):
        from torchvision import transforms as T
        self.paths = paths
        self.tf = T.Compose([T.Resize(resize), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def resolve_device(device, allow_cpu=False):
    """GPU is required unless allow_cpu: SSCD on the login-node CPU takes ~1 h."""
    if device == "cuda" and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("SSCD embedding needs a GPU: submit slurm/prepare_data.sh on a GPU node "
                               "(or pass --allow-cpu to run it on CPU)")
        log.warning("no GPU available; running SSCD on CPU (--allow-cpu)")
        device = "cpu"
    return torch.device(device)


@torch.no_grad()
def sscd_embed(model, paths, resize=288, num_workers=8, threads=16, device="cpu"):
    """One image per forward pass (variable aspect ratios, as in the SSCD small_288 protocol); fp32."""
    device = torch.device(device)
    if device.type == "cpu":
        torch.set_num_threads(threads)
    model = model.to(device)
    dl = torch.utils.data.DataLoader(_SSCDImages(paths, resize), batch_size=None, num_workers=num_workers,
                                     pin_memory=device.type == "cuda")
    out = np.zeros((len(paths), 512), np.float32)
    for k, x in enumerate(dl):
        out[k] = torch.nn.functional.normalize(model(x[None].to(device, non_blocking=True)), dim=1)[0].cpu().numpy()
        if k % 1000 == 0:
            log.info("SSCD %d/%d", k, len(paths))
    return out


def _candidate_pairs(ids_a, ids_b, h_a, h_b, sim, phash_max, sscd_thr, cols, upper_only=False):
    """Pairs with pHash Hamming <= phash_max OR SSCD cosine >= sscd_thr, with both scores."""
    ph = hamming_pairs(h_a, h_b, phash_max)
    si, sj = np.nonzero(sim >= sscd_thr)
    pairs = set(map(tuple, ph[:, :2].tolist())) | set(zip(si.tolist(), sj.tolist()))
    if upper_only:
        pairs = {(i, j) for i, j in pairs if i < j}
    rows = [(ids_a[i], ids_b[j], float(sim[i, j]), int(np.bitwise_count(h_a[i] ^ h_b[j]))) for i, j in sorted(pairs)]
    df = pd.DataFrame(rows, columns=[*cols, "sscd_sim", "phash_hamming"])
    df["flag_sscd"] = df.sscd_sim >= sscd_thr
    df["flag_phash"] = df.phash_hamming <= phash_max
    return df.sort_values("sscd_sim", ascending=False, ignore_index=True)


def duplicate_table(test_ids, train_ids, h_test, h_train, e_test, e_train, phash_max, sscd_thr):
    """Test-vs-train candidate pairs flagged by either criterion. Also returns the per-test NN summary."""
    sim = e_test @ e_train.T
    nn = sim.argmax(1)
    nn_summary = pd.DataFrame({"test_id": test_ids, "sscd_nn_train_id": np.asarray(train_ids)[nn],
                               "sscd_nn_sim": sim[np.arange(len(test_ids)), nn], "phash_min_hamming": hamming_min(h_test, h_train)})
    return _candidate_pairs(test_ids, train_ids, h_test, h_train, sim, phash_max, sscd_thr, ("test_id", "train_id")), nn_summary


def within_table(ids, h, e, phash_max, sscd_thr):
    """Near-duplicate pairs within one set of images (id_a < id_b by position)."""
    return _candidate_pairs(ids, ids, h, h, e @ e.T, phash_max, sscd_thr, ("id_a", "id_b"), upper_only=True)
