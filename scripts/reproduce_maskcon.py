"""Reproduce MaskCon on Stanford Cars196 with its published recipe (configs/maskcon_repro.yaml): ResNet-18, 8 body-type coarse labels,
200 epochs, whole raw images; reports Recall@K of the 196 fine classes on the test set, next to the paper's Table 6. Needs a GPU:
    sbatch slurm/reproduce_maskcon.sh
Differences from the released code: the 67 + 87 conflicting images (data/splits/exclude.txt) are left out; the key-encoder batch shuffle
(a no-op on one GPU) is dropped; the schedule, loss and queue warm-up are the original ones, in fp32.
Output: reports/maskcon_repro/results.json, and embeddings <derived_root>/features/maskcon_repro_224/{train,val,test}.csv of the
trained backbone (test-time transform of the original: Resize 224x224 of the whole image).
"""
import argparse
import logging
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import models
from torchvision import transforms as T

from lsgen.data.labels import load_labels
from lsgen.data.splits import load_split
from lsgen.eval.representation import recall_at_k
from lsgen.features.extract import embed_images, write_csv
from lsgen.methods.repr.common import EMA
from lsgen.methods.repr.maskcon import maskcon_loss
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, write_json

log = logging.getLogger("reproduce_maskcon")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class RandomBlur:
    """SimCLR Gaussian blur, sigma ~ U(0.1, 2) (utils.GaussianBlur)."""

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2.0)))


def augmentations(size):
    """(key = weak, query = strong), as `get_augment('cars196', ...)`."""
    crop = T.RandomResizedCrop(size, scale=(0.2, 1))
    tail = [T.ToTensor(), T.Normalize(MEAN, STD)]
    weak = T.Compose([crop, T.RandomPerspective(0.5, 0.5), T.RandomHorizontalFlip(), *tail])
    strong = T.Compose([crop, T.RandomHorizontalFlip(), T.RandomPerspective(0.5, 0.5),
                        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), T.RandomGrayscale(0.2),
                        T.RandomApply([RandomBlur()], p=0.5), *tail])
    return weak, strong


class TwoViews(torch.utils.data.Dataset):
    """(key view, query view, coarse label) of a raw image. Training sees coarse labels only."""

    def __init__(self, paths, coarse, key_tf, query_tf):
        self.paths, self.coarse, self.key_tf, self.query_tf = paths, coarse, key_tf, query_tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        return self.key_tf(im), self.query_tf(im), int(self.coarse[i])


def build(cfg, device):
    """(backbone, projector): ImageNet-initialised ResNet-18 without its fc layer, and the MLP projection head."""
    backbone = getattr(models, cfg["arch"])(weights="IMAGENET1K_V1")
    feat = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    projector = torch.nn.Sequential(torch.nn.Linear(feat, cfg["hidden"]), torch.nn.ReLU(True), torch.nn.Linear(cfg["hidden"], cfg["dim"]))
    return backbone.to(device), projector.to(device)


def lr_at(step, steps_per_epoch, cfg):
    """Linear warm-up then cosine to 0 (utils.adjust_learning_rate)."""
    warm, total = cfg["warmup_epochs"] * steps_per_epoch, (cfg["epochs"] - cfg["warmup_epochs"]) * steps_per_epoch
    return cfg["lr"] * step / warm if step < warm else 0.5 * cfg["lr"] * (1 + math.cos((step - warm) / total * math.pi))


@torch.no_grad()
def warm_up_queue(ema, loader, queue, queue_make, device):
    """Fill the queue with keys of real images (initiate_memorybank), cycling through the loader if it is shorter than the queue."""
    filled = 0
    while filled < len(queue):
        for x_k, _, make in loader:
            n = min(len(x_k), len(queue) - filled)
            queue[filled:filled + n] = F.normalize(ema.net(x_k[:n].to(device)), dim=1)
            queue_make[filled:filled + n] = make[:n].to(device)
            filled += n
            if filled == len(queue):
                break


def evaluate(backbone, paths, fine, tf, cfg, device):
    backbone.eval()
    emb = embed_images(backbone, paths, tf, device, 256, cfg["num_workers"])
    backbone.train()
    return emb, {k: 100 * v for k, v in recall_at_k(emb, fine, cfg["recall_ks"]).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/maskcon_repro.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("training needs a GPU: sbatch slurm/reproduce_maskcon.sh")
    device = torch.device("cuda")
    random.seed(cfg["seed"]), np.random.seed(cfg["seed"]), torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    derived, splits_dir, raw = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits", Path(dcfg["raw_root"])
    labels = load_labels(derived, dcfg["data_root"], "maskcon")
    rel = pd.read_parquet(derived / "metadata.parquet", columns=["id", "rel_path"]).set_index("id").rel_path
    ids = {"train": load_split(splits_dir, "train"), "val": load_split(splits_dir, "val"), "test": load_split(splits_dir, "test")}
    train_ids = ids["train"] + ids["val"]  # the official train set (val is not held out: nothing is tuned)
    paths = lambda s: [raw / rel[i] for i in s]  # noqa: E731
    log.info("official train %d, test %d after excluding conflicting images", len(train_ids), len(ids["test"]))

    key_tf, query_tf = augmentations(cfg["size"])
    test_tf = T.Compose([T.Resize((cfg["size"],) * 2), T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = TwoViews(paths(train_ids), labels.make_id[train_ids].to_numpy(), key_tf, query_tf)
    dl = torch.utils.data.DataLoader(ds, cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["num_workers"],
                                     pin_memory=True, persistent_workers=True)
    fine_test = labels.fine_id[ids["test"]].to_numpy()

    backbone, projector = build(cfg, device)
    net = torch.nn.Sequential(backbone, projector)
    ema = EMA(net, cfg["momentum"])
    queue, queue_make = torch.zeros(cfg["queue"], cfg["dim"], device=device), torch.zeros(cfg["queue"], dtype=torch.long, device=device)
    assert cfg["queue"] % cfg["batch_size"] == 0, "the queue size must be a multiple of the batch size"
    warm_up_queue(ema, dl, queue, queue_make, device)
    opt = torch.optim.SGD(net.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])

    history, ptr, step = [], 0, 0
    for epoch in range(cfg["epochs"] + 1):
        if epoch % cfg["eval_every"] == 0:
            _, recall = evaluate(backbone, paths(ids["test"]), fine_test, test_tf, cfg, device)
            history.append({"epoch": epoch, **{f"R@{k}": v for k, v in recall.items()}})
            log.info("epoch %d test %s", epoch, {f"R@{k}": round(v, 2) for k, v in recall.items()})
        if epoch == cfg["epochs"]:
            break
        losses = []
        for x_k, x_q, make in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, len(dl), cfg)
            x_k, x_q, make = x_k.to(device, non_blocking=True), x_q.to(device, non_blocking=True), make.to(device)
            ema.update()
            q = F.normalize(net(x_q), dim=1)
            with torch.no_grad():
                k = F.normalize(ema.net(x_k), dim=1)
            loss = maskcon_loss(q, k, queue.clone(), queue_make.clone(), make, cfg["temperature"], cfg["soft_temperature"], cfg["w"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            queue[ptr:ptr + len(k)], queue_make[ptr:ptr + len(k)] = k, make  # enqueue after the loss, as in the original
            ptr, step = (ptr + len(k)) % cfg["queue"], step + 1
            losses.append(loss.item())
        log.info("epoch %d/%d loss %.4f lr %.5f", epoch + 1, cfg["epochs"], np.mean(losses), opt.param_groups[0]["lr"])

    final = history[-1]
    paper = {f"R@{k}": v for k, v in cfg["paper"]["recall"].items()}
    out = derived / "features" / f"maskcon_repro_{cfg['size']}"
    out.mkdir(parents=True, exist_ok=True)
    for split, split_ids in ids.items():
        emb, _ = evaluate(backbone, paths(split_ids), labels.fine_id[split_ids].to_numpy(), test_tf, cfg, device)
        write_csv(out / f"{split}.csv", [f"{i}.png" for i in split_ids], emb)
    write_json(Path(dcfg["reports_root"]) / "maskcon_repro" / "results.json", {
        **provenance(cfg, cfg["seed"]), "final_epoch": final, "paper": paper,
        "difference_to_paper": {k: round(final[k] - v, 2) for k, v in paper.items()}, "history": history,
        "n_train": len(train_ids), "n_test": len(ids["test"]),
        "note": "conflicting images excluded (data/splits/exclude.txt); native preprocessing, not comparable to bbox15 tables"})
    log.info("final %s | paper %s", {k: round(v, 2) for k, v in final.items() if k != "epoch"}, paper)


if __name__ == "__main__":
    main()
