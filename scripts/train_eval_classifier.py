"""Milestone 3: train the evaluator classifiers (fine 196 / make 49). Needs a GPU:
    sbatch slurm/train_eval_classifier.sh
Model selection and temperature scaling use val only; test is scored once at the end. Conflicting images
(data/splits/exclude.txt) are left out of train, val and test.
Output: <derived_root>/evaluators/<label>/model.pt and reports/evaluator_<label>.json.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from lsgen.data.splits import load_split
from lsgen.eval.classifier import LabelledImages, ece, fit_temperature, predict, train_classifier
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, write_json

log = logging.getLogger("train_eval_classifier")
MEAN_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classifier.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("classifier training needs a GPU: sbatch slurm/train_eval_classifier.sh")
    device = torch.device("cuda")
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    meta = pd.read_parquet(derived / "metadata.parquet", columns=["id", "fine_id", "make_id"]).set_index("id")
    dataset_hash = json.loads((Path(dcfg["data_root"]) / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    size = int(cfg["crop"].split("_")[-1])
    train_tf = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(*MEAN_STD)])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(*MEAN_STD)])

    for label in cfg["labels"]:
        torch.manual_seed(cfg["seed"])
        n_classes = meta[f"{label}_id"].nunique()
        dls = {}
        for split, tf in (("train", train_tf), ("val", eval_tf), ("test", eval_tf)):
            ids = load_split(splits_dir, split)
            ds = LabelledImages([derived / cfg["crop"] / f"{i}.png" for i in ids], meta.loc[ids, f"{label}_id"].tolist(), tf)
            dls[split] = torch.utils.data.DataLoader(ds, cfg["batch_size"], shuffle=split == "train", drop_last=split == "train",
                                                     num_workers=cfg["num_workers"])
        log.info("%s (%d classes): %s images", label, n_classes, {s: len(d.dataset) for s, d in dls.items()})
        model, val_acc, history = train_classifier(cfg["arch"], n_classes, dls["train"], dls["val"], device,
                                                   cfg["epochs"], cfg["lr"], cfg["weight_decay"])
        val_logits, val_y = predict(model, dls["val"], device)
        temperature = fit_temperature(val_logits, val_y)
        test_logits, test_y = predict(model, dls["test"], device)
        metrics = {
            "label": label, "n_classes": n_classes, "n": {s: len(d.dataset) for s, d in dls.items()},
            "val_top1": val_acc, "temperature": temperature,
            "test_top1": (test_logits.argmax(1) == test_y).float().mean().item(),
            "test_ece_uncalibrated": ece(test_logits.softmax(1), test_y),
            "test_ece_calibrated": ece((test_logits / temperature).softmax(1), test_y),
            "test_nll_uncalibrated": F.cross_entropy(test_logits, test_y).item(),
            "test_nll_calibrated": F.cross_entropy(test_logits / temperature, test_y).item(),
            "history": history}
        out = derived / "evaluators" / label
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "temperature": temperature, "arch": cfg["arch"], "n_classes": n_classes}, out / "model.pt")
        write_json(Path(dcfg["reports_root"]) / f"evaluator_{label}.json", {**provenance(cfg, cfg["seed"], dataset_hash), **metrics})
        log.info("%s: val %.4f, test top-1 %.4f, ECE %.4f -> %.4f (T=%.3f)", label, val_acc, metrics["test_top1"],
                 metrics["test_ece_uncalibrated"], metrics["test_ece_calibrated"], temperature)


if __name__ == "__main__":
    main()
