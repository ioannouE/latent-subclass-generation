"""Milestone 3: train the evaluator classifiers (hidden subclass `fine` / coarse class `make` of the config's label scheme). Needs a GPU:
    sbatch slurm/train_eval_classifier.sh
Model selection and temperature scaling use val only; test is scored once at the end. Conflicting images (the label scheme's exclusion
list, configs/labels.yaml) are left out of train, val and test.
Output: <derived_root>/evaluators[_<labels>]/<head>/model.pt and <reports_root>/evaluator[_<labels>]_<head>.json (suffix only for a scheme
other than the data config's own, lsgen.data.labels.suffix).
"""
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms as T

from lsgen.data.labels import load_labels, suffix
from lsgen.data.splits import load_split
from lsgen.eval.classifier import MEAN_STD, LabelledImages, ece, eval_transform, fit_temperature, predict, train_classifier
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, write_json

log = logging.getLogger("train_eval_classifier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classifier.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("classifier training needs a GPU: sbatch slurm/train_eval_classifier.sh")
    device = torch.device("cuda")
    labels = cfg.get("labels", "make")
    derived, splits_dir, sfx = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits", suffix(labels, dcfg)
    meta = load_labels(derived, dcfg["data_root"], labels)
    dataset_hash = json.loads((Path(dcfg["data_root"]) / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    size = cfg["input_size"]
    train_tf = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(*MEAN_STD)])

    for head in cfg["heads"]:
        torch.manual_seed(cfg["seed"])
        n_classes = meta[f"{head}_id"].nunique()
        dls = {}
        for split, tf in (("train", train_tf), ("val", eval_transform(size)), ("test", eval_transform(size))):
            ids = load_split(splits_dir, split, labels=labels)
            ds = LabelledImages([derived / cfg["crop"] / f"{i}.png" for i in ids], meta.loc[ids, f"{head}_id"].tolist(), tf)
            dls[split] = torch.utils.data.DataLoader(ds, cfg["batch_size"], shuffle=split == "train", drop_last=split == "train",
                                                     num_workers=cfg["num_workers"])
        log.info("%s %s (%d classes): %s images", labels, head, n_classes, {s: len(d.dataset) for s, d in dls.items()})
        model, val_acc, history = train_classifier(cfg["arch"], n_classes, dls["train"], dls["val"], device,
                                                   cfg["epochs"], cfg["lr"], cfg["weight_decay"])
        val_logits, val_y = predict(model, dls["val"], device)
        temperature = fit_temperature(val_logits, val_y)
        test_logits, test_y = predict(model, dls["test"], device)
        metrics = {
            "label": head, "labels": labels, "n_classes": n_classes, "n": {s: len(d.dataset) for s, d in dls.items()},
            "val_top1": val_acc, "temperature": temperature,
            "test_top1": (test_logits.argmax(1) == test_y).float().mean().item(),
            "test_ece_uncalibrated": ece(test_logits.softmax(1), test_y),
            "test_ece_calibrated": ece((test_logits / temperature).softmax(1), test_y),
            "test_nll_uncalibrated": F.cross_entropy(test_logits, test_y).item(),
            "test_nll_calibrated": F.cross_entropy(test_logits / temperature, test_y).item(),
            "history": history}
        out = derived / f"evaluators{sfx}" / head
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "temperature": temperature, "arch": cfg["arch"], "n_classes": n_classes,
                    "input_size": size, "labels": labels}, out / "model.pt")
        write_json(Path(dcfg["reports_root"]) / f"evaluator{sfx}_{head}.json", {**provenance(cfg, cfg["seed"], dataset_hash), **metrics})
        log.info("%s: val %.4f, test top-1 %.4f, ECE %.4f -> %.4f (T=%.3f)", head, val_acc, metrics["test_top1"],
                 metrics["test_ece_uncalibrated"], metrics["test_ece_calibrated"], temperature)


if __name__ == "__main__":
    main()
