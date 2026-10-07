"""Side experiment: do Gram statistics carry fine-class (subclass) signal beyond the CLS embedding, or mostly colour and
background? CPU, after `sbatch slurm/extract_features.sh` has cached the gram_* embeddings:
    sbatch slurm/gram_check.sh
Within-make fine-class probe (fit on train, scored on test) for: CLS, Gram, CLS+Gram, and Gram with the linear
colour/background component regressed out. `gain_*` = CLS+Gram minus CLS per image (CIs: bootstrap within makes).
Also: how well each embedding predicts colour / background (ridge R2 on test).
Output: <reports_root>/gram_check/{probe_fine,colour_r2}.csv + summary.json.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

from lsgen.eval.report import write_results
from lsgen.eval.representation import probe_correct
from lsgen.eval.stats import bootstrap_table
from lsgen.features.extract import read_csv
from lsgen.utils import REPO_ROOT, load_config, setup_logging

log = logging.getLogger("gram_check")


def colour(path):
    """Mean RGB of the central half of the crop (car body) and of its outer border ring (background): 6 numbers."""
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    h, w = a.shape[:2]
    inner = np.zeros((h, w), bool)
    inner[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    ring = np.ones((h, w), bool)
    ring[h // 8:-h // 8, w // 8:-w // 8] = False
    return np.concatenate([a[inner].mean(0), a[ring].mean(0)])


def standardise(train, test):
    """Per-dimension z-score with train statistics, scaled so every embedding has the same total variance."""
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [((x - mu) / sd) / np.sqrt(train.shape[1]) for x in (train, test)]


def load_split_data(derived, cfg, split, meta):
    names = [cfg["cls"], *cfg["gram"]]
    feats = {n: read_csv(derived / "features" / f"{n}_{cfg['crop']}" / f"{split}.csv") for n in names}
    files = feats[cfg["cls"]][0]
    assert all(f[0] == files for f in feats.values()), "embeddings must list the same images in the same order"
    ids = [f[:-4] for f in files]
    rgb = np.stack([colour(derived / f"bbox15_{cfg['crop']}" / f"{i}.png") for i in ids])
    return meta.loc[ids], {n: f[1] for n, f in feats.items()}, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gram_check.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    derived, splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
    dataset_hash = json.loads((splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    meta = pd.read_parquet(derived / "metadata.parquet", columns=["id", "make_id", "fine_id"]).set_index("id")
    (mtr, ftr, ctr), (mte, fte, cte) = [load_split_data(derived, cfg, s, meta) for s in ("train", "test")]

    emb = {n: standardise(ftr[n], fte[n]) for n in ftr}  # name -> (train, test)
    for g in cfg["gram"]:
        emb[f"{cfg['cls']}+{g}"] = tuple(np.hstack([c, x]) for c, x in zip(emb[cfg["cls"]], emb[g]))
        fit = LinearRegression().fit(ctr, emb[g][0])
        emb[f"{g}_nocolour"] = tuple(x - fit.predict(c) for x, c in zip(emb[g], (ctr, cte)))

    make, fine = mte.make_id.to_numpy(), mte.fine_id.to_numpy()
    multi = [c for c in np.unique(make) if len(np.unique(fine[make == c])) >= 2]
    probe = pd.DataFrame({"make_id": make})
    for name, (xtr, xte) in emb.items():
        correct = np.full(len(make), np.nan)
        for c in multi:
            in_tr, in_te = (mtr.make_id == c).to_numpy(), make == c
            correct[in_te] = probe_correct(xtr[in_tr], mtr.fine_id[in_tr], xte[in_te], fine[in_te])
        probe[f"acc_{name}"] = correct
        log.info("%s: within-make fine probe %.4f", name, np.nanmean(correct))
    for g in cfg["gram"]:
        probe[f"gain_{g}"] = probe[f"acc_{cfg['cls']}+{g}"] - probe[f"acc_{cfg['cls']}"]
    probe = probe[probe[f"acc_{cfg['cls']}"].notna()]
    table = bootstrap_table(probe, "make_id", [c for c in probe if c != "make_id"], cfg["n_boot"], cfg["seed"], cfg["n_jobs"])

    r2 = []
    for name in [cfg["cls"], *cfg["gram"]]:
        xtr, xte = emb[name]
        pred = Ridge(alpha=1.0).fit(xtr, ctr).predict(xte)
        r2.append({"embedding": name, "r2_car_colour": r2_score(cte[:, :3], pred[:, :3]),
                   "r2_background": r2_score(cte[:, 3:], pred[:, 3:])})
    write_results(Path(dcfg["reports_root"]) / "gram_check", {"probe_fine": table, "colour_r2": pd.DataFrame(r2)}, cfg,
                  cfg["seed"], dataset_hash, splits_dir, n_train=len(mtr), n_test=len(mte))


if __name__ == "__main__":
    main()
