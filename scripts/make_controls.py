"""Build the control sets of docs/PLAN.md A5 (CPU, minutes): use case A: C1 copy-train, C2 oracle split, C3 coarse-only,
C4 prototype collapse, C5 majority-only, C6 degraded; use case B: C1b copy-support, C1b retrieve-NN, C2 real images of
the subclass, C4 prototype collapse. Score them with slurm/eval_controls.sh.
    python scripts/make_controls.py --config configs/controls.yaml
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from lsgen.data.splits import load_split
from lsgen.eval import controls as C
from lsgen.features.extract import read_csv
from lsgen.utils import REPO_ROOT, git_hash, load_config, setup_logging

log = logging.getLogger("make_controls")


class Builder:
    def __init__(self, cfg, dcfg):
        self.cfg, self.derived = cfg, Path(dcfg["derived_root"])
        self.splits_dir, self.episodes_dir = Path(dcfg["data_root"]) / "splits", Path(dcfg["data_root"]) / "episodes"
        self.images = self.derived / f"bbox15_{cfg['crop_size']}"
        self.meta = pd.read_parquet(self.derived / "metadata.parquet", columns=["id", "make_id", "fine_id"]).set_index("id")
        self.make_of_fine = self.meta.groupby("fine_id").make_id.first().sort_index().to_numpy()
        self.ids = {s: load_split(self.splits_dir, s) for s in ("train", "test", "test_A")}
        self.feats = {}
        for s in ("train", "test"):
            files, x = read_csv(self.derived / "features" / f"{cfg['features']}_{cfg['crop_size']}" / f"{s}.csv")
            self.feats |= dict(zip((f[:-4] for f in files), x))
        train, fine = np.array(self.ids["train"]), self.meta.fine_id[self.ids["train"]].to_numpy()
        self.by_fine = {g: train[fine == g] for g in np.unique(fine)}
        self.medoids = {g: ids[C.medoid(np.stack([self.feats[i] for i in ids]))] for g, ids in self.by_fine.items()}

    def write(self, name, case, items, uses=(), **method):
        """items: [(columns of samples.csv, source image id, transform or None)]. Unmodified images are symlinked."""
        out = self.derived / self.cfg["out_dir"] / f"{name}_{case}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        rows = []
        for k, (cols, src, transform) in enumerate(items):
            f = f"{k:06d}.png"
            if transform is None:
                (out / f).symlink_to((self.images / f"{src}.png").resolve())
            else:
                transform(Image.open(self.images / f"{src}.png").convert("RGB")).save(out / f)
            rows.append({"filename": f, **cols})
        pd.DataFrame(rows).to_csv(out / "samples.csv", index=False)
        (out / "method.json").write_text(json.dumps({"name": name, "uses": list(uses), "git_hash": git_hash(), "seed": self.cfg["seed"], **method}, indent=1))
        log.info("%s_%s: %d images", name, case, len(rows))

    # ---- use case A
    def use_case_a(self):
        seed, mk = self.cfg["seed"], self.meta.make_id
        requested = mk[self.ids["test"]].to_numpy()  # one image per test image, same make prevalence
        train_fine = self.meta.fine_id[self.ids["train"]].to_numpy()
        random_image = C.random_image(self.by_fine)
        specs = {  # name: (subclass weights, image of a subclass, transform)
            "c1_copy_train": ("prevalence", random_image, None),
            "c3_coarse_only": ("uniform", random_image, None),
            "c4_prototype_collapse": ("prevalence", lambda g, rng: self.medoids[g], None),
            "c5_majority_only": ("majority", random_image, None),
            "c6_degraded": ("prevalence", random_image, C.degrade)}
        for i, (name, (mode, image_of, transform)) in enumerate(specs.items()):
            rng = np.random.default_rng([seed, i])
            ids = C.draw_set(requested, self.make_of_fine, C.fine_weights(train_fine, self.make_of_fine, mode), image_of, rng)
            self.write(name, "A", [({"make_id": m}, i, transform) for m, i in zip(requested, ids)],
                       uses=[self.cfg["features"]] if name == "c4_prototype_collapse" else [])
        half = self.ids["test_A"]  # C2: scored against the other half of test
        self.write("c2_oracle_split", "A", [({"make_id": mk[i]}, i, None) for i in half], reference_split="test_B")

    # ---- use case B
    def use_case_b(self):
        n, seed = self.cfg["n_episodes_images"], self.cfg["seed"]
        episodes = [e for s in self.cfg["episode_sets"] for e in json.loads((self.episodes_dir / f"{s}.json").read_text())["episodes"]]
        train, rng = self.ids["train"], np.random.default_rng(seed)
        train_feats = np.stack([self.feats[i] for i in train])
        sets = {k: [] for k in ("c1b_copy_support", "c1b_retrieve_nn", "c2_oracle_real", "c4_prototype_collapse")}
        for e in episodes:
            support = e["support"] if "support" in e else sum(e["support_by_class"].values(), [])
            counts = C.split_counts(n, e["fine_ids"], e["ratio"])
            augment = lambda im: C.augment(im, rng)  # noqa: E731
            picked = {"c1b_copy_support": [(i, augment) for i in C.copy_support_ids(support, n, rng)],
                      "c1b_retrieve_nn": [(i, None) for i in C.retrieve_nn_ids(np.stack([self.feats[i] for i in support]), train, train_feats, n)],
                      "c2_oracle_real": [(i, None) for i in C.real_ids(e["fine_ids"], counts, self.by_fine, rng)],
                      "c4_prototype_collapse": [(i, None) for i in C.collapse_ids(e["fine_ids"], counts, self.medoids)]}
            for name, imgs in picked.items():
                sets[name] += [({"episode_id": e["episode_id"]}, i, t) for i, t in imgs]
        for name, items in sets.items():
            self.write(name, "B", items, uses=[self.cfg["features"]] if name in ("c1b_retrieve_nn", "c4_prototype_collapse") else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/controls.yaml")
    cfg = load_config(ap.parse_args().config)
    setup_logging()
    b = Builder(cfg, load_config(REPO_ROOT / cfg["data_config"]))
    b.use_case_a()
    b.use_case_b()


if __name__ == "__main__":
    main()
