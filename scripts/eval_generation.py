"""Score generated images (docs/METRICS.md): use case A (G1-G7) or B (B1-B6), per configs/eval_generation.yaml. GPU:
    sbatch slurm/eval_generation.sh
Refuses to run if the method declares an encoder that is also an evaluator (lsgen/eval/evaluators.py).
Output: <reports_root>/generation/<method name>_<use case>/*.csv + summary.json. Reference = test (and train for G4/G5/G7).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

from lsgen.data.splits import load_split
from lsgen.eval import memorization, support, synthesis
from lsgen.eval.classifier import calibrated_probs, load_classifier
from lsgen.eval.evaluators import check_disjoint, read_method
from lsgen.eval.report import write_results
from lsgen.eval.stats import bootstrap, bootstrap_table
from lsgen.features.encoders import Embedder
from lsgen.features.extract import embed_images, read_csv
from lsgen.utils import REPO_ROOT, load_config, setup_logging

log = logging.getLogger("eval_generation")
CLASSIFIER_TF = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # as in training


class Context:
    """Reference embeddings (cached), labels, and the evaluators needed to embed generated images."""

    def __init__(self, cfg, dcfg, device):
        self.cfg, self.device = cfg, device
        self.derived, self.splits_dir = Path(dcfg["derived_root"]), Path(dcfg["data_root"]) / "splits"
        self.sscd = self.derived / "weights" / Path(dcfg["duplicates"]["sscd_url"]).name
        self.meta = pd.read_parquet(self.derived / "metadata.parquet", columns=["id", "make_id", "fine_id"]).set_index("id")
        self.make_of_fine = self.meta.groupby("fine_id").make_id.first().sort_index().to_numpy()
        self.dataset_hash = json.loads((self.splits_dir.parent / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]

    def reference(self, space, split):
        """(ids, embeddings) of a split, from the cache written by scripts/extract_features.py."""
        files, x = read_csv(self.derived / "features" / f"{space}_{self.cfg['crop_size']}" / f"{split}.csv")
        return [f[:-4] for f in files], x

    def labels(self, ids, col):
        return self.meta.loc[ids, col].to_numpy()

    def embed(self, space, paths):
        model = Embedder(space, self.sscd).to(self.device)
        x = embed_images(model, paths, model.transform, self.device, self.cfg["batch_size"], self.cfg["num_workers"])
        del model
        torch.cuda.empty_cache()
        return x

    def probs(self, label, paths):
        model, temperature = load_classifier(self.derived / "evaluators" / label / "model.pt", self.device)
        return calibrated_probs(model, temperature, paths, CLASSIFIER_TF, self.device, self.cfg["batch_size"], self.cfg["num_workers"])


def subsample(x, n, seed):
    return x if len(x) <= n else x[np.random.default_rng(seed).choice(len(x), n, replace=False)]


def run_a(ctx, samples, paths):
    cfg, seed = ctx.cfg, ctx.cfg["seed"]
    req = samples.make_id.to_numpy()
    feats = {s: ctx.embed(s, paths) for s in ("inception", "clip_l", "sscd")}
    fine_probs, make_probs = ctx.probs("fine", paths), ctx.probs("make", paths)
    test_ids, train_ids = (ctx.reference("clip_l", s)[0] for s in ("test", "train"))
    test_fine, train_fine = ctx.labels(test_ids, "fine_id"), ctx.labels(train_ids, "fine_id")
    test_make = ctx.labels(test_ids, "make_id")
    ref = {(s, sp): ctx.reference(s, sp)[1] for s in feats for sp in ("test", "train")}
    groups = {c: np.flatnonzero(req == c) for c in np.unique(req)}
    for c in np.setdiff1d(np.unique(test_make), list(groups)):
        log.warning("no samples requested for make %d: skipped in G2-G6", c)
    t = {}

    n = min(len(samples), len(test_ids))  # G1 at matched n
    inc, inc_ref = subsample(feats["inception"], n, seed), subsample(ref["inception", "test"], n, seed)
    t["g1"] = pd.DataFrame([{"n": n, "fid": synthesis.frechet_distance(inc, inc_ref), "kid": synthesis.kid(inc, inc_ref, seed=seed),
                             **synthesis.prdc(inc_ref, inc, cfg["prdc_k"])}])

    g2 = pd.DataFrame({"make_id": req, "make_acc": synthesis.make_correct(make_probs, req)})
    t["g2_make_acc"] = bootstrap_table(g2, "make_id", ["make_acc"], cfg["n_boot"], seed, cfg["n_jobs"])
    test_make_counts = np.bincount(test_make, minlength=make_probs.shape[1])
    t["g2_prevalence"] = pd.DataFrame([{"tv": synthesis.prevalence_tv(make_probs.argmax(1), test_make_counts)}])

    test_fine_counts = np.bincount(test_fine, minlength=len(ctx.make_of_fine))
    t["g3"] = bootstrap(synthesis.g3_metric(fine_probs, req, ctx.make_of_fine, test_fine_counts), groups, cfg["n_boot"], seed, cfg["n_jobs"])

    val_ids = load_split(ctx.splits_dir, "val")
    val_probs = ctx.probs("fine", [ctx.derived / f"bbox15_{cfg['crop_size']}" / f"{i}.png" for i in val_ids])
    tau = synthesis.choose_tau(val_probs, ctx.labels(val_ids, "fine_id"), cfg["tau_target"])
    rare = synthesis.rare_classes(np.bincount(train_fine, minlength=len(ctx.make_of_fine)))
    cf = synthesis.ClassFidelity(feats["clip_l"], fine_probs, req, ctx.make_of_fine, tau, ref["clip_l", "test"], test_fine,
                                 ref["clip_l", "train"], train_fine, rare, cfg["min_samples"], cfg["coverage_k"], seed)
    t["g4_g5"] = bootstrap(cf, groups, cfg["n_boot"], seed, cfg["n_jobs"])
    t["g4_g5_per_class"] = pd.concat([cf.table(i) for i in groups.values()], ignore_index=True)

    if "group" in samples:
        group = samples.group.to_numpy()
        t["g6"] = bootstrap(synthesis.g6_metric(feats["clip_l"], group, fine_probs, req, ctx.make_of_fine, ref["clip_l", "test"], test_fine),
                            groups, cfg["n_boot"], seed, cfg["n_jobs"])

    sim, nn = memorization.nearest_similarity(feats["sscd"], ref["sscd", "train"])
    null = memorization.nearest_similarity(ref["sscd", "test"], ref["sscd", "train"])[0]
    t["g7"] = pd.DataFrame([memorization.copy_summary(sim, null, cfg["copy_threshold"]) | {"tau": tau, "n": len(sim)}])
    t["g7_samples"] = pd.DataFrame({"filename": samples.filename, "sscd_sim": sim, "nn_train_id": np.array(train_ids)[nn]})
    train_paths = [ctx.derived / f"bbox15_{cfg['crop_size']}" / f"{i}.png" for i in train_ids]
    return t, lambda out: memorization.nn_grid(paths, train_paths, sim, nn, out / "g7_nn_grid.png")


def load_episodes(episodes_dir):
    eps = {}
    for f in sorted(Path(episodes_dir).glob("*_k*.json")):
        eps |= {e["episode_id"]: e for e in json.loads(f.read_text())["episodes"]}
    return eps


def run_b(ctx, samples, paths):
    cfg, seed = ctx.cfg, ctx.cfg["seed"]
    episodes = load_episodes(ctx.splits_dir.parent / "episodes")
    clip, sscd, fine_probs = ctx.embed("clip_l", paths), ctx.embed("sscd", paths), ctx.probs("fine", paths)
    ref = {s: {i: v for sp in ("train", "test") for i, v in zip(*ctx.reference(s, sp))} for s in ("clip_l", "sscd")}
    rows = []
    for eid, idx in samples.groupby("episode_id").indices.items():
        e = episodes[eid]
        if len(idx) != cfg["n_episodes_images"]:
            log.warning("episode %s has %d images, expected %d", eid, len(idx), cfg["n_episodes_images"])
        S = e["support"] if "support" in e else sum(e["support_by_class"].values(), [])
        Tg = e["target"] if "target" in e else sum(e["target_by_class"].values(), [])
        get = lambda space, ids: np.stack([ref[space][i] for i in ids])  # noqa: E731
        row = {"episode_id": eid, "set": f"{e['family']}_k{e['k']}", "k": e["k"], "make_id": e["make_id"], "n": len(idx)}
        if len(e["fine_ids"]) == 1:
            row["hit"] = support.hit_rate(fine_probs[idx], e["fine_ids"][0])
        else:
            row["mixture_error"] = support.mixture_error(fine_probs[idx], e["fine_ids"], e["ratio"])
        row |= support.beyond_support(clip[idx], get("clip_l", S), get("clip_l", Tg), cfg["coverage_k"])
        row |= support.exemplar_dependence(sscd[idx], get("sscd", S), get("sscd", Tg))
        rows.append(row | support.diversity(clip[idx], get("clip_l", Tg)))
    per_episode = pd.DataFrame(rows)
    cols = [c for c in per_episode.columns if c not in ("episode_id", "set", "k", "make_id", "n")]
    by_set = []
    for s, df in per_episode.groupby("set"):  # per make + macro for every episode set; the macro rows over k are B6
        by_set.append(bootstrap_table(df, "make_id", cols, cfg["n_boot"], seed, cfg["n_jobs"]).assign(set=s, k=df.k.iloc[0]))
    return {"episodes": per_episode, "by_set": pd.concat(by_set, ignore_index=True)}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_generation.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    samples_dir = Path(cfg["samples_dir"])
    method = read_method(samples_dir)
    warnings = check_disjoint(method["uses"])  # raises on overlap
    for w in warnings:
        log.warning(w)
    if not torch.cuda.is_available():
        raise RuntimeError("generation evaluation needs a GPU: sbatch slurm/eval_generation.sh")
    ctx = Context(cfg, dcfg, torch.device("cuda"))
    samples = pd.read_csv(samples_dir / "samples.csv")
    paths = [samples_dir / f for f in samples.filename]
    with Image.open(paths[0]) as im:
        if im.size != (cfg["crop_size"],) * 2:
            raise ValueError(f"generated images must be {cfg['crop_size']} px PNGs like the real ones, got {im.size}")
    log.info("%s, use case %s: %d samples", method["name"], cfg["use_case"], len(samples))
    tables, extra_output = (run_a if cfg["use_case"] == "A" else run_b)(ctx, samples, paths)
    out = Path(dcfg["reports_root"]) / "generation" / f"{method['name']}_{cfg['use_case']}"
    write_results(out, tables, cfg, cfg["seed"], ctx.dataset_hash, ctx.splits_dir, method=method, same_family_warnings=warnings)
    if extra_output:
        extra_output(out)


if __name__ == "__main__":
    main()
