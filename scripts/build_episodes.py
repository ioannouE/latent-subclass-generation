"""Milestone 2: build and freeze use-case-B episodes from the Milestone 1 outputs.

CPU only, a few seconds (no GPU, no images read). Needs prepare_data.py stages up to `finalize`
(metadata.parquet with `exclude`, duplicates.csv, duplicates_within_test.csv).
    python scripts/build_episodes.py --config configs/episodes.yaml
Refuses to overwrite existing episode files unless --overwrite (episodes are frozen).
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from lsgen.data.episodes import build_mixed, build_single, flat, near_dup_index
from lsgen.utils import REPO_ROOT, load_config, provenance, setup_logging, sha256_file, write_json

log = logging.getLogger("build_episodes")


def write_episodes(path, header, episodes):
    """Header + one episode per line (diff-friendly, compact)."""
    body = ",\n".join(json.dumps(e, separators=(",", ":")) for e in episodes)
    path.write_text(json.dumps(header, indent=1, default=str)[:-2] + ',\n "episodes": [\n' + body + "\n]}\n")


def pools(meta, split):
    m = meta[(meta.split == split) & ~meta.exclude]
    return {int(g): sorted(ids) for g, ids in m.groupby("fine_id").id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/episodes.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    out = REPO_ROOT / cfg["out_dir"]
    if out.exists() and any(out.glob("*.json")) and not args.overwrite:
        raise SystemExit(f"{out} already holds frozen episodes; pass --overwrite to rebuild (documented change only)")
    out.mkdir(parents=True, exist_ok=True)

    data = Path(dcfg["data_root"])
    meta = pd.read_parquet(Path(dcfg["derived_root"]) / "metadata.parquet",
                           columns=["id", "split", "fine_id", "make_id", "exclude"])
    dataset_hash = json.loads((data / "manifests" / "raw_manifest.json").read_text())["manifest_hash"]
    cross = pd.read_csv(data / "duplicates.csv")
    within = pd.read_csv(data / "duplicates_within_test.csv")
    near_dup = near_dup_index(list(zip(cross.test_id, cross.train_id)) + list(zip(within.id_a, within.id_b)))
    make_of = meta.groupby("fine_id").make_id.first().astype(int).to_dict()
    test_pools, train_pools = pools(meta, "test"), pools(meta, "train")
    log.info("pools: %d test images in %d classes, %d train images (%d conflicting images excluded)",
             sum(map(len, test_pools.values())), len(test_pools), sum(map(len, train_pools.values())), int(meta.exclude.sum()))

    seed, s, mx = cfg["seed"], cfg["single"], cfg["mixed"]
    prov = provenance(cfg, seed, dataset_hash)
    prov["inputs_sha256"] = {"duplicates.csv": sha256_file(data / "duplicates.csv"),
                             "duplicates_within_test.csv": sha256_file(data / "duplicates_within_test.csv"),
                             "exclude.txt": sha256_file(data / "splits" / "exclude.txt")}
    summary, files = {}, {}
    for src in s["support_sources"]:
        family = f"single_{src}"
        eps, elig, skipped = build_single(test_pools if src == "test" else train_pools, test_pools, make_of, s["ks"],
                                          s["n_episodes"], s["min_target"], seed, family, near_dup, cfg["max_redraws"])
        for k in s["ks"]:
            e = [x for x in eps if x["k"] == k]
            name = f"{family}_k{k}.json"
            write_episodes(out / name, {**prov, "family": family, "k": k, "n_episodes": len(e),
                                        "n_eligible_classes": elig[k],
                                        "skipped": [x[1:] for x in skipped if x[0] == k]}, e)
            files[name] = sha256_file(out / name)
            summary[name] = {"episodes": len(e), "eligible_classes": elig[k], "skipped_classes": len(test_pools) - elig[k],
                             "with_near_dup_S_T": sum(x["n_near_dup_S_T"] > 0 for x in e),
                             "redrawn": sum(x["redraws"] > 0 for x in e),
                             "target_size_min_median": _minmed([len(x["target"]) for x in e])}
    eps, pairs_used, skipped = build_mixed(test_pools, make_of, mx["k"], mx["ratios"], mx["n_episodes"], mx["min_target"],
                                           mx["max_pairs_per_make"], seed, near_dup, cfg["max_redraws"])
    name = f"mixed_k{mx['k']}.json"
    write_episodes(out / name, {**prov, "family": "mixed", "k": mx["k"], "n_episodes": len(eps),
                                "pairs_per_make": pairs_used, "skipped": skipped}, eps)
    files[name] = sha256_file(out / name)
    summary[name] = {"episodes": len(eps), "makes": len(pairs_used), "pairs": sum(pairs_used.values()),
                     "skipped_pair_ratios": len(skipped), "with_near_dup_S_T": sum(x["n_near_dup_S_T"] > 0 for x in eps),
                     "redrawn": sum(x["redraws"] > 0 for x in eps),
                     "target_size_min_median": _minmed([len(flat(x, "target")) for x in eps])}
    write_json(out / "manifest.json", {**prov, "files": files, "summary": summary})
    _report(dcfg["reports_root"], prov, summary, make_of, pairs_used)
    for k, v in summary.items():
        log.info("%s: %s", k, v)


def _minmed(x):
    x = sorted(x)
    return [x[0], x[len(x) // 2]] if x else [None, None]


def _report(reports_root, prov, summary, make_of, pairs_used):
    kc = Counter(make_of.values())
    lines = ["# Episodes (Milestone 2)", "", "<!-- generated by scripts/build_episodes.py; do not edit -->",
             f"Provenance: git `{prov['git_hash']}`, seed {prov['seed']}, dataset manifest `{prov['dataset_manifest_hash'][:16]}`, "
             f"created {prov['created_utc']}", "",
             "Frozen files: `data/episodes/*.json` (sha256 in `data/episodes/manifest.json`). Label-conflict test images "
             "are excluded from S and T; label-conflict train images from train supports. S is redrawn to avoid known "
             "near-duplicate pairs with T.", "",
             "| file | " + " | ".join(sorted({k for v in summary.values() for k in v})) + " |",
             "|---|" + "---|" * len({k for v in summary.values() for k in v})]
    keys = sorted({k for v in summary.values() for k in v})
    lines += [f"| {f} | " + " | ".join(str(v.get(k, "")) for k in keys) + " |" for f, v in summary.items()]
    lines += ["", f"Mixed episodes: makes with K_c >= 2: {sum(1 for c in kc.values() if c >= 2)}; "
              f"pairs used per make: {json.dumps({str(k): v for k, v in sorted(pairs_used.items())})}", ""]
    Path(reports_root, "episodes.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
