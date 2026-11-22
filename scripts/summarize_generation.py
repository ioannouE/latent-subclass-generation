"""Summary of one rung of the generation ladder (CPU, seconds): the test scores of every seed of `ldm.name` and their mean and standard
deviation across seeds, from <reports_root>/generation[_<labels>]/<name>_s<seed>_step*_A/ (slurm/generate.sh):
    python scripts/summarize_generation.py --config configs/gen_coarse_maskcon_new.yaml
Output: <reports_root>/generation[_<labels>]/<name>.{csv,md}, with links to the seed-0 figures.
"""
import argparse
from pathlib import Path

import pandas as pd

from lsgen.data.labels import suffix
from lsgen.eval.report import exclusion_note, generation_row
from lsgen.utils import REPO_ROOT, load_config, provenance

METRICS = ["G1 FID", "G1 KID", "G1 recall", "G2 make acc", "G3 TV", "G3 KL", "G4 covered", "G4 rare covered", "G5 excess KID",
           "G5 coverage ratio", "G7 copies", "G7 copies, real"]
OPTIONAL = ["G6 matched KID", "G6 purity", "G6 unmatched real", "E1 fine acc (generated)", "E1 fine acc (real val)"]  # cluster rungs; oracle (E1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_coarse_maskcon_new.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    reports, name = Path(dcfg["reports_root"]) / f"generation{suffix(cfg['labels'], dcfg)}", cfg["ldm"]["name"]
    dirs = sorted(reports.glob(f"{name}_s*_step*_A"))
    rows = pd.DataFrame([generation_row(d) for d in dirs])
    metrics = METRICS + [m for m in OPTIONAL if m in rows]
    per_seed = rows[[c for c in ("seed", "step", "train_hours", "ddim_steps", "guidance", "n_labels") if c in rows] + metrics]
    per_seed.to_csv(reports / f"{name}.csv", index=False)
    agg = per_seed[metrics].agg(["mean", "std"]).T.rename_axis("metric").reset_index()
    p = provenance(cfg, cfg["seed"])
    ex = exclusion_note(Path(dcfg["data_root"]) / "splits", labels=cfg["labels"])
    figs = ", ".join(f"[{f}]({dirs[0].name}/{f})" for f in ("samples.png", "nearest_real.png", "embedding.png", "g7_nn_grid.png")) if dirs else ""
    md = [f"# Generation: {name} ({cfg['labels']})\n", f"Provenance: git `{p['git_hash']}`, config `{cfg['_config_path']}`, seeds "
          f"{per_seed.seed.tolist()}. {ex['note'].capitalize()}: {ex['excluded_conflicting_images']} (train / val / test). Scored on test.\n",
          f"## Mean and standard deviation over {len(per_seed)} seeds\n", agg.to_markdown(index=False, floatfmt=".4g"),
          "\n## Per seed\n", per_seed.to_markdown(index=False, floatfmt=".4g"), f"\nFigures (seed {per_seed.seed.iloc[0]}): {figs}\n"]
    (reports / f"{name}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
