"""Controls x metrics (docs/PLAN.md A5, Milestone 5; CPU, seconds). Needs the control results of slurm/eval_controls.sh.
Writes <out>.csv (one row per control and metric), <out>.md and <out>.png (heat-map). Exits with 1 if a control does not
fail a metric it must fail: stop and report, do not tune.
    python scripts/metric_validation.py --config configs/metric_validation.yaml
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from lsgen.eval.report import exclusion_note
from lsgen.eval.validation import read_metric, verdict, worse_than
from lsgen.utils import REPO_ROOT, git_hash, load_config, setup_logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("metric_validation")
COLORS = {"ok": "#7fbf7b", "MISSED": "#e66101", "also": "#fdd49e", "same": "#f0f0f0"}
SYMBOL = {"ok": "ok", "MISSED": "MISSED", "also": "also", "same": ""}


def collect(cfg, runs_dir):
    rows = []
    for case, specs in cfg["metrics"].items():
        base_dir = runs_dir / f"{cfg['baseline'][case]}_{case}"
        controls = sorted({p.name[:-2] for p in runs_dir.glob(f"*_{case}")} - {cfg["baseline"][case]})
        for name, (table, column, better) in specs.items():
            b_set = cfg["b_set"] if case == "B" else None
            base = read_metric(base_dir, table, column, b_set)
            for control in [cfg["baseline"][case]] + controls:
                value = read_metric(runs_dir / f"{control}_{case}", table, column, b_set)
                expected = name in cfg["expected"][case].get(control, [])
                is_worse = control != cfg["baseline"][case] and worse_than(*value, *base, better, cfg["margin"])
                rows.append(dict(use_case=case, control=control, metric=name, better=better, value=value[0], lo=value[1], hi=value[2],
                                 floor=base[0], floor_lo=base[1], floor_hi=base[2], expected=expected, worse_than_floor=is_worse,
                                 verdict="floor" if control == cfg["baseline"][case] else verdict(is_worse, expected)))
    return pd.DataFrame(rows)


def heatmap(df, path, b_set):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [12, 5]})
    for ax, case in zip(axes, "AB"):
        d = df[(df.use_case == case) & (df.verdict != "floor")]
        controls, metrics = list(dict.fromkeys(d.control)), list(dict.fromkeys(d.metric))
        for i, c in enumerate(controls):
            for j, m in enumerate(metrics):
                r = d[(d.control == c) & (d.metric == m)].iloc[0]
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=COLORS[r.verdict], ec="white"))
                ax.text(j + 0.5, i + 0.5, f"{r.value:.3g}\n{SYMBOL[r.verdict]}", ha="center", va="center", fontsize=7)
        ax.set_xlim(0, len(metrics)), ax.set_ylim(len(controls), 0)
        ax.set_xticks(np.arange(len(metrics)) + 0.5, metrics, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(controls)) + 0.5, controls, fontsize=8)
        ax.set_title(f"Use case {case}" + (f" ({b_set})" if case == "B" else ""), fontsize=10)
    fig.suptitle("Controls x metrics. green ok: expected failure, worse than the real-vs-real floor | orange MISSED: expected, not worse "
                 "| light: worse, not expected | grey: unchanged", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/metric_validation.yaml")
    cfg = load_config(ap.parse_args().config)
    setup_logging()
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    runs_dir, out = Path(dcfg["reports_root"]) / "generation", REPO_ROOT / cfg["out"]
    df = collect(cfg, runs_dir)
    df.to_csv(out.with_suffix(".csv"), index=False)
    heatmap(df, out.with_suffix(".png"), cfg["b_set"])
    missed = df[df.verdict == "MISSED"]
    floor = df[df.verdict == "floor"].pivot(index="metric", columns="use_case", values="value")
    out.with_suffix(".md").write_text(
        f"# Metric validation (controls x metrics)\n\nProvenance: git `{git_hash()}`, seed {json.loads((runs_dir / (cfg['baseline']['A'] + '_A') / 'summary.json').read_text())['seed']}, "
        f"config `{cfg['_config_path']}`. {exclusion_note(Path(dcfg['data_root']) / 'splits')['note']}: "
        f"{exclusion_note(Path(dcfg['data_root']) / 'splits')['excluded_conflicting_images']} (train / val / test).\n\n"
        f"Floor = {cfg['baseline']['A']} (A), {cfg['baseline']['B']} (B, set {cfg['b_set']}). Verdicts: ok = expected and clearly worse than the floor; "
        f"MISSED = expected, not worse; also = worse though not expected; same = unchanged.\n\n"
        f"## Expected failures that were missed: {len(missed)}\n\n{missed[['use_case', 'control', 'metric', 'value', 'floor']].to_markdown(index=False) if len(missed) else 'none'}\n\n"
        f"## All controls\n\n{df.round(4).drop(columns=['better', 'floor_lo', 'floor_hi']).to_markdown(index=False)}\n")
    log.info("wrote %s.{csv,md,png}; %d expected failures missed", out, len(missed))
    sys.exit(1 if len(missed) else 0)


if __name__ == "__main__":
    main()
