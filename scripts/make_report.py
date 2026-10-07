"""Milestone 6: assemble reports/report.md (data summary, metric validation, real-vs-real floor, frozen-encoder table, per-make
breakdown) plus reports/repr_frozen.csv and reports/repr_per_make.csv from the stored results. CPU, seconds:
    python scripts/make_report.py --config configs/report.yaml
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from lsgen.data.splits import load_split
from lsgen.eval.report import exclusion_note, macro_row, per_make_table, worst_makes
from lsgen.utils import REPO_ROOT, git_hash, load_config, setup_logging

log = logging.getLogger("make_report")
ORACLE_LABEL = "supervised oracle features (not a baseline)"
CAVEATS = """\
- **Single seed.** Every number comes from one seed (0); differences of a few hundredths between encoders can be within the CIs.
- **K-hat over-fragments.** The label-free K-hat (silhouette sweep, gap-statistic fallback) never returned 1 on the single-subclass
  makes for any DINOv3 or CLIP embedding (checked on `train`, docs/METRICS.md). Treat the K-hat columns as the behaviour of a label-free
  per-make pipeline, not as a property of the encoder alone.
- **CIs.** Oracle-K, R1, R3: bootstrap over images within each make (1000 resamples). K-hat metrics and R4 `var_ratio`: draws of 63.2% of a
  make's images without replacement (resamples with duplicates bias them). R4 effective rank (`pr_subclass`) is a point estimate.
- **Padding.** 74% of the crops have more than 10% padding (a car's longer side exceeds the photo's short side); the whole car is kept
  (docs/DATA.md). The padding gate of the spec is therefore a warning, not a stop.
- **Evaluator accuracy.** The fine classifier reaches the accuracy reported above on test; its mistakes propagate into G3-G5 and B1, B5.
"""


def fmt(row, name, digits=3):
    lo, hi = row.get(f"{name}_lo"), row.get(f"{name}_hi")
    ci = "" if pd.isna(lo) or pd.isna(hi) else f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    return f"{row[name]:.{digits}f}{ci}"


def frozen_table(rows):
    """Markdown table of the macro metrics of {label: row}."""
    cols = {"kNN make": "knn_make", "probe make": "probe_make", "Recall@1 (in make)": "recall1", "fine probe (in make)": "probe_fine",
            "ACC oracle K": "acc_oracle", "NMI oracle K": "nmi_oracle", "ACC K-hat": "acc_khat", "mean K-hat": "khat",
            "d0/d1": "d0/d1", "d1/d2": "d1/d2", "% Eq.(2)": "eq2", "var. ratio": "var_ratio", "ARI boot": "ari_boot", "K-hat err (K_c=1)": "khat_error"}
    out = []
    for label, row in rows.items():
        row = row | {f"eq2{s}": row[f"eq2{s}"] * 100 for s in ("", "_lo", "_hi")}  # share of makes, in %
        out.append({"encoder": label} | {c: fmt(row, k, 1 if c in ("% Eq.(2)", "mean K-hat") else 3) for c, k in cols.items()})
    return pd.DataFrame(out).to_markdown(index=False)


def data_summary(dcfg, hierarchy):
    splits = Path(dcfg["data_root"]) / "splits"
    n = {s: len(load_split(splits, s)) for s in ("train", "val", "test")}
    removed = exclusion_note(splits)
    dup = json.loads((Path(dcfg["data_root"]) / "manifests" / "duplicates.json").read_text())
    k = hierarchy.groupby("make_id").K_c.first()
    hist = k.value_counts().sort_index()
    ev = {l: json.loads((Path(dcfg["reports_root"]) / f"evaluator_{l}.json").read_text()) for l in ("fine", "make")}
    return f"""\
- Images after excluding conflicting ones: {n} (removed: {removed['excluded_conflicting_images']}, {sum(removed['excluded_conflicting_images'].values())} in total). {removed['note'].capitalize()}.
- 49 makes, 196 fine classes. **K_c histogram** (subclasses per make: number of makes): {dict(zip(hist.index.tolist(), hist.tolist()))}; {int((k == 1).sum())} makes have K_c = 1.
- **Near-duplicates (official train vs test):** {dup['n_pairs']} candidate pairs ({dup['n_exact_duplicate']} byte-identical, {dup['n_exact_duplicate_label_conflict']} of them with different fine labels), marked in the metadata, never removed; {dup['n_pairs_within_test']} pairs inside test.
- **Evaluator classifiers** (ConvNeXt-T, 128 px, temperature-scaled on val; evaluation only): fine (196) test top-1 {ev['fine']['test_top1']:.3f}, ECE {ev['fine']['test_ece_calibrated']:.3f}; make (49) test top-1 {ev['make']['test_top1']:.3f}, ECE {ev['make']['test_ece_calibrated']:.3f}.

![data summary](data_summary.png)

Details: [data_summary.md](data_summary.md), [preprocessing.md](preprocessing.md), [episodes.md](episodes.md).
"""


def floor_table(validation_csv):
    v = pd.read_csv(validation_csv)
    f = v[v.verdict == "floor"]
    f = f.assign(text=lambda d: d.apply(lambda r: f"{r.value:.4g}" + ("" if pd.isna(r.lo) else f" [{r.lo:.3g}, {r.hi:.3g}]"), axis=1))
    return f.pivot(index="metric", columns="use_case", values="text").rename(columns={"A": "A: test_A vs test_B", "B": "B: real episodes"}).reset_index().to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/report.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    reports, repr_dir = Path(dcfg["reports_root"]), Path(dcfg["reports_root"]) / "repr"
    rcfg = load_config(REPO_ROOT / cfg["repr_config"])
    validation = REPO_ROOT / cfg["validation"]
    for suffix in (".csv", ".md", ".png"):
        if not validation.with_suffix(suffix).exists():
            raise FileNotFoundError(f"{validation.with_suffix(suffix)} missing: run scripts/metric_validation.py first")
    hierarchy = pd.read_csv(Path(dcfg["data_root"]) / "hierarchy.csv")

    runs = {f"{enc}_{size}": repr_dir / f"{enc}_{size}" for enc in rcfg["encoders"] for size in rcfg["crop_sizes"]}
    rows = {name: macro_row(d) for name, d in runs.items()}
    oracle_dir = repr_dir / cfg["oracle"]
    if oracle_dir.exists():
        rows_with_oracle = rows | {f"{cfg['oracle']}: {ORACLE_LABEL}": macro_row(oracle_dir)}
    else:
        log.warning("%s not found: the supervised oracle row is left out (run configs/eval_repr_oracle.yaml)", oracle_dir)
        rows_with_oracle = rows
    pd.DataFrame(rows_with_oracle).T.rename_axis("encoder").to_csv(REPO_ROOT / cfg["tables"]["frozen"])

    names = hierarchy.drop_duplicates("make_id").set_index("make_id")[["make", "K_c"]]
    per_make = pd.concat({n: per_make_table(d) for n, d in (runs | ({cfg["oracle"]: oracle_dir} if oracle_dir.exists() else {})).items()}, names=["encoder"])
    per_make = per_make.join(names, on="make_id")
    per_make.to_csv(REPO_ROOT / cfg["tables"]["per_make"])
    primary = per_make.loc[cfg["primary"]]
    show = lambda t, cols: t[["make", "K_c", "n_test"] + cols].round(3).reset_index(drop=True).to_markdown(index=False)  # noqa: E731
    r2 = worst_makes(primary, "acc_oracle", cfg["worst_n"])
    r3 = worst_makes(primary, "d0/d1", cfg["worst_n"], lowest=False)

    removed = exclusion_note(Path(dcfg["data_root"]) / "splits")["excluded_conflicting_images"]
    removed_total, removed_text = sum(removed.values()), " / ".join(f"{s} {n}" for s, n in removed.items())
    text = f"""\
# Subclass-preserving generation benchmark: Step 1 report

Generated by `scripts/make_report.py` (git `{git_hash()}`, config `{cfg['_config_path']}`). **{removed_total} conflicting images ({removed_text}) are left out of
every experiment below** (docs/DATA.md). Metric definitions: docs/METRICS.md.

## 1. Data
{data_summary(dcfg, hierarchy)}
## 2. Metric validation: controls x metrics
Every control must fail the metrics it is built to break, relative to the real-vs-real floor (C2). Fixed in `configs/metric_validation.yaml` before the first run.

![metric validation](metric_validation.png)

Full table, verdicts and missed expectations: [metric_validation.md](metric_validation.md), [metric_validation.csv](metric_validation.csv).

## 3. Real-vs-real floor
What a perfect generator scores (C2: `test_A` as samples against `test_B` for use case A, test images drawn like generated ones for use case B). Values with 95% CIs where the metric has them.

{floor_table(validation.with_suffix('.csv'))}

## 4. Frozen encoders (R1-R5, test)
Macro means over makes with 95% CIs (kNN / probes fitted on train; per-make clustering of test embeddings). `ACC oracle K` uses the true number of subclasses, `ACC K-hat` does not. d0/d1 and d1/d2 are macro means of the per-make ratios (< 1 is the desired order), `% Eq.(2)` is the share of makes with d1 < d2. The last row, if present, is a reference trained with the fine labels.

{frozen_table(rows_with_oracle)}

Tables: [repr_frozen.csv](repr_frozen.csv) (with all CIs), per-encoder results in `repr/<encoder>_<crop>/`.

## 5. Per-make breakdown ({cfg['primary']})
Full table for every encoder and make: [repr_per_make.csv](repr_per_make.csv).

**{cfg['worst_n']} worst makes on R2 (lowest ACC of k-means with the true K):**

{show(r2, ['acc_oracle', 'nmi_oracle', 'acc_khat', 'khat', 'recall1'])}

**{cfg['worst_n']} worst makes on R3 (highest d0/d1: subclasses least separated within the make):**

{show(r3, ['d0/d1', 'd1/d2', 'eq1', 'eq2', 'acc_oracle'])}

## 6. Caveats
{CAVEATS}
"""
    out = REPO_ROOT / cfg["out"]
    out.write_text(text)
    log.info("wrote %s, %s, %s", out, cfg["tables"]["frozen"], cfg["tables"]["per_make"])


if __name__ == "__main__":
    main()
