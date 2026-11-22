"""G-0c summary (CPU, seconds): the autoencoder comparison (val reconstructions) and the pilot curves (val metrics against training step,
per number of DDIM steps) from the scored sets in <reports_root>/generation[_<labels>]/ (slurm/autoencoders.sh, slurm/gen_pilot.sh):
    python scripts/summarize_pilot.py --config configs/gen_pilot_maskcon_new.yaml
Output: <reports_root>/generation[_<labels>]/<run>_pilot.{csv,md,png}. Everything is scored against val; test is not used.
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

from lsgen.data.labels import suffix
from lsgen.data.splits import load_split
from lsgen.eval.report import exclusion_note, generation_row
from lsgen.utils import REPO_ROOT, load_config, provenance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CURVES = ["G1 FID", "G1 KID", "G3 TV", "G5 excess KID", "G7 copies", "G2 make acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_pilot_maskcon_new.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    reports = Path(dcfg["reports_root"]) / f"generation{suffix(cfg['labels'], dcfg)}"
    run = cfg["ldm"]["name"]
    recon = pd.DataFrame([generation_row(d) for d in sorted(reports.glob("recon_*_A"))])
    pilot = pd.DataFrame([generation_row(d) for d in sorted(reports.glob(f"{run}_step*_A"))])
    if len(pilot):
        pilot = pilot.sort_values(["ddim_steps", "step"])
        pilot.to_csv(reports / f"{run}_pilot.csv", index=False)
        fig, axes = plt.subplots(1, len(CURVES), figsize=(4 * len(CURVES), 3.2))
        for ax, m in zip(axes, CURVES):
            for steps, df in pilot.groupby("ddim_steps"):
                ax.plot(df.step, df[m], marker="o", label=f"DDIM {steps}")
            if m == "G7 copies":
                ax.axhline(pilot["G7 copies, real"].iloc[0], color="grey", ls="--", label="real val")
            ax.set(title=m, xlabel="training step")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(reports / f"{run}_pilot.png", dpi=120)

    p = provenance(cfg, cfg["seed"])
    ex = exclusion_note(Path(dcfg["data_root"]) / "splits", labels=cfg["labels"])
    n_val = len(load_split(Path(dcfg["data_root"]) / "splits", "val", labels=cfg["labels"]))
    note = f"{ex['note'].capitalize()}: {ex['excluded_conflicting_images']} (train / val / test)."
    cols_recon = ["name", "psnr", "G1 FID", "G1 KID", "G1 recall", "G2 make acc", "E1 fine acc (generated)", "E1 fine acc (real val)"]
    cols_pilot = ["step", "train_hours", "ddim_steps", "sampling_seconds", "G1 FID", "G1 KID", "G1 recall", "G2 make acc", "G3 TV",
                  "G4 covered", "G5 excess KID", "G5 coverage ratio", "G7 copies", "G7 copies, real"]
    md = [f"# G-0c pilot: {run}\n", f"Provenance: git `{p['git_hash']}`, seed {cfg['seed']}, config `{cfg['_config_path']}`, label scheme "
          f"`{cfg['labels']}`. {note}\n", "Every set is scored against **val** (G1, G3: val reference; G4/G5/G7: train as usual); test is not used. "
          f"G1 is computed at n = {n_val} (the val size), so FID is biased upwards: compare rows, not absolute values; KID is unbiased.\n",
          "## Autoencoders: val reconstructions\n", recon[cols_recon].to_markdown(index=False, floatfmt=".4g") if len(recon) else "none yet",
          f"\n## Coarse-only latent diffusion ({run}): val metrics against training step\n",
          pilot[cols_pilot].to_markdown(index=False, floatfmt=".4g") if len(pilot) else "none yet", ""]
    (reports / f"{run}_pilot.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
