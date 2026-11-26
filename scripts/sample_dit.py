"""Sample use-case-A sets from the official DiT checkpoints of scripts/train_dit.py with DiT's own code: its model (EMA weights),
`forward_with_cfg` and `ddim_sample_loop` (eta = 0). Only the labels are ours: DiT's sample scripts draw them uniformly or from a fixed list,
use case A needs them in train prevalence (lsgen.methods.gen.ldm.prevalence_labels, from the run's labels.json). The initial and the sampler
noise are seeded by the run's seed, the same for every set. Latents are decoded by the same SD-VAE and resized to `image_size` (the scored
size). Sets are scored against `sampling.reference_split` (default val). Every checkpoint of every run in the config is sampled with each
number of steps in `sampling.ddim_steps`. GPU:
    sbatch slurm/sample_dit.sh
Output: <derived_root>/generation[_<labels>]/<run>/step<step>_ddim<steps>/ with samples.csv + method.json, as scripts/sample_ldm.py
(paths printed to stdout).
"""
import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from lsgen.data.labels import suffix
from lsgen.methods.gen.ldm import load_latents, prevalence_labels, runs
from lsgen.utils import REPO_ROOT, git_hash, load_config, setup_logging

log = logging.getLogger("sample_dit")
STAMP = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def train_hours(log_txt, step):
    """Hours from DiT's "Training for" log line to the line that saved checkpoint `step` (log.txt of DiT's train.py)."""
    t = {}
    for line in Path(log_txt).read_text().splitlines():
        if "Training for" in line or f"/{step:07d}.pt" in line:
            t.setdefault("start" if "Training for" in line else "end", datetime.strptime(STAMP.search(line).group(1), "%Y-%m-%d %H:%M:%S"))
    return (t["end"] - t["start"]).total_seconds() / 3600


@torch.no_grad()
def sample(model, diffusion, y, latent_size, w, null, device):
    """DiT's sample.py with DDIM: classifier-free guidance by forward_with_cfg on [z, z] with [y, null]."""
    z = torch.randn(len(y), 4, latent_size, latent_size, device=device)
    z, y = torch.cat([z, z]), torch.cat([y, torch.full_like(y, null)])
    out = diffusion.ddim_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs={"y": y, "cfg_scale": w}, device=device)
    return out.chunk(2)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen_dit_pilot_maskcon_new.yaml")
    cfg = load_config(ap.parse_args().config)
    dcfg = load_config(REPO_ROOT / cfg["data_config"])
    setup_logging()
    D, s = cfg["dit"], cfg["sampling"]
    sys.path.insert(0, str(REPO_ROOT / D["repo"]))
    from diffusion import create_diffusion  # noqa: E402  (external/DiT)
    from models import DiT_models  # noqa: E402
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # as DiT's sample.py
    device, derived, sfx = torch.device("cuda"), Path(dcfg["derived_root"]), suffix(cfg["labels"], dcfg)
    ae = load_latents("sd_vae", cfg["autoencoders"]["sd_vae"], cfg["image_size"], None).to(device)
    cond = cfg.get("conditioning", {"kind": "coarse"})

    for seed, run in runs(cfg, "dit"):
        out_run = derived / f"generation{sfx}" / run
        info = json.loads((out_run / "labels.json").read_text())
        pairs, n_classes = torch.tensor(info["pairs"]), len(info["pairs"])
        latest = sorted((out_run / "results").glob("*-*"))[-1]
        labels = torch.from_numpy(prevalence_labels(info["train_counts"], s["n"], seed)).to(device)
        requested = pd.DataFrame({"filename": [f"{i:06d}.png" for i in range(s["n"])], "make_id": pairs[labels.cpu(), 0].numpy()})
        if cond["kind"] != "coarse":
            requested["cluster"], requested["group"] = pairs[labels.cpu(), 1].numpy(), labels.cpu().numpy()
        for ckpt in sorted((latest / "checkpoints").glob("*.pt")):
            step = int(ckpt.stem)
            model = DiT_models[D["model"]](input_size=D["image_size"] // 8, num_classes=n_classes)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["ema"])
            model = model.to(device).eval()
            for steps in s["ddim_steps"]:
                name = f"step{step:06d}_ddim{steps}"
                out = out_run / name
                out.mkdir(parents=True, exist_ok=True)
                diffusion = create_diffusion(str(steps))
                torch.manual_seed(seed)
                t0 = time.time()
                for b in range(0, s["n"], s["batch_size"]):
                    x = ae.decode(sample(model, diffusion, labels[b:b + s["batch_size"]], D["image_size"] // 8, s["guidance"], n_classes, device))
                    for i, im in enumerate((x * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()):
                        Image.fromarray(im).save(out / f"{b + i:06d}.png")
                seconds = time.time() - t0
                requested.to_csv(out / "samples.csv", index=False)
                (out / "method.json").write_text(json.dumps({
                    "name": f"{run}_{name}", "generator": f"official {D['model']} ({D['repo']})", "uses": cfg["autoencoders"]["sd_vae"]["uses"] + cond.get("uses", []),
                    "reference_split": s.get("reference_split", "val"), "conditioning": cond, "n_labels": n_classes, "autoencoder": "sd_vae",
                    "train_image_size": D["image_size"], "step": step, "train_hours": train_hours(latest / "log.txt", step), "ddim_steps": steps,
                    "guidance": s["guidance"], "sampling_seconds": seconds, "git_hash": git_hash(), "seed": seed}, indent=1))
                log.info("%s/%s: %d samples in %.0f s", run, name, s["n"], seconds)
                print(out, flush=True)


if __name__ == "__main__":
    main()
