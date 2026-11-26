"""Controlled latent diffusion generator (docs/EXPERIMENT_GENERATION.md): an autoencoder (own MONAI AutoencoderKL or the frozen SD-VAE),
a MONAI latent UNet conditioned on one class label with classifier-free guidance, and DDIM sampling. Only the conditioning label changes
between variants: the coarse class (G-0c pilot, coarse rung), or one label per (coarse class, cluster) pair (condition_labels)."""
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler


def build_autoencoder(arch):
    """Own autoencoder: MONAI AutoencoderKL on RGB images in [-1, 1]; len(channels) - 1 downsamplings."""
    return AutoencoderKL(spatial_dims=2, in_channels=3, out_channels=3, **arch)


def kl_loss(mu, sigma):
    """KL(N(mu, sigma^2) || N(0, 1)), summed over latent dimensions, mean over the batch."""
    return 0.5 * (mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2)) - 1).flatten(1).sum(1).mean()


class Latents(torch.nn.Module):
    """Frozen autoencoder: images in [0, 1] at the data resolution <-> scaled latents (unit variance on train).
    kind "sd_vae": diffusers AutoencoderKL `weights`, images resized to `input_size` first (CIFAR-100: 32 -> 64 px for an 8 x 8 latent);
    kind "own": the checkpoint written by scripts/train_autoencoder.py."""

    def __init__(self, kind, image_size, weights=None, input_size=None, ckpt=None):
        super().__init__()
        self.kind, self.image_size, self.input_size = kind, image_size, input_size or image_size
        if kind == "sd_vae":
            from diffusers import AutoencoderKL as SDVAE
            self.net = SDVAE.from_pretrained(weights)
            self.scale = self.net.config.scaling_factor
        else:
            ck = torch.load(ckpt, map_location="cpu")
            self.net = build_autoencoder(ck["arch"])
            self.net.load_state_dict(ck["state_dict"])
            self.scale = ck["scale"]
        self.net.eval().requires_grad_(False)

    @staticmethod
    def _resize(x, size):
        return x if x.shape[-1] == size else F.interpolate(x, size=(size, size), mode="bicubic", antialias=True, align_corners=False)

    @torch.no_grad()
    def encode(self, x):
        x = self._resize(x, self.input_size) * 2 - 1
        mu = self.net.encode(x).latent_dist.mean if self.kind == "sd_vae" else self.net.encode(x)[0]
        return mu * self.scale

    @torch.no_grad()
    def decode(self, z):
        x = self.net.decode(z / self.scale)
        x = x.sample if self.kind == "sd_vae" else x
        return self._resize((x.float().clamp(-1, 1) + 1) / 2, self.image_size).clamp(0, 1)


def load_latents(name, spec, image_size, ae_dir):
    """The frozen autoencoder `name` of the config's `autoencoders` (own: <ae_dir>/<name>/model.pt from scripts/train_autoencoder.py)."""
    ckpt = f"{ae_dir}/{name}/model.pt" if spec["kind"] == "own" else None
    return Latents(spec["kind"], image_size, spec.get("weights"), spec.get("input_size"), ckpt)


def runs(cfg, section="ldm"):
    """[(seed, run name)]: `<section.name>_s<seed>` for each of `seeds`; a config without `seeds` (a pilot) is one run, `<section.name>`.
    section: the generator's config section (`ldm`, or `dit` for scripts/train_dit.py)."""
    if "seeds" not in cfg:
        return [(cfg["seed"], cfg[section]["name"])]
    return [(s, f"{cfg[section]['name']}_s{s}") for s in cfg["seeds"]]


def build_unet(arch, latent_channels, n_classes):
    """Class-conditional latent UNet; label n_classes is the null label of classifier-free guidance."""
    return DiffusionModelUNet(spatial_dims=2, in_channels=latent_channels, out_channels=latent_channels, num_class_embeds=n_classes + 1, **arch)


def schedulers(cfg):
    """(DDPM for training, DDIM for sampling) on the same noise schedule."""
    return DDPMScheduler(clip_sample=False, **cfg), DDIMScheduler(clip_sample=False, **cfg)


def drop_labels(y, p, null, generator=None):
    """Classifier-free guidance training: each label is replaced by the null label with probability p."""
    drop = torch.rand(len(y), generator=generator, device=y.device) < p
    return torch.where(drop, torch.full_like(y, null), y)


def diffusion_loss(unet, ddpm, z, y, generator=None):
    """Epsilon-prediction MSE at a uniformly drawn timestep."""
    noise = torch.randn(z.shape, generator=generator, device=z.device)
    t = torch.randint(0, ddpm.num_train_timesteps, (len(z),), generator=generator, device=z.device)
    return F.mse_loss(unet(ddpm.add_noise(z, noise, t), t, class_labels=y).float(), noise)


def guided_eps(unet, z, t, y, null, w):
    """eps_u + w (eps_c - eps_u); w = 1 is the plain conditional model (one forward pass)."""
    if w == 1:
        return unet(z, t, class_labels=y)
    eps_c, eps_u = unet(torch.cat([z, z]), torch.cat([t, t]), class_labels=torch.cat([y, torch.full_like(y, null)])).chunk(2)
    return eps_u + w * (eps_c - eps_u)


@torch.no_grad()
def sample(unet, ddim, y, shape, steps, w, null, generator):
    """DDIM (eta = 0) from Gaussian noise drawn with `generator`: latents (len(y), *shape)."""
    ddim.set_timesteps(steps)
    z = torch.randn((len(y), *shape), generator=generator, device=y.device)
    for t in ddim.timesteps:
        z, _ = ddim.step(guided_eps(unet, z, torch.full((len(y),), int(t), device=y.device), y, null, w).float(), int(t), z)
    return z


@torch.no_grad()
def ema_update(ema, model, step, decay):
    """EMA of the weights with warm-up min(decay, (1 + step) / (10 + step)), so early checkpoints are not dominated by the initialisation."""
    d = min(decay, (1 + step) / (10 + step))
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - d)


def prevalence_labels(counts, n, seed):
    """n labels in proportion to `counts` (largest-remainder rounding), in a seeded random order: prevalence-matching sampling (use case A)."""
    q = n * np.asarray(counts, float) / np.sum(counts)
    k = np.floor(q).astype(int)
    k[np.argsort(k - q, kind="stable")[: n - k.sum()]] += 1
    return np.random.default_rng(seed).permutation(np.repeat(np.arange(len(k)), k))


def condition_labels(coarse, cluster):
    """(label per image, (n_labels, 2) table of (coarse, cluster) pairs): one conditioning label per pair present, numbered in (coarse, cluster)
    order. With a single cluster per class the labels are the coarse classes."""
    pairs, label = np.unique(np.stack([coarse, cluster], 1), axis=0, return_inverse=True)
    return label.ravel(), pairs


def image_labels(cfg, derived, ids, coarse, seed):
    """(conditioning label per train image, (n_labels, 2) table of (coarse, cluster) pairs) for one seed: the coarse class, or with
    `conditioning` the (coarse, cluster) pairs of the per-image cluster file <derived>/<conditioning.labels> ({seed} = the run's seed)."""
    if "conditioning" not in cfg:
        return condition_labels(coarse, np.zeros_like(coarse))
    clusters = pd.read_csv(derived / cfg["conditioning"]["labels"].format(seed=seed)).set_index("id").loc[ids]
    if not (clusters.make_id.to_numpy() == coarse).all():
        raise ValueError("the cluster file's coarse classes differ from the label scheme's")
    return condition_labels(coarse, clusters.cluster.to_numpy())


def write_imagefolder(out, images, ids, label):
    """Training data for the official generators that read a torchvision ImageFolder (scripts/train_dit.py): out/<label>/<id>.png, a
    symlink to images/<id>.png. Folder names are zero-padded to 4 digits, so ImageFolder's sorted class index is the label itself."""
    shutil.rmtree(out, ignore_errors=True)
    for i, y in zip(ids, label):
        (out / f"{y:04d}").mkdir(parents=True, exist_ok=True)
        (out / f"{y:04d}" / f"{i}.png").symlink_to((images / f"{i}.png").resolve())
