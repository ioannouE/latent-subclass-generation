"""DreamSim embeddings (Fu et al., NeurIPS 2023; github.com/ssundaram21/dreamsim, package `dreamsim` 0.2.1) through the package's own
code, so backbone checkpoints, feature location (DINO: CLS of block 11; CLIP / OpenCLIP: projection head; ensemble: their concatenation),
normalisation (mean-centred, then L2) and the 224 px input come from its config. `dreamsim_<type>` is the LoRA-aligned model, `dreamsim_base_<type>`
the same model built without the LoRA adapters (the pre-alignment backbone). Same interface as features.encoders.Embedder.
"""
import torch
import torch.nn.functional as F
from dreamsim import dreamsim
from dreamsim.config import dreamsim_args
from dreamsim.model import PerceptualModel, download_weights
from torchvision import transforms as T

TYPES = ["dinov2_vitb14", "dino_vitb16", "clip_vitb32", "open_clip_vitb32", "ensemble", "dino_vitb16_patch", "dinov2_vitb14_patch"]
DREAMSIM = {f"dreamsim_{t}": (t, True) for t in TYPES} | {f"dreamsim_base_{t}": (t, False) for t in TYPES}  # name: (type, aligned)


class DreamSimEmbedder(torch.nn.Module):
    family, timm_name = "dreamsim", None

    def __init__(self, name, weights, device="cuda"):
        """weights: cache folder of the DreamSim release (checkpoints and LoRA adapters); downloaded on first use."""
        super().__init__()
        self.name = name
        kind, aligned = DREAMSIM[name]
        self.patch = kind.endswith("_patch")
        self.input_size = dreamsim_args["img_size"]
        if aligned:
            self.model, _ = dreamsim(device=device, cache_dir=str(weights), dreamsim_type=kind.removesuffix("_patch"),
                                     use_patch_model=self.patch)
        else:
            download_weights(str(weights), kind)
            self.model = PerceptualModel(**dreamsim_args["model_config"][kind], device=device, load_dir=str(weights),
                                         normalize_embeds=True)  # lora=True in the config: identity head, no adapters
        self.eval().requires_grad_(False)
        self.transform = T.Compose([T.Resize((self.input_size,) * 2, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])

    def forward(self, x):
        z = self.model.embed(x)
        if self.patch:  # patch models: per-token embeddings; their distance compares CLS || mean of the patch tokens
            z = F.normalize(torch.cat([z[:, 0], z[:, 1:].mean(1)], -1), dim=-1)
        assert torch.allclose(z.norm(dim=-1), torch.ones(len(z), device=z.device), atol=1e-3), "embeddings are not unit norm"
        return z
