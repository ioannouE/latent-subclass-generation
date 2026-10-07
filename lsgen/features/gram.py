"""Gram-statistic embeddings (diagnostic, scripts/gram_check.py): channel-projected Gram matrices of intermediate
feature maps of a frozen VGG16 or DINOv3 (patch tokens). Same interface as features.encoders.Embedder."""
import timm
import torch
import torchvision
from torchvision import transforms as T

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
GRAM = {  # name: (family, timm model, taps = layer indices, channels)
    "gram_vgg16": ("vgg", None, (8, 15, 22), (128, 256, 512)),                      # relu2_2, relu3_3, relu4_3
    "gram_dinov3_base": ("dinov3", "vit_base_patch16_dinov3.lvd1689m", (3, 7, 11), (768,) * 3),
}
RANK = 64  # channels kept by the fixed random projection: Gram dimension RANK * (RANK + 1) / 2 per layer


class GramEmbedder(torch.nn.Module):
    input_size = 224

    def __init__(self, name, weights=None):
        super().__init__()
        self.name = name
        self.family, self.timm_name, self.taps, channels = GRAM[name]
        if self.family == "vgg":
            self.model = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:max(self.taps) + 1]
            mean, std = IMAGENET
        else:
            self.model = timm.create_model(self.timm_name, pretrained=True, num_classes=0, img_size=self.input_size)
            mean, std = self.model.pretrained_cfg["mean"], self.model.pretrained_cfg["std"]
        g = torch.Generator().manual_seed(0)
        for i, c in enumerate(channels):
            self.register_buffer(f"proj{i}", torch.randn(c, RANK, generator=g) / c ** 0.5)
        self.register_buffer("tri", torch.triu_indices(RANK, RANK))
        self.eval().requires_grad_(False)
        self.transform = T.Compose([T.Resize((self.input_size,) * 2, interpolation=T.InterpolationMode.BICUBIC),
                                    T.ToTensor(), T.Normalize(mean, std)])

    def feature_maps(self, x):
        if self.family == "dinov3":
            return self.model.forward_intermediates(x, indices=list(self.taps), intermediates_only=True)
        maps = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.taps:
                maps.append(x)
        return maps

    def forward(self, x):
        grams = []
        for i, f in enumerate(self.feature_maps(x)):
            f = torch.einsum("bchw,cr->brhw", f.float(), getattr(self, f"proj{i}")).flatten(2)
            grams.append((f @ f.transpose(1, 2) / f.shape[-1])[:, self.tri[0], self.tri[1]])
        return torch.cat(grams, 1)
