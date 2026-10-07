"""Frozen encoders. Embedding = CLS / pooled output, as in med-img-gen/zero_shot/encoders.py.

DINOv2, DINOv3 and CLIP are loaded through timm (weights from the HF cache); SSCD is the TorchScript model used in
the duplicate audit; Inception pool3 is clean-fid's TorchScript network with its own resizer. Real and generated images
must go through the same Embedder.transform. `fine_classifier` is the penultimate layer of the fine evaluator classifier: the
"supervised oracle features" reference row of the representation table (it saw the fine labels; not a baseline).
"""
import numpy as np
import timm
import torch
from cleanfid.inception_torchscript import InceptionV3W
from cleanfid.resize import build_resizer
from torchvision import transforms as T

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
ENCODERS = {  # name: (family, timm model, input size)
    "dinov2_large": ("dinov2", "vit_large_patch14_dinov2.lvd142m", 224),
    "dinov3_base": ("dinov3", "vit_base_patch16_dinov3.lvd1689m", 224),
    "dinov3_large": ("dinov3", "vit_large_patch16_dinov3.lvd1689m", 224),
    "clip_l": ("clip", "vit_large_patch14_clip_224.openai", 224),
    "sscd": ("sscd", None, 288),
    "inception": ("inception", None, 299),
    "fine_classifier": ("classifier", "convnext_tiny.fb_in1k", 128),  # trained on 128 px crops: feed those only
}


class Embedder(torch.nn.Module):
    def __init__(self, name, weights=None):
        """weights: path of the SSCD TorchScript model, or of the fine classifier checkpoint (train_eval_classifier.py)."""
        super().__init__()
        self.name = name
        self.family, self.timm_name, self.input_size = ENCODERS[name]
        if self.family == "inception":
            self.model = InceptionV3W("/tmp", download=True, resize_inside=False)
        elif self.family == "sscd":
            self.model, (mean, std) = torch.jit.load(str(weights), map_location="cpu"), IMAGENET
        elif self.family == "classifier":
            ck = torch.load(weights, map_location="cpu")
            self.model = timm.create_model(ck["arch"], pretrained=False, num_classes=ck["n_classes"])
            self.model.load_state_dict(ck["state_dict"])
            mean, std = IMAGENET
        else:
            self.model = timm.create_model(self.timm_name, pretrained=True, num_classes=0, img_size=self.input_size)
            mean, std = self.model.pretrained_cfg["mean"], self.model.pretrained_cfg["std"]
        self.eval().requires_grad_(False)
        if self.family == "inception":  # clean-fid: PIL bicubic to 299, pixel values stay in [0, 255]
            resize = build_resizer("clean")
            self.transform = lambda im: torch.from_numpy(resize(np.asarray(im)).transpose(2, 0, 1).copy())
            return
        self.transform = T.Compose([T.Resize((self.input_size,) * 2, interpolation=T.InterpolationMode.BICUBIC),
                                    T.ToTensor(), T.Normalize(mean, std)])

    def forward(self, x):
        if self.family == "classifier":
            return self.model.forward_head(self.model.forward_features(x), pre_logits=True)
        if self.family == "dinov3":  # timm's default DINOv3 output is the patch average; we use the CLS token
            return self.model.forward_features(x)[:, 0]
        return self.model(x)
