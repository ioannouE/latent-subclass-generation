"""Evaluator registry and the circularity rule (docs/PLAN.md A3): evaluators must be disjoint from every encoder a method
uses for conditioning or training."""
import json
from pathlib import Path

from lsgen.features.encoders import ENCODERS

EVALUATORS = {  # role -> name. Names are keys of ENCODERS, or "classifier" for the ConvNeXt-T fine/make evaluators
    "fid_kid": "inception", "semantic": "clip_l", "copy": "sscd", "classifier": "classifier"}
LINEAGE = {"dinov2": "dino", "dinov3": "dino", "classifier": "convnext"}  # families that count as "same family"


def lineage(name):
    family = ENCODERS[name][0] if name in ENCODERS else name
    return LINEAGE.get(family, family)


def check_disjoint(uses, evaluators=EVALUATORS):
    """uses: encoder names a method declares. Raises if one is also an evaluator; returns warnings for same-family pairs
    (e.g. a DINOv3 method scored with DINOv2)."""
    uses = set(uses)
    clash = sorted(uses & set(evaluators.values()))
    if clash:
        raise ValueError(f"method uses {clash}, which is also an evaluator: refusing to evaluate (circularity rule)")
    return [f"method encoder '{u}' and evaluator '{e}' ({r}) are the same family: flag in the report"
            for u in sorted(uses) for r, e in evaluators.items() if lineage(u) == lineage(e)]


def read_method(samples_dir):
    """method.json next to the generated images: {"name": ..., "uses": [encoder names, any string for non-lsgen ones]}."""
    return json.loads((Path(samples_dir) / "method.json").read_text())
