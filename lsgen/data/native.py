"""Native preprocessing of published baselines. ONLY for reproducing published numbers (Step 3).
Never use these for comparison tables: every compared number comes from the stored bbox15 PNGs.
"""
from PIL import Image

from lsgen.data.crops import devkit_bbox_to_px


def native_finegan(img, bbox, rng=None, load_size=152, crop_size=128):
    """FineGAN: square of side 1.5 * max(w, h) centred on the bbox, CLAMPED at the borders (may be
    non-square), short side -> 152 (bilinear), random 128 crop + random flip (rng given) or centre crop."""
    img = img.convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = devkit_bbox_to_px(bbox, W, H)
    r = int(max(x2 - x1, y2 - y1) * 0.75)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    img = img.crop((max(0, cx - r), max(0, cy - r), min(W, cx + r), min(H, cy + r)))
    return _short_side_then_crop(img, load_size, crop_size, rng)


def native_c3gan(img, rng=None, size=128):
    """C3-GAN: whole image, short side -> 128 (bilinear), random crop + flip (train, rng given) / centre crop."""
    return _short_side_then_crop(img.convert("RGB"), size, size, rng)


def _short_side_then_crop(img, load_size, crop_size, rng):
    w, h = img.size
    k = load_size / min(w, h)
    img = img.resize((max(load_size, round(w * k)), max(load_size, round(h * k))), Image.BILINEAR)
    w, h = img.size
    if rng is None:
        x, y = (w - crop_size) // 2, (h - crop_size) // 2
    else:
        x, y = int(rng.integers(0, w - crop_size + 1)), int(rng.integers(0, h - crop_size + 1))
    img = img.crop((x, y, x + crop_size, y + crop_size))
    if rng is not None and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
