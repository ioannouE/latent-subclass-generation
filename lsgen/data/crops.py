"""Image preprocessing (docs/PLAN.md, Part A2b). Pure functions; no file I/O except save_png.

bbox15 : square window of side 1.5 * max(bbox w, h), centred on the bbox, shifted (never shrunk)
         to stay inside the image; constant padding only where the side exceeds an image dimension.
full_cc: whole image, ADM center_crop_arr (BOX halving, BICUBIC short side -> size, centre crop).
"""
import io

import numpy as np
from PIL import Image

PAD_COLOR = (124, 116, 104)


def devkit_bbox_to_px(bbox, W, H):
    """Devkit bbox (1-indexed, inclusive) -> 0-indexed half-open pixel box [x1, x2) x [y1, y2), clipped."""
    x1, y1, x2, y2 = bbox
    x1, y1 = x1 - 1, y1 - 1
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def _place(lo_bbox, hi_bbox, S, L):
    """Place a window of integer length S on an axis of length L. Returns (window_start, pad_lo, pad_hi)."""
    if S > L:  # keep the whole axis, split padding equally (lower side gets the floor)
        pad = S - L
        return -(pad // 2), pad // 2, pad - pad // 2
    lo = int(round((lo_bbox + hi_bbox) / 2 - S / 2))
    lo = min(max(lo, hi_bbox - S), lo_bbox)  # guard against rounding cutting the bbox
    return min(max(lo, 0), L - S), 0, 0  # shift inside the image


def bbox15_window(W, H, bbox_px, margin=1.5, min_margin=1.02):
    """Geometry only. bbox_px is 0-indexed half-open. Returns meta dict (window in original coords)."""
    x1, y1, x2, y2 = bbox_px
    bw, bh = x2 - x1, y2 - y1
    m = max(bw, bh)
    s = margin * m
    if s > min(W, H):
        s = max(min(W, H), min_margin * m)
    S = max(int(round(s)), m)
    wx, pad_l, pad_r = _place(x1, x2, S, W)
    wy, pad_t, pad_b = _place(y1, y2, S, H)
    pad_area = S * S - (S - pad_l - pad_r) * (S - pad_t - pad_b)
    return {
        "side": S,  # side of the square window in original pixels
        "crop_box": (wx, wy, wx + S, wy + S),  # may extend outside the image where padded
        "pad_l": pad_l, "pad_t": pad_t, "pad_r": pad_r, "pad_b": pad_b,
        "pad_fraction": pad_area / (S * S),
        "margin_used": S / m,
        "bbox_px": (x1, y1, x2, y2),
    }


def bbox_in_crop(meta, size):
    """The bbox in output-pixel coordinates of the `size` x `size` crop."""
    wx, wy = meta["crop_box"][:2]
    k = size / meta["side"]
    x1, y1, x2, y2 = meta["bbox_px"]
    return ((x1 - wx) * k, (y1 - wy) * k, (x2 - wx) * k, (y2 - wy) * k)


def crop_bbox15(img, bbox, sizes=(128, 256), margin=1.5, min_margin=1.02, pad_color=PAD_COLOR):
    """img: PIL image (any mode; no EXIF transpose). bbox: devkit (1-indexed, inclusive).
    Returns ({size: square RGB PIL image}, meta). Each size is resized directly from the square."""
    was_grayscale = img.mode in ("L", "LA", "I", "I;16", "F", "1")
    img = img.convert("RGB")
    W, H = img.size
    meta = bbox15_window(W, H, devkit_bbox_to_px(bbox, W, H), margin, min_margin)
    S, (wx, wy) = meta["side"], meta["crop_box"][:2]
    x0, y0 = max(wx, 0), max(wy, 0)
    region = img.crop((x0, y0, min(wx + S, W), min(wy + S, H)))
    if meta["pad_fraction"] > 0:
        square = Image.new("RGB", (S, S), tuple(pad_color))
        square.paste(region, (meta["pad_l"], meta["pad_t"]))
    else:
        square = region
    if square.size != (S, S):
        raise AssertionError(f"non-square crop {square.size} for side {S}")
    out = {sz: square.resize((sz, sz), Image.LANCZOS) for sz in sizes}
    meta.update(was_grayscale=was_grayscale, orig_W=W, orig_H=H, scale={sz: sz / S for sz in sizes})
    return out, meta


def crop_full_cc(img, size):
    """ADM guided-diffusion center_crop_arr, on an RGB copy of the whole image."""
    img = img.convert("RGB")
    while min(*img.size) >= 2 * size:
        img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
    scale = size / min(*img.size)
    img = img.resize(tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)
    arr = np.array(img)
    cy, cx = (arr.shape[0] - size) // 2, (arr.shape[1] - size) // 2
    return Image.fromarray(arr[cy:cy + size, cx:cx + size])


def png_bytes(img):
    """Deterministic lossless PNG encoding (fixed compression, no metadata chunks)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()
