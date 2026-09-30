import numpy as np
import pytest
from PIL import Image

from lsgen.data.crops import (PAD_COLOR, bbox15_window, bbox_in_crop, crop_bbox15, crop_full_cc,
                              devkit_bbox_to_px, png_bytes)
from lsgen.data.native import native_c3gan, native_finegan

RED = (255, 0, 0)


def synthetic(W, H, bbox_devkit, mode="RGB"):
    """Grey noise background with the devkit bbox (1-indexed inclusive) painted solid red."""
    rng = np.random.default_rng(0)
    arr = rng.integers(60, 140, (H, W, 3), dtype=np.uint8)
    arr[..., 0] = arr[..., 1]  # no strongly red background pixels
    x1, y1, x2, y2 = bbox_devkit
    arr[y1 - 1:y2, x1 - 1:x2] = RED
    img = Image.fromarray(arr)
    return img.convert("L") if mode == "L" else img


def red_extent(img):
    a = np.asarray(img).astype(float)
    red = (a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] < 60)
    ys, xs = np.nonzero(red)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def red_size(img):
    """Sub-pixel width/height of the red box: integral of redness along its central row/column
    (resampling filters preserve the integral of a step)."""
    a = np.asarray(img).astype(float)
    red = np.clip((a[..., 0] - a[..., 1]) / 255, 0, 1)
    x1, y1, x2, y2 = red_extent(img)
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    return red[cy - 2:cy + 3].sum(1).mean(), red[:, cx - 2:cx + 3].sum(0).mean()


def test_devkit_conversion_is_one_indexed_inclusive():
    img = synthetic(100, 80, (11, 21, 50, 60))
    box = devkit_bbox_to_px((11, 21, 50, 60), 100, 80)
    assert box == (10, 20, 50, 60)
    region = np.asarray(img.crop(box))
    assert region.shape[:2] == (40, 40)
    assert (region == RED).all()
    a = np.asarray(img)  # nothing red just outside the box
    assert not (a[20:60, 9] == RED).all(axis=-1).any() and not (a[20:60, 50] == RED).all(axis=-1).any()
    assert not (a[19, 10:50] == RED).all(axis=-1).any() and not (a[60, 10:50] == RED).all(axis=-1).any()


def test_devkit_conversion_clips():
    assert devkit_bbox_to_px((0, 1, 120, 90), 100, 80) == (0, 0, 100, 80)


@pytest.mark.parametrize("W,H,bbox", [
    (800, 600, (201, 151, 600, 400)),   # fits with full 1.5 margin, no shift
    (800, 600, (11, 101, 400, 350)),    # near left border: must shift, not shrink
    (800, 600, (401, 301, 790, 590)),   # near bottom-right corner
    (800, 400, (51, 41, 750, 380)),     # wide car: needs padding in y
    (600, 800, (101, 51, 500, 780)),    # tall bbox: padding in x
])
def test_bbox15_pixels_aspect_and_containment(W, H, bbox):
    img = synthetic(W, H, bbox)
    # neutral padding (R == G) so the redness integral in red_size only measures the car box
    out, meta = crop_bbox15(img, bbox, sizes=(128, 256), pad_color=(110, 110, 110))
    for sz, crop in out.items():
        assert crop.size == (sz, sz) and crop.mode == "RGB"
    w, h = red_size(out[256])
    bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
    assert abs((w / h) / (bw / bh) - 1) < 0.01
    x1, y1, x2, y2 = red_extent(out[256])
    bx = bbox_in_crop(meta, 256)
    assert bx[0] >= 0 and bx[1] >= 0 and bx[2] <= 256 and bx[3] <= 256
    assert np.allclose((x1, y1, x2, y2), bx, atol=1.5)


def test_bbox15_geometry_properties_random():
    rng = np.random.default_rng(1)
    for _ in range(5000):
        W, H = int(rng.integers(40, 3000)), int(rng.integers(40, 3000))
        x1, x2 = sorted(rng.choice(W + 1, 2, replace=False))
        y1, y2 = sorted(rng.choice(H + 1, 2, replace=False))
        m = bbox15_window(W, H, (x1, y1, x2, y2))
        S = m["side"]
        wx0, wy0, wx1, wy1 = m["crop_box"]
        assert wx1 - wx0 == S and wy1 - wy0 == S  # always square
        assert wx0 <= x1 and wy0 <= y1 and x2 <= wx1 and y2 <= wy1  # car fully inside
        assert m["margin_used"] <= 1.5 + 1 / max(x2 - x1, y2 - y1) + 1e-9
        for L, lo, pl, pr in ((W, wx0, m["pad_l"], m["pad_r"]), (H, wy0, m["pad_t"], m["pad_b"])):
            if S <= L:  # fits: inside the image, no padding
                assert 0 <= lo and lo + S <= L and pl == pr == 0
            else:  # whole axis kept, padding split equally
                assert pl + pr == S - L and abs(pl - pr) <= 1 and lo == -pl
        if 1.5 * max(x2 - x1, y2 - y1) <= min(W, H):
            assert m["pad_fraction"] == 0
        if m["pad_fraction"] > 0:  # padding only when even the reduced side exceeds a dimension
            assert S > min(W, H)


def test_bbox15_margin_reduced_before_padding():
    # m = 500, 1.5 m = 750 > H = 600 -> s = max(600, 510) = 600, no padding
    m = bbox15_window(1000, 600, (100, 50, 600, 550))
    assert m["side"] == 600 and m["pad_fraction"] == 0 and m["margin_used"] == pytest.approx(1.2)
    # m = 700 wide, H = 600 -> s = max(600, 714) = 714 -> pad 114 in y, split 57/57
    m = bbox15_window(1000, 600, (100, 100, 800, 400))
    assert m["side"] == 714 and (m["pad_t"], m["pad_b"]) == (57, 57) and m["pad_l"] == m["pad_r"] == 0


def test_padding_colour_is_constant():
    img = synthetic(800, 300, (21, 41, 780, 260))
    out, meta = crop_bbox15(img, (21, 41, 780, 260), sizes=(256,))
    assert meta["pad_t"] > 0
    top = np.asarray(out[256])[:int(meta["pad_t"] * 256 / meta["side"]) - 3]
    assert (top == PAD_COLOR).all()


def test_grayscale_becomes_rgb():
    out, meta = crop_bbox15(synthetic(300, 200, (51, 51, 250, 150), mode="L"), (51, 51, 250, 150))
    assert meta["was_grayscale"] and out[128].mode == "RGB"


def test_sizes_resized_independently_from_square():
    img = synthetic(640, 480, (101, 101, 500, 380))
    out, meta = crop_bbox15(img, (101, 101, 500, 380))
    S = meta["side"]
    x0, y0 = meta["crop_box"][:2]
    square = img.crop((x0, y0, x0 + S, y0 + S))
    assert png_bytes(out[128]) == png_bytes(square.resize((128, 128), Image.LANCZOS))
    assert png_bytes(out[256]) == png_bytes(square.resize((256, 256), Image.LANCZOS))


def test_full_cc_sizes_and_centre():
    img = synthetic(1000, 500, (401, 101, 600, 400))
    for sz in (128, 256):
        c = crop_full_cc(img, sz)
        assert c.size == (sz, sz) and c.mode == "RGB"
    x1, _, x2, _ = red_extent(crop_full_cc(img, 256))
    assert abs((x1 + x2) / 2 - 128) <= 2


def test_png_encoding_deterministic():
    img = synthetic(400, 300, (51, 51, 350, 250))
    a = [png_bytes(crop_bbox15(img, (51, 51, 350, 250))[0][128]) for _ in range(2)]
    assert a[0] == a[1]
    assert png_bytes(crop_full_cc(img, 128)) == png_bytes(crop_full_cc(img, 128))


def test_native_adapters_shapes():
    img = synthetic(640, 400, (101, 101, 500, 350))
    rng = np.random.default_rng(0)
    assert native_finegan(img, (101, 101, 500, 350), rng).size == (128, 128)
    assert native_finegan(img, (101, 101, 500, 350)).size == (128, 128)
    assert native_c3gan(img, rng).size == (128, 128)
    assert native_c3gan(img).size == (128, 128)
