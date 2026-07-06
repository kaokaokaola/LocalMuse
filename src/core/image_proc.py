"""
Pure image processing utilities.
No UI, No File IO — accepts/returns PIL Images and numpy arrays.
"""

from __future__ import annotations
import hashlib
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

STRUCTURE_FEATURE_SIZE = 128


# ------------------------------------------------------------------ #
#  Thumbnail
# ------------------------------------------------------------------ #

def generate_thumbnail(image: Image.Image, size: int = 360) -> Image.Image:
    """
    Return a thumbnail copy that fits within a (size x size) bounding box.
    Preserves aspect ratio. Uses LANCZOS resampling.
    """
    thumb = image.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    return thumb


def pil_to_thumbnail_bytes(image: Image.Image, size: int = 360) -> bytes:
    """Return thumbnail as PNG bytes."""
    import io
    thumb = generate_thumbnail(image, size)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ------------------------------------------------------------------ #
#  Dominant color extraction
# ------------------------------------------------------------------ #

def extract_dominant_color(image: Image.Image) -> Tuple[int, int, int]:
    """
    Extract the single most dominant RGB color using color quantization.
    Returns (R, G, B) ints in [0, 255].
    """
    try:
        small = image.copy().convert("RGB")
        small.thumbnail((64, 64), Image.NEAREST)
        quantized = small.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()
        pixels = list(quantized.getdata())
        counts = [0] * 8
        for p in pixels:
            if p < 8:
                counts[p] += 1
        best = counts.index(max(counts))
        r = palette[best * 3]
        g = palette[best * 3 + 1]
        b = palette[best * 3 + 2]
        return (r, g, b)
    except Exception:
        return (128, 128, 128)


def extract_color_palette(image: Image.Image, n: int = 6) -> list:
    """
    Extract top-n dominant colors as a list of (R, G, B) tuples,
    ordered by frequency descending.
    Kept for backward-compat; prefer extract_color_palette_with_ratio().
    """
    try:
        small = image.copy().convert("RGB")
        small.thumbnail((128, 128), Image.NEAREST)
        quantized = small.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()
        pixels = list(quantized.getdata())
        counts = [0] * n
        for p in pixels:
            if p < n:
                counts[p] += 1
        order = sorted(range(n), key=lambda i: counts[i], reverse=True)
        result = []
        for i in order:
            r = palette[i * 3]
            g = palette[i * 3 + 1]
            b = palette[i * 3 + 2]
            result.append((r, g, b))
        return result
    except Exception:
        return [(128, 128, 128)] * n


def extract_color_palette_with_ratio(
    image: Image.Image,
    n: int = 12,
    file_size: int = 0,
) -> list:
    """
    Eagle-style color palette extraction.

    Returns up to n colors as:
        [{"rgb": [R, G, B], "ratio": float}, ...]   sorted by ratio desc

    Key differences from basic extract_color_palette():
    - Adaptive sampling step based on file_size (Eagle behaviour):
        < 100 KB  → step 1  (full sampling)
        < 1 MB    → step 5
        < 10 MB   → step 10
        ≥ 10 MB   → step 20
    - Skips transparent pixels (alpha < 170)     — Eagle threshold
    - Filters colors with ratio < 0.25%          — Eagle threshold
    - Stores frequency ratio per color
    - Quantizes up to 12 colors (Eagle max)
    """
    try:
        img_rgba = image.copy().convert("RGBA")
        # Resize to at most 360 px for performance (consistent canvas size)
        img_rgba.thumbnail((360, 360), Image.NEAREST)

        pixels = list(img_rgba.getdata())
        total  = len(pixels)

        # Adaptive sampling step (Eagle-style)
        if file_size < 100_000:
            step = 1
        elif file_size < 1_000_000:
            step = 5
        elif file_size < 10_000_000:
            step = 10
        else:
            step = 20

        # Sample pixels; skip transparent ones (Eagle: alpha < 170)
        sampled_rgb: list = []
        for i in range(0, total, step):
            r, g, b, a = pixels[i]
            if a >= 170:
                sampled_rgb.append((r, g, b))

        if not sampled_rgb:
            return [{"rgb": [128, 128, 128], "ratio": 1.0}]

        # Build a 1-row image from sampled pixels for PIL quantize
        sample_img = Image.new("RGB", (len(sampled_rgb), 1))
        sample_img.putdata(sampled_rgb)

        colors = min(n, len(sampled_rgb))
        quantized = sample_img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        palette   = quantized.getpalette()
        indices   = list(quantized.getdata())

        counts = [0] * colors
        for idx in indices:
            if idx < colors:
                counts[idx] += 1

        n_sampled = len(sampled_rgb)
        result = []
        for i in sorted(range(colors), key=lambda x: counts[x], reverse=True):
            if counts[i] == 0:
                continue
            ratio = round(counts[i] / n_sampled, 4)
            if ratio < 0.0025:      # Eagle: filter < 0.25%
                continue
            result.append({
                "rgb":   [palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]],
                "ratio": ratio,
            })

        return result if result else [{"rgb": [128, 128, 128], "ratio": 1.0}]
    except Exception:
        return [{"rgb": [128, 128, 128], "ratio": 1.0}]


# ------------------------------------------------------------------ #
#  Edge / Sketch extraction (for Structure modality)
# ------------------------------------------------------------------ #

def extract_edges(image: Image.Image, long_side: int = 512) -> Image.Image:
    """
    Extract a Canny-edge sketch from an image.
    Returns a PIL Image (RGB, white background, black lines)
    suitable for CLIP encoding.

    ``long_side`` controls the working resolution. Keep the default (512)
    for anything that feeds CLIP or ``sketch_local_match``; higher values
    (e.g. 1024) are for display-quality edge maps only.
    """
    # Resize long edge to ``long_side``
    w, h = image.size
    scale = long_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    # Convert to grayscale numpy
    gray = np.array(resized.convert("L"))

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection (architecture-optimized thresholds)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # Slight dilation to thicken lines
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Invert: white background, black lines
    sketch = 255 - edges

    # Convert back to PIL RGB (CLIP needs 3-channel)
    sketch_pil = Image.fromarray(sketch, mode="L").convert("RGB")
    return sketch_pil


def resize_long_side_pad_square(
    image: Image.Image,
    size: int = STRUCTURE_FEATURE_SIZE,
    fill: int = 255,
) -> Image.Image:
    """
    Resize so the long side becomes ``size`` and pad to a square.

    This preserves the whole composition. For edge sketches the padding should
    be white, because the sketch background is white.
    """
    src = image.convert("L")
    w, h = src.size
    if w <= 0 or h <= 0:
        return Image.new("L", (size, size), color=fill)
    scale = size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = src.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (size, size), color=fill)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def resize_short_side_crop_square(
    image: Image.Image,
    size: int = STRUCTURE_FEATURE_SIZE,
) -> Image.Image:
    """
    Resize so the short side becomes ``size`` and center-crop to a square.

    This keeps local structure larger, useful when a non-square source image is
    compared against a square query drawing.
    """
    src = image.convert("L")
    w, h = src.size
    if w <= 0 or h <= 0:
        return Image.new("L", (size, size), color=255)
    scale = size / min(w, h)
    new_w = max(size, int(round(w * scale)))
    new_h = max(size, int(round(h * scale)))
    resized = src.resize((new_w, new_h), Image.LANCZOS)
    left = max(0, (new_w - size) // 2)
    top = max(0, (new_h - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def structure_dual_vectors_from_edges(
    edge_image: Image.Image,
    size: int = STRUCTURE_FEATURE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the two structure vectors used by scheme B:
      1. fit vector: long side -> size, white padded to size x size
      2. crop vector: short side -> size, center-cropped to size x size
    """
    fit = resize_long_side_pad_square(edge_image, size=size, fill=255)
    crop = resize_short_side_crop_square(edge_image, size=size)
    return (_normalized_gray_vector(fit), _normalized_gray_vector(crop))


def structure_dual_vectors(
    image: Image.Image,
    size: int = STRUCTURE_FEATURE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract edges and return the two scheme-B structure vectors."""
    return structure_dual_vectors_from_edges(extract_edges(image), size=size)


def _normalized_gray_vector(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("L"), dtype=np.float32) / 255.0
    vec = arr.flatten()
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def pil_from_opencv(arr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL RGB image."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ------------------------------------------------------------------ #
#  Sketch <-> image local matching (orientation-aware chamfer)
# ------------------------------------------------------------------ #

_MATCH_BINS = 4          # orientation channels (mod 180 deg)
_MATCH_TAU = 0.10        # chamfer tolerance, fraction of window diagonal
_ADJ_PENALTY = 4.0       # extra distance (px) for adjacent-orientation matches
_MAX_SKETCH_PTS = 500    # sketch stroke points sampled per window score
_MAX_EDGE_PTS = 400      # image edge points sampled for the reverse term
_TOP_REVERSE = 40        # best forward windows re-scored with the reverse term


def _orientation_bins(mask: np.ndarray, k: int = _MATCH_BINS) -> np.ndarray:
    """
    Per-pixel orientation bin (0..k-1) for edge pixels, -1 elsewhere.

    Uses the structure tensor so stroke centerlines (where the raw
    gradient cancels to zero) still receive the correct line orientation.
    """
    m = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (7, 7), 0)
    jyy = cv2.GaussianBlur(gy * gy, (7, 7), 0)
    jxy = cv2.GaussianBlur(gx * gy, (7, 7), 0)
    ang = (0.5 * np.arctan2(2.0 * jxy, jxx - jyy)) % np.pi   # 0..pi
    bins = (np.floor(ang / (np.pi / k) + 0.5).astype(np.int32)) % k
    return np.where(mask, bins, -1)


def _channel_dts(bins: np.ndarray, k: int = _MATCH_BINS) -> np.ndarray:
    """Stack of distance transforms, one per orientation channel: (k, H, W)."""
    h, w = bins.shape
    dts = np.empty((k, h, w), dtype=np.float32)
    for i in range(k):
        ch = (bins == i)
        if not ch.any():
            dts[i].fill(1e6)
            continue
        inv = np.where(ch, 0, 255).astype(np.uint8)
        dts[i] = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dts


def _oriented_lookup(dts: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                     b: np.ndarray, k: int = _MATCH_BINS) -> np.ndarray:
    """min(own-bin distance, adjacent-bin distance + penalty) per point."""
    d_own = dts[b, ys, xs]
    d_prev = dts[(b - 1) % k, ys, xs]
    d_next = dts[(b + 1) % k, ys, xs]
    return np.minimum(d_own, np.minimum(d_prev, d_next) + _ADJ_PENALTY)


def _match_subsample(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.linspace(0, n - 1, cap).astype(np.int64)


def sketch_local_match(edge_image: Image.Image, sketch_image: Image.Image) -> Optional[dict]:
    """
    Find the region of ``edge_image`` (white bg, black Canny lines -- output of
    :func:`extract_edges`) that best matches the user's sketch, by SHAPE.

    Orientation-aware chamfer matching:
      * edges are split into orientation channels via the structure tensor,
        so vertical strokes only match near-vertical edges, etc.;
      * each channel gets a distance transform -- scores fall off smoothly
        with misalignment instead of requiring pixel overlap;
      * forward term (sketch -> image) is balanced across orientation bins,
        so every stroke direction in the sketch must find support;
      * reverse term (image edges -> sketch) penalises cluttered windows;
      * windows are searched at multiple scales AND multiple aspect ratios
        (0.72x / 1.0x / 1.38x of the sketch aspect), so a slightly wider or
        narrower building still matches.

    Returns a dict::

        {
          "bbox":        [x, y, w, h],      # best window, edge-image pixels
          "score":       float,             # 0..1
          "full_score":  float,             # score of the whole frame
          "sketch_crop": [sx, sy, sw, sh],  # stroke bounding box in sketch px
        }

    or None if the sketch contains no strokes / edges are empty.
    """
    try:
        E = np.array(edge_image.convert("L")) < 128
        H, W = E.shape
        if H < 8 or W < 8 or not E.any():
            return None

        # ---- sketch stroke mask, cropped to content ------------------
        s_gray = np.array(sketch_image.convert("L"))
        strokes = s_gray < 200
        if not strokes.any():
            return None
        ys_, xs_ = np.where(strokes)
        pad = 4
        sy0 = max(0, int(ys_.min()) - pad)
        sy1 = min(strokes.shape[0], int(ys_.max()) + 1 + pad)
        sx0 = max(0, int(xs_.min()) - pad)
        sx1 = min(strokes.shape[1], int(xs_.max()) + 1 + pad)
        s_crop = strokes[sy0:sy1, sx0:sx1]
        sh, sw = s_crop.shape
        if sh < 4 or sw < 4:
            return None

        # ---- sketch: oriented stroke points + own channel DTs --------
        s_bins = _orientation_bins(s_crop)
        sp_y, sp_x = np.where(s_bins >= 0)
        if sp_y.size == 0:
            return None
        idx = _match_subsample(sp_y.size, _MAX_SKETCH_PTS)
        sp_u = sp_x[idx].astype(np.float32) / max(sw - 1, 1)   # 0..1
        sp_v = sp_y[idx].astype(np.float32) / max(sh - 1, 1)
        sp_b = s_bins[sp_y[idx], sp_x[idx]]

        s_dts = _channel_dts(s_bins)
        s_tau = max(4.0, _MATCH_TAU * float(np.hypot(sw, sh)))

        # Balanced forward scoring: every stroke orientation present in
        # the sketch must find a match (sqrt-count weighted).
        bin_masks = [sp_b == i for i in range(_MATCH_BINS)]
        bin_w = np.array([np.sqrt(m.sum()) for m in bin_masks], dtype=np.float64)
        bin_w = bin_w / max(bin_w.sum(), 1e-9)

        # ---- image: oriented edge points + per-channel DTs -----------
        e_bins = _orientation_bins(E)
        ep_y, ep_x = np.where(e_bins >= 0)
        ep_b = e_bins[ep_y, ep_x]
        dts = _channel_dts(e_bins)

        def _forward(x: int, y: int, ww: int, wh: int) -> float:
            """Sketch strokes -> image edges: does the shape exist here?"""
            px = np.clip((x + sp_u * (ww - 1)).astype(np.int64), 0, W - 1)
            py = np.clip((y + sp_v * (wh - 1)).astype(np.int64), 0, H - 1)
            d = _oriented_lookup(dts, py, px, sp_b)
            tau = max(4.0, _MATCH_TAU * float(np.hypot(ww, wh)))
            g = 1.0 - np.minimum(d, tau) / tau
            total = 0.0
            for i in range(_MATCH_BINS):
                if bin_w[i] > 0:
                    total += bin_w[i] * float(g[bin_masks[i]].mean())
            return total

        def _reverse(x: int, y: int, ww: int, wh: int) -> float:
            """Image edges in window -> sketch strokes: clutter penalty."""
            m = (ep_x >= x) & (ep_x < x + ww) & (ep_y >= y) & (ep_y < y + wh)
            n = int(m.sum())
            if n == 0:
                return 0.0
            wx, wy, wb = ep_x[m], ep_y[m], ep_b[m]
            sub = _match_subsample(n, _MAX_EDGE_PTS)
            su = np.clip(((wx[sub] - x).astype(np.float32) / max(ww - 1, 1)
                          * (sw - 1)).astype(np.int64), 0, sw - 1)
            sv = np.clip(((wy[sub] - y).astype(np.float32) / max(wh - 1, 1)
                          * (sh - 1)).astype(np.int64), 0, sh - 1)
            d = _oriented_lookup(s_dts, sv, su, wb[sub])
            return float(np.mean(1.0 - np.minimum(d, s_tau) / s_tau))

        def _combined(x, y, ww, wh, fwd=None):
            f = _forward(x, y, ww, wh) if fwd is None else fwd
            return 0.7 * f + 0.3 * _reverse(x, y, ww, wh)

        # ---- candidate windows: multi-scale AND multi-aspect ----------
        aspect0 = sw / sh
        base = min(H, W)
        cands = [(_forward(0, 0, W, H), 0, 0, W, H)]   # full frame
        for scale in (1.0, 0.8, 0.62, 0.47, 0.35):
            for a_mul in (0.72, 1.0, 1.38):
                aspect = aspect0 * a_mul
                wh = int(base * scale)
                ww = int(wh * aspect)
                if ww > W:
                    ww = W
                    wh = max(8, int(ww / aspect))
                if wh > H:
                    wh = H
                    ww = max(8, int(wh * aspect))
                if wh < 24 or ww < 24 or wh > H or ww > W:
                    continue
                stride = max(12, min(wh, ww) // 4)
                for y in range(0, H - wh + 1, stride):
                    for x in range(0, W - ww + 1, stride):
                        cands.append((_forward(x, y, ww, wh), x, y, ww, wh))

        full_fwd = cands[0][0]
        full_score = _combined(0, 0, W, H, full_fwd)

        # Re-score only the best forward candidates with the reverse term.
        cands.sort(key=lambda c: -c[0])
        best = (full_score, (0, 0, W, H))
        for fwd, x, y, ww, wh in cands[:_TOP_REVERSE]:
            sc = _combined(x, y, ww, wh, fwd)
            if sc > best[0]:
                best = (sc, (x, y, ww, wh))

        # ---- refinement pass around the winner -------------------------
        score, (bx, by, bw, bh) = best
        if (bw, bh) != (W, H):
            stride = max(12, min(bw, bh) // 4)
            fine = max(4, stride // 4)
            for y in range(max(0, by - stride), min(H - bh, by + stride) + 1, fine):
                for x in range(max(0, bx - stride), min(W - bw, bx + stride) + 1, fine):
                    sc = _combined(x, y, bw, bh)
                    if sc > best[0]:
                        best = (sc, (x, y, bw, bh))

        score, (bx, by, bw, bh) = best
        return {
            "bbox": [int(bx), int(by), int(bw), int(bh)],
            "score": round(float(score), 4),
            "full_score": round(float(full_score), 4),
            "sketch_crop": [int(sx0), int(sy0), int(sw), int(sh)],
        }
    except Exception:
        return None


# ------------------------------------------------------------------ #
#  Duplicate detection hashes
# ------------------------------------------------------------------ #

def compute_file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash for a file path, or "" on failure."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def perceptual_hash(
    image: Image.Image,
    hash_size: int = 8,
    highfreq_factor: int = 4,
) -> str:
    """
    Compute a 64-bit pHash as a 16-char hex string.

    The hash is stable across resize/recompression and is used with Hamming
    distance for near-duplicate detection.
    """
    try:
        img_size = hash_size * highfreq_factor
        gray = image.convert("L").resize((img_size, img_size), Image.LANCZOS)
        pixels = np.array(gray, dtype=np.float32)
        dct = cv2.dct(pixels)
        low = dct[:hash_size, :hash_size]
        vals = low.flatten()
        median = np.median(vals[1:]) if vals.size > 1 else np.median(vals)
        bits = vals > median
        value = 0
        for bit in bits:
            value = (value << 1) | int(bool(bit))
        return f"{value:0{hash_size * hash_size // 4}x}"
    except Exception:
        return ""


def duplicate_hashes(image: Image.Image) -> Tuple[str, str]:
    """Return (phash, phash_flip) for normal and horizontal-flipped image."""
    try:
        normal = image.convert("RGB")
        flipped = normal.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return perceptual_hash(normal), perceptual_hash(flipped)
    except Exception:
        return "", ""


def hamming_distance_hex(a: str, b: str) -> Optional[int]:
    """Return Hamming distance between two hex hashes, or None if invalid."""
    try:
        if not a or not b or len(a) != len(b):
            return None
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return None


# ------------------------------------------------------------------ #
#  Color space conversions (for perceptual color distance)
# ------------------------------------------------------------------ #

def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert 8-bit RGB to CIELAB (D65 illuminant).
    Returns (L*, a*, b*).
    """
    # Normalize to [0,1]
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0

    # Linearize (sRGB gamma removal)
    def linearize(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_l, g_l, b_l = linearize(r_), linearize(g_), linearize(b_)

    # RGB -> XYZ (D65 matrix)
    x = r_l * 0.4124564 + g_l * 0.3575761 + b_l * 0.1804375
    y = r_l * 0.2126729 + g_l * 0.7151522 + b_l * 0.0721750
    z = r_l * 0.0193339 + g_l * 0.1191920 + b_l * 0.9503041

    # Normalize by D65 white point
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    fx = _f_lab(x / xn)
    fy = _f_lab(y / yn)
    fz = _f_lab(z / zn)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    return (L, a, b_)


def _f_lab(t: float) -> float:
    return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16.0 / 116.0


def color_distance_lab(
    rgb1: Tuple[int, int, int],
    rgb2: Tuple[int, int, int],
) -> float:
    """
    Perceptual color distance in CIELAB space (CIE76).
    Returns a non-negative float (0 = identical, ~100 = very different).
    """
    L1, a1, b1 = rgb_to_lab(*rgb1)
    L2, a2, b2 = rgb_to_lab(*rgb2)
    return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


def color_score_from_distance(dist: float, max_dist: float = 100.0) -> float:
    """
    Convert a Lab distance (0–100) to a similarity score in [0, 1].
    Uses a Gaussian kernel.
    """
    sigma = max_dist / 3.0
    return float(np.exp(-(dist ** 2) / (2 * sigma ** 2)))
