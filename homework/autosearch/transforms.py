"""Transformation library for the false-positive search.

Each transform corresponds to one of the three user complaints in HOMEWORK.md:

  UF1 "grid or collage ... lots of lines running through them"  -> collage
  UF2 "watermarks and text"                                     -> watermark
  UF3 "messy ... hard to make out ... low quality"              -> degrade

Every transform is a plausible, non-adversarial image edit: the kind of thing a real user or
pipeline could produce. No pixel-level optimisation against the model, no gradient access. The
search explores *transformation parameters* only (see README).

Transform signature is uniform:

    fn(sources: list[Image], params: dict) -> Image

`sources` is the pool of provided base images. Per-image transforms use ``sources[0]``;
multi-source transforms (the mixed collage) may draw tiles from all of them.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS = 512

# Resolved lazily so a missing font never hard-fails the search.
_FONT_DIRS = [
    "/System/Library/Fonts/Supplemental/",
    "/System/Library/Fonts/",
    "/Library/Fonts/",
    "/usr/share/fonts/truetype/dejavu/",
]

NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
}


def _resolve_color(c):
    if isinstance(c, str):
        return NAMED_COLORS[c]
    return tuple(c)


def _load_font(font_file: str, size: int):
    """Find a font by filename across the usual locations; fall back to PIL's default."""
    for d in _FONT_DIRS:
        p = Path(d) / font_file
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                pass
    try:
        return ImageFont.truetype(font_file, size)
    except OSError:
        return ImageFont.load_default()


def _fit(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((CANVAS, CANVAS), Image.LANCZOS)


# ---------------------------------------------------------------------------
# UF1 — grid / collage
# ---------------------------------------------------------------------------

def collage(sources: list[Image.Image], params: dict) -> Image.Image:
    """Grid of image crops with optional gutters.

    Findings that motivated this parameterisation (see experiment-log.md):
      - the gutters are REQUIRED: identical tiles with line_width=0 never cross
      - grid density has a resonance around 10-32; coarser than 8 never crosses
      - line colour preference inverts with density (black wins coarse, white wins fine)
      - mixed sources beat any single source (0.93 vs 0.76 mean dog probability)
    """
    n = int(params["grid"])
    line_w = int(params.get("line_width", 0))
    color = _resolve_color(params.get("line_color", "black"))
    seed = int(params.get("seed", 0))
    min_crop = float(params.get("min_crop_frac", 0.25))
    mode = params.get("source_mode", "mixed")

    pool = [_fit(s) for s in sources] if mode == "mixed" else [_fit(sources[0])]
    rng = random.Random(seed)
    step = CANVAS / n
    tile_px = max(1, int(round(step)) + 1)
    out = Image.new("RGB", (CANVAS, CANVAS))

    for gy in range(n):
        for gx in range(n):
            src = rng.choice(pool)
            cw = rng.randint(max(1, int(CANVAS * min_crop)), CANVAS - 1)
            x0 = rng.randint(0, CANVAS - cw)
            y0 = rng.randint(0, CANVAS - cw)
            crop = src.crop((x0, y0, x0 + cw, y0 + cw)).resize((tile_px, tile_px), Image.LANCZOS)
            out.paste(crop, (int(gx * step), int(gy * step)))

    if line_w > 0:
        d = ImageDraw.Draw(out)
        for i in range(n + 1):
            p = int(i * step)
            d.line([(p, 0), (p, CANVAS)], fill=color, width=line_w)
            d.line([(0, p), (CANVAS, p)], fill=color, width=line_w)
    return out


# ---------------------------------------------------------------------------
# UF2 — watermark / text
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _watermark_layer(font_file: str, size: int, spacing: int, opacity: float, angle: float,
                     color: tuple, text: str) -> Image.Image:
    """Build (and cache) the watermark overlay.

    The overlay depends only on its own parameters, never on the base image, so it is rendered
    once per unique parameter set and composited onto each image. This matters: drawing ~900
    individual glyphs per image was the dominant cost of the UF2 search, and the same layer is
    reused across every base image in a `per_image` sweep.
    """
    layer = Image.new("RGBA", (CANVAS * 2, CANVAS * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = _load_font(font_file, size)
    a = int(255 * opacity)
    for y in range(0, CANVAS * 2, max(1, spacing)):
        for x in range(0, CANVAS * 2, max(1, spacing * 3)):
            d.text((x, y), text, font=f, fill=(*color, a))
    layer = layer.rotate(angle, resample=Image.BICUBIC, center=(CANVAS, CANVAS))
    return layer.crop((CANVAS // 2, CANVAS // 2, CANVAS // 2 + CANVAS, CANVAS // 2 + CANVAS))


def watermark(sources: list[Image.Image], params: dict) -> Image.Image:
    """Tiled diagonal watermark, stock-photo style.

    Findings that motivated this parameterisation:
      - text CONTENT is irrelevant (the model has no OCR ability), so `text` is not searched
      - glyph SIZE and tiling DENSITY dominate; OPACITY is a weak, non-monotonic lever
      - rotation ANGLE is unexpectedly strong (~11.6 logits between the worst and best angle)
      - a single large wordmark never crosses, at any size or opacity
    """
    img = _fit(sources[0])
    layer = _watermark_layer(
        params.get("font", "Arial.ttf"),
        int(params["font_size"]),
        int(params["spacing"]),
        float(params["opacity"]),
        float(params.get("angle", 30)),
        _resolve_color(params.get("color", "white")),
        params.get("text", "© SAMPLE"),
    )
    out = img.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


# ---------------------------------------------------------------------------
# UF3 — messy / low quality
# ---------------------------------------------------------------------------

def degrade(sources: list[Image.Image], params: dict) -> Image.Image:
    """Quality degradation: sensor-noise style artefacts.

    Findings that motivated this parameterisation:
      - additive gaussian noise crosses on every base image; strength is monotonic
      - alpha-blending toward uniform noise also works
      - SALT-AND-PEPPER NEVER CROSSES at any density -- kept in the search space anyway as a
        documented control, since its failure is what isolates *smooth* fine-grained variation
        as the trigger rather than disruption in general
      - blur alone only reaches the model's no-evidence prior; it cannot cross
    """
    img = _fit(sources[0])
    kind = params.get("kind", "gaussian")
    strength = float(params["strength"])
    seed = int(params.get("seed", 0))
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32)

    if kind == "gaussian":
        arr = arr + rng.normal(0, strength, arr.shape)
    elif kind == "uniform_blend":
        noise = rng.integers(0, 256, arr.shape).astype(np.float32)
        arr = (1 - strength) * arr + strength * noise
    elif kind == "salt_pepper":
        mask = rng.random(arr.shape[:2])
        arr[mask < strength / 2] = 0.0
        arr[(mask >= strength / 2) & (mask < strength)] = 255.0
    else:
        raise ValueError(f"unknown degrade kind: {kind!r}")

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


REGISTRY = {
    "collage": collage,
    "watermark": watermark,
    "degrade": degrade,
}


def apply_transform(name: str, sources: list[Image.Image], params: dict) -> Image.Image:
    try:
        fn = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown transform {name!r}; known: {sorted(REGISTRY)}") from None
    return fn(sources, params)
