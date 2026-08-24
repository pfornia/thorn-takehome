"""E002 — Does the model respond to SHAPE/SILHOUETTE, or only to local texture?

⚠️ DIAGNOSTIC ONLY — these images intentionally *depict* dogs/cats, so they are NOT valid
deliverables (the assignment requires images that "definitely do not depict a cat or dog").
They exist to establish an upper bound: if an explicit dog silhouette barely moves the model,
shape is irrelevant and the whole search should be texture/content-driven.

Two probes:
  A. ASCII art of a dog/cat rendered as text (with non-animal ASCII controls).
  B. Photo-mosaic: tiles cropped from the provided base images, brightness-modulated by a
     dog silhouette mask, so the mosaic "reads" as a dog at low resolution.
"""

import random

from PIL import Image, ImageDraw, ImageFont

from score import OUTPUTS_DIR, format_score, score_image

MONO_FONT = "/System/Library/Fonts/Supplemental/Andale Mono.ttf"
HOMEWORK_IMAGES = None  # set in __main__
CANVAS = 512

ASCII_ART = {
    "dog_ascii": r"""
      / \__
     (    @\___
     /         O
    /   (_____/
   /_____/   U
""",
    "cat_ascii": r"""
     /\_/\
    ( o.o )
     > ^ <
""",
    "dog_ascii_big": r"""
        __
   (\,--------'()'--o
    (_    ___    /~"
     (_)_)  (_)_)
""",
    # ---- non-animal ASCII controls ----
    "tree_ascii": r"""
       /\
      /  \
     /    \
    /______\
       ||
       ||
""",
    "house_ascii": r"""
      /\
     /  \
    /____\
    |    |
    | [] |
    |____|
""",
    "noise_ascii": r"""
   x#@%&*x#@
   @%x*#&@%x
   *&#@x%*&#
   %x@&*#%x@
""",
}


def render_ascii(art: str, font_size: int = 28) -> Image.Image:
    """Render multi-line ASCII art in monospace, black on white, centred."""
    img = Image.new("RGB", (CANVAS, CANVAS), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(MONO_FONT, font_size)
    except OSError:
        font = ImageFont.load_default()
    lines = [l for l in art.split("\n") if l.strip()]
    text = "\n".join(lines)
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font)
    draw.multiline_text(
        ((CANVAS - (right - left)) / 2 - left, (CANVAS - (bottom - top)) / 2 - top),
        text,
        fill="black",
        font=font,
    )
    return img


def dog_silhouette_mask(size: int = CANVAS) -> Image.Image:
    """Crude but recognisable dog silhouette as an 'L' mask (255 = dog body)."""
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    s = size / 512.0
    # body
    d.ellipse([120 * s, 230 * s, 380 * s, 360 * s], fill=255)
    # head
    d.ellipse([300 * s, 150 * s, 430 * s, 280 * s], fill=255)
    # snout
    d.ellipse([390 * s, 210 * s, 470 * s, 265 * s], fill=255)
    # ears
    d.polygon([(320 * s, 165 * s), (300 * s, 90 * s), (365 * s, 140 * s)], fill=255)
    d.polygon([(390 * s, 160 * s), (405 * s, 85 * s), (430 * s, 150 * s)], fill=255)
    # legs
    for x in (150, 210, 290, 345):
        d.rectangle([x * s, 340 * s, (x + 35) * s, 450 * s], fill=255)
    # tail
    d.polygon([(120 * s, 260 * s), (60 * s, 180 * s), (95 * s, 175 * s), (140 * s, 250 * s)], fill=255)
    return mask


def photo_mosaic(base_images, mask: Image.Image, grid: int = 16, size: int = CANVAS,
                 rng: random.Random | None = None) -> Image.Image:
    """Mosaic of crops from the provided base images, modulated by `mask`.

    Tiles inside the silhouette are kept bright; tiles outside are darkened, so the
    arrangement reads as the mask's shape at low resolution.
    """
    rng = rng or random.Random(0)
    canvas = Image.new("RGB", (size, size), "white")
    tile = size // grid
    mask_small = mask.resize((grid, grid), Image.LANCZOS)

    for gy in range(grid):
        for gx in range(grid):
            src = rng.choice(base_images)
            cw = rng.randint(min(80, src.width - 1), src.width - 1)
            ch = rng.randint(min(80, src.height - 1), src.height - 1)
            x0 = rng.randint(0, src.width - cw)
            y0 = rng.randint(0, src.height - ch)
            crop = src.crop((x0, y0, x0 + cw, y0 + ch)).resize((tile, tile), Image.LANCZOS)

            inside = mask_small.getpixel((gx, gy)) > 127
            # brighten inside the silhouette, darken outside -> shape becomes visible
            factor = 1.0 if inside else 0.25
            crop = crop.point(lambda v, f=factor: int(min(255, v * f)))
            canvas.paste(crop, (gx * tile, gy * tile))
    return canvas


if __name__ == "__main__":
    from score import HOMEWORK_DIR, OUTPUTS_DIR

    print("E002 — SHAPE/SILHOUETTE probe (diagnostic only; these depict animals by design)")
    print("=" * 112)

    print("\n--- A. ASCII art ---")
    results = []
    for name, art in ASCII_ART.items():
        img = render_ascii(art)
        r = score_image(img)
        m = r["logits"]["dog"] - r["logits"]["other"]
        results.append((name, m))
        print(f"{format_score(name, r)}  dog_margin={m:+.3f}")

    print("\n--- B. Photo-mosaic shaped like a dog (tiles from the 5 base images) ---")
    bases = [
        Image.open(p).convert("RGB")
        for p in sorted((HOMEWORK_DIR / "images").glob("*"))
        if p.is_file()
    ]
    mask = dog_silhouette_mask()
    out_dir = OUTPUTS_DIR
    out_dir.mkdir(exist_ok=True)

    for grid in (8, 16, 32, 64):
        img = photo_mosaic(bases, mask, grid=grid, rng=random.Random(42))
        img.save(out_dir / f"e002_mosaic_dog_grid{grid}.png")
        r = score_image(img)
        m = r["logits"]["dog"] - r["logits"]["other"]
        results.append((f"mosaic_dog_grid{grid}", m))
        print(f"{format_score(f'mosaic_dog grid={grid}', r)}  dog_margin={m:+.3f}")

    # control: same mosaic machinery, no silhouette (uniform mask) -> isolates shape effect
    flat = Image.new("L", (CANVAS, CANVAS), 255)
    for grid in (16, 32):
        img = photo_mosaic(bases, flat, grid=grid, rng=random.Random(42))
        img.save(out_dir / f"e002_mosaic_flat_grid{grid}.png")
        r = score_image(img)
        m = r["logits"]["dog"] - r["logits"]["other"]
        results.append((f"mosaic_FLAT_grid{grid}", m))
        print(f"{format_score(f'mosaic_FLAT(control) grid={grid}', r)}  dog_margin={m:+.3f}")

    print("\n" + "=" * 112)
    print("Ranked by dog_margin (baseline refs: blank_white=-2.950, ocean.jpg=-11.933):")
    for name, m in sorted(results, key=lambda t: -t[1]):
        print(f"  {name:<26} dog_margin={m:+8.3f}")
