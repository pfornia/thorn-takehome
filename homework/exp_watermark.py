"""E005 — Watermark / text overlay sweeps (User Feedback 2).

Tests the hypothesis recorded in strategy.md: if the trigger is high-frequency *texture*
(established in E003/E004), then watermark DENSITY and GLYPH SCALE should dominate OPACITY.
A single large wordmark is low-frequency and should do little; dense small repeated glyphs
are high-frequency and should behave like the noise that already works.

Text *content* is known irrelevant (E001), so wording is held fixed except in one control.
"""

from PIL import Image, ImageDraw, ImageFont

from score import HOMEWORK_DIR, score_image

CANVAS = 512
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
BASES = ["ocean.jpg", "woman.jpg", "forest.jpg", "man.jpeg", "rav4.jpg"]


def load_base(name: str) -> Image.Image:
    return Image.open(HOMEWORK_DIR / "images" / name).convert("RGB").resize((CANVAS, CANVAS))


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def tiled_watermark(img, text="© SAMPLE", font_size=28, spacing=None, opacity=0.5,
                    angle=30, color=(255, 255, 255)):
    """Repeated diagonal watermark tiled across the frame (stock-photo style)."""
    spacing = spacing if spacing is not None else font_size * 4
    layer = Image.new("RGBA", (CANVAS * 2, CANVAS * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = _font(font_size)
    alpha = int(255 * opacity)
    for y in range(0, CANVAS * 2, spacing):
        for x in range(0, CANVAS * 2, spacing * 3):
            d.text((x, y), text, font=f, fill=(*color, alpha))
    layer = layer.rotate(angle, resample=Image.BICUBIC, center=(CANVAS, CANVAS))
    layer = layer.crop((CANVAS // 2, CANVAS // 2, CANVAS // 2 + CANVAS, CANVAS // 2 + CANVAS))
    out = img.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def single_wordmark(img, text="SAMPLE", font_size=90, opacity=0.5, color=(255, 255, 255)):
    """One big centred low-frequency wordmark."""
    layer = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = _font(font_size)
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    d.text(((CANVAS - (r - l)) / 2 - l, (CANVAS - (b - t)) / 2 - t), text, font=f,
           fill=(*color, int(255 * opacity)))
    out = img.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def run(label, variants, out_sub, bases=BASES):
    out_dir = HOMEWORK_DIR / "outputs" / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*118}\n{label}\n{'='*118}")
    hdr = f"{'variant':>26} | " + " | ".join(f"{b.split('.')[0]:>15}" for b in bases)
    print(hdr)
    print("-" * len(hdr))
    best = []
    for vname, fn in variants:
        cells = []
        for b in bases:
            img = fn(load_base(b))
            img.save(out_dir / f"{b.split('.')[0]}__{vname}.png")
            r = score_image(img)
            m = r["logits"]["dog"] - r["logits"]["other"]
            flag = "*" if r["pred"] != "other" else " "
            cells.append(f"{m:+6.2f} {r['probs']['dog']:.2f}{flag}")
            best.append((m, vname, b, r["pred"], r["probs"]["dog"]))
        print(f"{vname:>26} | " + " | ".join(f"{c:>15}" for c in cells))
    print("\n  Top 5 by dog_margin:")
    for m, vname, b, pred, dp in sorted(best, reverse=True)[:5]:
        print(f"    {m:+7.2f}  {vname:<24} on {b:<12} pred={pred} dog={dp:.3f}")
    return best


if __name__ == "__main__":
    print("E005 — watermark / text overlay (UF2)")
    print("Refs: prior~-2.5 | woman+noise60=+6.70 (dog 1.00) | best base=ocean -11.93")

    # --- 1. GLYPH SCALE at fixed high density and opacity ---
    run(
        "1. GLYPH SCALE sweep (dense tiling, opacity 0.6)",
        [
            (f"size{s}_dense", (lambda s: lambda im: tiled_watermark(
                im, font_size=s, spacing=max(6, int(s * 1.4)), opacity=0.6))(s))
            for s in (8, 12, 16, 24, 36, 56, 90)
        ],
        "e005_scale",
    )

    # --- 2. DENSITY sweep at fixed small glyph size ---
    run(
        "2. DENSITY sweep (font 14, opacity 0.6; smaller spacing = denser)",
        [
            (f"spacing{sp}", (lambda sp: lambda im: tiled_watermark(
                im, font_size=14, spacing=sp, opacity=0.6))(sp))
            for sp in (100, 60, 40, 28, 20, 14, 10)
        ],
        "e005_density",
    )

    # --- 3. OPACITY sweep at fixed dense small glyphs ---
    run(
        "3. OPACITY sweep (font 14, spacing 20)",
        [
            (f"opacity{o}", (lambda o: lambda im: tiled_watermark(
                im, font_size=14, spacing=20, opacity=o))(o))
            for o in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)
        ],
        "e005_opacity",
    )

    # --- 4. Low-frequency control: one big wordmark ---
    run(
        "4. CONTROL — single large wordmark (low frequency)",
        [
            (f"wordmark{s}_op{o}", (lambda s, o: lambda im: single_wordmark(
                im, font_size=s, opacity=o))(s, o))
            for s, o in ((90, 0.5), (90, 1.0), (140, 1.0), (200, 1.0))
        ],
        "e005_wordmark",
    )
