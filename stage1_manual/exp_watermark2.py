"""E011 — Watermark grid search at REALISTIC density (UF2, second pass).

Paul (2026-08-24): spacing 10-14 is too dense to look like a real watermark. Freeze spacing at 20
and search the other levers instead.

Two carry-over winners from E005 at spacing >= 20, both barely over the line:
  woman.jpg  font 14, spacing 20, opacity 0.55, white, 30deg -> dog 0.614
  forest.jpg font 16, spacing 22, opacity 0.60, white, 30deg -> dog 0.602

Levers swept here, all at SPACING = 20:
  - font size
  - opacity
  - font colour (white / black / grey) -- never tested in E005, and E009 showed the colour
    preference can invert with scale
  - font family/weight -- never tested; stroke weight and glyph density vary a lot by face
  - rotation angle -- fixed at 30deg throughout E005
"""

import itertools

from PIL import Image, ImageDraw, ImageFont

from score import HOMEWORK_DIR, OUTPUTS_DIR, score_image

CANVAS = 512
SPACING = 20
FONT_DIR = "/System/Library/Fonts/Supplemental/"

FONTS = {
    "Arial": "Arial.ttf",
    "ArialBold": "Arial Bold.ttf",
    "ArialBlack": "Arial Black.ttf",
    "ArialNarrow": "Arial Narrow.ttf",
    "Courier": "Courier New.ttf",
    "CourierBold": "Courier New Bold.ttf",
    "Georgia": "Georgia.ttf",
    "Impact": "Impact.ttf",
    "Verdana": "Verdana.ttf",
    "Times": "Times New Roman.ttf",
    "ComicSans": "Comic Sans MS.ttf",
    "Copperplate": "Copperplate.ttc",
}

COLORS = {"white": (255, 255, 255), "black": (0, 0, 0), "grey": (128, 128, 128)}


def load_base(name):
    return Image.open(HOMEWORK_DIR / "images" / name).convert("RGB").resize((CANVAS, CANVAS))


def _font(path, size):
    try:
        return ImageFont.truetype(FONT_DIR + path, size)
    except OSError:
        return ImageFont.load_default()


def watermark(img, text="© SAMPLE", font_file="Arial.ttf", font_size=14, spacing=SPACING,
              opacity=0.6, angle=30, color=(255, 255, 255)):
    layer = Image.new("RGBA", (CANVAS * 2, CANVAS * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = _font(font_file, font_size)
    a = int(255 * opacity)
    for y in range(0, CANVAS * 2, spacing):
        for x in range(0, CANVAS * 2, spacing * 3):
            d.text((x, y), text, font=f, fill=(*color, a))
    layer = layer.rotate(angle, resample=Image.BICUBIC, center=(CANVAS, CANVAS))
    layer = layer.crop((CANVAS // 2, CANVAS // 2, CANVAS // 2 + CANVAS, CANVAS // 2 + CANVAS))
    out = img.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


def sweep(label, base_name, param_grid, fixed, out_sub, top_n=8):
    """param_grid: dict of name -> list of values. Cartesian product."""
    out_dir = OUTPUTS_DIR / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_base(base_name)
    keys = list(param_grid)
    results = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        kw = dict(fixed)
        kw.update(dict(zip(keys, combo)))
        cname = kw.pop("color_name")
        kw["color"] = COLORS[cname]
        fname = kw.pop("font_name")
        kw["font_file"] = FONTS[fname]
        img = watermark(base, **kw)
        r = score_image(img)
        tag = f"{fname}_s{kw['font_size']}_o{kw['opacity']}_{cname}_a{kw['angle']}"
        results.append((r["probs"]["dog"], r["pred"], tag, r["logits"]["dog"] - r["logits"]["other"]))
        if r["probs"]["dog"] > 0.60 and r["pred"] == "dog":
            img.save(out_dir / f"{base_name.split('.')[0]}__{tag}_dog{r['probs']['dog']:.3f}.png")

    dogs = [x for x in results if x[1] == "dog"]
    print(f"\n{'='*100}\n{label}  ({len(results)} configs, spacing={SPACING})\n{'='*100}")
    print(f"  crossings: {len(dogs)}/{len(results)} predict dog")
    print(f"  Top {top_n}:")
    for dp, pred, tag, m in sorted(results, reverse=True)[:top_n]:
        print(f"    dog={dp:.4f}  margin={m:+6.2f}  pred={pred:<5}  {tag}")
    return results


if __name__ == "__main__":
    print("E011 — watermark grid search at realistic density (spacing=20)")
    print("Baseline to beat: woman dog=0.614 | forest dog=0.602 | best-ever (dense) dog=0.822")

    grid = {
        "font_name": ["Arial", "ArialBold", "ArialBlack", "ArialNarrow", "Courier",
                      "CourierBold", "Georgia", "Impact", "Verdana", "Times",
                      "ComicSans", "Copperplate"],
        "font_size": [10, 12, 14, 16, 18],
        "opacity": [0.4, 0.55, 0.7, 0.85],
        "color_name": ["white", "black", "grey"],
        "angle": [30],
    }
    fixed = {"text": "© SAMPLE", "spacing": SPACING}

    r_woman = sweep("A. woman.jpg", "woman.jpg", grid, fixed, "e011_wm2_woman")
    r_forest = sweep("B. forest.jpg", "forest.jpg", grid, fixed, "e011_wm2_forest")

    # angle sweep on whichever base did better
    best_w = max([x[0] for x in r_woman] + [0])
    best_f = max([x[0] for x in r_forest] + [0])
    winner_base = "woman.jpg" if best_w >= best_f else "forest.jpg"
    winner_res = r_woman if best_w >= best_f else r_forest
    top = max(winner_res, key=lambda t: t[0])
    fname, s, o, cname, _ = top[2].split("_")
    print(f"\n>>> Best base: {winner_base} ({top[2]}, dog {top[0]:.4f}). Sweeping rotation angle:")

    grid2 = {
        "font_name": [fname],
        "font_size": [int(s[1:])],
        "opacity": [float(o[1:])],
        "color_name": [cname],
        "angle": [0, 15, 30, 45, 60, 75, 90],
    }
    sweep(f"C. angle sweep on {winner_base}", winner_base, grid2, fixed,
          "e011_wm2_angle", top_n=7)
