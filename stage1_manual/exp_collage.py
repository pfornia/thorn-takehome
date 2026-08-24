"""E007 — Collage / grid refinement (User Feedback 1).

UF1: "sort of like a grid or collage and they seem to have lots of lines running through them."
E006 found rav4 tiled 8x8 -> dog 0.63. This pushes further on the three levers UF1 names:
  A. finer grids (tile density)
  B. explicit grid LINES between tiles (width, colour) -- UF1 mentions lines specifically
  C. mixed-source collage (crops drawn from all five base images)
  D. control: lines ONLY, no tiling -- isolates the line contribution
"""

import random

from PIL import Image, ImageDraw

from score import HOMEWORK_DIR, OUTPUTS_DIR, score_image

CANVAS = 512
BASES = ["ocean.jpg", "woman.jpg", "forest.jpg", "man.jpeg", "rav4.jpg"]


def load_base(name: str) -> Image.Image:
    return Image.open(HOMEWORK_DIR / "images" / name).convert("RGB").resize((CANVAS, CANVAS))


def tile_n(img, n, line_w=0, line_color=(255, 255, 255)):
    """Repeat image n x n, optionally with grid lines drawn between tiles."""
    step = CANVAS / n
    small = img.resize((max(1, int(step)), max(1, int(step))), Image.LANCZOS)
    out = Image.new("RGB", (CANVAS, CANVAS))
    for gy in range(n):
        for gx in range(n):
            out.paste(small, (int(gx * step), int(gy * step)))
    if line_w > 0:
        d = ImageDraw.Draw(out)
        for i in range(n + 1):
            p = int(i * step)
            d.line([(p, 0), (p, CANVAS)], fill=line_color, width=line_w)
            d.line([(0, p), (CANVAS, p)], fill=line_color, width=line_w)
    return out


def mixed_collage(bases, n, line_w=0, line_color=(255, 255, 255), seed=0):
    """Grid of random crops drawn from ALL base images."""
    rng = random.Random(seed)
    step = CANVAS / n
    out = Image.new("RGB", (CANVAS, CANVAS))
    for gy in range(n):
        for gx in range(n):
            src = rng.choice(bases)
            cw = rng.randint(CANVAS // 4, CANVAS - 1)
            x0 = rng.randint(0, CANVAS - cw)
            y0 = rng.randint(0, CANVAS - cw)
            crop = src.crop((x0, y0, x0 + cw, y0 + cw)).resize(
                (max(1, int(step)), max(1, int(step))), Image.LANCZOS
            )
            out.paste(crop, (int(gx * step), int(gy * step)))
    if line_w > 0:
        d = ImageDraw.Draw(out)
        for i in range(n + 1):
            p = int(i * step)
            d.line([(p, 0), (p, CANVAS)], fill=line_color, width=line_w)
            d.line([(0, p), (CANVAS, p)], fill=line_color, width=line_w)
    return out


def lines_only(img, n, line_w=2, line_color=(255, 255, 255)):
    """Grid lines drawn over the UNMODIFIED image -- isolates the line effect."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    step = CANVAS / n
    for i in range(n + 1):
        p = int(i * step)
        d.line([(p, 0), (p, CANVAS)], fill=line_color, width=line_w)
        d.line([(0, p), (CANVAS, p)], fill=line_color, width=line_w)
    return out


def run(label, variants, out_sub, bases=BASES):
    out_dir = OUTPUTS_DIR / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*112}\n{label}\n{'='*112}")
    hdr = f"{'variant':>22} | " + " | ".join(f"{b.split('.')[0]:>14}" for b in bases)
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
            cells.append(f"{m:+7.2f} {r['probs']['dog']:.2f}{flag}")
            best.append((m, vname, b, r["pred"], r["probs"]["dog"]))
        print(f"{vname:>22} | " + " | ".join(f"{c:>14}" for c in cells))
    print("\n  Top 5:")
    for m, vname, b, pred, dp in sorted(best, reverse=True)[:5]:
        print(f"    {m:+7.2f}  {vname:<20} on {b:<12} pred={pred} dog={dp:.3f}")
    return best


if __name__ == "__main__":
    print("E007 — collage / grid refinement (UF1)")
    print("Refs: prior~-2.5 | rav4 tile8x8=+1.32 (dog .63) | woman+noise60=+6.70 (dog 1.00)")

    # A. finer grids, no lines
    run("A. TILE DENSITY (no lines)",
        [(f"tile{n}", (lambda n: lambda im: tile_n(im, n))(n))
         for n in (8, 10, 12, 16, 20, 24, 32)],
        "e007_density")

    # B. grid lines added to a fixed 12x12 tiling
    run("B. GRID LINES on 12x12 tiling (width, colour)",
        [(f"t12_w{w}_{cn}", (lambda w, c: lambda im: tile_n(im, 12, line_w=w, line_color=c))(w, c))
         for w, c, cn in [
             (1, (255, 255, 255), "wht"), (2, (255, 255, 255), "wht"),
             (4, (255, 255, 255), "wht"), (2, (0, 0, 0), "blk"),
             (4, (0, 0, 0), "blk"), (2, (128, 128, 128), "gry")]],
        "e007_lines")

    # C. mixed-source collage
    bases_imgs = [load_base(b) for b in BASES]
    run("C. MIXED-SOURCE collage (crops from all 5 images)",
        [(f"mix{n}_w{w}", (lambda n, w: lambda im: mixed_collage(
            bases_imgs, n, line_w=w, seed=42))(n, w))
         for n, w in [(8, 0), (12, 0), (16, 0), (24, 0), (12, 2), (16, 2), (24, 2)]],
        "e007_mixed",
        bases=BASES[:1])  # source-independent; one column is enough

    # D. control: lines only, no tiling
    run("D. CONTROL — grid LINES only, image untiled",
        [(f"lines{n}_w{w}", (lambda n, w: lambda im: lines_only(im, n, line_w=w))(n, w))
         for n, w in [(8, 2), (16, 2), (24, 2), (32, 2), (32, 1), (48, 1)]],
        "e007_linesonly")
