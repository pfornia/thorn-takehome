"""E006 — Is it SPATIAL FREQUENCY (texture scale), independent of content?

Paul's test (2026-08-21): compare an N x N collage of an image against the individual sub-tiles
classified alone. Same content, different scale.

Mechanism prediction: after the model's fixed 224x224 resize,
  - tiling N x N  shrinks every feature by N  -> spatial frequency x N   -> MORE dog-like
  - cropping 1/N and upscaling magnifies features -> frequency / N      -> LESS dog-like
Content is held constant in both directions; only scale changes. If the margins move in opposite
directions, frequency (not content) is the driver.

Conditions per base image:
  A. original
  B. tile_N: the SAME image repeated N x N (no borders)
  C. crop_N: a single 1/N-linear crop upscaled back to full canvas (the "sub-tile alone")
  D. down_N: whole image downscaled by N then upscaled back (frequency reduced, framing kept)
"""

from PIL import Image

from score import HOMEWORK_DIR, score_image

CANVAS = 512
BASES = ["ocean.jpg", "woman.jpg", "forest.jpg", "man.jpeg", "rav4.jpg"]
FACTORS = [2, 4, 8]


def load_base(name: str) -> Image.Image:
    return Image.open(HOMEWORK_DIR / "images" / name).convert("RGB").resize((CANVAS, CANVAS))


def tile_n(img: Image.Image, n: int) -> Image.Image:
    """Repeat the same image n x n, no borders. Features shrink by n."""
    small = img.resize((CANVAS // n, CANVAS // n), Image.LANCZOS)
    out = Image.new("RGB", (CANVAS, CANVAS))
    for gy in range(n):
        for gx in range(n):
            out.paste(small, (gx * (CANVAS // n), gy * (CANVAS // n)))
    return out


def crop_n(img: Image.Image, n: int) -> Image.Image:
    """One centre sub-tile (1/n linear) blown back up. Features magnify by n."""
    s = CANVAS // n
    off = (CANVAS - s) // 2
    return img.crop((off, off, off + s, off + s)).resize((CANVAS, CANVAS), Image.LANCZOS)


def down_n(img: Image.Image, n: int) -> Image.Image:
    """Downscale then upscale: same framing, reduced high-frequency content."""
    return img.resize((CANVAS // n, CANVAS // n), Image.LANCZOS).resize(
        (CANVAS, CANVAS), Image.NEAREST
    )


if __name__ == "__main__":
    out_dir = HOMEWORK_DIR / "outputs" / "e006_scale"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E006 — spatial frequency vs content")
    print("tile_N  = same image repeated NxN      -> frequency x N")
    print("crop_N  = one 1/N sub-tile, upscaled   -> frequency / N")
    print("down_N  = downscale+upscale, same view -> frequency / N (framing preserved)")
    print("Refs: prior~-2.5 | pure uniform noise=+4.48 | woman+noise60=+6.70")

    conditions = [("original", lambda im: im)]
    for n in FACTORS:
        conditions.append((f"tile_{n}x{n}", (lambda n: lambda im: tile_n(im, n))(n)))
    for n in FACTORS:
        conditions.append((f"crop_1/{n}", (lambda n: lambda im: crop_n(im, n))(n)))
    for n in FACTORS:
        conditions.append((f"down_{n}x", (lambda n: lambda im: down_n(im, n))(n)))

    hdr = f"{'condition':>14} | " + " | ".join(f"{b.split('.')[0]:>14}" for b in BASES)
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    table = {}
    for cname, fn in conditions:
        cells = []
        for b in BASES:
            img = fn(load_base(b))
            img.save(out_dir / f"{b.split('.')[0]}__{cname.replace('/', '')}.png")
            r = score_image(img)
            m = r["logits"]["dog"] - r["logits"]["other"]
            table[(cname, b)] = m
            flag = "*" if r["pred"] != "other" else " "
            cells.append(f"{m:+7.2f} {r['probs']['dog']:.2f}{flag}")
        print(f"{cname:>14} | " + " | ".join(f"{c:>14}" for c in cells))

    print("=" * len(hdr))
    print("\nDelta vs original (positive = more dog-like):")
    print(f"{'condition':>14} | " + " | ".join(f"{b.split('.')[0]:>14}" for b in BASES))
    print("-" * len(hdr))
    for cname, _ in conditions[1:]:
        cells = [f"{table[(cname, b)] - table[('original', b)]:+14.2f}" for b in BASES]
        print(f"{cname:>14} | " + " | ".join(cells))

    print("\nIf frequency drives it: tile_* deltas should be POSITIVE, crop_*/down_* NEGATIVE.")
