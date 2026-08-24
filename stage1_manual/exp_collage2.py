"""E008 — Proper mixed-source COLLAGE sweep (UF1, second pass).

E007 sweep C under-explored the true multi-source collage: only 7 configs, one seed, and line
widths of 0 or 2 only -- never 1px, which was the clear winner in the lines-only sweep, and never
finer than 24x24.

UF1 describes "a grid or collage ... with lots of lines running through them", so a genuine
multi-image collage is a better match to the complaint than lines drawn on a single photo.
This sweeps grid density x line width x line colour x seed to find a collage that crosses
convincingly to `dog`.
"""

import random

from PIL import Image, ImageDraw

from score import HOMEWORK_DIR, OUTPUTS_DIR, score_image

CANVAS = 512
BASES = ["ocean.jpg", "woman.jpg", "forest.jpg", "man.jpeg", "rav4.jpg"]


def load_bases():
    return [
        Image.open(HOMEWORK_DIR / "images" / b).convert("RGB").resize((CANVAS, CANVAS))
        for b in BASES
    ]


def mixed_collage(bases, n, line_w=1, line_color=(255, 255, 255), seed=0, min_crop_frac=0.25):
    """Grid of random square crops drawn from ALL base images, with optional grid lines."""
    rng = random.Random(seed)
    step = CANVAS / n
    out = Image.new("RGB", (CANVAS, CANVAS))
    tile_px = max(1, int(round(step)) + 1)
    for gy in range(n):
        for gx in range(n):
            src = rng.choice(bases)
            cw = rng.randint(int(CANVAS * min_crop_frac), CANVAS - 1)
            x0 = rng.randint(0, CANVAS - cw)
            y0 = rng.randint(0, CANVAS - cw)
            crop = src.crop((x0, y0, x0 + cw, y0 + cw)).resize((tile_px, tile_px), Image.LANCZOS)
            out.paste(crop, (int(gx * step), int(gy * step)))
    if line_w > 0:
        d = ImageDraw.Draw(out)
        for i in range(n + 1):
            p = int(i * step)
            d.line([(p, 0), (p, CANVAS)], fill=line_color, width=line_w)
            d.line([(0, p), (CANVAS, p)], fill=line_color, width=line_w)
    return out


if __name__ == "__main__":
    out_dir = OUTPUTS_DIR / "e008_collage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    bases = load_bases()

    print("E008 — mixed-source collage, expanded sweep")
    print("Refs: prior~-2.5 | E007 best true collage = mix16_w2 dog .515 | man+lines16_w2 dog 1.000")
    print("=" * 104)

    results = []
    grids = [12, 16, 20, 24, 32, 40, 48]
    widths = [0, 1, 2, 3]

    hdr = f"{'grid':>6} | " + " | ".join(f"{'w='+str(w):>21}" for w in widths)
    print(hdr)
    print("-" * len(hdr))
    for n in grids:
        cells = []
        for w in widths:
            img = mixed_collage(bases, n, line_w=w, seed=42)
            img.save(out_dir / f"mix_g{n}_w{w}_white.png")
            r = score_image(img)
            m = r["logits"]["dog"] - r["logits"]["other"]
            results.append((m, r["probs"]["dog"], r["pred"], f"g{n}_w{w}_white"))
            flag = "*" if r["pred"] != "other" else " "
            cells.append(f"{m:+7.2f} {r['pred']:<5} d{r['probs']['dog']:.2f}{flag}")
        print(f"{n:>6} | " + " | ".join(f"{c:>21}" for c in cells))

    # black lines at the promising densities
    print("\nBlack lines:")
    print(hdr)
    print("-" * len(hdr))
    for n in (16, 24, 32, 40):
        cells = []
        for w in widths:
            img = mixed_collage(bases, n, line_w=w, line_color=(0, 0, 0), seed=42)
            img.save(out_dir / f"mix_g{n}_w{w}_black.png")
            r = score_image(img)
            m = r["logits"]["dog"] - r["logits"]["other"]
            results.append((m, r["probs"]["dog"], r["pred"], f"g{n}_w{w}_black"))
            flag = "*" if r["pred"] != "other" else " "
            cells.append(f"{m:+7.2f} {r['pred']:<5} d{r['probs']['dog']:.2f}{flag}")
        print(f"{n:>6} | " + " | ".join(f"{c:>21}" for c in cells))

    # seed variation on the best config found so far
    dog_hits = [r for r in results if r[2] == "dog"]
    if dog_hits:
        best = max(dog_hits, key=lambda t: t[1])
        print(f"\nBest dog config: {best[3]} (dog {best[1]:.3f}) — trying 8 more seeds:")
        n = int(best[3].split("_")[0][1:])
        w = int(best[3].split("_")[1][1:])
        color = (0, 0, 0) if "black" in best[3] else (255, 255, 255)
        for seed in range(1, 9):
            img = mixed_collage(bases, n, line_w=w, line_color=color, seed=seed)
            img.save(out_dir / f"mix_g{n}_w{w}_seed{seed}.png")
            r = score_image(img)
            m = r["logits"]["dog"] - r["logits"]["other"]
            results.append((m, r["probs"]["dog"], r["pred"], f"g{n}_w{w}_seed{seed}"))
            print(f"    seed {seed}: {m:+7.2f}  {r['pred']:<5} dog={r['probs']['dog']:.3f}")

    print("\n" + "=" * 104)
    print("Top 10 by dog probability (dog predictions only):")
    for m, dp, pred, name in sorted([r for r in results if r[2] == "dog"], key=lambda t: -t[1])[:10]:
        print(f"  dog={dp:.3f}  margin={m:+7.2f}  {name}")
