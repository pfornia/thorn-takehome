"""E009 — COARSE collage sweep (UF1, third pass).

Paul (2026-08-24): a 32x32 grid of 144+ tiles doesn't look like a realistic collage. Real
collages have relatively few, visibly distinct photos. Find the best result at grid <= 16,
ideally much coarser.

E008 capped line width at 3px, which is likely too thin for coarse grids -- at 32x32 the winning
1px line is ~6% of a tile's width, but at 6x6 a 1px line is ~1%. This sweeps proportionally
thicker lines at coarse densities.
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
    out_dir = OUTPUTS_DIR / "e009_coarse"
    out_dir.mkdir(parents=True, exist_ok=True)
    bases = load_bases()

    print("E009 — COARSE collage sweep (grid <= 16, wider lines)")
    print("Refs: E008 best g32_w1 dog .968 | best coarse so far g12_w3 dog .79")
    print("=" * 118)

    results = []
    grids = [3, 4, 5, 6, 8, 10, 12, 16]
    widths = [0, 2, 4, 6, 8, 12, 16]

    for color, cname in [((255, 255, 255), "white"), ((0, 0, 0), "black")]:
        print(f"\n{cname.upper()} lines — dog probability (pred shown if not 'other')")
        hdr = f"{'grid':>5} | " + " | ".join(f"{'w='+str(w):>16}" for w in widths)
        print(hdr)
        print("-" * len(hdr))
        for n in grids:
            cells = []
            for w in widths:
                img = mixed_collage(bases, n, line_w=w, line_color=color, seed=42)
                img.save(out_dir / f"mix_g{n}_w{w}_{cname}.png")
                r = score_image(img)
                m = r["logits"]["dog"] - r["logits"]["other"]
                results.append((r["probs"]["dog"], m, r["pred"], n, f"g{n}_w{w}_{cname}"))
                tag = r["pred"] if r["pred"] != "other" else "."
                cells.append(f"{r['probs']['dog']:.2f} {tag:<5} {m:+6.1f}")
            print(f"{n:>5} | " + " | ".join(f"{c:>16}" for c in cells))

    coarse_dogs = [r for r in results if r[2] == "dog" and r[3] <= 16]
    print("\n" + "=" * 118)
    print("Top 10 DOG predictions at grid <= 16:")
    for dp, m, pred, n, name in sorted(coarse_dogs, key=lambda t: -t[0])[:10]:
        print(f"  dog={dp:.3f}  margin={m:+7.2f}  {name}")

    if coarse_dogs:
        best = max(coarse_dogs, key=lambda t: t[0])
        n = best[3]
        w = int(best[4].split("_")[1][1:])
        cname = best[4].split("_")[2]
        color = (0, 0, 0) if cname == "black" else (255, 255, 255)
        print(f"\nSeed robustness for best coarse config ({best[4]}, dog {best[0]:.3f}):")
        hits = 0
        for seed in range(1, 11):
            img = mixed_collage(bases, n, line_w=w, line_color=color, seed=seed)
            img.save(out_dir / f"best_{best[4]}_seed{seed}.png")
            r = score_image(img)
            m = r["logits"]["dog"] - r["logits"]["other"]
            hits += r["pred"] == "dog"
            print(f"    seed {seed:>2}: {r['pred']:<5} dog={r['probs']['dog']:.3f}  margin={m:+6.2f}")
        print(f"  -> {hits}/10 seeds predict dog")
