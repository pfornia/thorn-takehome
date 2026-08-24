"""E010 — Seed / source sensitivity of the g10_w8_black collage (UF1).

Paul (2026-08-24): how sensitive is the winning collage to the random seed, and does it need
tiles from MULTIPLE base images, or does a single source work just as well?

This separates two things the winning candidate confounds:
  - the periodic grid structure (10x10 tiles + 8px black gutters)
  - the diversity of image content across tiles

A. 30 seeds, tiles drawn randomly from all five base images.
B. 10 seeds per base image, tiles drawn from ONE image only (all five tested).

If single-source performs comparably, content diversity is irrelevant and the trigger is purely
structural -- a cleaner and stronger claim for the write-up.
"""

import random
import statistics

from PIL import Image, ImageDraw

from score import HOMEWORK_DIR, score_image

CANVAS = 512
BASES = ["ocean.jpg", "woman.jpg", "forest.jpg", "man.jpeg", "rav4.jpg"]
GRID, LINE_W, LINE_COLOR = 10, 8, (0, 0, 0)


def load_bases():
    return {
        b: Image.open(HOMEWORK_DIR / "images" / b).convert("RGB").resize((CANVAS, CANVAS))
        for b in BASES
    }


def collage(sources, n=GRID, line_w=LINE_W, line_color=LINE_COLOR, seed=0, min_crop_frac=0.25):
    """`sources` is a list of PIL images to draw tiles from (1 or many)."""
    rng = random.Random(seed)
    step = CANVAS / n
    out = Image.new("RGB", (CANVAS, CANVAS))
    tile_px = max(1, int(round(step)) + 1)
    for gy in range(n):
        for gx in range(n):
            src = rng.choice(sources)
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


def summarize(label, probs, preds):
    hits = sum(1 for p in preds if p == "dog")
    print(
        f"  {label:<22} dog: mean={statistics.mean(probs):.3f} "
        f"median={statistics.median(probs):.3f} "
        f"min={min(probs):.3f} max={max(probs):.3f} "
        f"sd={statistics.pstdev(probs):.3f} | {hits}/{len(preds)} predict dog"
    )
    return hits


if __name__ == "__main__":
    out_dir = HOMEWORK_DIR / "outputs" / "e010_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = load_bases()

    print(f"E010 — seed/source sensitivity, grid={GRID} width={LINE_W} colour=black")
    print("=" * 100)

    # ---- A. mixed sources, 30 seeds ----
    print("\nA. MIXED sources (all 5 base images), 30 seeds")
    all_srcs = list(imgs.values())
    probs, preds = [], []
    for seed in range(30):
        img = collage(all_srcs, seed=seed)
        if seed < 5:
            img.save(out_dir / f"mixed_seed{seed}.png")
        r = score_image(img)
        probs.append(r["probs"]["dog"])
        preds.append(r["pred"])
    summarize("mixed (n=30)", probs, preds)
    non_dog = [(s, p, pr) for s, (p, pr) in enumerate(zip(probs, preds)) if pr != "dog"]
    if non_dog:
        print("    seeds NOT predicting dog: " + ", ".join(
            f"seed{s}({pr} d={p:.2f})" for s, p, pr in non_dog))

    # ---- B. single source, 10 seeds each ----
    print("\nB. SINGLE source (all tiles from one image), 10 seeds each")
    single_results = {}
    for name, im in imgs.items():
        probs, preds = [], []
        for seed in range(10):
            img = collage([im], seed=seed)
            if seed < 2:
                img.save(out_dir / f"single_{name.split('.')[0]}_seed{seed}.png")
            r = score_image(img)
            probs.append(r["probs"]["dog"])
            preds.append(r["pred"])
        hits = summarize(name, probs, preds)
        single_results[name] = (statistics.mean(probs), hits)

    print("\n" + "=" * 100)
    print("Single-source ranked by mean dog probability:")
    for name, (mean_p, hits) in sorted(single_results.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:<14} mean dog={mean_p:.3f}  {hits}/10 predict dog")
