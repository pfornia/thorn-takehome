"""E003 — Is the blank-white result just the model's PRIOR (no-evidence fallback)?

Paul's hypothesis (2026-08-21): blank white scores ~4.6% dog / 8.3% cat not because white is
dog-like, but because a featureless image carries no evidence, so the model falls back to
something like its training prior. If true, blank white is an ASYMPTOTE, not a building block:
destroying information can only walk you toward the prior, never past it — and the prior is
nowhere near the 50% needed.

Test: score many *different* information-free images. If they all cluster at the same margin,
that's a prior/fallback. If they vary widely, the model is still responding to features and
"blank" isn't special.
"""

import numpy as np
from PIL import Image, ImageFilter

from score import HOMEWORK_DIR, format_score, score_image

CANVAS = 512


def solid(color) -> Image.Image:
    return Image.new("RGB", (CANVAS, CANVAS), color)


def uniform_noise(seed: int, low: int = 0, high: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(low, high, size=(CANVAS, CANVAS, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def gray_noise(seed: int, sigma: float = 10.0, mean: int = 128) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.normal(mean, sigma, (CANVAS, CANVAS, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def blurred_to_death(path, radius: int) -> Image.Image:
    img = Image.open(path).convert("RGB").resize((CANVAS, CANVAS))
    return img.filter(ImageFilter.GaussianBlur(radius))


if __name__ == "__main__":
    print("E003 — prior / no-information probe")
    print("Reference points: blank_white=-2.950 | best base image ocean.jpg=-11.933")
    print("=" * 112)

    cases = []

    # --- solid colours: no information at all, but different colours ---
    for name, c in [
        ("solid_white", (255, 255, 255)),
        ("solid_black", (0, 0, 0)),
        ("solid_gray50", (128, 128, 128)),
        ("solid_gray25", (64, 64, 64)),
        ("solid_gray75", (191, 191, 191)),
        ("solid_red", (200, 30, 30)),
        ("solid_green", (30, 200, 30)),
        ("solid_blue", (30, 30, 200)),
        ("solid_tan", (190, 150, 100)),      # dog-fur-ish colour
        ("solid_brown", (110, 75, 45)),      # dog-fur-ish colour
    ]:
        cases.append((name, solid(c)))

    # --- structureless noise ---
    cases.append(("uniform_noise_full", uniform_noise(0)))
    cases.append(("gaussian_noise_sigma10", gray_noise(0, sigma=10)))
    cases.append(("gaussian_noise_sigma40", gray_noise(0, sigma=40)))

    # --- real photos blurred until featureless ---
    for base in ("ocean.jpg", "woman.jpg", "forest.jpg"):
        p = HOMEWORK_DIR / "images" / base
        for radius in (32, 96, 200):
            cases.append((f"blur{radius}_{base.split('.')[0]}", blurred_to_death(p, radius)))

    out_dir = HOMEWORK_DIR / "outputs" / "e003_prior"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, img in cases:
        img.save(out_dir / f"{name}.png")
        r = score_image(img)
        m = r["logits"]["dog"] - r["logits"]["other"]
        results.append((name, r, m))
        print(f"{format_score(name, r)}  dog_margin={m:+.3f}")

    print("=" * 112)
    print("Ranked by dog_margin:")
    for name, r, m in sorted(results, key=lambda t: -t[2]):
        p = r["probs"]
        print(
            f"  {name:<26} dog_margin={m:+8.3f}   dog={p['dog']:.4f} cat={p['cat']:.4f} other={p['other']:.4f}"
        )

    margins = [m for _, _, m in results]
    print(
        f"\nSpread across all no-information images: min={min(margins):+.3f} "
        f"max={max(margins):+.3f} range={max(margins)-min(margins):.3f} logits"
    )
    print("Tight cluster => prior/fallback (Paul's hypothesis). Wide spread => still feature-driven.")
