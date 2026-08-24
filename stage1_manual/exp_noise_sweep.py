"""E004 — Noise sweep on the REAL base images.

E003 showed pure synthetic uniform noise is classified `dog` at 68%, but that isn't a valid
deliverable (not a transformation of a provided image). This sweeps noise applied *to the base
photos* and answers: how much original photograph can survive while still crossing 50% dog?

Three parameterisations:
  A. Additive gaussian noise, sigma sweep.
  B. Alpha blend toward uniform noise (alpha=0 -> original, 1 -> pure noise).
  C. Salt-and-pepper noise, density sweep.
"""

import numpy as np
from PIL import Image

from score import HOMEWORK_DIR, OUTPUTS_DIR, score_image

CANVAS = 512
BASES = ["ocean.jpg", "woman.jpg", "forest.jpg", "man.jpeg", "rav4.jpg"]


def load_base(name: str) -> Image.Image:
    return Image.open(HOMEWORK_DIR / "images" / name).convert("RGB").resize((CANVAS, CANVAS))


def additive_gaussian(img: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def blend_uniform_noise(img: Image.Image, alpha: float, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.integers(0, 256, arr.shape).astype(np.float32)
    out = (1 - alpha) * arr + alpha * noise
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def salt_pepper(img: Image.Image, density: float, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).copy()
    mask = rng.random(arr.shape[:2])
    arr[mask < density / 2] = 0
    arr[(mask >= density / 2) & (mask < density)] = 255
    return Image.fromarray(arr)


def run_sweep(label, fn, values, out_sub, bases=BASES):
    out_dir = OUTPUTS_DIR / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*116}\n{label}\n{'='*116}")
    header = f"{'value':>8} | " + " | ".join(f"{b.split('.')[0]:>22}" for b in bases)
    print(header)
    print("-" * len(header))
    crossings = {}
    for v in values:
        cells = []
        for b in bases:
            img = fn(load_base(b), v)
            img.save(out_dir / f"{b.split('.')[0]}_{v}.png")
            r = score_image(img)
            m = r["logits"]["dog"] - r["logits"]["other"]
            p = r["probs"]
            best = "dog" if p["dog"] >= p["cat"] else "cat"
            flag = "*" if r["pred"] != "other" else " "
            cells.append(f"{m:+7.2f} {best[0]}{p[best]:.2f}{flag}")
            if r["pred"] != "other" and b not in crossings:
                crossings[b] = (v, r["pred"], p[r["pred"]])
        print(f"{v:>8} | " + " | ".join(f"{c:>22}" for c in cells))
    if crossings:
        print("\n  First crossing into a cat/dog prediction:")
        for b, (v, pred, prob) in crossings.items():
            print(f"    {b:<12} at value={v:<8} -> {pred} @ {prob:.3f}")
    else:
        print("\n  (no crossings)")
    return crossings


if __name__ == "__main__":
    print("E004 — noise applied to REAL base images")
    print("Legend: dog_margin, then best animal class + its probability. '*' = predicted cat/dog.")
    print("Refs: best base image ocean.jpg=-11.933 | prior~-2.5 | pure uniform noise=+4.482 (dog .68)")

    run_sweep(
        "A. Additive gaussian noise (sigma)",
        lambda im, v: additive_gaussian(im, v),
        [5, 10, 20, 40, 60, 80, 120, 160, 200],
        "e004_gaussian",
    )

    run_sweep(
        "B. Alpha blend toward uniform noise (alpha: 0=original, 1=pure noise)",
        lambda im, v: blend_uniform_noise(im, v),
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "e004_blend",
    )

    run_sweep(
        "C. Salt-and-pepper noise (density)",
        lambda im, v: salt_pepper(im, v),
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
        "e004_saltpepper",
    )
