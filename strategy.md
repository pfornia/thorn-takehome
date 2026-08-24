# Search Strategy — Thorn Take-Home

Written 2026-08-21, before running the search. Results land in `experiment-log.md`.

## Objective

Maximize the **dog margin**: `logit_dog − logit_other`.

Logits, not probabilities — probabilities saturate (four of five base images print as exactly
`dog=0.000000`, see E000) and destroy the resolution we need to detect incremental progress.
Log the full probability vector and all three logits for every candidate regardless, so no
evaluation is wasted.

Reference targets, from E000 baselines and the thresholds in `tests/test_false_positives.py`:

| target | margin needed (from ocean.jpg, the closest base image) |
|---|---|
| easy tier (10% dog) | ≈ +9.7 |
| medium tier (50% dog) | ≈ +11.9 — the point the model actually *predicts* dog |
| hard tier (99% dog) | ≈ +16.5 |

## Search design: staged hybrid, not one big search

Neither pure grid nor pure random is right on its own.

- **Grid search alone is infeasible.** The full space is effectively infinite (crop regions are
  continuous), and a coarse grid over the collage family alone is ~10⁵ cells before crops.
- **Random search alone doesn't answer the question.** The write-up must explain *what
  characteristics trigger the false positives*. A winning 14-parameter config with arbitrary values
  doesn't explain anything. Single-variable dose-response curves do.

So: grid where interpretability matters, random where dimensionality dominates, greedy to finish.

### Stage 0 — Crop pre-scoring (cheap, high information)
Score a few hundred random crops across the five base images; record the dog-margin distribution
per image and per region.

Purpose: (a) find whether specific *regions* are already dog-leaning, (b) produce "hot tiles" to
seed collages with later, (c) directly quotable evidence for the write-up.

### Stage 1 — Univariate sweeps (grid search)
One parameter at a time, all else fixed, run on **ocean.jpg** and **woman.jpg** (the two closest to
the boundary per E000). ~50 configs per sweep. Output: interpretable dose-response curves.

Degradation (UF3):
- JPEG quality: `[1,2,3,5,8,12,20,30,50,80]`
- Gaussian noise σ: `[0,2,5,10,20,40,80]`
- Blur radius: `[0,0.5,1,2,4,8]`
- Downsample→upsample factor: `[1,2,4,8,16,32]`
- Posterize bits: `[1..8]`
- Contrast / brightness scaling

Filters (ingredient, not a standalone category — see Decision 2 in `experiment-log.md`):
- Sepia strength, saturation, hue shift, vignette strength, colour temperature: `[0…1]`

Watermark/text (UF2) — see expanded section below.

**Explicitly deferred from Stage 1: collage.** See Stage 2A.

### Stage 2A — Collage, its own stage (deferred)
Collage is not a single parameter; it's a compound generator with many interacting dimensions:
which source images, which crop regions, tile order/arrangement, grid dimensions, per-tile
flip/rotate, border/line width, line colour, gutter spacing, per-tile independent transformations.
Sweeping "grid size" while holding the rest fixed would be close to meaningless.

**Key idea (Paul, 2026-08-21): collage is likely the *final* compositional step, not an early
one.** Rather than collaging raw base-image crops, collage together the *already-dog-leaning*
outputs discovered in Stages 0/1 — i.e. use the winners as tiles. This makes collage a
force-multiplier on whatever else works, and matches UF1's description (a grid/collage with lines
running through it) while stacking gains rather than starting from scratch.

Deferred until Stage 0/1 tell us what's actually dog-leaning and worth tiling.

### Stage 2B — Random search over compositions
Sample transformation *pipelines*: choose 1–4 ops, sample each op's parameters from the Stage 1
ranges, apply in a sampled order (order matters — JPEG-then-noise ≠ noise-then-JPEG). Budget a few
thousand candidates per user-complaint category, constrained so each candidate still visually
belongs to its category (a UF1 candidate must actually contain a grid/collage).

### Stage 3 — Greedy refinement
Take top-K from Stage 2, perturb one parameter at a time, keep improvements, iterate. This is
coordinate ascent **over transformation parameters, never over pixels** — the legitimate side of
the adversarial line (see Decision in `experiment-log.md`).

## Watermarking, expanded (UF2)

UF2 is the complaint with the most design surface, and "opaque text dropped on top" is only one
point in it. Real-world watermarks vary along several axes worth testing:

**Style / technique**
- Tiled diagonal repeated text (stock-photo style: Shutterstock/Getty/Alamy)
- Single large centred semi-transparent wordmark
- Corner/edge attribution mark or handle (social-media style)
- Full-frame diagonal band
- Outlined/stroked text (no fill)
- Embossed / bevelled (offset light+dark copies of the same text)
- Drop-shadow text
- Camera timestamp burn-in (digital-clock font, corner, orange/yellow)
- Meme-style top/bottom caption (heavy condensed face, white with black stroke)
- Subtitle burn-in
- Dense fine-print / "PROOF" / "SAMPLE" / "DO NOT COPY" repetition

**Parameters within style**
- Opacity `[0.05…1.0]`, font size, tiling density/spacing, rotation angle
- Colour (white / black / grey / coloured), stroke width, blend mode
- Font family — real variety matters here; macOS ships many faces under
  `/System/Library/Fonts` and `/Library/Fonts`

**Hypothesis worth testing explicitly:** dense, small, high-contrast repeated glyphs should behave
like *texture* to an ImageNet backbone (fine repeated high-frequency edges ≈ fur statistics),
whereas a single large low-frequency wordmark should not. If true, watermark **density and glyph
scale** should dominate opacity in the dose-response curves. That's a crisp, checkable claim for
the write-up.

### Library choice: Pillow + numpy only — deliberate constraint

No new dependencies. `pyproject.toml` pins exactly `torch, torchvision, pillow, pytest` (numpy
arrives via torch). **Thorn will re-run this code to verify the scores**, so anything requiring
`opencv`, `scikit-image`, `wand`/ImageMagick, or a pip-installed watermark package risks not
reproducing in their environment. Pillow's `ImageDraw` / `ImageFont` / `ImageFilter` / `ImageEnhance`
/ `ImageOps` cover every style listed above with full parameter control.

Fonts are the one external resource; prefer fonts present by default on macOS, and fail gracefully
to Pillow's built-in bitmap font so the pipeline still runs on another machine.

## Reproducibility

- Seed every random draw; record the seed with each result.
- Log every candidate to CSV: transformation name, full parameter dict, all three logits, all three
  probabilities, margin, seed.
- Keep failures. Negative results are required content for "which transformations you tested."

## Hard constraint checked outside the search

Every finalist must still **clearly not depict a cat or dog** to a human. The search optimises a
number and cannot enforce this — it's a manual review gate on the 2–3 selected deliverables.
