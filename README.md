# Thorn — Senior ML Engineer take-home

Investigation into why an image classifier (`cat` / `dog` / `other`) produces false positives,
reproducing the behaviour described in three user complaints.

Assignment: [`homework/HOMEWORK.md`](homework/HOMEWORK.md)

## Layout

| path | what it is |
|---|---|
| [`experiment-log.md`](experiment-log.md) | full record of every experiment, **including failures and corrections** |
| [`strategy.md`](strategy.md) | search design, written before running |
| [`homework/exp_*.py`](homework/) | the manual investigation, E000–E011 |
| [`homework/autosearch/`](homework/autosearch/) | automated pipeline that formalises the manual process |
| [`homework/model.py`](homework/model.py) | provided by Thorn, unmodified |
| [`homework/tests/`](homework/tests/) | provided FP tests, plus tests for the search pipeline |
| `ai-usage-log.md` | AI tool disclosure |

Not in git: `model.pt` and `images/` (provided in the assignment zip), `homework/outputs/`
(thousands of generated candidates; regenerate by re-running the scripts).

## Setup

```bash
cd homework
uv sync
```

Restore `model.pt` and `images/` from the assignment zip.

## What was found

The model is a **frozen ImageNet-pretrained MobileNetV3-Small** with only the final 3-way
classifier retrained. Roughly 120 of ImageNet's 1000 classes are dog breeds, so the feature space
is heavily tuned to fine-grained dog texture, and the retrained head can only read out those
features.

The trigger is **high-frequency texture**. Established by elimination, each step ruling out an
alternative:

| ruled out | evidence |
|---|---|
| semantics | the rendered word "Dog" scored *worse* than "Tree" and worse than gibberish; the model has no OCR ability |
| shape | ASCII art of a dog scored worse than ASCII art of a tree; a photo-mosaic arranged into a dog silhouette did worse than the same mosaic unshaped |
| colour | solid fur-toned tan and brown scored no better than solid blue |
| information loss | blur, cropping and downscaling all asymptote at the model's no-evidence prior (≈ −2.5 logits) and never cross |
| **→ texture** | uniform random noise classifies as `dog` at 0.68, monotonic in noise energy |

Two negative controls do most of the work:

- **Salt-and-pepper noise never crosses** at any density, while gaussian noise of comparable
  visual severity crosses on every image. The trigger is *smooth, continuous, fine-grained*
  variation, not disruption.
- **A large opaque wordmark does nothing** while a faint dense watermark reaches 0.99+. Same ink,
  same colour, same opacity, same word: only spatial frequency differs.

Two results also came from controls that were only included to isolate a variable:

- grid *lines alone*, drawn over an otherwise untouched photo, beat every elaborate construction
- an unshaped mosaic beat one deliberately arranged into a dog

## Results

All three complaints reproduce, each above 99% `dog` confidence, from images that plainly contain
no animal:

| complaint | transformation | `dog` |
|---|---|---|
| UF1 "grid or collage … lots of lines running through them" | 10×10 mixed-source collage, 8px black gutters | 0.9992 |
| UF2 "watermarks and text" | tiled watermark, Arial 10pt, spacing 20 | 0.9998 |
| UF3 "messy … hard to make out … low quality" | additive gaussian noise, σ=60 | ~1.000 |

Constraint respected throughout: **no adversarial perturbations, and the model is never modified.**
Every candidate is a plausible image edit; the search only chooses parameters. See
[`homework/autosearch/README.md`](homework/autosearch/README.md) for where that line is drawn.

## Automated pipeline

```bash
cd homework
python -m autosearch --uf all --images images/*.jpg images/*.jpeg --out results/
```

Two stages: a coarse grid over the space declared in `autosearch/config.json`, then local
refinement around the best results (numeric steps plus additional random seeds).

It has already found configurations the manual sweep missed — a grid-8 collage reaches
`dog` 0.9991 with the right seed, where the single-seed manual sweep measured 0.46 and moved on.
