# Deliverables — before / after scores

All scores produced by the provided `model.py` + `model.pt`, unmodified. Reproduce with:

```bash
cd homework
uv run python -c "
from autosearch.scoring import Scorer; from PIL import Image; from pathlib import Path
s = Scorer()
for p in sorted(Path('deliverables').glob('*.png')) + sorted(Path('deliverables').glob('*.jpg')):
    r = s.score(Image.open(p)); print(p.name, r.pred, r.probs)
"
```

Or run the provided test suite against them:

```bash
FALSE_POSITIVE_DIR=deliverables/ uv run pytest tests/test_false_positives.py
```

## The three false positives

| user complaint | image | pred | dog | cat | other | logit margin |
|---|---|---|---|---|---|---|
| — | `UF2_watermark_ORIGINAL_woman.jpg` (unmodified) | `other` | 0.0000 | 0.0000 | 1.0000 | −13.68 |
| **UF1** grid/collage | `UF1_collage_MODIFIED_dog0.9992.png` | **`dog`** | **0.9992** | 0.0004 | 0.0004 | **+7.81** |
| **UF2** watermark/text | `UF2_watermark_MODIFIED_dog0.9998.png` | **`dog`** | **0.9998** | 0.0000 | 0.0002 | **+8.34** |
| **UF3** messy/low quality | `UF3_noise_MODIFIED_dog1.000.png` | **`dog`** | **0.9986** | 0.0002 | 0.0012 | **+6.70** |

All three clear the provided suite's hard (99%) threshold.

### Transformations

| | base image(s) | transformation |
|---|---|---|
| UF1 | all five (mixed) | 10×10 collage of random square crops, 8px black gutters, seed 17 |
| UF2 | `woman.jpg` | tiled diagonal watermark "© SAMPLE", Arial 10pt, spacing 20, opacity 0.7, white, 30° |
| UF3 | `woman.jpg` | additive gaussian noise, σ=60 |

UF1 has no single "original" — it is composed from crops of all five provided images. Its paired
before-image is the **no-gutter control** below, which is the more meaningful comparison anyway:
identical tiles, identical arrangement, only the gutters differ.

## Controls — the evidence for *why*

These are not deliverables. They are included because each isolates a variable, and together they
are the argument that the trigger is high-frequency texture rather than anything else.

| control | pred | dog | what it rules out |
|---|---|---|---|
| `UF1_collage_CONTROL_nogutters_dog0.00.png` | `other` | 0.0000 | Same tiles, same arrangement, **no gutters** → no false positive. The gutters are the trigger, not the collage composition. |
| `EVIDENCE_wordmark_control_dog0.00.png` | `other` | 0.0000 | A **large opaque** wordmark, the most visually obvious watermark tested, does nothing. Same ink and colour as the dense version that reaches 0.9998; only spatial frequency differs. Rules out coverage and contrast. |
| `EVIDENCE_saltpepper_control_dog0.00.png` | `other` | 0.0046 | **Salt-and-pepper noise at 50% density** never crosses, while gaussian noise of comparable visual severity crosses on every image. Rules out "disruption" or information loss; the trigger is *smooth, continuous* fine-grained variation. |

## Note on the UF1 seed

The UF1 collage draws tiles at random. Seed 17 was selected from a 30-seed search, so it should be
read as the best of a distribution, not a typical draw:

- **30/30** seeds produce a `dog` prediction
- mean dog probability **0.931**, min 0.515, max 0.999
- **3/30** exceed 0.99

The effect is a property of the transformation; the seed only determines how far past the
threshold it lands.
