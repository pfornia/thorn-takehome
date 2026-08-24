# Thorn — Sr ML Eng Take-Home Assignment

## Quick start
```
cd projects/thorn-takehome-08-20/homework
```
Env not yet set up. Stack is confirmed (see below): PyTorch/torchvision, uv-managed
(`pyproject.toml` + `uv.lock` present) — the same `uv venv` approach used for the AE.Studio
interview folder should work here too.

## Materials received 2026-08-20 — all present
Extracted from the submitted zip into `homework/`:
- `HOMEWORK.md` — the assignment, matches what Paul pasted (task section is slightly more precise:
  "transforming the images in the `images` directory so that they produce false positive
  predictions for the `dog` or `cat` classes" — both classes are in scope per this copy, not dog
  only; the submission bullet list still says "false-positive dog prediction" specifically, worth
  producing dog false positives as the primary target but this confirms cat is a valid secondary
  target too if that's easier for a given transformation).
- `model.py` — `MobileNetSmall`: torchvision `mobilenet_v3_small` backbone (frozen, pretrained
  ImageNet weights by default) with the final classifier layer replaced for 3 labels
  `["cat", "dog", "other"]`. Standard ImageNet preprocessing (resize 224x224, normalize). Has
  `preprocess_image`, `forward`, `save_model`, `load_model` methods.
- `model.pt` — the trained checkpoint to load via `load_model`.
- `images/` — 5 benign base images, all correctly classified "other": `man.jpeg`, `woman.jpg`,
  `rav4.jpg`, `forest.jpg`, `ocean.jpg`.
- `tests/test_false_positives.py` — provided tests to evaluate modified images.
- `pyproject.toml` + `uv.lock` — uv-managed Python env, not yet created locally.

## Gitignore convention for this project
`model.pt` and `homework/images/` (the provided base images) are gitignored — large, provided by
Thorn, not Paul's own work product, and re-extractable from the original zip if ever needed. Any
generated/intermediate images from an automated transformation-search pipeline should go in
`homework/outputs/` (also gitignored) so bulk experimentation doesn't bloat git history. **The
final 2-3 selected deliverable false-positive images should NOT live in `images/` or `outputs/`**
— put them somewhere else (e.g. `homework/deliverables/`) so they stay tracked as the actual
submission artifacts.

## Context
- Take-home for Thorn's Senior ML Engineer round 2, following a first-round call with Wil Dyer on
  2026-08-19 that went well. Full role context, comp, and process history:
  `projects/job-search/active-applications/thorn-sr-ml-eng/notes.md`.
- **Due ~1 week from receipt: 2026-08-25/26 (Tue/Wed).** Leave real buffer before the deadline,
  don't submit last-minute — see `todo_2026h2.md` reminder under Monday 2026-08-24.
- Full prompt, verbatim: `prompt.md` in this folder.

## Task summary
Given: an image classifier (cat / dog / other), its weights, an eval repo, and a set of benign
base images the model correctly labels "other."

Given: three qualitative user complaints about false positives:
1. Grid/collage images with lots of lines running through them.
2. Images with watermarks and text.
3. Messy, low-quality, hard-to-make-out images.

**Task:** use non-adversarial *transformations* of the base images to reproduce these false
positives. Produce 2-3 modified images that clearly are *not* cats or dogs but that the model
classifies as a **dog** (the submission list is explicit about "dog prediction," even though the
framing earlier in the prompt talks about cat/dog broadly — default to targeting dog, but
double-check against the actual base images/model once in hand in case that changes the read).

**Explicitly forbidden:** adversarial perturbations, modifying the model itself. This has to be
achieved through plausible, realistic-looking image transformations only — the kind of thing that
could genuinely happen to a real user-submitted image, not an attack crafted to fool the model.

**Bonus points:** an automated pipeline that applies transformations and searches for
misclassifications, rather than one-off manual edits.

**Deliverables:**
- [ ] 2-3 images producing a false-positive **dog** prediction
- [ ] The original base image paired with each modified image
- [ ] Predicted class + confidence scores, before and after, for each
- [ ] The code/scripts used to generate and evaluate the modified images
- [ ] Written explanation covering: interpretation of the user feedback, transformations tested,
      what characteristics seem to trigger the false positives, production mitigation approach,
      and how to design a system to catch future false positives

## Project files
- `prompt.md` — the assignment as emailed (verbatim).
- `homework/HOMEWORK.md` — **authoritative** version, bundled in the zip; more specific than the
  email and differs from it in four substantive places. See the discrepancy table in
  `experiment-log.md`.
- `strategy.md` — search design (staged: crop pre-scoring → univariate sweeps → collage →
  random composition search → greedy refinement), objective function, watermark taxonomy,
  dependency constraint.
- `experiment-log.md` — running record of every experiment, including failures. Also holds the
  resolved scope decisions and parked write-up ideas.
- `ai-usage-log.md` — disclosure log for the write-up.

## Dependency constraint
**Do not add dependencies beyond `torch, torchvision, pillow, pytest`** (numpy comes via torch).
Thorn will re-run this code to verify the scores; anything requiring opencv/scikit-image/wand or a
pip-installed watermark package risks not reproducing in their environment. Pillow covers
everything needed.

## ⚠️ Always give Paul the file path of top-performing images
Whenever reporting a winning or notable result, **include the file path** so Paul can open it
himself. Don't make him ask. Applies to every result summary, not just when images are sent via
SendUserFile. Repo-relative is fine for context, but give the full absolute path when he's likely
to open it directly.

Corollary: **encode config and score in the filename** (e.g.
`woman__Arial_s10_o0.7_white_a30_dog1.000.png`) so the outputs folders are browsable without
cross-referencing the experiment log.

## ⚠️ Always save generated test images
Every experiment script must write its generated images to `homework/outputs/<expNNN>_<name>/`
(gitignored). Needed for visual inspection, for reproducing results, and because the submission
requires before/after image pairs. Don't score an image without saving it.

## ⚠️ Keep `experiment-log.md` updated continuously
Every transformation tried gets logged in `experiment-log.md` **as it happens** — including
failures and dead ends, with the full probability vector before/after. Do not batch this up to
write at the end. The write-up has to answer "which transformations you tested" and "what
characteristics appear to trigger the false positives," and the negative results are what make
that argument credible rather than a just-so story. Update the log in the same turn as running an
experiment, not later.

## CRITICAL collaboration constraint — read before helping with this

AI tool use is explicitly welcome for code and technical work, but **not for the write-up**. The
prompt says directly: *"we want to hear your reasoning in your own words"* and requires disclosure
of what tools were used, for what, and how output was verified/modified.

Practical rule for this project:
- Fine for Claude to help: build/debug the transformation pipeline, run evaluations, propose
  transformation ideas to test, discuss and react to draft write-up text, sanity-check reasoning.
- **Not fine for Claude to do: draft the substantive reasoning or conclusions in the written
  explanation.** That has to come from Paul's own understanding of what the pipeline actually
  showed. Claude can ask questions and push back to help him think it through, not supply the
  answer.
- Keep a running, honest log of what AI helped with as we go (see `ai-usage-log.md`), so the
  disclosure section at the end is accurate and doesn't need to be reconstructed from memory.

## Still missing
- [ ] Submission link — prompt references one but it wasn't included in what Paul pasted; get this
      from him before the work is done

## Why this task shape makes sense for Thorn specifically
Worth keeping in mind for the written explanation (Paul's own reasoning, not something to hand
him): this is structurally a false-positive/robustness investigation on an image classifier,
directly analogous to Thorn's actual CSAM-detection problem — a review queue clogged with
non-actionable false positives is a real, costly failure mode for exactly the kind of system Thorn
builds (per the ROOST/Coop and Thorn/Safer research earlier this week, human reviewer capacity is
the bottleneck a lot of the tooling is designed around). This isn't a generic ML exercise, it's
close to the actual day-to-day problem shape of the role.
