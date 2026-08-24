# Experiment Log — Thorn Take-Home

Running record of **everything tried**, including things that didn't work. Negative results are
as valuable as positive ones here: the write-up asks explicitly "which transformations you tested"
and "what characteristics appear to trigger the false positives," and that argument is much
stronger with real evidence of what *didn't* move the needle as a contrast.

**Keep this updated continuously as work happens, not reconstructed at the end.**

Format per entry: date, transformation tried, base image(s), full probability vector before/after,
verdict, and any observation worth carrying forward.

---

## Baseline facts (established 2026-08-20, from reading the provided code)

- Model: `MobileNetSmall` — torchvision `mobilenet_v3_small`, ImageNet-pretrained backbone
  (frozen), final classifier layer replaced with a 3-way head: `["cat", "dog", "other"]`.
- Preprocessing: resize to **224x224 (non-aspect-preserving)**, ImageNet normalize
  (mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`).
- Base images (all currently classified "other" with >99% confidence, per the provided test):
  `man.jpeg`, `woman.jpg`, `rav4.jpg`, `forest.jpg`, `ocean.jpg`.

### Scoring is graded — discovered in `tests/test_false_positives.py`
The provided tests check the false-positive images at **three difficulty tiers**, all computed in
logit space against `logsumexp`:
- **Easy:** best cat/dog probability > **10%**
- **Medium:** best cat/dog probability > **50%** (this is the real "the model predicts dog" bar)
- **Hard:** best cat/dog probability > **99%**

Implication: this isn't pass/fail. Aim for the hard tier (99%) if achievable; medium (50%) is the
threshold at which the model actually *predicts* cat/dog rather than merely assigning it some mass.
Tests read the directory from the `FALSE_POSITIVE_DIR` env var.

---

## Ideas parked for the write-up (Paul's to develop, not drafted here)

- **Thresholding as a production mitigation.** Production T&S classifiers generally don't use plain
  argmax; they threshold the class probability at a level tuned to a target precision/recall
  tradeoff. Thorn's own test file implies they think this way — it scores at 10% / 50% / 99%
  probability tiers rather than "is dog the argmax." Raising the flag threshold is one of the
  cheapest available levers against exactly the false-positive complaint users are reporting.
  Relevant to the "how would you mitigate in production" section.
- **The frozen-backbone thesis.** Backbone is ImageNet-pretrained and frozen (`model.py` L44-49);
  only the 3-way head was trained. ImageNet-1k devotes ~120 of its 1000 classes to dog breeds, so
  the feature space is heavily tuned toward fine-grained dog texture (fur, repeated high-frequency
  edges, tan/brown mid-tones). Hypothesis: all three user complaints are surface variations of one
  underlying failure — transformations that push texture statistics toward "dog-ish" while
  destroying the semantic content the head would otherwise rely on. Verify against real data before
  committing to this in the write-up.
- **Adversarial vs. parameter search.** Hill-climbing over *pixels* = adversarial perturbation,
  explicitly forbidden. Hill-climbing over *transformation parameters* (grid size, tile choice,
  text opacity, JPEG quality, noise seed) = legitimate automated search, and is what the "extra
  points for automated pipelines" line is inviting. Worth stating the distinction explicitly in the
  write-up to show the constraint was understood rather than skirted.
- **Why temperature doesn't apply here** (considered and rejected 2026-08-20): the model is
  deterministic, argmax over fixed logits, no sampling step to exploit. Temperature is also
  monotonic, so it cannot reorder classes, and the test recomputes probabilities from raw logits
  anyway, so any client-side temperature wouldn't be evaluated. Randomness has to live in image
  generation, not the readout.

---

## Scope decisions + prompt version discrepancy (resolved 2026-08-21)

**⚠️ There are two versions of the prompt and they are not identical.**
- `prompt.md` — the email text Paul pasted.
- `homework/HOMEWORK.md` — bundled inside the zip with the materials.

Verified by diff. Nothing conflicts; the zip is a strict superset of specificity — every point of
vagueness in the email is *narrowed* in the zip, never contradicted. **Treat `HOMEWORK.md` as
authoritative** (it ships with the actual materials and is more precise). Substantive zip-only
additions:

| | Email | Zip |
|---|---|---|
| Complaints | three unlabeled quotes | labeled **User Feedback 1 / 2 / 3** |
| Task | "reproduce the reported false-positive behavior" | "…**by transforming the images in the `images` directory** so that they produce false positives for the **`dog` or `cat` classes**" |
| Materials | "a repository containing the code" | names `model.pt`, `model.py`, `tests` dir |
| Deliverable | "Two to three images that produce a false-positive dog prediction." | "…**based on each user's feedback**." |

### Decision 1: base images — CLOSED, not ambiguous
Zip task sentence says explicitly "by transforming the images in the `images` directory," and the
deliverables require "the original base image associated with each modified image." The five
provided images are the required source. **Do not introduce outside photographs.**

Sub-question that *is* real, and the line drawn: compositing in **synthetic** elements (text,
watermarks, grid lines, borders, noise, filters) is clearly intended — UF1 names lines and UF2
names watermarks/text, none of which exist in the five photos. Compositing in **other
photographs** is out: it breaks the base-image pairing and risks smuggling in genuinely dog-like
content, colliding with "images that definitely do not depict a cat or dog."

### Decision 2: transformations beyond the three complaints — GENUINELY AMBIGUOUS, sidestepped
"Two to three images that produce a false-positive dog prediction based on each user's feedback"
(zip only) admits two readings: (a) one image per complaint, or (b) 2–3 images total drawing on the
complaints in any combination. "Two to three" doesn't divide cleanly into three complaints, so
neither reading is clearly right. _(Claude initially called this resolved in favor of (a) — that
was overconfident and Paul correctly pushed back.)_

**Resolution: make the ambiguity moot.** Target three images that between them cover all three
complaints — one clearly grid/collage, one clearly watermark/text, one clearly messy/low-quality.
That satisfies both readings simultaneously. If only two clear the threshold convincingly, submit
two strong ones rather than three with a weak entry.

Corollary: transformations not named in any complaint (e.g. Instagram-style filters) are legitimate
as **ingredients within** a complaint category — a heavy filter stacked with noise and JPEG crush is
a plausible route to UF3's "messy / low quality / hard to make out" — but should not be presented as
a standalone fourth category.

### Decision 3: target class
Zip task sentence permits "`dog` or `cat`," but the deliverable bullet specifies **dog**. Baselines
(E000) show dog > cat on all five base images anyway. **Target dog; treat a cat false positive as a
bonus, not a substitute.**

---

## Transformation backlog (to try)

Mapped to the three user complaints where applicable. Add to this list as ideas come up; move
entries into the Experiments section with results once actually tested.

- **Collage / grid tiling** (User Feedback 1) — tile N×M crops into one frame; crops sourced from
  the base images. Exploits the 224×224 non-aspect-preserving resize: each tile ends up tiny, so
  semantic content is destroyed while texture statistics survive.
- **Grid lines / borders** (UF1) — thin lines between tiles, periodic high-frequency edges.
- **Watermark / text overlay** (UF2) — tiled semi-transparent text, varying opacity, font size,
  density, rotation.
- **Quality degradation** (UF3) — aggressive JPEG (q5–15, 8×8 blocking artifacts),
  downsample→upsample, gaussian/salt-pepper noise, posterize, contrast reduction, blur.
- **Instagram-style filters / photo enhancement** (Paul's idea, 2026-08-21) — sepia, vintage/faded,
  heavy saturation, color-channel shifts, vignette, warm/cool grading, "enhance" style contrast
  curves. Rationale: these are plausible real edits a non-adversarial user would apply to a photo,
  which fits the assignment's "plausible examples of the issue" requirement well. Also a good
  probe of the frozen-backbone thesis specifically — if the model is keying on *texture*, color
  grading alone should move the needle much less than collage/degradation does; if warm/sepia tones
  (tan/brown, dog-fur-colored) move it substantially, that points at a color-statistics component
  rather than pure texture. Either result is informative for the write-up.

---

## Experiments

### E000 — Environment setup + baseline scoring (2026-08-21)

Env created with `uv sync` in `homework/` (torch 2.13.0, torchvision 0.28.0, pillow 12.3.0,
pytest 9.1.1). Note `uv run` emits a harmless `VIRTUAL_ENV` mismatch warning because Paul's shell
has anaconda active; ignore it, `.venv` is what's used.

Wrote `homework/score.py` — loads the checkpoint once, returns the full softmax vector, raw logits,
argmax, and the best non-"other" probability (the quantity the provided tests actually grade).

**Baseline probabilities (unmodified base images):**

| image | cat | dog | other | pred |
|---|---|---|---|---|
| forest.jpg | 0.000000 | 0.000000 | 1.000000 | other |
| man.jpeg | 0.000000 | 0.000000 | 1.000000 | other |
| ocean.jpg | 0.000005 | 0.000007 | 0.999988 | other |
| rav4.jpg | 0.000000 | 0.000000 | 1.000000 | other |
| woman.jpg | 0.000000 | 0.000001 | 0.999999 | other |

Probabilities saturate, so the useful view is **logit margins**:

| image | logit_cat | logit_dog | logit_other | margin (other − best animal) |
|---|---|---|---|---|
| **ocean.jpg** | -3.447 | -3.238 | 8.695 | **11.933** ← closest to boundary |
| **woman.jpg** | -7.952 | -2.643 | 11.036 | **13.679** ← 2nd closest |
| man.jpeg | -11.110 | -5.161 | 17.872 | 23.032 |
| rav4.jpg | -10.063 | -6.551 | 16.774 | 23.325 |
| forest.jpg | -9.272 | -7.434 | 17.207 | 24.641 |

**Findings:**

1. **Confirms part of the frozen-backbone thesis:** `dog` logit > `cat` logit on *all five* base
   images, without exception. Dog is consistently the nearer animal class. Consistent with ImageNet
   dog-breed overrepresentation, and convenient since the deliverable asks for dog false positives.

2. **❌ HYPOTHESIS KILLED — "forest.jpg will be closest to dog because foliage has fur-like texture
   statistics."** Wrong, and not marginally: forest.jpg is the *farthest* of all five (24.6 logits).
   Predicted it would be the easiest starting point; it looks like the hardest. Don't build the
   strategy on intuitions about which content "looks fur-like" to a human eye — measure instead.

3. **Two-tier starting field.** ocean.jpg (11.9) and woman.jpg (13.7) sit roughly 10 logits closer
   to the boundary than the other three (23–25). Start transformation search on ocean and woman.

4. **The gap to close is large.** Even the best starting point needs ~12 logits of movement to reach
   a coin-flip, and the tests' medium tier (50%) is exactly that crossover. The 99% "hard" tier
   needs ~+16.5. Small perturbations will not be enough — this needs transformations that
   substantially replace image content/texture, not subtle degradations.

**Carry forward:** start on ocean.jpg and woman.jpg; target the `dog` class; expect to need
aggressive transformations.

_Note on sign convention: E000 reported `other − best_animal` (positive, lower is better). From E001
onward the log uses **`dog_margin = logit_dog − logit_other`** (negative, **higher/less negative is
better**, 0 ≈ coin flip). ocean.jpg baseline is `dog_margin = −11.933`._

---

### E001 — Does rendered *text content* affect the prediction? (2026-08-21)

Paul's idea: render plain black words on white and score them. If "Dog" beats control words, the
model has OCR-like semantic sensitivity and watermark *wording* matters. If not, any watermark
effect is texture/layout-driven.

Setup: `homework/exp_text_only.py`. Arial 160pt, black, centred, 512×512 white canvas. Includes a
**blank white control** to separate "effect of the word" from "effect of a mostly-white canvas."

| rendered | dog_margin | dog_prob | cat_prob |
|---|---|---|---|
| **(blank white)** | **−2.950** | **0.045569** | **0.083482** |
| `\|\|\|\|\|\|\|` (vertical strokes) | −8.820 | 0.000148 | 0.000260 |
| "DOG" | −13.511 | 0.000001 | — |
| "Ocean" | −13.707 | 0.000001 | — |
| "Dogs" | −14.061 | 0.000001 | — |
| "Xqzptl" (gibberish) | −14.305 | 0.000001 | — |
| "Tree" | −14.775 | — | — |
| "dog" | −14.899 | — | — |
| "Other" | −15.231 | — | — |
| "Labrador" | −15.917 | — | — |
| "Car" | −16.296 | — | — |
| "Dog" | −16.318 | — | — |
| "Puppy" | −16.801 | — | — |
| "Cat" | −18.027 | — | — |

**Finding 1 — ❌ NO semantic/OCR sensitivity. Idea killed.**
"Dog" (−16.318) ranks 12th of 14, *worse* than "Ocean", "Tree", and the gibberish string "Xqzptl".
"Cat" is dead last. "Puppy" and "Labrador" are near the bottom. The word content is irrelevant —
spread across all real words is ~4.5 logits and is uncorrelated with meaning. **Do not bother
optimising watermark wording.** Watermark effects, if any, will come from density, scale, contrast
and layout — i.e. texture — not from what the text says. Consistent with the frozen-backbone
thesis: an ImageNet feature extractor has no reading capability.

**Finding 2 — 🔥 BLANK WHITE IS THE STRONGEST RESULT SO FAR, by a wide margin.**
A featureless white canvas scores `dog_margin = −2.950` (dog_prob 4.6%, cat_prob 8.3%) — roughly
**9 logits closer to a dog prediction than ocean.jpg** (−11.933), the best of the five base photos,
and ~11–15 logits better than any text image. cat_prob 8.3% is already within striking distance of
the tests' 10% "easy" tier, from an image containing *nothing at all*.

Implication, and it inverts the working assumption: the model's "other" confidence appears to
depend on the presence of real photographic content. **Destroying content may matter more than
adding dog-like texture.** Adding text to the white canvas made it *worse* (−2.95 → ~−15), i.e.
marks on the canvas pushed it back toward "other".

**Finding 3 — texture beats semantics, weakly.** `|||||||` (−8.820) outscored every real word by
~5 logits, supporting the texture-over-semantics reading, though it's far behind blank white.

**⚠️ Tension to watch:** on blank white, **cat (0.0835) > dog (0.0456)** — the reverse of E000,
where dog > cat on all five photos. Content-destruction may steer toward *cat*, while the
deliverable asks for *dog*. The zip permits "dog or cat" in the task sentence but the deliverable
bullet says dog. Worth tracking which class each transformation family favours.

**Carry forward:**
- Drop watermark-wording optimisation entirely.
- **New priority: content-destruction transformations** — heavy blur to near-uniform, extreme
  contrast/saturation reduction, washout, over-exposure, downsample-to-near-flat. These plausibly
  satisfy UF3 ("messy looking, hard to make out, low quality") *and* now have the strongest
  evidence behind them.
- Re-frame the Stage 1 degradation sweeps around "how far toward featureless does this push the
  image", not just "how much artifact does it add".

---

### E002 — Does the model respond to SHAPE/SILHOUETTE at all? (2026-08-21)

Paul's idea: probe with images that *deliberately depict* a dog — ASCII art, and a photo-mosaic
arranged into a dog silhouette. **Diagnostic only, not deliverable candidates** (the assignment
requires images that "definitely do not depict a cat or dog"; these fail that clause by design).
Purpose: establish an upper bound. If an explicit dog shape doesn't move the model, shape is
irrelevant and the entire search should be texture/content-driven.

Setup: `homework/exp_shape_probe.py`. ASCII rendered in Andale Mono, black on white, 512².
Mosaic = tiles cropped from the five base images, brightness-modulated by a dog-silhouette mask
(inside tiles kept at 1.0×, outside darkened to 0.25×). Flat-mask control uses identical machinery
with a uniform mask, isolating the effect of the shape.

**A. ASCII art**

| rendered | dog_margin |
|---|---|
| noise_ascii (`x#@%&*`) | −15.486 |
| tree_ascii | −16.249 |
| cat_ascii | −17.472 |
| dog_ascii_big | −20.481 |
| **dog_ascii** | **−21.131** |
| house_ascii | −22.028 |

**B. Photo-mosaic** (baseline refs: blank_white −2.950, best base image ocean.jpg −11.933)

| image | dog_margin |
|---|---|
| **mosaic FLAT (control), grid=32** | **−7.168** |
| mosaic dog-shaped, grid=8 | −11.069 |
| mosaic dog-shaped, grid=64 | −11.571 |
| mosaic FLAT (control), grid=16 | −14.411 |
| mosaic dog-shaped, grid=32 | −19.469 |
| mosaic dog-shaped, grid=16 | −19.514 |

**Finding 1 — ❌❌ SHAPE IS IRRELEVANT. Hypothesis dead, twice over.**
ASCII art of a dog (−21.131) scored *worse* than ASCII art of a tree (−16.249) and worse than
meaningless punctuation noise (−15.486). Drawing an actual dog is among the least dog-like things
tested. Combined with E001 (word content irrelevant), the model demonstrably has **no semantic,
symbolic, or shape-level sensitivity** — entirely consistent with a frozen ImageNet backbone
reading local texture statistics.

**Finding 2 — 🔥 Shape-matched comparison: making the mosaic look like a dog made it dramatically
WORSE.** At matched grid size:
- grid=16: flat −14.411 vs dog-shaped −19.514 → **flat better by 5.1 logits**
- grid=32: flat −7.168 vs dog-shaped −19.469 → **flat better by 12.3 logits**

⚠️ **Confound, stated honestly:** the dog-shaped variant darkens 75% of tiles to 0.25×, so it is
also much darker and higher-contrast overall. Shape and brightness are not cleanly separated in
this run. The finding "shape doesn't help" is safe (it's corroborated independently by the ASCII
result); the finding "shape actively hurts by 12 logits" is **not** safe to attribute to shape —
darkening is the more likely driver. Would need a brightness-matched mask to separate. Do not put
the 12-logit claim in the write-up without rerunning that control.

**Finding 3 — 🔥 Uniform collage is the second-strongest result overall, and it's directly UF1.**
`mosaic_FLAT grid=32` at **−7.168** beats every base image by ~4.8 logits and is behind only blank
white (−2.950). This is a plain grid/collage of random crops with no attempt to make it resemble
anything — exactly what User Feedback 1 describes ("sort of like a grid or collage").

**Finding 4 — grid density is a strong, non-monotonic-looking parameter.** Flat mosaic: grid=16
gives −14.411, grid=32 gives −7.168 — a 7.2-logit swing from tile count alone. Dog-shaped mosaic
peaked at the extremes (8 and 64) rather than the middle, though that series is confounded per
Finding 2. **Grid density deserves a proper sweep on the flat/unshaped mosaic.**

**Carry forward:**
- Stop testing anything semantic or shape-based; it is conclusively a dead end (E001 + E002).
- **Two live leads now, both content-destroying rather than content-adding:** (a) featureless/
  near-uniform images (blank white, −2.950), (b) fine-grained uniform collage (−7.168).
  These are plausibly the *same underlying effect* — both destroy coherent photographic structure.
- Next: sweep flat-mosaic grid density properly (8/16/24/32/48/64/96/128) with brightness held
  constant, and sweep the content-destruction transformations (blur/washout/contrast collapse).
- Re-run the shape-vs-brightness control if the 12-logit claim is ever needed.

---

### E003 — 🎯 BREAKTHROUGH: is blank-white just the model's prior? (2026-08-21)

Paul's hypothesis: blank white isn't dog-like, it's the model's **no-evidence fallback** ≈ training
prior. If so it's an *asymptote*, not a building block — destroying information can only walk you
toward the prior (~4.6% dog), never to the 50% needed. Test: score many different information-free
images. Tight cluster ⇒ prior. Wide spread ⇒ still feature-driven.

Setup: `homework/exp_prior_probe.py`. Solid colours, structureless noise, and real photos blurred
to featurelessness.

**Result 1 — ✅ PAUL'S HYPOTHESIS CONFIRMED for *smooth* featureless images.**
Everything smooth clusters in a narrow band regardless of colour or origin:

| image | dog_margin |
|---|---|
| solid_red | −1.700 |
| solid_brown (fur-coloured) | −2.256 |
| solid_green | −2.407 |
| solid_gray50 | −2.574 |
| solid_blue | −2.750 |
| solid_gray75 | −2.774 |
| solid_tan (fur-coloured) | −2.819 |
| solid_white | −2.950 |
| solid_black | −3.315 |
| blur32/96/200 of ocean, woman, forest | −1.744 … −3.042 |

Range ≈ **1.6 logits** across white, black, every grey level, every hue, and three different photos
blurred into mush. That is a prior/fallback, exactly as predicted. **Blank white is not special and
not a stepping stone.**

Corollary: **colour is not the driver either.** `solid_tan` (−2.819) and `solid_brown` (−2.256),
chosen as dog-fur colours, sit right in the pack with blue and green. Kills any colour-statistics
sub-hypothesis from the filters backlog.

**❌ Correction to E001's carry-forward.** E001 concluded "destroying content may matter more than
adding dog-like texture." **That was wrong.** Blur destroys content thoroughly and only ever reaches
the prior — every blur variant tested landed between −1.7 and −3.0, no better than a solid colour.
Content destruction alone asymptotes at the prior and cannot cross.

**Result 2 — 🔥🔥 FIRST SUCCESSFUL FALSE POSITIVE. High-frequency noise crosses, decisively.**

| image | dog | cat | other | dog_margin | prediction |
|---|---|---|---|---|---|
| **uniform_noise_full** (random RGB per pixel) | **0.6824** | 0.3099 | 0.0077 | **+4.482** | **`dog`** ✅ |
| gaussian_noise σ=40 (about mid-grey) | 0.3619 | 0.5595 | 0.0787 | +1.526 | `cat` |
| gaussian_noise σ=10 | 0.0715 | 0.1207 | 0.8078 | −2.424 | other (prior) |

Uniform random noise is classified **dog at 68.2% confidence** — clearing the tests' *medium* (50%)
tier outright, from a starting point where the best base photo sat 12 logits away. σ=40 noise
crosses to `cat` at 56%.

**Result 3 — the mechanism is HIGH-FREQUENCY TEXTURE, and the dose-response is monotonic.**
σ=10 → −2.424 (prior) → σ=40 → +1.526 (cat) → full uniform → +4.482 (dog). More high-frequency
energy ⇒ more animal. This is the **frozen-backbone thesis confirmed with a clean mechanism**: an
ImageNet backbone (~120 dog-breed classes) reads dense fine-grained high-frequency detail as fur.
Not colour, not shape, not semantics, not the mere absence of content — *texture*.

The three probes together are a tidy elimination argument for the write-up:
- E001: word content irrelevant ⇒ not semantic
- E002: dog silhouette irrelevant/harmful ⇒ not shape
- E003 solids: fur colours no better than blue ⇒ not colour; blur ⇒ not mere content-destruction
- E003 noise: crosses to `dog` ⇒ **high-frequency texture**

**⚠️ Caveat before celebrating:** `uniform_noise_full` is *not a transformation of a base image* —
it's synthetic noise from scratch, so it is **not a valid deliverable** (`HOMEWORK.md`: "by
transforming the images in the `images` directory"). It proves the mechanism and gives a target to
aim at; the real task is now to reach the same region *starting from* the five base photos, with a
result that plausibly matches a user complaint.

**Carry forward — this reshapes the whole search:**
- Target **high-frequency noise/texture added to base images**, not content destruction.
- Sweep noise strength on each base image (σ sweep, uniform vs gaussian vs salt-pepper), and check
  how much original photo can remain while still crossing 50%.
- This maps naturally onto **UF3** ("messy looking, hard to make out, low quality") — heavy sensor
  noise/grain is exactly that, and is a completely plausible real-world image defect.
- Likely also explains the E002 collage result: fine grids (grid=32 ≫ grid=16) inject more edge
  density = more high-frequency energy. Re-read the collage lead through the texture lens.
- Open question worth checking: does noise steer toward `dog` or `cat`? Uniform→dog, σ40→cat.
  Understanding that split matters since the deliverable wants dog.

---

### E004 — ✅✅ NOISE SWEEP ON REAL BASE IMAGES — deliverable-quality results (2026-08-21)

Setup: `homework/exp_noise_sweep.py`. Three parameterisations applied to all five base photos,
resized to 512². Images saved under `outputs/e004_*`.

#### A. Additive gaussian noise (σ) — **all five base images cross to `dog`**

| σ | ocean | woman | forest | man | rav4 |
|---|---|---|---|---|---|
| 20 | −5.78 | −4.59 | −22.96 | −17.80 | −18.80 |
| 40 | −0.64 | **+2.27 dog .90** ✅ | −19.32 | −10.97 | −14.14 |
| 60 | **+1.75 dog .53** ✅ | **+6.70 dog 1.00** 🏆 | −14.50 | −4.81 | −10.72 |
| 80 | +2.84 dog .67 | **+8.66 dog 1.00** 🏆 | −9.10 | **+0.47 dog .60** ✅ | −6.54 |
| 120 | +2.89 dog .58 | **+9.34 dog 1.00** 🏆 | **+1.21 dog .75** ✅ | +4.63 dog .96 | **+0.34 dog .50** ✅ |
| 160 | +3.02 dog .54 | +9.33 dog .99 | +6.13 dog .98 | +6.92 dog .98 | +4.19 dog .95 |
| 200 | +3.34 dog .58 | +7.71 dog .91 | +7.85 dog .99 | +7.32 dog .96 | +7.25 dog .99 |

First crossing per image: **woman σ=40** (dog .905) → ocean σ=60 → man σ=80 → forest/rav4 σ=120.

#### B. Alpha blend toward uniform noise — also crosses everywhere

| α | ocean | woman | forest | man | rav4 |
|---|---|---|---|---|---|
| 0.4 | +0.25 dog .47 | **+1.40 dog .80** ✅ | −17.88 | −8.12 | −13.83 |
| 0.5 | +1.39 dog .64 | **+6.38 dog 1.00** 🏆 | −11.66 | −3.31 | −6.95 |
| 0.6 | +1.72 dog .64 | **+9.15 dog 1.00** 🏆 | −4.46 | **+3.35 dog .96** ✅ | −0.58 |
| 0.7 | +2.33 dog .67 | +7.48 dog .99 | **+2.57 dog .93** ✅ | +4.07 dog .88 | **+3.77 dog .95** ✅ |

#### C. Salt-and-pepper noise — ❌ **NEVER crosses, at any density**

Best result across the entire sweep was forest @ density 0.7: −0.26 (dog 0.43) — close but no
crossing. At density 0.9 (90% of pixels replaced with pure black/white!) everything sits at ≈−3,
i.e. back at the prior.

**Finding 1 — 🏆 The hard (99%) tier is cleared.** `woman.jpg` + additive gaussian σ=60–120 gives
dog probability ≈1.00 (margin +6.7 to +9.3). σ=120 is the peak at **+9.34**. Multiple deliverable
candidates now exist.

**Finding 2 — a genuinely plausible candidate, not a degenerate one.** `woman.jpg` at **σ=40**
scores **dog @ 90.5%** while the photograph is still clearly recognisable as a woman — heavy sensor
grain, not destroyed content. This is close to ideal for the assignment: it obviously does not
depict a dog, it is a completely plausible real-world low-quality image (UF3: "messy looking images
that are hard to make out and sort of low quality"), and the model is 90% confident it's a dog.

**Finding 3 — ❌ SALT-AND-PEPPER FAILS COMPLETELY, and this is the most informative result here.**
Salt-and-pepper injects *maximum* per-pixel disruption (pure 0 / pure 255) yet never crosses at any
density, while gaussian noise of comparable visual severity crosses everywhere. So it is **not
"disruption" or "information destruction" that triggers the false positive** — it is specifically
*smooth, continuous-valued, fine-grained variation*. Binary extreme pixels don't look like fur to
the backbone; continuous mid-tone granularity does. This sharpens the texture thesis considerably
and is strong material for the write-up's "what characteristics trigger the false positives."

**Finding 4 — base images differ enormously in susceptibility, and it tracks E000.** woman (crosses
σ=40) and ocean (σ=60) are the easy pair; forest and rav4 need σ=120, 3× more noise. Matches the
E000 baseline margins almost exactly — the model's starting confidence predicts how much noise is
needed to break it.

**Finding 5 — noise steers to `dog`, not `cat`.** Nearly every crossing in A and B lands on `dog`.
The E003 σ=40-around-grey→cat result appears to be specific to noise on a flat grey field, not noise
on a photograph. Convenient: the deliverable wants dog.

**Carry forward:**
- **UF3 (messy/low quality) is effectively solved.** Candidate: `woman.jpg` + gaussian σ≈40–60.
  Pick the lowest σ that clears the desired tier so the image stays maximally plausible.
- Still needed: **UF1** (grid/collage) and **UF2** (watermark/text) candidates. Now retest both
  through the texture lens — the E002 collage lead (−7.168 at grid=32) plausibly works by the same
  mechanism (fine grid = more edge density = more high-frequency energy).
- Keep salt-and-pepper as a documented negative result; it's the cleanest evidence that the
  mechanism is texture statistics rather than degradation per se.

---

### E005 — ✅ WATERMARK / TEXT OVERLAY (UF2) — texture hypothesis confirmed (2026-08-21)

Setup: `homework/exp_watermark.py`. Tiled diagonal watermark (stock-photo style), 30°, white,
`"© SAMPLE"`. Wording held fixed — E001 established text content is irrelevant. Four sweeps
designed to test the prediction from `strategy.md`: **if texture is the mechanism, glyph scale and
density should dominate opacity, and a single large wordmark should fail.**

#### 1. Glyph scale (dense tiling, opacity 0.6) — smaller is dramatically better

| font size | ocean | woman | forest | man | rav4 |
|---|---|---|---|---|---|
| 8 | −6.01 | **+1.41 dog .80** ✅ | −4.70 | −5.78 | −10.50 |
| 12 | −2.82 | −1.94 | **+1.72 dog .81** ✅ | −5.82 | −7.54 |
| 16 | −6.59 | −1.38 | **+1.48 dog .60** ✅ | −10.72 | −7.81 |
| 24 | −10.50 | −1.98 | −6.27 | −15.47 | −14.01 |
| 36 | −8.84 | −6.61 | −8.65 | −12.99 | −20.51 |
| 56 | −18.99 | −10.57 | −19.25 | −18.98 | −29.89 |
| 90 | −27.53 | −10.71 | −10.81 | −17.61 | −17.76 |

Monotonic and enormous: ~21 logits of swing on ocean between size 12 and size 90. Large glyphs are
*worse than the untouched baseline* (ocean size90 = −27.53 vs baseline −11.93).

#### 2. Density (font 14, opacity 0.6) — denser is better

spacing 100 → −15…−21 across the board; spacing 10 → forest **+1.72 (dog .82)** ✅. Clear monotonic
trend toward denser tiling.

#### 3. Opacity (font 14, spacing 20) — weak and NON-monotonic

Peaks mid-range and falls off at full opacity: woman goes +0.47 @ 0.55 → −8.77 @ 1.0. On ocean the
entire opacity range spans only ~3.7 logits, versus ~21 for glyph scale.

#### 4. ❌ CONTROL — single large wordmark: fails completely

| variant | best result (any base) |
|---|---|
| wordmark 90px, opacity 0.5 | −14.18 |
| wordmark 90px, opacity 1.0 | −13.50 |
| wordmark 140px, opacity 1.0 | −11.78 |
| wordmark 200px, opacity 1.0 | −12.18 |

Never crosses on any base image at any size or opacity. A big opaque wordmark — the most *visually
obvious* watermark tested — has essentially no effect, and often makes things worse than baseline.

**Finding 1 — ✅ HYPOTHESIS CONFIRMED. Density and glyph scale dominate; opacity is secondary.**
Predicted in `strategy.md` before running. Glyph scale spans ~21 logits, density ~17, opacity ~4
and non-monotonically. **How much of the image the watermark covers matters far less than how fine
its structure is.** This is a strong, counter-intuitive, quotable result: a faint dense watermark
breaks the model while a bold obvious one doesn't.

**Finding 2 — the wordmark control is the cleanest evidence in the whole investigation.** Same ink,
same colour, same opacity, same semantic content — only the *spatial frequency* differs — and the
outcome flips from "no effect" to "confident dog". Pairs perfectly with the salt-and-pepper negative
from E004: both isolate *fine continuous texture* as the trigger, ruling out coverage, contrast,
and content.

**Finding 3 — 🔄 SUSCEPTIBILITY ORDER INVERTS vs. noise.** `forest.jpg` is the **best** base for
watermarks (+1.72, dog .82) despite being the **worst** baseline (E000: −24.6) and the most
noise-resistant (E004: needed σ=120). Plausibly its existing foliage detail combines constructively
with dense small glyphs. Worth stating carefully in the write-up — "which images are vulnerable"
depends on the transformation, not on a single per-image robustness score.

**Finding 4 — UF2 reaches the medium tier, not the hard tier.** Best watermark results are
dog ≈ 0.80–0.82 (margin ≈ +1.7). Clears easy (10%) and medium (50%); does **not** clear hard (99%).
Compare UF3 noise, which reached dog ≈ 1.00. If a hard-tier UF2 candidate is wanted, try stacking a
dense fine watermark *with* mild gaussian noise.

**Best UF2 candidates so far:**
- `forest.jpg` + tiled watermark, font 12, dense spacing, opacity 0.6 → **dog 0.807**
- `forest.jpg` + tiled watermark, font 14, spacing 10, opacity 0.6 → **dog 0.822**
- `woman.jpg` + tiled watermark, font 8, dense, opacity 0.6 → **dog 0.802**

---

### E006 — Spatial frequency vs. content: tiling vs. cropping vs. downscaling (2026-08-21)

Paul's test: classify an N×N collage of an image against its sub-tiles alone. Same content,
different scale — isolates scale from content.

Claude's stated prediction beforehand: after the fixed 224² resize, **tiling multiplies spatial
frequency by N (→ more dog-like)** while **cropping/upscaling divides it (→ less dog-like)**, so the
deltas should have opposite signs.

Setup: `homework/exp_scale_frequency.py`. `tile_N` = same image repeated N×N, no borders.
`crop_1/N` = one centre sub-tile blown back up. `down_Nx` = downscale+upscale, framing preserved.

**Absolute dog_margin** (prior ≈ −2.5):

| condition | ocean | woman | forest | man | rav4 |
|---|---|---|---|---|---|
| original | −11.83 | −13.64 | −24.67 | −22.95 | −23.27 |
| tile_2×2 | −9.95 | −18.44 | −22.51 | −19.88 | −11.55 |
| tile_4×4 | −6.04 | −5.00 | −6.50 | −17.25 | −9.40 |
| **tile_8×8** | **−3.06** | −5.00 | **+0.88** ✅ | −9.03 | **+1.32 dog .63** ✅ |
| crop_1/2 | −9.01 | −13.47 | −14.32 | −17.70 | −19.90 |
| crop_1/4 | −5.53 | −10.81 | −10.17 | −12.11 | −28.10 |
| crop_1/8 | −3.42 | −8.01 | −2.07 | −8.29 | −9.36 |
| down_2× | −10.94 | −13.38 | −25.36 | −22.17 | −22.70 |
| down_4× | −9.90 | −12.01 | −21.84 | −19.69 | −19.05 |
| down_8× | −4.20 | −9.91 | −14.32 | −12.28 | −14.94 |

**Finding 1 — ⚠️ MY PREDICTION WAS HALF WRONG, and the deltas are a trap.**
Measured as *deltas vs. original*, **every** condition came out positive — tiling, cropping, and
downscaling alike (e.g. crop_1/8 on forest: +22.60). Read that way it looks like everything helps
and the frequency story collapses.

It doesn't, because the baselines differ enormously (forest starts at −24.67, ocean at −11.83).
**Read in absolute terms the three families separate cleanly:**
- `tile_8×8` → **+0.88 / +1.32 — crosses into cat/dog territory** on forest and rav4.
- `crop_1/8` → best −2.07, i.e. *at the prior* (−2.5). Approaches, never exceeds.
- `down_8×` → best −4.20, *below the prior*.

**Lesson: judge against the prior (≈ −2.5), not against each image's own baseline.** A large
positive delta that lands at −2 is just "information removed, fell back to prior" — it is not
evidence of dog-like signal. Only crossing *above* the prior demonstrates positive dog evidence.
This corrects the framing and re-confirms E003.

**Finding 2 — ✅ Tiling is genuinely different from information removal.** Cropping and downscaling
also destroy information, yet asymptote at/below the prior exactly as blur did in E003. Tiling
*exceeds* the prior. So tiling adds something the others don't: shrunken high-frequency detail plus
periodic tile-seam edges. Consistent with the texture thesis, and it answers Paul's question — it
isn't only that content is removed, something must **add** fine periodic structure to cross.

**Finding 3 — 🔥 UF1 CANDIDATE FOUND: `rav4.jpg` tiled 8×8 → dog 0.63** (margin +1.32), predicted
`dog`. This is literally "sort of like a grid or collage" per User Feedback 1 — a photo of a car
repeated in a grid. Clears easy + medium tiers.

**Finding 4 — tile density is monotonic and strong.** 2×2 → 4×4 → 8×8 improves consistently
(forest: −22.51 → −6.50 → +0.88; rav4: −11.55 → −9.40 → +1.32). Matches E002's flat-mosaic result
(grid 16 → 32 gave a 7-logit gain) and the E005 watermark density finding. **Three independent
experiments now agree: finer repeated structure ⇒ more dog-like.**

**Finding 5 — susceptibility ordering shifts again.** rav4 and forest, the two *worst* baselines,
are the *best* tiling substrates; woman (best noise substrate) is the worst here (tile_8×8 = −5.00).
Reinforces E005's Finding 3: vulnerability is transformation-specific, not a fixed per-image
property. Worth being precise about this in the write-up rather than claiming "image X is fragile."

**Carry forward:**
- UF1 candidate exists (rav4 8×8, dog 0.63) but is only medium tier. Push further: finer grids
  (12×12, 16×16), add grid lines/borders (UF1 explicitly mentions "lots of lines running through
  them"), and try mixed-source tiles rather than one repeated image.
- Consider stacking mild noise onto the tiled candidate to reach the hard tier, as with UF2.

---

### E007 — 🏆 COLLAGE/GRID (UF1) — the control won: GRID LINES ALONE are the strongest trigger found

Setup: `homework/exp_collage.py`. Four sweeps: (A) finer tiling, (B) grid lines on 12×12 tiling,
(C) mixed-source collage, (D) **control — grid lines drawn over the *unmodified* photo, no tiling
at all**, included purely to isolate the line contribution.

#### D. CONTROL — lines only, image otherwise untouched 🏆

| variant | ocean | woman | forest | man | rav4 |
|---|---|---|---|---|---|
| lines8_w2 | −4.34 | −8.23 | +2.37 dog .89 | −6.40 | −13.83 |
| lines16_w2 | −3.20 | **+7.65 dog 1.00** | +5.49 dog .93 | **+6.24 dog 1.00** | −8.22 |
| lines24_w2 | −2.52 | +5.52 dog .99 | +5.68 dog .97 | +2.50 dog .92 | −5.20 |
| lines32_w2 | −2.44 | +5.35 dog .99 | +5.85 dog .62 | +1.75 dog .84 | −8.06 |
| **lines32_w1** | −1.58 | **+10.30 dog 1.000** 🏆 | +3.62 | +4.55 dog .99 | −5.07 |
| lines48_w1 | +0.10 dog .48 | +1.07 dog .75 | +4.97 dog .87 | +4.33 dog .99 | −4.75 |

#### A. Tile density (no lines) — best: forest tile32 **+5.56 dog .81**, forest tile24 +4.69 dog .61
#### B. Lines on 12×12 tiling — best +2.11, and mostly predicts **cat**, not dog. Worse than either lever alone.
#### C. Mixed-source collage — mix16_w2 → dog .515; mix24_w2 → +4.47 but predicts **cat** (dog .29).

**Finding 1 — 🏆 BEST RESULT OF THE WHOLE INVESTIGATION: `woman.jpg` + a 32×32 grid of 1px white
lines → `dog` at 1.000 (margin +10.30).** This beats the previous best (woman + gaussian σ=60,
+6.70) and does so while leaving the photograph **completely intact and recognisable** — no noise,
no tiling, no degradation. Just thin lines drawn over an ordinary portrait.

**Finding 2 — ✅ PERFECT match to User Feedback 1's actual wording.** UF1 says "…a grid or collage
and they seem to have **lots of lines running through them**." The winning artefact is literally
that: lines running through the image. The complaint named the trigger directly, and the *lines*
mattered far more than the *collage*.

**Finding 3 — the control beat the treatment, for the second time.** As in E002 (flat mosaic beat
the dog-shaped mosaic), the deliberately-minimal control outperformed the elaborate construction.
Combining lines *with* tiling (sweep B, best +2.11) was **worse than either alone** — and mostly
flipped the prediction to `cat`. More transformation ≠ more dog.

**Finding 4 — thinner lines beat thicker; density has an optimum.** On woman, 1px lines at 32×32
(+10.30) beat 2px at 32×32 (+5.35). But density isn't monotonic — 16×16 (+7.65) beat 32×32 (+5.35)
at 2px, and 48×48 at 1px collapsed to +1.07. There's a resonance: an optimum line pitch/width pair,
consistent with matching some preferred spatial frequency of the backbone rather than "more edges
always wins." **This qualifies the earlier "finer is always better" reading from E002/E005/E006.**

**Finding 5 — rav4 is immune to lines** (every variant negative, best −4.75) despite being the best
*tiling* substrate in E006. Third independent confirmation that susceptibility is
transformation-specific, not a per-image property.

**⚠️ Reporting caution:** several starred cells predict **cat**, not dog, while still showing a
positive dog-margin (e.g. sweep B forest: margin +2.11 but dog prob 0.005). Always check the
predicted class, not just the margin, before calling something a dog false positive.

**Deliverable status — all three user complaints now have candidates:**

| complaint | candidate | dog prob | tier |
|---|---|---|---|
| **UF1** grid/lines | `woman.jpg` + 32×32 grid, 1px white lines | **1.000** | **hard (99%)** ✅ |
| **UF2** watermark/text | `forest.jpg` + dense tiled watermark (font 12–14, spacing 10) | 0.807–0.822 | medium ✅ |
| **UF3** messy/low quality | `woman.jpg` + gaussian noise σ=60 | ~1.000 | **hard (99%)** ✅ |

Note UF1 and UF3 both currently use `woman.jpg`. Consider swapping UF1 to `man.jpeg` +
lines16_w2 (dog 0.998, also hard tier) so the three deliverables use three different base images —
more convincing as evidence of a general failure mode than three variants of one photo.

---

### E008 — ✅ TRUE mixed-source COLLAGE, expanded sweep (UF1, second pass) (2026-08-24)

**Motivation (Paul, 2026-08-24):** the E007 winner is grid *lines drawn on a single photo*, which
reads as "grid" but not really as "collage." UF1 says "sort of like a **grid or collage**," so a
genuine multi-image collage is the better match to the complaint. E007's sweep C had badly
under-explored this: 7 configs, one seed, and line widths of only 0 or 2 — **never 1px**, despite
1px being the clear winner in the lines-only sweep, and never finer than 24×24.

Setup: `homework/exp_collage2.py`. Each tile is a random square crop drawn from **all five** base
images, arranged on an N×N grid, with optional grid lines. Swept grid × line width × line colour ×
seed.

**White lines** (dog probability; `pred` shown where it isn't `other`):

| grid | w=0 | w=1 | w=2 | w=3 |
|---|---|---|---|---|
| 12 | .00 | .01 | .26 | **.79 dog** |
| 16 | .00 | .04 | **.55 dog** | **.66 dog** |
| 20 | .00 | .00 | .45 dog | **.78 dog** |
| 24 | .00 | **.58 dog** | **.82 dog** | **.85 dog** |
| **32** | .02 | **🏆 .97 dog** | **.89 dog** | **.96 dog** |
| 40 | .11 | .05 cat | .19 cat | .21 cat |
| 48 | .12 | .05 cat | .15 cat | **.86 dog** |

Black lines peaked lower (best: g32_w2 → dog .76; g24_w3 → dog .90).

**Finding 1 — 🏆 `g32_w1_white` → dog 0.968 (margin +7.39).** A 32×32 collage of random crops from
all five base images with 1px white grid lines. Nearly doubles E007's best true-collage result
(0.515) and clears the medium tier decisively; just short of the 99% hard tier.

**Finding 2 — the lines are REQUIRED, not decorative.** The `w=0` column never crosses at *any*
grid density (best: 0.12). Same collage, same tiles, no lines ⇒ no false positive. Adding 1px lines
at grid 32 takes it from 0.02 to 0.97. **This is a clean, isolated demonstration that the periodic
line structure itself is the trigger**, not the collage composition — and it explains why UF1's
users described lines rather than describing a collage.

**Finding 3 — 32×32 is the resonant density, and it replicates the lines-only result exactly.**
E007's lines-only sweep peaked at 32×32/1px on `woman.jpg`; this independent sweep on a completely
different image type (multi-source collage) peaks at the *same* grid pitch and line width. Two
independent experiments converging on 32×32/1px is strong evidence of a genuine preferred spatial
frequency in the backbone rather than a per-image fluke.

**Finding 4 — 40×40 flips to `cat`.** Beyond the resonance the prediction switches class rather than
simply weakening (grid 40: all widths predict cat). Reinforces E007's "finer is not monotonically
better" correction.

**Finding 5 — robust across seeds, not a lucky arrangement.** Re-running `g32_w1_white` with 8 fresh
random seeds: **7 of 8 predict `dog`**, dog probability 0.709–0.938 (one seed landed on cat at
0.336). The effect is a property of the transformation, not of one fortunate tile layout. Worth
citing in the write-up — it's the difference between "I found an image that breaks it" and "I found
a transformation that breaks it."

**Updated UF1 candidate options:**
| option | image | dog | tier | matches "collage"? |
|---|---|---|---|---|
| A | `man.jpeg` + 16×16 grid, 2px lines | 1.000 | hard | grid only |
| B | `woman.jpg` + 32×32 grid, 1px lines | 1.000 | hard | grid only |
| **C** | **mixed 32×32 collage + 1px lines** | **0.968** | medium | ✅ genuine collage + lines |

Option C trades ~3 points of confidence for a much better match to the actual user complaint.

---

### E009 — 🏆 COARSE collage (UF1, final) — realistic collage AND the most robust result yet (2026-08-24)

**Motivation (Paul, 2026-08-24):** a 32×32 grid is 1,024 tiles — that doesn't look like a collage a
real user would produce. Real collages have relatively few, visibly distinct photos. Find the best
result at grid ≤ 16.

Design note: E008 capped line width at 3px, which is likely too thin at coarse densities. At 32×32 a
1px line is ~6% of a tile's width; at 10×10 it's ~2%. This sweep scales line widths up accordingly
(2–16px) across grids 3–16, both line colours.

**Best coarse results (grid ≤ 16), dog probability:**

| config | dog | margin |
|---|---|---|
| 🏆 **g10_w8_black** | **0.984** | +4.93 |
| g10_w12_black | 0.982 | +4.37 |
| g12_w6_black | 0.974 | +3.89 |
| g12_w8_black | 0.969 | +4.17 |
| g10_w16_black | 0.965 | +3.81 |
| g10_w6_black | 0.963 | +3.49 |
| g12_w4_white | 0.888 | +2.34 |
| g16_w6_black | 0.854 | +3.15 |

**Finding 1 — 🏆 `g10_w8_black` → dog 0.984.** A 10×10 collage (100 tiles) with 8px black gutters.
Visually a plausible contact-sheet / photo-collage layout, and it scores *higher* than the 32×32
fine collage (0.968).

**Finding 2 — 🔄 LINE COLOUR INVERTS WITH GRID DENSITY.** At fine grids white lines won decisively
(E008 g32: white 0.97 vs black 0.76). At coarse grids **black wins decisively** (g10: black 0.98 vs
white 0.37 at w=8; g12: black 0.97 vs white 0.55 at w=8). Not a small effect — it reverses. Worth
flagging in the write-up as evidence that this is about the resulting *contrast/frequency profile*
after downsampling, not about "white lines" as such.

**Finding 3 — line width has its own resonance, interacting with grid.** At g10 black: w=2 → 0.01,
w=4 → 0.82, **w=8 → 0.98**, w=12 → 0.98, w=16 → 0.96. At g16 black the peak shifts thinner (w=4–6 →
0.85, w=12 → 0.10). So the optimum is a *grid × width* pair, consistent with hitting a preferred
spatial frequency rather than maximising edge count.

**Finding 4 — very coarse grids fail entirely.** Grids 3, 4, 5 never cross at any width or colour
(best dog 0.02). Grid 8 also fails (best 0.46). The effect needs enough repetitions to register as
periodic texture. **Usable window is roughly grid 10–32.**

**Finding 5 — ✅ MOST ROBUST RESULT IN THE INVESTIGATION: 10/10 seeds predict `dog`**, range
0.889–0.989 across completely different random tile arrangements. Compare the fine collage's 7/8.
This is a property of the transformation, not of any particular tile layout, which is the strongest
possible framing for the write-up.

**FINAL UF1 SELECTION: `mix_g10_w8_black` (dog 0.984).** Beats every alternative on the combination
of confidence, robustness, and visual plausibility as an actual user-generated collage. Paired
control `mix_g10_w0_black` (same tiles, no gutters) → dog 0.00, predicts `other`.

---

### E010 — Seed and source sensitivity of the winning collage (2026-08-24)

Paul's question: how seed-sensitive is `g10_w8_black`, and does it need tiles from *multiple* base
images or would a single source do? This separates two things the candidate confounds — the
**periodic grid structure** vs. the **diversity of tile content**.

Setup: `homework/exp_collage_seeds.py`. Grid/width/colour held fixed at the winner (10×10, 8px,
black). (A) 30 seeds drawing tiles from all five images. (B) 10 seeds per base image, tiles drawn
from **one** image only.

#### A. Mixed sources — extremely stable

| n | mean | median | min | max | sd | crossings |
|---|---|---|---|---|---|---|
| 30 seeds | **0.931** | 0.961 | 0.515 | 0.999 | 0.095 | **30/30 predict `dog`** |

Every single seed crosses. Worst case still clears the 50% medium tier (0.515). This is a property
of the transformation, full stop.

#### B. Single source — much weaker, and highly source-dependent

| source | mean dog | min | max | sd | crossings |
|---|---|---|---|---|---|
| rav4.jpg | 0.757 | 0.550 | 0.904 | 0.113 | **10/10** |
| forest.jpg | 0.569 | 0.041 | 0.979 | 0.292 | 7/10 |
| woman.jpg | 0.300 | 0.107 | 0.764 | 0.186 | 2/10 |
| man.jpeg | 0.214 | 0.102 | 0.517 | 0.126 | 1/10 |
| ocean.jpg | 0.134 | 0.052 | 0.198 | 0.047 | **0/10** |

**Finding 1 — ✅ The winning candidate is not seed-luck. 30/30.** Strongest robustness claim
available: the effect survives 30 completely different random tile arrangements with a worst case
that still crosses.

**Finding 2 — 🔥 SOURCE DIVERSITY MATTERS, and it is not a small effect.** Mixed sources average
**0.931**; the *best* single source averages 0.757, and three of five single-source variants fail
to cross on most seeds. So the trigger is **not purely structural** — the grid lines are necessary
(E009's w=0 control → dog 0.00) but not sufficient on their own. Heterogeneous tile content
contributes materially.

Mechanistically this fits the texture thesis: adjacent tiles drawn from *different* photographs
differ sharply at every tile boundary, so mixing sources adds a second layer of high-frequency
discontinuity on top of the gutters. Tiles from one photo resemble each other, so the boundaries are
softer.

**Finding 3 — single-source ranking tracks image busy-ness, and inverts the noise ranking again.**
`rav4` (hard edges, panel lines, high detail) is the best single source at 0.757; `ocean` (smooth
gradients, low detail) is the worst and never crosses at 0.134. Note `ocean` was the *easiest* base
for gaussian noise in E004 (crossed at σ=60) and is the *hardest* for collage. **Fourth independent
confirmation that susceptibility is transformation-specific, not a fixed per-image property** —
worth stating carefully in the write-up rather than claiming any image is "fragile."

**Finding 4 — variance is informative too.** `forest` single-source has by far the widest spread
(sd 0.292, range 0.041–0.979), i.e. it sometimes works brilliantly and sometimes not at all
depending on which crops get drawn. Mixed sources have the *lowest* spread (sd 0.095) of any
condition tested. Diversity buys both a higher mean and greater consistency.

**Implication for the deliverable:** keep the mixed-source version. It's stronger, far more
consistent, and the single-source comparison is itself good write-up material.

#### 🏆 Best-seed recovery — UF1 now clears the HARD tier

Paul's point (2026-08-24): given the seed spread, searching seeds and taking the best should reach
the 99% tier while keeping the realistic 10×10 collage. Correct. Re-scored all 30 seeds and saved
the top three (the original run only wrote seeds 0–4 to disk).

| seed | dog | margin |
|---|---|---|
| **17** | **0.9992** | **+7.81** 🏆 |
| 25 | 0.9928 | +5.14 |
| 15 | 0.9902 | +4.67 |
| 6 | 0.9895 | +4.91 |

**3 of 30 seeds clear 0.99** (~10%), while 30/30 clear 0.50. So the hard tier is reachable at grid
10 without any change to the transformation — only to which random arrangement is drawn.

Saved: `outputs/e010_seeds/BEST_g10_w8_black_seed17_dog0.999.png`

**⚠️ Disclosure obligation for the write-up.** Selecting the best of 30 seeds is a legitimate search
over the transformation's parameter space (not over pixels — see the adversarial-vs-parameter-search
decision above), and it is exactly the "automated pipeline that applies transformations and gets
incorrect predictions" the assignment offers extra credit for. **But it must be reported honestly:**
state that N=30 seeds were searched, report the full distribution (mean 0.931, 30/30 crossing 0.50,
3/30 crossing 0.99), and present the selected image as the best of that distribution rather than as
a typical result. Reporting only the 0.9992 without context would misrepresent the effect size.

**UPDATED FINAL UF1 SELECTION:** `BEST_g10_w8_black_seed17_dog0.999.png` — realistic 10×10
mixed-source collage with 8px black gutters, **dog 0.9992, hard tier**, backed by a 30-seed
distribution and a no-gutter control at dog 0.00.

---

### E011 — 🏆 WATERMARK grid search at realistic density (UF2, second pass) (2026-08-24)

**Motivation (Paul, 2026-08-24):** spacing 10–14 is too dense to pass as a real watermark. Freeze
**spacing = 20** and search the levers E005 never touched.

Setup: `homework/exp_watermark2.py`. 720 configs per base image: 12 font families × 5 sizes × 4
opacities × 3 colours, at spacing 20 / 30°. Then a rotation sweep on the winner.

Baselines to beat: woman 0.614, forest 0.602 at spacing ≥20 (0.822 was the *unrealistically* dense
spacing-10 result).

#### A. woman.jpg — **157 of 720 configs cross to `dog`**

| config | dog | margin |
|---|---|---|
| Copperplate s10 o0.85 white | 0.9998 | +8.76 |
| Copperplate s10 o0.7 white | 0.9998 | +8.64 |
| Arial s10 o0.7 white | 0.9998 | +8.34 |
| ArialBold s10 o0.55 white | 0.9995 | +7.64 |
| CourierBold s10 o0.7 white | 0.9989 | +6.78 |

#### B. forest.jpg — 60 of 720 cross; much weaker (best 0.9233, Georgia s18 o0.85 white)

#### C. Rotation sweep on the winner 🏆

| angle | dog | margin |
|---|---|---|
| 15° | 0.7832 | +1.29 |
| 0° | 0.9724 | +3.87 |
| 90° | 0.9625 | +3.24 |
| 45° | 0.9969 | +5.76 |
| 30° | 0.9998 | +8.76 |
| 60° | 1.0000 | +10.65 |
| **75°** | **1.0000** | **+12.90** 🏆 |

**Finding 1 — 🏆 BEST MARGIN IN THE ENTIRE INVESTIGATION: +12.90** (`woman.jpg`, Copperplate,
size 10, opacity 0.85, white, spacing 20, **75°**). Beats the previous best (lines-only, +10.30) and
does it at a *realistic* watermark density. UF2 goes from the weakest deliverable (0.822, medium
tier) to the strongest single result.

**Finding 2 — font SIZE dominates everything else.** Every one of the top 8 configs on woman uses
**size 10**, the smallest tested, across six different font families. Combined with fixed spacing,
smaller glyphs mean finer high-frequency structure — consistent with E005's original scale finding
and with the whole texture thesis.

**Finding 3 — white dominates at this density.** All top configs are white; black and grey are
absent from the leaderboard. Contrast with E009, where black won at coarse collage grids. The colour
preference tracks the resulting frequency/contrast profile, not any intrinsic property of the colour.

**Finding 4 — 🔄 rotation angle matters far more than expected, and non-monotonically.** Same
watermark, same everything, rotated: 15° → 0.78, 75° → 1.0000. That's ~11.6 logits from rotation
alone. Worst is 15°, best is 60–75°, with a local dip at 45° relative to 30°. Likely an interaction
between the text's periodic pitch and the 224² resampling grid — some angles alias into a more
"fur-like" frequency band than others. **Never tested in E005 (fixed at 30°) and it turned out to be
one of the strongest levers.**

**Finding 5 — 🔄 base-image ranking inverted vs. E005.** At dense spacing, forest was the best
watermark substrate; at realistic spacing 20, woman crushes it (0.9998 vs 0.9233) and crosses on
2.6× as many configs. Fifth confirmation that susceptibility is specific to the exact transformation
*parameters*, not just the transformation family.

**UF2 candidate options (all spacing 20, all realistic density):**
| option | config | dog | margin | note |
|---|---|---|---|---|
| **A** | Copperplate s10 o0.85 white **75°** | **1.0000** | **+12.90** | strongest; unusual angle |
| B | Copperplate s10 o0.85 white 45° | 0.9969 | +5.76 | conventional watermark angle |
| C | Arial s10 o0.7 white 30° | 0.9998 | +8.34 | most ordinary-looking config |

Recommend **C** or **A**: C is the most plausible-looking as a stock watermark (Arial, 30°, moderate
opacity) while still hitting 0.9998; A is the headline number.

---

### E012 — ✅ AUTOMATED PIPELINE validated end to end (2026-08-24)

Built `homework/autosearch/` to formalise the manual process. Two stages: coarse grid over
`config.json`, then local refinement around the top-K (numeric deltas + fresh random seeds).
Full design rationale in `autosearch/README.md`.

**Single command, all three complaints, five base images:**

```bash
python -m autosearch --uf all --images images/*.jpg images/*.jpeg --out results/
```

| complaint | target met | best dog | stage 1 evaluated | stage 1 crossings | stage 2 evaluated |
|---|---|---|---|---|---|
| uf1 | ✅ | 0.9991 | 150 | 59 | 193 |
| uf2 | ✅ | **1.0000** | 1620 | 136 | 89 |
| uf3 | ✅ | 0.9998 | 220 | 64 | 101 |

**2,388 candidates scored in 6m07s** on CPU/MPS, all three targets met.

Winning parameter sets found autonomously:
- uf1: `grid=8, line_width=8, black, mixed, seed=283372`
- uf2: `Courier New 10pt, spacing=20, opacity=0.85, white, angle=70°`
- uf3: `gaussian, strength=160, seed=896677`

**Finding 1 — 🔥 The pipeline found a configuration the manual sweep dismissed.** It selected
**grid=8** for UF1 at dog 0.9991. E009's manual sweep tested grid 8 at a *single seed* (42), got a
maximum of 0.46 across all widths and colours, and concluded "grid 8 also fails; usable window is
roughly 10–32." That conclusion was **wrong, and wrong for a methodological reason**: one sample
per cell cannot distinguish "this configuration doesn't work" from "this seed didn't work." The
automated seed search corrected it. Worth stating plainly in the write-up — it's a concrete
argument for automating the search rather than hand-sweeping.

Note grid 8 is also *more* visually plausible as a real collage than grid 10 (64 tiles vs 100).

**Finding 2 — independently converged on the manual UF2 optimum.** The search landed on Courier New
10pt at **70°**; the manual E011 sweep landed on Copperplate 10pt at **75°**. Different font
families, near-identical angle and identical size, both at dog ≈ 1.0000. Two independent searches
agreeing on `size=10` and a ~70–75° rotation is meaningful corroboration that those are the real
levers, not artefacts.

**Finding 3 — performance work was necessary, and the bottleneck was not the model.** The first
full run was killed after >10 minutes stuck on UF2. Cause: the watermark overlay was being
re-rendered (~900 individual glyph draws) for every base image, when the overlay depends only on
its own parameters. Caching the layer by parameter set, plus trimming an over-large UF2 grid
(1,920 configs was not the "quick" first stage the design intends), brought the whole three-complaint
run to about 6 minutes. Scoring was never the constraint; image generation was.

**Finding 4 — verification.** All three manual candidates pass Thorn's provided
`tests/test_false_positives.py` at the **hard (99%) threshold**:
```bash
FALSE_POSITIVE_DIR=deliverables/ pytest tests/test_false_positives.py   # 4 passed
```
Plus 22 tests for the pipeline itself, including one asserting batched scoring agrees with
one-at-a-time scoring, and one asserting transforms are deterministic so any reported result is
reproducible from its recorded parameters and seed.

---

### E013 — Per-image coverage: can EVERY base image be pushed to a dog FP? (2026-08-24)

Paul's question: rather than one winner per complaint, get a false positive for *each* of the five
base images.

Required a pipeline change first. Stage 2 previously refined only around the single globally-best
candidate, so one strong image monopolised refinement and the rest were never tuned. It now refines
around **each image's own top-K**, writes a winner per input, and reports per-image coverage.
(Running the CLI once per image instead would work but is much slower — the watermark layer cache
can't be shared across separate processes, ~7 min/image vs ~6 min total.)

#### UF3 — degradation: **4/5 images ≥ 0.99**, whole run 16 seconds

| image | dog | transformation |
|---|---|---|
| man | **1.0000** | gaussian σ=120 |
| woman | **0.9998** | gaussian σ=160 |
| rav4 | **0.9989** | uniform_blend α=0.8 |
| forest | **0.9980** | uniform_blend α=0.8 |
| ocean | 0.8445 🟡 | gaussian σ=120 |

#### UF2 — watermark: **2/5 ≥ 0.99**, 4/5 cross to `dog`, run 5m48s

| image | dog | transformation |
|---|---|---|
| woman | **1.0000** | Courier New 10pt, spacing 20, op 0.85, white, 70° |
| man | **0.9956** | Copperplate 14pt, spacing 20, op 0.6, white, 60° |
| forest | 0.9877 🟡 | Arial 14pt, spacing 16, op 0.85, white, 0° |
| ocean | 0.9497 🟡 | Arial 20pt, spacing 20, op 0.85, black, 0° |
| rav4 | **0.1164 ❌** | best found; never crosses |

**Finding 1 — the search generalises across images, but not uniformly.** UF3 reaches the hard tier
on 4/5; UF2 on 2/5 with two more in the 0.95–0.99 band. Every image except one can be pushed to a
confident dog prediction by at least one of the two transformations.

**Finding 2 — 🔄 the two holdouts are DIFFERENT images, and it's the expected inversion.**
`ocean` is the UF3 laggard (0.84) but reaches 0.95 on watermarks; `rav4` is fine under noise
(0.9989) but **fails watermarks entirely** (0.1164, never crosses). Sixth independent confirmation
that susceptibility is transformation-specific. Practically: **no single image is robust, and no
single transformation covers every image** — a platform can't defend by hardening against one
artefact type.

**Finding 3 — `ocean` is the most resistant image overall**, consistent with everything since E000
(it needed σ=60 vs woman's σ=40, never crossed as a single-source collage, and now tops out at 0.84
under noise). Its content is smooth gradients with little fine detail, so there is less existing
high-frequency structure for a transformation to amplify. The complement of the texture thesis.

**Finding 4 — stage 2 does most of the work on the hard cases.** UF3 stage 1 best was 0.9995 on the
easiest image; the per-image refinement is what lifted man to 1.0000 and rav4/forest above 0.99.
For UF2, stage 2 evaluated 426 additional configs across the five images and produced the winner
(+14.38 vs stage 1's best).

---

### E014 — 🐛 Targeting `cat`: a real objective-function bug, and cat is genuinely harder (2026-08-24)

Paul flagged that `HOMEWORK.md`'s task sentence permits false positives for "`dog` **or** `cat`",
so the target class was made a first-class parameter (`--target-label`, config `target_label`,
`Search(target_label=...)`) rather than being hardcoded.

#### 🐛 The bug this exposed

First cat run returned nonsense: `woman` reported **margin +7.15 with cat probability 0.0044.**

Cause: the ranking objective was `logit(cat) − logit(other)`, which **ignores the third class**.
On a 3-class model that is gameable — those candidates were confidently **`dog`**, so cat beat
"other" easily while dog took all the probability mass. The search was maximising a quantity that
does not imply the target class wins.

Fix: objective is now **`logit(label) − max(logit of every other class)`** — margin against the
strongest *competitor*, which is positive if and only if the target class is actually predicted.
Added `Score.margin_vs_best_other()` plus a regression test asserting
`margin_vs_best_other(label) > 0 ⟺ pred == label`.

**This bug was invisible while targeting `dog`,** because dog is the model's natural attractor for
these transformations — the two objectives happened to agree. It only surfaced once the target
changed. Worth citing in the write-up: a metric that looks fine on the happy path can be silently
wrong, which is exactly the failure mode the assignment's own false-positive problem illustrates.

#### Results after the fix

**UF3 (degradation), cat:** 4/5 cross to `cat`, **0/5 reach 0.99**

| image | cat | transformation |
|---|---|---|
| rav4 | 0.9771 🟡 | gaussian σ=160 |
| man | 0.9142 🟡 | gaussian σ=180 |
| ocean | 0.8482 🟡 | gaussian σ=120 |
| woman | 0.6106 🟡 | gaussian σ=180 |
| forest | 0.1776 ❌ | gaussian σ=120 |

**UF1 (collage), cat:** best **0.9804** — close, but short of 0.99.

**Finding 1 — 🔥 `cat` is materially harder to induce than `dog`.** Best cat results are ~0.98
against dog's 1.0000, and *nothing* reached the 99% tier for cat, while dog cleared it on all three
complaints and on 4/5 images for UF3. Same transformations, same images, same search budget.

This is direct evidence for the frozen-backbone thesis, and arguably the cleanest available:
**ImageNet-1k contains roughly 120 dog-breed classes but only about 7 cat classes.** The frozen
feature extractor therefore carries far more fine-grained dog-texture machinery than cat machinery,
so generic high-frequency texture has a much shorter path to "dog" than to "cat". The asymmetry was
visible from the very first baseline (E000: dog > cat on all five untouched images) and holds all
the way through to the automated searches.

**Finding 2 — the per-image ordering changes again under a different target.** For cat, `rav4` is
the *best* substrate (0.977) and `forest` the worst (0.178). Under dog+UF3, rav4 and forest were
both ≥ 0.998. Susceptibility depends on the transformation *and* the target class.

**Practical conclusion:** stick with `dog` for the deliverables. It is what the submission bullet
asks for, it clears the hard tier comfortably, and the cat results are better used as *evidence*
for the mechanism than as submission artefacts.
