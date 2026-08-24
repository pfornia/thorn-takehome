#!/usr/bin/env python3
"""Print the classifier's probability for each class, for one or more images.

    uv run python classify.py deliverables/false_positives/UF1_collage_MODIFIED_dog0.9992.png
    uv run python classify.py homework/images/*.jpg
    uv run python classify.py --json some_image.png

Uses the same loading and preprocessing path as Thorn's `homework/tests/test_false_positives.py`
(their `model.py`, their checkpoint, their `preprocess_image`), reached through the already-tested
`stage2_autosearch.scoring.Scorer` so there is one implementation of "how an image gets scored"
rather than two that could drift apart.

Probabilities only, by request. The search internals rank on logits (probabilities saturate to
0.000000 across several base images, which destroys the resolution a search needs), but for
eyeballing a single image the probabilities are what you want. `--logits` adds them if needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from stage2_autosearch.scoring import Scorer  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="classify",
        description="Report per-class probabilities for one or more images.",
    )
    ap.add_argument("images", nargs="+", help="image paths (relative or absolute; globs are fine)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--logits", action="store_true", help="also show raw logits")
    ap.add_argument("--device", default=None, help="cpu | mps (default: auto)")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.images]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        for m in missing:
            print(f"error: not a file: {m}", file=sys.stderr)
        return 2

    scorer = Scorer(device=args.device)
    labels = scorer.labels

    from PIL import Image
    results = scorer.score_batch([Image.open(p) for p in paths])

    if args.json:
        print(json.dumps(
            [
                {
                    "path": str(p),
                    "pred": r.pred,
                    "probs": {k: round(v, 6) for k, v in r.probs.items()},
                    **({"logits": {k: round(v, 4) for k, v in r.logits.items()}}
                       if args.logits else {}),
                }
                for p, r in zip(paths, results)
            ],
            indent=2,
        ))
        return 0

    name_w = max(len(p.name) for p in paths)
    name_w = max(name_w, 4)
    header = f"{'file':<{name_w}}  {'pred':<6}" + "".join(f"  {l:>9}" for l in labels)
    if args.logits:
        header += "   |" + "".join(f"  {l + '_logit':>12}" for l in labels)
    print(header)
    print("-" * len(header))
    for p, r in zip(paths, results):
        row = f"{p.name:<{name_w}}  {r.pred:<6}" + "".join(f"  {r.probs[l]:>9.6f}" for l in labels)
        if args.logits:
            row += "   |" + "".join(f"  {r.logits[l]:>12.4f}" for l in labels)
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
