"""CLI entry point.

    python -m autosearch --uf uf1 --images images/*.jpg --out results/
    python -m autosearch --uf all --images images/*.jpg --out results/ --target 0.99
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .scoring import Scorer
from .search import Search


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="autosearch",
        description="Search for non-adversarial transformations that induce false-positive "
                    "cat/dog predictions, guided by the three user complaints.",
    )
    ap.add_argument("--uf", default="all",
                    help="which user complaint to target: uf1, uf2, uf3, or 'all' (default)")
    ap.add_argument("--images", nargs="+", required=True,
                    help="base image paths (shell globs are fine)")
    ap.add_argument("--out", default="results", help="output directory (default: results/)")
    ap.add_argument("--config", default=None, help="path to config.json")
    ap.add_argument("--target", type=float, default=None,
                    help="target probability for the label (overrides config)")
    ap.add_argument("--top-k", type=int, default=5,
                    help="how many coarse results to refine around (default: 5)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cpu | mps (default: auto)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.images]
    missing = [p for p in paths if not p.exists()]
    if missing:
        ap.error(f"image(s) not found: {', '.join(str(m) for m in missing)}")

    scorer = Scorer(device=args.device, batch_size=args.batch_size)
    search = Search(config_path=args.config, scorer=scorer, verbose=not args.quiet)
    if args.target is not None:
        search.target = args.target

    ufs = ["uf1", "uf2", "uf3"] if args.uf == "all" else [args.uf]
    unknown = [u for u in ufs if u not in search.config]
    if unknown:
        ap.error(f"unknown --uf {unknown}; config has: "
                 f"{[k for k in search.config if k.startswith('uf')]}")

    reports = {}
    for uf in ufs:
        reports[uf] = search.run(uf, paths, top_k=args.top_k, out_dir=args.out)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(reports, indent=2))

    if not args.quiet:
        print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
        for uf, r in reports.items():
            b = r.get("best")
            if b:
                flag = "✅" if r["target_met"] else "⚠️ "
                print(f"{flag} {uf}: {r['target_label']}={b['probs'][r['target_label']]:.4f} "
                      f"on {b['image']}  ({r['candidates_scored']} scored)")
                print(f"     {b['params']}")
            else:
                print(f"❌ {uf}: no candidate found")
        print(f"\nreport: {out / 'report.json'}")

    return 0 if all(r["target_met"] for r in reports.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
