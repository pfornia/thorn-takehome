"""Two-stage search for false-positive-inducing transformations.

Stage 1 (coarse): full grid over the parameter space declared in config.json.
Stage 2 (refine): take the top-K coarse results and explore locally around each -- step numeric
                  parameters by the configured deltas, and draw additional random seeds.

Why two stages rather than one big grid or pure random search:
  - a single fine grid is combinatorially infeasible (the collage space alone is effectively
    unbounded once crop regions are continuous)
  - pure random search finds *a* winner but doesn't produce interpretable dose-response curves,
    and the investigation needs to explain WHICH characteristics trigger the failure
  - coarse-then-local gets both: readable structure from stage 1, precision from stage 2

IMPORTANT -- this searches TRANSFORMATION PARAMETERS, never pixels. Hill-climbing over pixel
values would be an adversarial perturbation, which the assignment forbids. Every candidate here
is a plausible image edit (a collage, a watermark, sensor noise) that a real user or pipeline
could produce; the search only chooses the settings.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .scoring import Score, Scorer
from .transforms import apply_transform


@dataclass
class Candidate:
    """One evaluated transformation."""

    transform: str
    params: dict
    image_name: str
    score: Score = field(repr=False)

    @property
    def dog(self) -> float:
        return self.score.probs["dog"]

    def objective(self, label: str) -> float:
        return self.score.margin(label)

    def summary(self, label: str = "dog") -> dict:
        return {
            "transform": self.transform,
            "image": self.image_name,
            "params": {k: v for k, v in self.params.items() if not k.startswith("_")},
            "pred": self.score.pred,
            "probs": {k: round(v, 6) for k, v in self.score.probs.items()},
            "logit_margin_vs_other": round(self.score.margin(label), 4),
        }


def _expand_grid(space: dict) -> list[dict]:
    keys = [k for k in space if not k.startswith("_")]
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


def _is_valid(transform: str, params: dict) -> bool:
    """Drop nonsensical combinations before spending a forward pass on them.

    The `degrade` strength axis is shared across kinds with different natural units (sigma for
    gaussian, a 0-1 fraction for blend/salt-pepper), so most cross-products are meaningless.
    """
    if transform == "degrade":
        kind, s = params.get("kind"), params.get("strength", 0)
        if kind == "gaussian":
            return s > 1
        return 0 < s <= 1.0
    if transform == "collage":
        # a gutter wider than a third of the tile would obliterate the tiles
        return params.get("line_width", 0) <= (512 / params["grid"]) / 3
    return True


def _neighbors(params: dict, refine_cfg: dict, transform: str) -> list[dict]:
    """Local neighbourhood of a parameter set: numeric steps plus extra seeds."""
    out: list[dict] = []
    for key, deltas in refine_cfg.get("numeric_deltas", {}).items():
        if key not in params:
            continue
        base = params[key]
        for d in deltas:
            cand = dict(params)
            val = base + d
            if isinstance(base, int) and isinstance(d, int):
                val = int(val)
                if val < 1:
                    continue
            else:
                val = round(float(val), 4)
                if val <= 0:
                    continue
            cand[key] = val
            if _is_valid(transform, cand):
                out.append(cand)

    n_seeds = int(refine_cfg.get("seeds", 0))
    if n_seeds and "seed" in params:
        rng = random.Random(12345)
        for _ in range(n_seeds):
            cand = dict(params)
            cand["seed"] = rng.randrange(1_000_000)
            if _is_valid(transform, cand):
                out.append(cand)
    return out


class Search:
    def __init__(self, config_path: str | Path | None = None, scorer: Scorer | None = None,
                 verbose: bool = True):
        cfg_path = Path(config_path) if config_path else Path(__file__).parent / "config.json"
        self.config = json.loads(cfg_path.read_text())
        self.scorer = scorer or Scorer()
        self.label = self.config.get("target_label", "dog")
        self.target = float(self.config.get("target_prob", 0.99))
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _evaluate(self, transform: str, param_sets: list[dict],
                  image_groups: list[tuple[str, list[Image.Image]]]) -> list[Candidate]:
        """Render and score every (param set x image group) pair, batched."""
        jobs, rendered = [], []
        for name, sources in image_groups:
            for p in param_sets:
                try:
                    img = apply_transform(transform, sources, p)
                except Exception as exc:  # a bad param combo shouldn't kill the run
                    self._log(f"    ! skipped {p}: {exc}")
                    continue
                jobs.append((name, p))
                rendered.append(img)
        if not rendered:
            return []
        scores = self.scorer.score_batch(rendered)
        return [
            Candidate(transform, p, name, s) for (name, p), s in zip(jobs, scores)
        ]

    def run(self, uf: str, image_paths: list[str | Path], top_k: int = 5,
            out_dir: str | Path | None = None) -> dict:
        """Run the two-stage search for one user complaint. Returns a report dict."""
        block = self.config[uf]
        transform = block["transform"]
        per_image = bool(block.get("per_image", True))

        loaded = [(Path(p).stem, Image.open(p).convert("RGB")) for p in image_paths]
        if per_image:
            groups = [(name, [img]) for name, img in loaded]
        else:
            groups = [("__mixed__", [img for _, img in loaded])]

        self._log(f"\n{'='*78}\n{block['name']}\n{'='*78}")
        self._log(f'complaint: "{block["complaint"]}"')

        # ---- baseline: confirm the untouched images really are "other" ----
        base_imgs = [img for _, img in loaded]
        base_scores = self.scorer.score_batch(base_imgs)
        self._log("\nbaseline (unmodified):")
        for (name, _), s in zip(loaded, base_scores):
            self._log(
                f"  {name:<12} pred={s.pred:<6} dog={s.probs['dog']:.6f} "
                f"margin={s.margin(self.label):+.2f}"
            )

        # ---- stage 1: coarse grid ----
        coarse = [p for p in _expand_grid(block["coarse"]) if _is_valid(transform, p)]
        self._log(f"\nstage 1 (coarse grid): {len(coarse)} param sets x {len(groups)} image group(s)")
        results = self._evaluate(transform, coarse, groups)
        results.sort(key=lambda c: -c.objective(self.label))
        n_cross = sum(1 for c in results if c.score.pred == self.label)
        best1 = results[0] if results else None
        self._log(f"  evaluated {len(results)} | {n_cross} predict '{self.label}'")
        if best1:
            self._log(f"  best: {self.label}={best1.dog:.4f} margin={best1.objective(self.label):+.2f}")
            self._log(f"        {best1.params}")

        # ---- stage 2: local refinement around the top-K ----
        seeds_for_refine = results[:top_k]
        neigh: list[dict] = []
        seen = {json.dumps(c.params, sort_keys=True, default=str) for c in results}
        for c in seeds_for_refine:
            for p in _neighbors(c.params, block.get("refine", {}), transform):
                key = json.dumps(p, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    neigh.append(p)

        refined: list[Candidate] = []
        if neigh:
            self._log(f"\nstage 2 (local refine around top {len(seeds_for_refine)}): "
                      f"{len(neigh)} new param sets")
            # refine only against the image that produced the best coarse result
            best_group = [g for g in groups if g[0] == best1.image_name] or groups
            refined = self._evaluate(transform, neigh, best_group)
            refined.sort(key=lambda c: -c.objective(self.label))
            if refined:
                self._log(f"  evaluated {len(refined)} | best: "
                          f"{self.label}={refined[0].dog:.4f} "
                          f"margin={refined[0].objective(self.label):+.2f}")

        allc = sorted(results + refined, key=lambda c: -c.objective(self.label))
        winners = [c for c in allc if c.score.pred == self.label]
        best = winners[0] if winners else (allc[0] if allc else None)

        self._log(f"\n{'-'*78}")
        if best and best.score.pred == self.label:
            hit = "✅ TARGET MET" if best.dog >= self.target else "⚠️  below target"
            self._log(f"{hit}: {self.label}={best.dog:.4f} (target {self.target}) "
                      f"margin={best.objective(self.label):+.2f}")
        else:
            self._log(f"❌ no {self.label} prediction found")
        self._log(f"total candidates scored: {self.scorer.n_scored}")

        report = {
            "user_feedback": uf,
            "name": block["name"],
            "complaint": block["complaint"],
            "transform": transform,
            "target_label": self.label,
            "target_prob": self.target,
            "target_met": bool(best and best.score.pred == self.label and best.dog >= self.target),
            "candidates_scored": self.scorer.n_scored,
            "stage1_evaluated": len(results),
            "stage1_crossings": n_cross,
            "stage2_evaluated": len(refined),
            "baseline": [
                {"image": n, "pred": s.pred, "probs": {k: round(v, 6) for k, v in s.probs.items()}}
                for (n, _), s in zip(loaded, base_scores)
            ],
            "best": best.summary(self.label) if best else None,
            "top": [c.summary(self.label) for c in allc[:top_k]],
        }

        if out_dir and best:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            sources = dict(groups)[best.image_name] if best.image_name in dict(groups) else base_imgs
            img = apply_transform(transform, sources, best.params)
            stem = f"{uf}_{best.image_name}_dog{best.dog:.4f}"
            img.save(out / f"{stem}.png")
            # pair the original alongside, as the submission requires
            if best.image_name != "__mixed__":
                dict(loaded)[best.image_name].save(out / f"{stem}__ORIGINAL.png")
            (out / f"{stem}.json").write_text(json.dumps(report, indent=2))
            report["output_image"] = str(out / f"{stem}.png")
            self._log(f"saved: {out / (stem + '.png')}")

        return report
