"""Batched model scoring.

Loads the provided checkpoint once and scores images in batches. Batching matters: the search
evaluates thousands of candidates, and a single forward pass over 64 images is far cheaper than
64 separate passes.

Always returns the FULL probability vector, never just the argmax. A candidate that fails is
still informative -- it tells the search which direction moved the model -- so nothing is
discarded.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

# model.py lives one level up (it is Thorn's provided module, left untouched)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import MobileNetSmall  # noqa: E402

HOMEWORK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = HOMEWORK_DIR / "model.pt"


@dataclass
class Score:
    """Full model output for one image."""

    probs: dict[str, float]
    logits: dict[str, float]
    pred: str

    def prob(self, label: str) -> float:
        return self.probs[label]

    def margin_vs_best_other(self, label: str) -> float:
        """logit(label) - max(logit of every OTHER class).

        This, not `margin(label, "other")`, is the correct search objective for a 3-class model.
        Maximising `logit(cat) - logit(other)` can be satisfied by images the model calls `dog`
        with near-certainty: cat beats "other" while dog takes all the probability mass. Measured
        directly -- a cat-targeted run produced candidates at margin +7.15 with cat probability
        0.0044, because they were confidently dog. Margin against the strongest *competitor*
        cannot be gamed that way: it is positive only when `label` is actually winning.
        """
        competitors = [v for k, v in self.logits.items() if k != label]
        return self.logits[label] - max(competitors)

    def margin(self, label: str, against: str = "other") -> float:
        """logit(label) - logit(against).

        Logit margin rather than probability is the search objective: probabilities saturate
        (four of the five provided base images score dog=0.000000, indistinguishable) while
        logits keep resolution where probabilities round to zero. Reference points measured on
        this checkpoint: the model's no-evidence prior sits near -2.5, and 0.0 is a coin flip.
        """
        return self.logits[label] - self.logits[against]


class Scorer:
    """Caches the model and scores PIL images in batches."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str | None = None,
                 batch_size: int = 64):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model = MobileNetSmall().load_model(str(checkpoint), map_location=str(self.device))
        self.model.to(self.device)
        self.model.eval()
        self.labels = list(self.model.labels)
        self._n_scored = 0

    @property
    def n_scored(self) -> int:
        """Total candidates evaluated; reported so search cost is auditable."""
        return self._n_scored

    @torch.no_grad()
    def score_batch(self, images: list[Image.Image]) -> list[Score]:
        out: list[Score] = []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            tensors = torch.stack(
                [self.model.preprocess_image(im.convert("RGB")) for im in chunk]
            ).to(self.device)
            logits = self.model(tensors)
            probs = torch.softmax(logits, dim=1)
            for row_logits, row_probs in zip(logits.cpu(), probs.cpu()):
                out.append(
                    Score(
                        probs={l: row_probs[j].item() for j, l in enumerate(self.labels)},
                        logits={l: row_logits[j].item() for j, l in enumerate(self.labels)},
                        pred=self.labels[int(torch.argmax(row_logits))],
                    )
                )
        self._n_scored += len(images)
        return out

    def score(self, image: Image.Image) -> Score:
        return self.score_batch([image])[0]
