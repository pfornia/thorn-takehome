"""Scoring helpers — load the model once, report full probability vectors.

Deliberately reports the *whole* softmax vector rather than just the argmax, so every
experiment (including failures) yields a usable signal about which direction a
transformation moved the image.
"""

import sys
from pathlib import Path

import torch
from PIL import Image

# Thorn's model.py lives in homework/ and is left untouched; put it on the path rather than
# copying it, so the provided module stays the single source of truth.
REPO_ROOT = Path(__file__).resolve().parent.parent
HOMEWORK_DIR = REPO_ROOT / "homework"
sys.path.insert(0, str(HOMEWORK_DIR))

from model import MobileNetSmall  # noqa: E402

# Generated images go to the repo-level outputs/, not inside homework/.
OUTPUTS_DIR = REPO_ROOT / "outputs"
_model = None


def get_model() -> MobileNetSmall:
    """Load the checkpoint once and cache it."""
    global _model
    if _model is None:
        _model = MobileNetSmall().load_model(str(HOMEWORK_DIR / "model.pt"))
    return _model


def score_image(image: Image.Image | str | Path) -> dict:
    """Return {label: probability} plus raw logits for a PIL image or path."""
    model = get_model()
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    tensor = model.preprocess_image(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)[0]
    probs = torch.softmax(logits, dim=0)

    return {
        "probs": {label: probs[i].item() for i, label in enumerate(model.labels)},
        "logits": {label: logits[i].item() for i, label in enumerate(model.labels)},
        "pred": model.labels[int(torch.argmax(logits))],
        # The tests score on the best non-"other" probability, so surface it directly.
        "best_animal": max(
            ((label, probs[i].item()) for i, label in enumerate(model.labels) if label != "other"),
            key=lambda kv: kv[1],
        ),
    }


def format_score(name: str, result: dict) -> str:
    p = result["probs"]
    animal_label, animal_prob = result["best_animal"]
    return (
        f"{name:<28} cat={p['cat']:.6f}  dog={p['dog']:.6f}  other={p['other']:.6f}  "
        f"pred={result['pred']:<6} best_animal={animal_label}@{animal_prob:.6f}"
    )


if __name__ == "__main__":
    image_dir = HOMEWORK_DIR / "images"
    print(f"{'IMAGE':<28} {'PROBABILITIES':<60}")
    print("-" * 110)
    for path in sorted(image_dir.glob("*")):
        if path.is_file():
            print(format_score(path.name, score_image(path)))
