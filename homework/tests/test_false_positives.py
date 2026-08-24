import math
import os

from pathlib import Path

import torch
from PIL import Image

from homework.model import MobileNetSmall

FALSE_POSITIVE_DIR = os.getenv("FALSE_POSITIVE_DIR", "")

HOMEWORK_DIR = Path(__file__).resolve().parent.parent
model = MobileNetSmall().load_model(str(HOMEWORK_DIR / "model.pt"))
original_image_dir = HOMEWORK_DIR / "images"

# 99% confidence in logits
LOGIT_OTHER_CONFIDENCE_THRESHOLD = math.log(0.99)
LOGIT_EASY_THRESHOLD = math.log(0.1)
LOGIT_MEDIUM_THRESHOLD = math.log(0.5)
LOGIT_HARD_THRESHOLD = math.log(0.99)

def test_other_images_predict_other():
    """
    Test that the original images in the images directory are correctly
    classified as "other" with at least 99% confidence, checked in logit space.
    """
    other_index = model.labels.index("other")
    image_paths = sorted(original_image_dir.glob("*"))
    assert image_paths, f"No images found in {original_image_dir}"

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = model.preprocess_image(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)[0]

        threshold = torch.logsumexp(logits, dim=0) + LOGIT_OTHER_CONFIDENCE_THRESHOLD
        assert logits[other_index] > threshold, (
            f"{image_path.name}: other logit {logits[other_index].item():.4f} "
            f"below 99% confidence threshold {threshold.item():.4f}"
        )

def _false_positive_logits():
    """Yield (name, best non-other logit, logsumexp) for each FP image.

    A false positive means the model assigns a cat/dog (i.e. non-"other")
    prediction. Working in logit space, an image reaches confidence ``p`` for a
    class when ``logit_class >= logsumexp(logits) + log(p)``. We report the best
    (max) non-other class logit against that per-image threshold.
    """
    fp_dir = Path(FALSE_POSITIVE_DIR)
    assert FALSE_POSITIVE_DIR and fp_dir.is_dir(), (
        f"FALSE_POSITIVE_DIR is not a directory: {FALSE_POSITIVE_DIR!r}"
    )

    other_index = model.labels.index("other")
    # Only keep cat or dog predictions
    non_other_indices = [i for i in range(len(model.labels)) if i != other_index]

    image_paths = sorted(p for p in fp_dir.glob("*") if p.is_file())
    assert image_paths, f"No images found in {fp_dir}"

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = model.preprocess_image(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)[0]

        best_logit = max(logits[i] for i in non_other_indices)
        logsumexp = torch.logsumexp(logits, dim=0)
        yield image_path.name, best_logit, logsumexp


def _assert_false_positive_confidence(log_threshold: float):
    for name, best_logit, logsumexp in _false_positive_logits():
        threshold = logsumexp + log_threshold
        best_prob = math.exp((best_logit - logsumexp).item())
        target_prob = math.exp(log_threshold)
        assert best_logit > threshold, (
            f"{name}: best cat/dog probability {best_prob:.4f} "
            f"below confidence threshold {target_prob:.4f}"
        )


def test_false_positive_images_predict_easy_threshold():
    """False-positive images clear an easy (10%) cat/dog confidence level."""
    _assert_false_positive_confidence(LOGIT_EASY_THRESHOLD)


def test_false_positive_images_predict_medium_threshold():
    """False-positive images clear a medium (50%) cat/dog confidence level."""
    _assert_false_positive_confidence(LOGIT_MEDIUM_THRESHOLD)


def test_false_positive_images_predict_hard_threshold():
    """False-positive images clear a hard (99%) cat/dog confidence level."""
    _assert_false_positive_confidence(LOGIT_HARD_THRESHOLD)
