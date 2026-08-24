from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

LABELS = ["cat", "dog", "other"]

# ImageNet normalization stats necessary for preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Resize, normalize, and cast to expected type as part of preprocessing
TRANSFORM = v2.Compose(
    [
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


class MobileNetSmall(nn.Module):
    """
    This is a torch wrapper for a MobileNetV3-Small model. Uses the torchvision
    implementation of MobileNetV3-Small, with the final linear classification
    layer set to predict on the given labels. For the homework assignment, the
    labels are ["cat", "dog", "other"].

    To run inference, call ``preprocess_image`` on a PIL image to get a model-ready tensor,
    then call ``forward`` on that tensor to get the model's output logits.
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        labels: list[str] | None = None,
    ):
        super().__init__()
        self.labels = labels if labels is not None else list(LABELS)
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, len(self.labels))

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Resize + normalize a single PIL image into a model-ready tensor."""
        return TRANSFORM(image)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def save_model(self, checkpoint_path: str | Path) -> "MobileNetSmall":
        """Save this model's weights and labels to a ``.pt`` checkpoint.

        Writes a checkpoint dict of the form
        ``{"model_state_dict": ..., "labels": ...}`` that ``load_model`` can
        read back. Returns ``self`` for chaining.
        """
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "labels": list(self.labels),
            },
            checkpoint_path,
        )
        return self

    def load_model(
        self, checkpoint_path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "MobileNetSmall":
        """Load weights from a ``.pt`` checkpoint into this model.

        Supports both a plain ``state_dict`` and a checkpoint dict saved as
        ``{"model_state_dict": ..., "labels": ...}`` (as produced by
        ``train_model.py``). Returns ``self`` for chaining and leaves the model
        in eval mode.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if checkpoint.get("labels"):
                self.labels = list(checkpoint["labels"])
        else:
            state_dict = checkpoint

        # Checkpoints may store keys with or without the ``backbone.`` prefix
        # depending on how the model was defined when saved.
        if not any(key.startswith("backbone.") for key in state_dict):
            state_dict = {f"backbone.{key}": value for key, value in state_dict.items()}

        self.load_state_dict(state_dict)
        self.eval()
        return self
