"""ResNet model implementation with integrated loss calculation functionality."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetType(Enum):
    """Supported ResNet model types."""

    RESNET50 = auto()
    RESNET101 = auto()


@dataclass
class ModelOutput:
    """Container for model outputs during training and inference."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class ResNetWithLoss(nn.Module):
    """ResNet model with integrated loss calculation functionality.

    This class wraps a ResNet model and provides an interface for both
    inference and training with loss calculation.
    """

    def __init__(
        self,
        model_type: ResNetType = ResNetType.RESNET50,
        num_classes: int = 1000,
        pretrained: bool = False,
    ):
        """Initialize a ResNet model with loss calculation capability.

        Args:
            model_type: Type of ResNet model (ResNetType.RESNET50 or ResNetType.RESNET101)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        # Create the base model
        self.model = self._create_base_model(model_type, pretrained)

        # Modify the final layer for dataset-specific number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def _create_base_model(self, model_type: ResNetType, pretrained: bool) -> nn.Module:
        """Create the base ResNet model.

        Args:
            model_type: Type of ResNet model
            pretrained: Whether to use pretrained weights

        Returns:
            Base ResNet model

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == ResNetType.RESNET50:
            return models.resnet50(pretrained=pretrained)
        elif model_type == ResNetType.RESNET101:
            return models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_type}")

    def forward(
        self, image: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, ModelOutput]:
        """Forward pass through the model.

        Args:
            image: Input image tensor
            label: Optional ground truth labels for loss calculation

        Returns:
            If label is provided, returns a ModelOutput with loss and logits.
            Otherwise, returns just the logits.
        """
        logits = self.model(image)

        if label is not None:
            loss = self.criterion(logits, label)
            return ModelOutput(logits=logits, loss=loss)
        return logits


def get_resnet(
    model_type: str = "resnet50", num_classes: int = 1000, pretrained: bool = False
) -> ResNetWithLoss:
    """Factory function to create a ResNet model with loss calculation interface.

    Args:
        model_type: Type of ResNet model ("resnet50" or "resnet101")
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        Configured ResNet model

    Raises:
        ValueError: If model_type is not supported
    """
    # Map string model type to enum
    if model_type == "resnet50":
        resnet_type = ResNetType.RESNET50
    elif model_type == "resnet101":
        resnet_type = ResNetType.RESNET101
    else:
        raise ValueError(
            f"Unsupported ResNet model: {model_type}. Choose 'resnet50' or 'resnet101'"
        )

    return ResNetWithLoss(
        model_type=resnet_type, num_classes=num_classes, pretrained=pretrained
    )


# Example usage (only runs when script is executed directly)
if __name__ == "__main__":
    # Create model
    model = get_resnet("resnet50", num_classes=1000)

    # Example forward pass
    batch_size, channels, height, width = 2, 3, 224, 224
    dummy_input = torch.randn(batch_size, channels, height, width)
    dummy_labels = torch.randint(0, 1000, (batch_size,))

    # Test inference
    with torch.no_grad():
        logits = model(dummy_input)
        print(f"Inference output shape: {logits.shape}")

    # Test training with loss
    output = model(dummy_input, dummy_labels)
    print(f"Training loss: {output.loss.item()}")
