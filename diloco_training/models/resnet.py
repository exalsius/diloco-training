import torch.nn as nn
import torchvision.models as models


class ResNetWithLoss(nn.Module):
    def __init__(self, model_type="resnet50", num_classes=1000, pretrained=False):
        super().__init__()
        if model_type == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif model_type == "resnet101":
            self.model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown ResNet model: {model_type}")

        # Modify the final layer for dataset-specific number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image, label=None):
        logits = self.model(image)
        if label is not None:
            loss = self.criterion(logits, label)
            return type("Output", (object,), {"loss": loss, "logits": logits})()
        return logits


def get_resnet(model_type="resnet50", num_classes=1000, pretrained=False):
    """Return ResNet model with a loss calculation interface"""
    return ResNetWithLoss(
        model_type=model_type, num_classes=num_classes, pretrained=pretrained
    )


# Example usage
if __name__ == "__main__":
    m = get_resnet("resnet50", num_classes=1000)
