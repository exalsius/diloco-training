import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample, BigGANConfig


class BigGANWithLoss(nn.Module):
    def __init__(self, model_type="biggan-deep-256", pretrained=True):
        super().__init__()
        if model_type == "biggan-deep-256":
            self.model = BigGAN.from_pretrained('biggan-deep-256')
        else:
            raise ValueError(f"Unknown BigGAN model: {model_type}")
        for module in self.model.modules():
            for module in self.model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.reset_parameters()
        self.criterion = nn.MSELoss()

    def forward(self, noise, class_vector, real_image=None):
        generated_image = self.model(noise, class_vector, truncation=0.4)
        if real_image is not None:
            loss = self.criterion(generated_image, real_image)
            return type("Output", (object,), {"loss": loss, "generated_image": generated_image})()
        return generated_image


def get_biggan(model_type="biggan-deep-256", pretrained=True):
    """Return BigGAN model with a loss calculation interface"""
    return BigGANWithLoss(
        model_type=model_type, pretrained=pretrained
    )


# Example usage
if __name__ == "__main__":
    model = get_biggan("biggan-deep-256")
    noise = torch.tensor(truncated_noise_sample(truncation=0.4, batch_size=1))
    class_vector = torch.tensor(one_hot_from_int([207], batch_size=1))  # Example class vector for "golden retriever"
    generated_image = model(noise, class_vector)
    print(generated_image)