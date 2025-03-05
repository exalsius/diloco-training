import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Any
from dataclasses import dataclass
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample


@dataclass
class BigGANOutput:
    """Output container for BigGAN model results"""
    loss: Optional[torch.Tensor] = None
    generated_image: Optional[torch.Tensor] = None


class BigGANWithLoss(nn.Module):
    """BigGAN model with integrated loss calculation"""
    
    def __init__(self, model_type: str = "biggan-deep-256", pretrained: bool = True, truncation: float = 0.4):
        """
        Initialize BigGAN model with loss function
        
        Args:
            model_type: Type of BigGAN model to use
            pretrained: Whether to use pretrained weights
            truncation: Truncation factor for sampling
        """
        super().__init__()
        self.truncation = truncation
        self._initialize_model(model_type, pretrained)
        self.criterion = nn.MSELoss()
        
    def _initialize_model(self, model_type: str, pretrained: bool) -> None:
        """Initialize the underlying BigGAN model"""
        if model_type == "biggan-deep-256":
            self.model = BigGAN.from_pretrained('biggan-deep-256') if pretrained else BigGAN.from_pretrained('biggan-deep-256', pretrained=False)
        else:
            raise ValueError(f"Unsupported BigGAN model: {model_type}")
            
        # Reset parameters if not using pretrained weights
        if not pretrained:
            self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Reset parameters of convolutional and linear layers"""
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.reset_parameters()
                
    def forward(self, 
                noise: torch.Tensor, 
                class_vector: torch.Tensor, 
                real_image: Optional[torch.Tensor] = None) -> Union[torch.Tensor, BigGANOutput]:
        """
        Generate images and optionally calculate loss
        
        Args:
            noise: Input noise tensor
            class_vector: One-hot encoded class vector
            real_image: Optional target image for loss calculation
            
        Returns:
            Either generated image tensor or BigGANOutput with loss and generated image
        """
        generated_image = self.model(noise, class_vector, truncation=self.truncation)
        
        if real_image is not None:
            loss = self.criterion(generated_image, real_image)
            return BigGANOutput(loss=loss, generated_image=generated_image)
        
        return generated_image
    
    @staticmethod
    def generate_noise(batch_size: int = 1, truncation: float = 0.4) -> torch.Tensor:
        """Generate truncated noise samples for BigGAN input"""
        return torch.tensor(truncated_noise_sample(truncation=truncation, batch_size=batch_size))
    
    @staticmethod
    def generate_class_vector(class_ids: list, batch_size: int = 1) -> torch.Tensor:
        """Generate one-hot encoded class vectors for BigGAN input"""
        return torch.tensor(one_hot_from_int(class_ids, batch_size=batch_size))


def get_biggan(model_type: str = "biggan-deep-256", pretrained: bool = True) -> BigGANWithLoss:
    """
    Factory function to create a BigGAN model with loss calculation
    
    Args:
        model_type: Type of BigGAN model to use
        pretrained: Whether to use pretrained weights
        
    Returns:
        Configured BigGAN model
    """
    return BigGANWithLoss(model_type=model_type, pretrained=pretrained)


# Example usage
if __name__ == "__main__":
    model = get_biggan("biggan-deep-256")
    
    # Generate inputs using helper methods
    noise = model.generate_noise(batch_size=1)
    class_vector = model.generate_class_vector([207], batch_size=1)  # 207 = "golden retriever"
    
    # Generate image
    generated_image = model(noise, class_vector)
    print(f"Generated image shape: {generated_image.shape}")