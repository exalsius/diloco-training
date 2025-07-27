# train_biggan_deep_imagenet.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from diloco_training.data.imagenet import get_imagenet
from dataclasses import dataclass
from typing import Optional, Union
# -------------------------------
# BigGAN-Deep Generator & Discriminator
# -------------------------------

class SpectralNormConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)


class SpectralNormLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

    @property
    def in_features(self):
        return self.linear.in_features

    def forward(self, x):
        return self.linear(x)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = SpectralNormLinear(embedding_dim, num_features)
        self.beta = SpectralNormLinear(embedding_dim, num_features)

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma(y).unsqueeze(2).unsqueeze(3)
        beta = self.beta(y).unsqueeze(2).unsqueeze(3)
        return gamma * out + beta


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, upsample=True):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_channels, embedding_dim)
        self.cbn2 = ConditionalBatchNorm2d(out_channels, embedding_dim)
        self.conv1 = SpectralNormConv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = SpectralNormConv2d(out_channels, out_channels, 3, 1, 1)
        self.upsample = upsample
        self.shortcut = SpectralNormConv2d(in_channels, out_channels, 1, 1, 0) \
            if in_channels != out_channels or upsample else nn.Identity()

    def forward(self, x, y):
        h = self.cbn1(x, y)
        h = F.relu(h)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2)
            x = F.interpolate(x, scale_factor=2)
        h = self.conv1(h)
        h = self.cbn2(h, y)
        h = F.relu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class Generator(nn.Module):
    def __init__(self, z_dim=128, class_dim=1000, ch=96):
        super().__init__()
        self.z_dim = z_dim
        self.class_dim = class_dim
        self.embed = nn.Embedding(class_dim, z_dim)
        self.linear = SpectralNormLinear(z_dim, 4*4*16*ch)
        self.blocks = nn.ModuleList([
            GBlock(16*ch, 16*ch, z_dim),   # 4x4 -> 8x8
            GBlock(16*ch, 8*ch, z_dim),    # 8x8 -> 16x16
            GBlock(8*ch, 4*ch, z_dim),     # 16x16 -> 32x32
            GBlock(4*ch, 2*ch, z_dim),     # 32x32 -> 64x64
            GBlock(2*ch, 1*ch, z_dim),     # 64x64 -> 128x128
            GBlock(1*ch, 1*ch, z_dim),     # 128x128 -> 256x256
        ])
        self.bn = nn.BatchNorm2d(1*ch)
        self.conv = nn.Conv2d(1*ch, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, z, y):
        y_embed = self.embed(y)
        h = self.linear(z).view(z.size(0), -1, 4, 4)
        for block in self.blocks:
            h = block(h, y_embed)
        h = self.bn(h)
        h = F.relu(h)
        h = self.conv(h)
        return self.tanh(h)


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.conv1 = SpectralNormConv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = SpectralNormConv2d(out_channels, out_channels, 3, 1, 1)
        self.downsample = downsample
        self.shortcut = SpectralNormConv2d(in_channels, out_channels, 1, 1, 0) \
            if in_channels != out_channels or downsample else nn.Identity()

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        return h + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, class_dim=1000, ch=96):
        super().__init__()
        self.blocks = nn.Sequential(
            DBlock(3, 1*ch),     # 256 -> 128
            DBlock(1*ch, 2*ch),  # 128 -> 64
            DBlock(2*ch, 4*ch),  # 64 -> 32
            DBlock(4*ch, 8*ch),  # 32 -> 16
            DBlock(8*ch, 16*ch), # 16 -> 8
            DBlock(16*ch, 16*ch) # 8 -> 4
        )
        self.relu = nn.ReLU()
        self.linear = SpectralNormLinear(16*ch, 1)
        self.embed = SpectralNormLinear(class_dim, 16*ch)

    def forward(self, x, y):
        h = self.blocks(x)
        h = self.relu(h)
        h = torch.sum(h, dim=(2, 3))  # Global sum pooling
        out = self.linear(h)
        y_embed = self.embed(F.one_hot(y, num_classes=self.embed.in_features).float())
        out += torch.sum(h * y_embed, dim=1, keepdim=True)
        return out


# -------------------------------
# GAN Model with Loss Calculation
# -------------------------------

@dataclass
class BigGANOutput:
    """Container for BigGAN outputs during training and inference."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class BigGANWithLoss(nn.Module):
    """BigGAN model with integrated loss calculation functionality.
    
    This class wraps the Generator and Discriminator models and provides 
    an interface for both inference and training with loss calculation.
    """
    
    def __init__(self, z_dim=128, class_dim=1000, ch=96):
        super().__init__()
        self.generator = Generator(z_dim=z_dim, class_dim=class_dim, ch=ch)
        self.discriminator = Discriminator(class_dim=class_dim, ch=ch)
        self.z_dim = z_dim
        self.class_dim = class_dim
        
    def forward(self, image: torch.Tensor, label: Optional[torch.Tensor] = None) -> Union[torch.Tensor, BigGANOutput]:
        """Forward pass through the BigGAN model.
        
        Args:
            image: Input image tensor (real images for discriminator training)
            label: Optional ground truth labels for loss calculation
            
        Returns:
            If label is provided, returns a BigGANOutput with loss and logits.
            Otherwise, returns just the discriminator logits.
        """
        batch_size = image.size(0)
        device = image.device
        
        if label is not None:
            # Training mode: compute GAN loss
            # Generate fake images
            z = torch.randn(batch_size, self.z_dim, device=device)
            fake_labels = torch.randint(0, self.class_dim, (batch_size,), device=device)
            fake_images = self.generator(z, fake_labels)
            
            # Discriminator predictions
            d_real = self.discriminator(image, label)
            d_fake = self.discriminator(fake_images.detach(), fake_labels)
            
            # Wasserstein GAN loss (can be modified for other GAN losses)
            d_loss = -(d_real.mean() - d_fake.mean())
            
            # Generator loss
            d_fake_for_gen = self.discriminator(fake_images, fake_labels)
            g_loss = -d_fake_for_gen.mean()
            
            # Combined loss (you may want to weight these differently)
            total_loss = d_loss + g_loss
            
            return BigGANOutput(logits=d_real, loss=total_loss)
        else:
            # Inference mode: just return discriminator output
            # For inference, we need a dummy label if not provided
            if label is None:
                label = torch.zeros(batch_size, dtype=torch.long, device=device)
            return self.discriminator(image, label)


def get_biggan(z_dim: int = 128, class_dim: int = 1000, ch: int = 96):
    """Factory function to create a BigGAN model with loss calculation interface.
    
    Args:
        z_dim: Dimension of the noise vector
        class_dim: Number of classes (for conditional generation)
        ch: Base channel multiplier
        
    Returns:
        Tuple of (config, model) where config is None and model is BigGANWithLoss
    """
    model = BigGANWithLoss(z_dim=z_dim, class_dim=class_dim, ch=ch)
    return None, model


# -------------------------------
# Training Loop & Dataset Loading (Reference Implementation)
# -------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """Example usage of BigGAN with loss calculation interface."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model using the factory function
    _, model = get_biggan(z_dim=128, class_dim=1000, ch=96)
    model = model.to(device)
    
    print(f"BigGAN Parameters: {count_parameters(model):,}")
    
    # Example forward pass
    batch_size, channels, height, width = 2, 3, 256, 256
    dummy_images = torch.randn(batch_size, channels, height, width, device=device)
    dummy_labels = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Test inference (without loss calculation)
    with torch.no_grad():
        logits = model(dummy_images)
        print(f"Inference output shape: {logits.shape}")
    
    # Test training with loss calculation
    model.train()
    output = model(dummy_images, dummy_labels)
    print(f"Training loss: {output.loss.item()}")
    print(f"Training logits shape: {output.logits.shape}")
    
    # Example of generating images
    with torch.no_grad():
        z = torch.randn(4, 128, device=device)
        labels = torch.randint(0, 1000, (4,), device=device)
        generated_images = model.generator(z, labels)
        print(f"Generated images shape: {generated_images.shape}")


if __name__ == '__main__':
    os.makedirs("samples", exist_ok=True)
    main()
