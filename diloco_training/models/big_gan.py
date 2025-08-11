# train_biggan_deep_imagenet.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from tqdm import tqdm
from diloco_training.data.imagenet import get_imagenet
from dataclasses import dataclass
from typing import Optional
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
    loss: torch.Tensor
    d_loss: Optional[torch.Tensor] = None
    g_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class BigGANWithLoss(nn.Module):
    """BigGAN model with integrated loss calculation functionality for DiLoCo framework."""
    
    def __init__(self, z_dim=128, class_dim=1000, ch=96):
        super().__init__()
        self.generator = Generator(z_dim=z_dim, class_dim=class_dim, ch=ch)
        self.discriminator = Discriminator(class_dim=class_dim, ch=ch)
        self.z_dim = z_dim
        self.class_dim = class_dim
        self.training_mode = "discriminator"  # "discriminator" or "generator"
        
    def set_training_mode(self, mode: str):
        """Set which component to train: 'discriminator' or 'generator'"""
        assert mode in ["discriminator", "generator"]
        self.training_mode = mode
        
    def forward(self, image: torch.Tensor, label: torch.Tensor) -> BigGANOutput:
        """Forward pass through the BigGAN model compatible with DiLoCo framework."""
        batch_size = image.size(0)
        device = image.device
        
        # Generate fake samples
        z = torch.randn(batch_size, self.z_dim, device=device)
        fake_labels = torch.randint(0, self.class_dim, (batch_size,), device=device)
        fake_images = self.generator(z, fake_labels)
        
        if self.training_mode == "discriminator":
            # Train discriminator
            d_real = self.discriminator(image, label)
            d_fake = self.discriminator(fake_images.detach(), fake_labels)
            d_loss = -(d_real.mean() - d_fake.mean())
            return BigGANOutput(loss=d_loss, d_loss=d_loss, logits=d_real)
            
        else:  # generator mode
            # Train generator
            d_fake_for_gen = self.discriminator(fake_images, fake_labels)
            g_loss = -d_fake_for_gen.mean()
            return BigGANOutput(loss=g_loss, g_loss=g_loss, logits=d_fake_for_gen)


def get_biggan(z_dim: int = 128, class_dim: int = 1000, ch: int = 96):
    """Factory function to create a BigGAN model compatible with DiLoCo framework."""
    model = BigGANWithLoss(z_dim=z_dim, class_dim=class_dim, ch=ch)
    return None, model


# -------------------------------
# Training Loop & Dataset Loading (Reference Implementation)
# -------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """Example usage - simplified for DiLoCo integration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, model = get_biggan(z_dim=128, class_dim=1000, ch=96)
    model = model.to(device)
    
    print(f"BigGAN Parameters: {count_parameters(model):,}")
    
    # Test with dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 256, 256, device=device)
    dummy_labels = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Test discriminator training
    model.set_training_mode("discriminator")
    d_output = model(dummy_images, dummy_labels)
    print(f"Discriminator loss: {d_output.loss.item()}")
    
    # Test generator training  
    model.set_training_mode("generator")
    g_output = model(dummy_images, dummy_labels)
    print(f"Generator loss: {g_output.loss.item()}")

    # Full training loop with data loading
    rank = 0
    world_size = 1
    train_batch_size = 32

    train_loader, _ = get_imagenet(
        world_size=world_size,
        local_rank=rank,
        per_device_train_batch_size=train_batch_size,
        split="train",
        image_size=256,
        dataset_name="ILSVRC/imagenet-1k"
    )

    # Optimizers for generator and discriminator components
    opt_G = torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.0, 0.999))
    opt_D = torch.optim.Adam(model.discriminator.parameters(), lr=2e-4, betas=(0.0, 0.999))

    z_dim = 128

    for epoch in range(1000):
        for i, batch in enumerate(tqdm(train_loader)):
            real, labels = batch["image"].to(device), batch["label"].to(device)
            batch_size = real.size(0)
    
            
            # Generator and discriminator steps using the wrapped model components
            z = torch.randn(batch_size, z_dim, device=device)
            fake_labels = torch.randint(0, 1000, (batch_size,), device=device)
            fake = model.generator(z, fake_labels)

            # Discriminator step
            D_real = model.discriminator(real, labels).mean()
            D_fake = model.discriminator(fake.detach(), fake_labels).mean()
            d_loss = -(D_real - D_fake)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Generator step  
            fake = model.generator(z, fake_labels)
            g_loss = -model.discriminator(fake, fake_labels).mean()
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            # Save generated images every 1000 iterations
            if i % 1000 == 0:
                with torch.no_grad():
                    z_sample = torch.randn(16, z_dim, device=device)
                    labels_sample = torch.randint(0, 1000, (16,), device=device)
                    sample_images = model.generator(z_sample, labels_sample)
                    utils.save_image(sample_images, f"samples/fake_epoch{epoch}_iter{i}.png", 
                                   normalize=True, scale_each=True, nrow=4)

        print(f"Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
        
        # Save final images for the epoch
        with torch.no_grad():
            z_sample = torch.randn(16, z_dim, device=device)
            labels_sample = torch.randint(0, 1000, (16,), device=device)
            sample_images = model.generator(z_sample, labels_sample)
            utils.save_image(sample_images, f"samples/fake_epoch{epoch}.png", 
                           normalize=True, scale_each=True, nrow=4)


if __name__ == '__main__':
    os.makedirs("samples", exist_ok=True)
    main()