"""
ImageNet dataset module for loading and processing ImageNet data.
Provides utilities for distributed training and data transformation.
"""

from itertools import islice
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader


class StreamingImageNetDataset(IterableDataset):
    """Streaming ImageNet dataset for distributed training."""

    def __init__(
        self,
        dataset_name: str,
        rank: int,
        world_size: int,
        split: str = "train",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the streaming dataset.

        Args:
            dataset_name: HuggingFace dataset name
            rank: Rank of the current process
            world_size: Total number of processes
            split: Dataset split to use ('train' or 'validation')
            cache_dir: Directory for caching datasets. If None, uses HuggingFace default
        """
        self.split = split
        if split == "validation":
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        else:
            # For training, we don't shuffle
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        self.rank = rank
        self.world_size = world_size
        self.transform = create_imagenet_transforms()

    def __iter__(self):
        """
        Iterate over the dataset, partitioned for distributed training.

        Yields:
            Samples from the dataset
        """
        if self.split == "validation":
            iterator = iter(self.dataset)
            batch = islice(iterator, self.rank, None, self.world_size)
            for item in batch:
                image = self.transform(item["image"])
                label = torch.tensor(item["label"], dtype=torch.long)
                yield {"image": image, "label": label}
        else:
            while True:
                iterator = iter(self.dataset)
                batch = islice(iterator, self.rank, None, self.world_size)
                for item in batch:
                    image = self.transform(item["image"])
                    label = torch.tensor(item["label"], dtype=torch.long)
                    yield {"image": image, "label": label}


def create_imagenet_transforms(image_size: int = 256) -> transforms.Compose:
    """
    Create standard ImageNet transforms.

    Args:
        image_size: Size to resize images to

    Returns:
        Composition of transforms
    """
    return transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.CenterCrop(224),
            transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),  # Ensure 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_imagenet(
    world_size: int,
    local_rank: int,
    per_device_train_batch_size: int,
    split: str = "train",
    image_size: int = 64,
    dataset_name: str = "imagenet-1k",
    cache_dir: Optional[Path] = None,
):
    """
    Load and prepare ImageNet dataset for training or evaluation.

    Args:
        world_size: Number of processes in distributed training
        local_rank: Rank of current process
        per_device_train_batch_size: Batch size per device
        split: Dataset split to use ('train' or 'validation')
        image_size: Size to resize images to
        dataset_name: HuggingFace dataset name
        cache_dir: Directory for caching datasets. If None, uses HuggingFace default

    Returns:
        DataLoader for the dataset
    """
    # Create transforms
    dataset_name = "ILSVRC/imagenet-1k"
    # Load streaming dataset
    dataset = StreamingImageNetDataset(
        dataset_name, local_rank, world_size, split, cache_dir=cache_dir
    )

    # Wrap with HuggingFaceDataset for PyTorch compatibility
    # torch_dataset = HuggingFaceDataset(dataset, transform)

    # Create DataLoader
    train_loader = StatefulDataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Create validation dataset (for simplicity, using the same dataset)
    val_dataset = StreamingImageNetDataset(
        dataset_name, 0, 1, "validation", cache_dir=cache_dir
    )
    val_loader = StatefulDataLoader(
        val_dataset,
        batch_size=per_device_train_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":

    rank, world_size = 0, 2
    train_loader, val_loader = get_imagenet(
        world_size=world_size, local_rank=rank, per_device_train_batch_size=32
    )

    count = 0
    for batch in val_loader:
        count += 1
        if count % 10 == 0:
            print(count)
