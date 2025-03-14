"""
ImageNet dataset module for loading and processing ImageNet data.
Provides utilities for distributed training and data transformation.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, Dataset


class HuggingFaceDataset(Dataset):
    """Wrapper for HuggingFace datasets to make them compatible with PyTorch DataLoader."""

    def __init__(self, hf_dataset: Any, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset wrapper.

        Args:
            hf_dataset: HuggingFace dataset to wrap
            transform: Torchvision transforms to apply to images
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to get

        Returns:
            Dict containing the image and label tensors
        """
        item = self.hf_dataset[idx]
        image = item["image"]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(item["label"], dtype=torch.long)
        return {"image": image, "label": label}


def create_imagenet_transforms(image_size: int = 64) -> transforms.Compose:
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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_imagenet(
    world_size: int,
    local_rank: int,
    per_device_train_batch_size: int,
    split: str = "train",
    num_samples: Optional[int] = 100,
    image_size: int = 64,
    dataset_name: str = "zh-plus/tiny-imagenet",
    shuffle: bool = True,
    num_workers: int = 4,
) -> Tuple[Dataset, DataLoader]:
    """
    Load and prepare ImageNet dataset for training or evaluation.

    Args:
        world_size: Number of processes in distributed training
        local_rank: Rank of current process
        per_device_train_batch_size: Batch size per device
        split: Dataset split to use ('train' or 'validation')
        num_samples: Number of samples to select (None for all)
        image_size: Size to resize images to
        dataset_name: HuggingFace dataset name
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (dataset, dataloader)
    """
    # Create transforms
    transform = create_imagenet_transforms(image_size)

    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split=split)
        if num_samples is not None:
            dataset = dataset.select(range(num_samples))
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    # Split dataset across nodes for distributed training
    if world_size > 1:
        dataset = split_dataset_by_node(dataset, rank=local_rank, world_size=world_size)

    # Convert to PyTorch dataset
    torch_dataset = HuggingFaceDataset(dataset, transform)

    # Create DataLoader
    dataloader = DataLoader(
        torch_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return torch_dataset, dataloader


if __name__ == "__main__":
    # Example usage
    dataset, dataloader = get_imagenet(
        world_size=1,
        local_rank=0,
        per_device_train_batch_size=32,
        num_samples=100,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Dataloader length: {len(dataloader)}")
    print(f"Dataloader batch size: {dataloader.batch_size}")

    # Verify data loading works
    try:
        for step, batch in enumerate(dataloader):
            print(f"Batch shape: {batch['image'].shape}")
            print(f"Label shape: {batch['label'].shape}")
            break
    except Exception as e:
        print(f"Error loading batch: {e}")
