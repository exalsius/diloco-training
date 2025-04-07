"""
ImageNet dataset module for loading and processing ImageNet data.
Provides utilities for distributed training and data transformation.
"""

from itertools import islice

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

# class HuggingFaceDataset(Dataset):
#     """Wrapper for HuggingFace datasets to make them compatible with PyTorch DataLoader."""

#     def __init__(self, hf_dataset: Any, transform: Optional[transforms.Compose] = None):
#         """
#         Initialize the dataset wrapper.

#         Args:
#             hf_dataset: HuggingFace dataset to wrap
#             transform: Torchvision transforms to apply to images
#         """
#         self.hf_dataset = hf_dataset
#         self.transform = transform

#     def __len__(self) -> int:
#         """Return the number of samples in the dataset."""
#         return len(self.hf_dataset)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         """
#         Get a sample from the dataset.

#         Args:
#             idx: Index of the sample to get

#         Returns:
#             Dict containing the image and label tensors
#         """
#         item = self.hf_dataset[idx]
#         image = item["image"]

#         if self.transform:
#             image = self.transform(image)

#         label = torch.tensor(item["label"], dtype=torch.long)
#         return {"image": image, "label": label}


class StreamingImageNetDataset(IterableDataset):
    """Streaming ImageNet dataset for distributed training."""

    def __init__(
        self, dataset_name: str, rank: int, world_size: int, split: str = "train"
    ):
        """
        Initialize the streaming dataset.

        Args:
            dataset_name: HuggingFace dataset name
            rank: Rank of the current process
            world_size: Total number of processes
            split: Dataset split to use ('train' or 'validation')
        """
        if split == "validation":
            self.dataset = load_dataset(
                dataset_name, split=split, streaming=True
            ).shuffle(buffer_size=10_000)
        else:
            # For training, we don't shuffle
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.rank = rank
        self.world_size = world_size
        self.transform = create_imagenet_transforms()

    def __iter__(self):
        """
        Iterate over the dataset, partitioned for distributed training.

        Yields:
            Samples from the dataset
        """
        iterator = iter(self.dataset)
        batch = islice(iterator, self.rank, None, self.world_size)
        for item in batch:
            image = self.transform(item["image"])
            label = torch.tensor(item["label"], dtype=torch.long)
            yield {"image": image, "label": label}


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
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader for the dataset
    """
    # Create transforms

    # Load streaming dataset
    dataset = StreamingImageNetDataset(dataset_name, local_rank, world_size, split)

    # Wrap with HuggingFaceDataset for PyTorch compatibility
    # torch_dataset = HuggingFaceDataset(dataset, transform)

    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
    )

    # Create validation dataset (for simplicity, using the same dataset)
    val_dataset = StreamingImageNetDataset(dataset_name, 0, 1, "validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_device_train_batch_size,
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
