"""
Module for loading and processing C4/Pile datasets for language model training.
Provides utilities for tokenization, batching, and distributed training setup.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar

from datasets import Dataset
from datasets import Dataset as HFDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from datasets.utils.info_utils import VerificationMode
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from diloco_training.utils.hf_download import (
    DEFAULT_HF_MAX_RETRIES,
    DEFAULT_HF_TIMEOUT,
    create_download_config,
    set_hf_timeout,
)

# Configure logging
logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


class C4PileDataset(TorchDataset[T_co]):
    """Wrapper class for HuggingFace Dataset to make it compatible with PyTorch DataLoader."""

    def __init__(self, hf_dataset: HFDataset):
        self.dataset = hf_dataset

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    dataset_name: str = "PrimeIntellect/c4-tiny"
    dataset_config: str = "en"
    model_name: str = "EleutherAI/gpt-neo-1.3B"
    max_length: int = 1024
    split: str = "train"
    pad_token: str = "</s>"
    columns_to_remove: List[str] = field(
        default_factory=lambda: ["text", "timestamp", "url"]
    )
    use_fast_tokenizer: bool = True
    ignore_verifications: bool = True
    download_timeout: int = DEFAULT_HF_TIMEOUT
    max_retries: int = DEFAULT_HF_MAX_RETRIES


def create_tokenizer(
    config: DatasetConfig,
    cache_dir: Optional[Path] = None,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """
    Create and configure a tokenizer based on the provided configuration.

    Args:
        config: Dataset configuration object
        cache_dir: Directory for caching tokenizer. If None, uses HuggingFace default

    Returns:
        Configured tokenizer
    """
    # Set timeout for HuggingFace Hub downloads
    set_hf_timeout(config.download_timeout)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, use_fast=config.use_fast_tokenizer, cache_dir=cache_dir
    )

    # Ensure pad token is set for models that need it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = config.pad_token

    return tokenizer


def load_and_process_dataset(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    config: DatasetConfig,
    cache_dir: Optional[Path] = None,
) -> Dataset:
    """
    Load and tokenize the dataset.

    Args:
        tokenizer: The tokenizer to use for processing
        config: Dataset configuration object
        cache_dir: Directory for caching datasets. If None, uses HuggingFace default

    Returns:
        Tokenized dataset
    """
    # Create robust download configuration
    download_config = create_download_config(
        config.download_timeout, config.max_retries
    )

    try:
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            verification_mode=VerificationMode.NO_CHECKS,
            split=config.split,
            trust_remote_code=True,
            cache_dir=cache_dir,
            download_config=download_config,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    def tokenize_fn(batch: Dict[str, Any]) -> Any:
        """Tokenize a batch of examples."""
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
        )

    # Tokenize the dataset and remove unnecessary columns
    tokenized_datasets = dataset.map(
        tokenize_fn, batched=True, remove_columns=config.columns_to_remove
    )
    if isinstance(tokenized_datasets, Dataset):
        return tokenized_datasets
    else:
        raise ValueError(
            f"Expected Dataset, got {type(tokenized_datasets)}. Please check the dataset format."
        )


def get_c4_pile_prime(
    world_size: int,
    local_rank: int,
    per_device_train_batch_size: int,
    split: str = "train",
    config: Optional[DatasetConfig] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[HFDataset, DataLoader]:
    """
    Loads C4/The Pile dataset for language model training with distributed support.

    Args:
        world_size: Number of processes in distributed training
        local_rank: Local rank of the current process
        per_device_train_batch_size: Batch size per device
        split: Dataset split to use (e.g., "train", "validation")
        config: Optional dataset configuration, uses default if not provided
        cache_dir: Directory for caching datasets. If None, uses HuggingFace default

    Returns:
        Tuple containing the dataset and dataloader
    """
    if config is None:
        config = DatasetConfig()

    # Override the split in the config with the provided split parameter
    config.split = split

    tokenizer = create_tokenizer(config, cache_dir=cache_dir)

    # Load and process the dataset
    tokenized_dataset = load_and_process_dataset(tokenizer, config, cache_dir=cache_dir)

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Split dataset for distributed training
    distributed_dataset = split_dataset_by_node(
        tokenized_dataset, world_size=world_size, rank=local_rank
    )

    # Create dataloader with wrapped dataset
    dataloader = DataLoader(
        C4PileDataset(distributed_dataset),
        collate_fn=data_collator,
        batch_size=per_device_train_batch_size,
    )

    return distributed_dataset, dataloader


if __name__ == "__main__":
    # Configure logging for the script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    dataset, dataloader = get_c4_pile_prime(
        world_size=1, local_rank=0, per_device_train_batch_size=32
    )

    logger.info(f"Dataset length: {len(dataset)}")
    logger.info(f"Dataloader length: {len(dataloader)}")
    logger.info(f"Dataloader batch size: {dataloader.batch_size}")

    # Print shape of first few batches
    for step, batch in enumerate(dataloader):
        logger.info(f"Batch {step} shape: {batch['input_ids'].shape}")
        if step >= 2:  # Only show first few batches
            break
