"""
Module for loading and processing C4/Pile datasets for language model training.
Provides utilities for tokenization, batching, and distributed training setup.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    dataset_name: str = "PrimeIntellect/c4-tiny"
    dataset_config: str = "en"
    model_name: str = "EleutherAI/gpt-neo-1.3B"
    max_length: int = 1024
    split: str = "train"
    pad_token: str = "</s>"
    columns_to_remove: Tuple[str, ...] = ("text", "timestamp", "url")
    use_fast_tokenizer: bool = True
    ignore_verifications: bool = True


def create_tokenizer(config: DatasetConfig) -> PreTrainedTokenizer:
    """
    Create and configure a tokenizer based on the provided configuration.

    Args:
        config: Dataset configuration object

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, use_fast=config.use_fast_tokenizer
    )

    # Ensure pad token is set for models that need it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = config.pad_token

    return tokenizer


def load_and_process_dataset(
    tokenizer: PreTrainedTokenizer, config: DatasetConfig
) -> Dataset:
    """
    Load and tokenize the dataset.

    Args:
        tokenizer: The tokenizer to use for processing
        config: Dataset configuration object

    Returns:
        Tokenized dataset
    """
    try:
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            ignore_verifications=config.ignore_verifications,
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
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

    return tokenized_datasets[config.split]


def get_c4_pile(
    world_size: int,
    local_rank: int,
    per_device_train_batch_size: int,
    split: str = "train",
    config: Optional[DatasetConfig] = None,
) -> Tuple[Dataset, DataLoader]:
    """
    Loads C4/The Pile dataset for language model training with distributed support.

    Args:
        world_size: Number of processes in distributed training
        local_rank: Local rank of the current process
        per_device_train_batch_size: Batch size per device
        split: Dataset split to use (e.g., "train", "validation")
        config: Optional dataset configuration, uses default if not provided

    Returns:
        Tuple containing the dataset and dataloader
    """
    if config is None:
        config = DatasetConfig()

    # Override the split in the config with the provided split parameter
    config.split = split

    tokenizer = create_tokenizer(config)

    # Load and process the dataset
    tokenized_dataset = load_and_process_dataset(tokenizer, config)

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Split dataset for distributed training
    distributed_dataset = split_dataset_by_node(
        tokenized_dataset, world_size=world_size, rank=local_rank
    )

    # Create dataloader
    dataloader = DataLoader(
        distributed_dataset,
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
    dataset, dataloader = get_c4_pile(
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
