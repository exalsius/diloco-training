from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
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


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    dataset_name: str = "allenai/c4"
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


class StreamingC4Dataset(IterableDataset):
    def __init__(
        self,
        dataset_name,
        rank,
        world_size,
        split="train",
        cache_dir: Optional[Path] = None,
    ):
        # Create robust download configuration
        download_config = create_download_config()

        if split == "validation":
            self.dataset = load_dataset(
                dataset_name,
                "en",
                split=split,
                streaming=True,
                trust_remote_code=True,
                cache_dir=cache_dir,
                download_config=download_config,
            )
        else:
            self.dataset = load_dataset(
                dataset_name,
                "en",
                split=split,
                streaming=True,
                trust_remote_code=True,
                cache_dir=cache_dir,
                download_config=download_config,
            )
        self.rank = rank
        self.split = split
        self.world_size = world_size
        self.config = DatasetConfig()
        self.tokenizer = create_tokenizer(self.config, cache_dir=cache_dir)

    def tokenize_fn(self, batch: Dict[str, Any]) -> Any:
        return self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
        )

    def __iter__(self):
        iterator = iter(self.dataset)
        batch = islice(iterator, self.rank, None, self.world_size)  # DDP partitioning
        # Tokenize and yield each batch
        for item in batch:
            tokenized_item = self.tokenize_fn(item)
            yield tokenized_item


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


def get_c4_pile(
    world_size: int,
    local_rank: int,
    per_device_train_batch_size: int,
    split: str = "train",
    config: Optional[DatasetConfig] = None,
    cache_dir: Optional[Path] = None,
):
    if config is None:
        config = DatasetConfig()
    # Load the dataset (each process gets its own portion)
    dataset = StreamingC4Dataset(
        "c4", local_rank, world_size, split, cache_dir=cache_dir
    )

    # Define Dataloader
    tokenizer = create_tokenizer(config, cache_dir=cache_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = StatefulDataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True,
    )

    dataset = StreamingC4Dataset("c4", 0, 1, "validation", cache_dir=cache_dir)

    # Define Dataloader
    tokenizer = create_tokenizer(config, cache_dir=cache_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    val_loader = StatefulDataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    rank, world_size = 0, 2
    train_loader, val_loader = get_c4_pile(rank, world_size, 16)
    import time

    count = 0
    for batch in val_loader:
        stime = time.time()
        print(batch.keys())
        batch["input_ids"] = batch["input_ids"].to(rank)
        batch["labels"] = batch["labels"].to(rank)
        batch["attention_mask"] = batch["attention_mask"].to(rank)
        print(batch["input_ids"].device)
        etime = time.time()
        print(f"Time taken: {etime - stime:.4f} seconds")
        count += 1
        if count % 10000 == 0:
            print(count)
