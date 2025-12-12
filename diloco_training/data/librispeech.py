from itertools import islice
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import Wav2Vec2Processor

from diloco_training.utils.hf_download import create_download_config, set_hf_timeout


class StreamingLibriSpeechDataset(IterableDataset):
    """Streaming LibriSpeech dataset for distributed training."""

    def __init__(
        self,
        dataset_name,
        rank,
        world_size,
        split="train.clean.100",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the streaming dataset.

        Args:
            dataset_name: HuggingFace dataset name
            rank: Rank of the current process
            world_size: Total number of processes
            split: Dataset split to use
            cache_dir: Directory for caching datasets. If None, uses HuggingFace default
        """
        # Create robust download configuration
        download_config = create_download_config()
        # Set timeout for HuggingFace Hub downloads
        set_hf_timeout()

        if split == "validation.clean":
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                trust_remote_code=True,
                cache_dir=cache_dir,
                download_config=download_config,
            )
        else:
            # For training, we don't shuffle
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                trust_remote_code=True,
                cache_dir=cache_dir,
                download_config=download_config,
            )
        self.rank = rank
        self.world_size = world_size
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base", cache_dir=cache_dir
        )

    def normalize_text(self, text: str) -> str:
        # Basic LibriSpeech style normalization (could be extended)
        text = text.lower()
        # Keep letters, space and apostrophe
        return "".join(ch for ch in text if ch.isalpha() or ch in " '")

    def preprocess(self, batch):
        """
        Preprocess a single batch by extracting input values and tokenizing text.

        Args:
            batch: A single data sample

        Returns:
            Preprocessed sample
        """
        audio = batch["audio"]
        batch["input_values"] = self.processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        norm_text = self.normalize_text(batch["text"])
        batch["labels"] = self.processor.tokenizer(norm_text).input_ids
        return batch

    def __iter__(self):
        """
        Iterate over the dataset, partitioned for distributed training.

        Yields:
            Preprocessed samples from the dataset
        """
        while True:
            iterator = iter(self.dataset)
            partitioned_iterator = islice(iterator, self.rank, None, self.world_size)
            for item in partitioned_iterator:
                yield self.preprocess(item)


def get_librispeech(
    world_size,
    local_rank,
    per_device_train_batch_size,
    split="train.clean.100",
    cache_dir: Optional[Path] = None,
):
    """
    Load and prepare LibriSpeech dataset for training or evaluation.

    Args:
        world_size: Number of processes in distributed training
        local_rank: Rank of current process
        per_device_train_batch_size: Batch size per device
        split: Dataset split to use
        cache_dir: Directory for caching datasets. If None, uses HuggingFace default

    Returns:
        DataLoader for the dataset
    """
    # Set timeout for HuggingFace Hub downloads
    set_hf_timeout()

    if split == "train":
        split = "train.clean.100"
    dataset = StreamingLibriSpeechDataset(
        "librispeech_asr", local_rank, world_size, split, cache_dir=cache_dir
    )
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base", cache_dir=cache_dir
    )

    def collate_fn(features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = processor.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        with processor.as_target_processor():
            labels_batch = processor.pad(
                label_features,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

    train_loader = StatefulDataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    dataset = StreamingLibriSpeechDataset(
        "librispeech_asr", 0, 1, "validation.clean", cache_dir=cache_dir
    )

    val_loader = StatefulDataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader


def main():
    # Test the get_librispeech function with a small number of samples
    split = "train.clean.100"

    train_loader, val_loader = get_librispeech(2, 0, 4)

    print(f"Loaded samples from the {split} split.")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i + 1}: {batch['input_values'].shape, batch['labels'].shape}")
        if i >= 2:  # Print only the first 3 batches
            break

    for i, batch in enumerate(val_loader):
        print(f"Batch {i + 1}: {batch['input_values'].shape, batch['labels'].shape}")
        if i >= 2:  # Print only the first 3 batches
            break


if __name__ == "__main__":
    main()
