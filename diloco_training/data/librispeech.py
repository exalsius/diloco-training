from itertools import islice

from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import Wav2Vec2Processor


class StreamingLibriSpeechDataset(IterableDataset):
    """Streaming LibriSpeech dataset for distributed training."""

    def __init__(self, dataset_name, rank, world_size, split="train.clean.100"):
        """
        Initialize the streaming dataset.

        Args:
            dataset_name: HuggingFace dataset name
            rank: Rank of the current process
            world_size: Total number of processes
            split: Dataset split to use
        """
        if split == "validation.clean":
            self.dataset = load_dataset(
                dataset_name, split=split, streaming=True, trust_remote_code=True
            ).shuffle(buffer_size=10000)
        else:
            # For training, we don't shuffle
            self.dataset = load_dataset(
                dataset_name, split=split, streaming=True, trust_remote_code=True
            )
        self.rank = rank
        self.world_size = world_size
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

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
        batch["labels"] = self.processor.tokenizer(batch["text"]).input_ids
        return batch

    def __iter__(self):
        """
        Iterate over the dataset, partitioned for distributed training.

        Yields:
            Preprocessed samples from the dataset
        """
        iterator = iter(self.dataset)
        partitioned_iterator = islice(iterator, self.rank, None, self.world_size)
        for item in partitioned_iterator:
            yield self.preprocess(item)


def get_librispeech(
    world_size, local_rank, per_device_train_batch_size, split="train.clean.100"
):
    """
    Load and prepare LibriSpeech dataset for training or evaluation.

    Args:
        world_size: Number of processes in distributed training
        local_rank: Rank of current process
        per_device_train_batch_size: Batch size per device
        split: Dataset split to use

    Returns:
        DataLoader for the dataset
    """
    if split == "train":
        split = "train.clean.100"
    dataset = StreamingLibriSpeechDataset(
        "librispeech_asr", local_rank, world_size, split
    )

    def collate_fn(features):
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = dataset.processor.pad(
            input_features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        with dataset.processor.as_target_processor():
            labels_batch = dataset.processor.pad(
                label_features,
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch

    train_loader = DataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=collate_fn,
    )

    dataset = StreamingLibriSpeechDataset("librispeech_asr", 0, 1, "validation.clean")

    val_loader = DataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=collate_fn,
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
