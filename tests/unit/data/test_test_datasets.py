import torch

from diloco_training.data.test_datasets import SequenceTestDataset


def test_sequence_test_dataset_iteration():
    # Test with finite number of samples
    num_samples = 5
    vocab_size = 100
    sequence_length = 10

    def mock_generator():
        while True:
            yield {
                "input_ids": torch.randint(0, vocab_size, (sequence_length,)),
                "attention_mask": torch.ones(sequence_length),
                "labels": torch.randint(0, vocab_size, (sequence_length,)),
            }

    dataset = SequenceTestDataset(mock_generator, num_samples)
    samples = list(dataset)

    assert len(samples) == num_samples
    for sample in samples:
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (sequence_length,)
        assert sample["attention_mask"].shape == (sequence_length,)
        assert sample["labels"].shape == (sequence_length,)


def test_get_test_sequence_dataloader():
    batch_size = 4
    sequence_length = 8
    vocab_size = 50
    num_samples = 10

    _, dataloader = SequenceTestDataset.get_test_sequence_dataloader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        num_samples=num_samples,
    )

    # Check dataloader configuration
    assert dataloader.batch_size == batch_size
    assert dataloader.num_workers == 0

    # Get first batch
    batch = next(iter(dataloader))

    # Check batch structure and shapes
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    expected_shape = (batch_size, sequence_length)
    assert batch["input_ids"].shape == expected_shape
    assert batch["attention_mask"].shape == expected_shape
    assert batch["labels"].shape == expected_shape

    # Verify values are within vocab range
    assert torch.all(batch["input_ids"] >= 0)
    assert torch.all(batch["input_ids"] < vocab_size)

    # Verify attention mask is all ones
    assert torch.all(batch["attention_mask"] == 1)

    # Verify labels match input_ids
    assert torch.equal(batch["input_ids"], batch["labels"])


def test_infinite_dataset():
    """Test that dataset with num_samples=-1 continues generating samples"""
    batch_size = 2
    sequence_length = 4
    vocab_size = 10
    _, dataset = SequenceTestDataset.get_test_sequence_dataloader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        num_samples=-1,  # infinite samples
    )

    # Try getting multiple batches
    iterator = iter(dataset)
    for _ in range(5):  # Get 5 batches
        batch = next(iterator)
        assert batch["input_ids"].shape == (batch_size, sequence_length)
