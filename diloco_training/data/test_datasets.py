import torch
from torchdata.stateful_dataloader import StatefulDataLoader


class SequenceTestDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_generator, num_samples):
        super().__init__()
        self.data_generator = data_generator
        self.num_samples = num_samples

    def __iter__(self):
        sample_count = 0
        generator = self.data_generator()

        while True:
            if self.num_samples != -1 and sample_count >= self.num_samples:
                break

            sample_count += 1
            yield next(generator)

    @classmethod
    def get_test_sequence_dataloader(
        cls, batch_size=32, sequence_length=100, vocab_size=1000, num_samples=-1
    ):
        """
        Factory method to create a test sequence dataloader with configurable parameters.

        Args:
            batch_size (int): Size of each batch
            sequence_length (int): Length of each sequence
            vocab_size (int): Size of the vocabulary for random tokens
            num_samples (int): Number of samples to generate. -1 for infinite samples.

        Returns:
            DataLoader: Configured dataloader with specified number of random sequences
        """

        def sequence_generator():
            while True:
                # Generate single sequence
                input_ids = torch.randint(0, vocab_size - 1, (sequence_length,))
                attention_mask = torch.ones(sequence_length)
                labels = input_ids.clone()

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        # Create a dataloader with specified batch size
        return sequence_generator, StatefulDataLoader(
            cls(sequence_generator, num_samples),
            batch_size=batch_size,
            num_workers=0,  # Important for infinite datasets
        )
