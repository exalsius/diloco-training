from typing import Callable, Dict

from .c4_pile import get_c4_pile
from .c4_prime import get_c4_pile_prime
from .test_datasets import SequenceTestDataset

DATASET_REGISTRY: Dict[str, Callable] = {
    "c4_prime": get_c4_pile_prime,
    "c4": get_c4_pile,
    "test_squence_dataset": SequenceTestDataset.get_test_sequence_dataloader,
}
