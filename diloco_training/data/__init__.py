from .c4_pile import get_c4_pile
from .c4_prime import get_c4_pile_prime
from .imagenet import get_imagenet
from .librispeech import get_librispeech
from .obgn_arxiv import get_ogbn_arxiv
from .tencent_ml import get_tencent_ml
from .test_datasets import SequenceTestDataset

DATASET_REGISTRY = {
    "imagenet": get_imagenet,
    "librispeech": get_librispeech,
    "c4_prime": get_c4_pile_prime,
    "c4": get_c4_pile,
    "ogbn_arxiv": get_ogbn_arxiv,
    "tencent_ml": get_tencent_ml,
    "test_squence_dataset": SequenceTestDataset.get_test_sequence_dataloader,
}
