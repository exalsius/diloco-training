from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from argparse import ArgumentTypeError

def __dataset_type(value):
    if value not in DATASET_REGISTRY:
        valid_datasets = ", ".join(DATASET_REGISTRY.keys())
        raise ArgumentTypeError(
            f"Invalid dataset: {value}. Must be one of: {valid_datasets}"
        )
    return value


def __model_type(value):
    if value not in MODEL_REGISTRY:
        valid_models = ", ".join(MODEL_REGISTRY.keys())
        raise ArgumentTypeError(
            f"Invalid model: {value}. Must be one of: {valid_models}"
        )
    return value


def __validate_optimizer(value):
    valid_optimizers = ["demo", "sgd", "sgd_quantized"]
    if value not in valid_optimizers:
        raise ArgumentTypeError(
            f"Invalid optimizer: {value}. Must be one of: {', '.join(valid_optimizers)}"
        )
    return value