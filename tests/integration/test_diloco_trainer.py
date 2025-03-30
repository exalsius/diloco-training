import os

import pytest
import torch
from transformers import get_cosine_schedule_with_warmup

from diloco_training.data.test_datasets import SequenceTestDataset
from diloco_training.models.gpt_neo import get_tiny_gpt_neo
from diloco_training.training.diloco_trainer import train
from diloco_training.utils.diloco_utils import ddp_setup


@pytest.fixture
def setup_training_env():
    # Set up mock distributed environment variables
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Create a tiny model
    config, model = get_tiny_gpt_neo()

    # Create dummy optimizers
    inner_optimizer = torch.optim.AdamW(
        model.parameters(), weight_decay=0.1, lr=0.1, betas=(0.9, 0.95)
    )
    outer_optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, nesterov=True
    )

    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer, num_warmup_steps=2, num_training_steps=10
    )

    return config, model, inner_optimizer, outer_optimizer, scheduler


def test_train(setup_training_env):
    model_config, model, inner_optimizer, outer_optimizer, scheduler = (
        setup_training_env
    )
    batch_size = 4
    # Training parameters
    local_rank = 0
    global_rank = 0
    world_size = 1
    local_steps = 2
    total_steps = 4
    per_device_train_batch_size = 2

    _, test_dataloader = SequenceTestDataset.get_test_sequence_dataloader(
        batch_size=batch_size,
        vocab_size=model_config.vocab_size,
        num_samples=-1,
    )

    ddp_setup(device="cpu")

    # Run training
    train(
        model=model,
        train_dataloader=test_dataloader,
        val_dataloader=test_dataloader,
        inner_optimizer=inner_optimizer,
        outer_optimizer=outer_optimizer,
        scheduler=scheduler,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        total_steps=total_steps,
        local_steps=local_steps,
        batch_size=batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        optim_method="sgd",
        checkpoint_path="",
        checkpoint_interval=10,
        model_name="test_model",
        dataset_name="test_dataset",
        device="cpu",
        start_step=0,
    )
