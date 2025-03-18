import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_cosine_schedule_with_warmup

from diloco_training.models.gpt_neo import get_tiny_gpt_neo
from diloco_training.training.diloco_trainer import train


@pytest.fixture
def setup_training_env():
    # Set up mock distributed environment variables
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Create a tiny model
    model = get_tiny_gpt_neo()

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

    return model, inner_optimizer, outer_optimizer, scheduler


@pytest.fixture
def dummy_dataloader():
    # Create a small dummy dataset
    # Input shape: (batch_size, sequence_length)
    # For tiny GPT-Neo, vocab_size=10 and max_position_embeddings=64
    batch_size = 2
    sequence_length = 8

    input_ids = torch.randint(0, 10, (batch_size, sequence_length))
    attention_mask = torch.ones(batch_size, sequence_length)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([x[0] for x in batch]),
            "attention_mask": torch.stack([x[1] for x in batch]),
            "labels": torch.stack([x[2] for x in batch]),
        },
    )
    return dataloader


def test_train_step(setup_training_env, dummy_dataloader):
    model, inner_optimizer, outer_optimizer, scheduler = setup_training_env

    # Training parameters
    local_rank = 0
    world_size = 1
    local_steps = 2
    batch_size = 2
    per_device_train_batch_size = 2

    # Initial loss
    initial_loss = 100
    for batch in dummy_dataloader:
        outputs = model(**batch)
        initial_loss = outputs.loss.item()
        break

    # Run training
    train(
        model=model,
        train_dataloader=dummy_dataloader,
        inner_optimizer=inner_optimizer,
        outer_optimizer=outer_optimizer,
        scheduler=scheduler,
        local_rank=local_rank,
        world_size=world_size,
        local_steps=local_steps,
        batch_size=batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
    )

    # Check final loss
    final_loss = 100
    for batch in dummy_dataloader:
        outputs = model(**batch)
        final_loss = outputs.loss.item()
        break

    # Assert that loss decreased
    assert final_loss < initial_loss, (
        f"Training did not reduce loss: initial={initial_loss}, final={final_loss}"
    )
