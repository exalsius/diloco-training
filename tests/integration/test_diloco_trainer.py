import os
from pathlib import Path

import pytest

from diloco_training.training.distributed_trainer import DistributedTrainer
from diloco_training.training.training_config import TrainingConfig
from diloco_training.utils.diloco_utils import ddp_setup


@pytest.fixture
def setup_training_env():
    # Set up mock distributed environment variables
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["MASTER_ADDR"] = "localhost"


def test_train(setup_training_env):
    # Training parameters
    local_rank = 0
    global_rank = 0
    world_size = 1
    local_steps = 2
    total_steps = 4
    batch_size = 4
    per_device_train_batch_size = 2

    # Create training configuration
    config = TrainingConfig(
        model="gpt-neo-tiny",  # Use the tiny model from registry
        dataset="test_squence_dataset",  # Use the test dataset from registry
        local_steps=local_steps,
        lr=0.1,
        outer_lr=0.1,
        warmup_steps=2,
        total_steps=total_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        batch_size=batch_size,
        optim_method="sgd",
        quantization=False,
        checkpoint_path=Path("test_checkpoint.pth"),
        checkpoint_interval=10,
        device="cpu",
        heterogeneous=False,
        wandb_project_name="test",
        wandb_run_id=None,
        wandb_group=None,
        experiment_description="Test run",
        experiment_tags=[],
        seed=42,
        wandb_logging=False,
    )

    # Setup distributed training environment
    ddp_setup(device="cpu")
    # wandb_setup(local_rank, None, "test", None)

    # Create trainer and run training
    trainer = DistributedTrainer(config, local_rank, global_rank, world_size)
    trainer.load_checkpoint()
    trainer.train()
