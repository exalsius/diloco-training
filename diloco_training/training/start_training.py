import logging.config
import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import login

from diloco_training.training.distributed_trainer import DistributedTrainer
from diloco_training.training.training_config import TrainingConfig, load_config
from diloco_training.utils.diloco_utils import wandb_setup
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger
from diloco_training.utils.metrics_logger import collect_environment_metadata

# ignore warnings
import warnings

warnings.filterwarnings("ignore")

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


def init_and_start_training(config: TrainingConfig):
    """
    Initialize and start distributed training using the provided configuration.

    Args:
        config: TrainingConfig object containing all training parameters
    """
    # Setup distributed training from environment variables
    hostname = socket.gethostname()
    master_address = os.environ.get("MASTER_ADDR", None)
    master_port = os.environ["MASTER_PORT"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    wandb_user_key = os.environ.get("WANDB_USER_KEY", None)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
    pgroup_backend = "nccl" if config.device == "cuda" else "gloo"
    os.environ["HF_HUB_HTTP_TIMEOUT"] = "60"
    os.environ["HF_HUB_DOWNLOAD_RETRY"] = "100"

    logger.info("Distributed training configuration:")
    logger.info(
        f"{hostname=} {local_rank=} {global_rank=} {world_size=} {master_port=} {master_address=}"
    )
    dist.init_process_group(backend=pgroup_backend)

    if config.device == "cuda":
        torch.cuda.set_device(local_rank)

    # Collect environment metadata for logging
    env_metadata = collect_environment_metadata(config)

    # Initialize WandB with enhanced metadata
    wandb_setup(
        local_rank=local_rank,
        global_rank=global_rank,
        user_key=wandb_user_key,
        project_name=config.wandb_project_name,
        run_id=config.wandb_run_id,
        group=config.wandb_group if config.wandb_group else "diloco_workers",
        experiment_description=config.experiment_description,
        metadata=env_metadata,
        args=config,
    )

    # HF login
    if hf_token:
        login(token=hf_token)

    # initialize trainer
    trainer = DistributedTrainer(config, local_rank, global_rank, world_size)

    # run heterogenous profiling
    if config.heterogeneous:
        logger.info("Running heterogeneous profiling")
        trainer.heterogeneous_profiling()
        config = config.model_copy(
            update={
                "heterogeneous": False,
                "local_steps": trainer.config.local_steps,
                "per_device_train_batch_size": trainer.config.per_device_train_batch_size,
                "total_steps": trainer.config.total_steps,  # total_steps is now distributed properly
                "checkpoint_interval": trainer.config.checkpoint_interval,
            }
        )

        # storing config to file (with updated heterogenous values)
        config.save_config(Path("training-config-updated.json"))

    logger.info(f"Starting training with config: {config.to_dict()}")

    trainer = DistributedTrainer(config, local_rank, global_rank, world_size)
    trainer.load_checkpoint()
    trainer.train()
    dist.destroy_process_group()


def main():
    """
    Main entry point for the training script.
    """
    logger.info("Starting training script")
    try:
        # Load configuration
        config: TrainingConfig = load_config()

        # Log the final configuration
        logger.info(f"Starting training with configuration: {config.model_dump()}")

        # Start training
        init_and_start_training(config)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
