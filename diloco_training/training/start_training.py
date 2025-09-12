import argparse
import logging.config
import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import login

from diloco_training.training.distributed_trainer import DistributedTrainer
from diloco_training.training.training_config import TrainingConfig
from diloco_training.utils.diloco_utils import wandb_setup
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger
from diloco_training.utils.metrics_logger import collect_environment_metadata

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
    dist.init_process_group(
        backend=pgroup_backend,
        init_method=f"tcp://10.42.0.63:29500",
        world_size=world_size,
        rank=global_rank    
    )

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
                "heterogeneous": False,  # srnbckr: not sure why we would set heterogenous to false here?
                "local_steps": trainer.local_steps,
                "per_device_train_batch_size": trainer.per_device_train_batch_size,
                "total_steps": trainer.total_steps,
                "checkpoint_interval": trainer.checkpoint_interval,
            }
        )

        # storing config to file (with updated heterogenous values)
        config.save_config(Path("training-config-updated.json"))

    logger.info(f"Starting training with config: {config.to_dict()}")

    trainer = DistributedTrainer(config, local_rank, global_rank, world_size)
    trainer.load_checkpoint()
    trainer.train()
    dist.destroy_process_group()


def load_config_from_env() -> TrainingConfig:
    """
    Load configuration from environment variables only.

    Returns:
        TrainingConfig: Configuration loaded from environment variables
    """
    logger.info("Loading configuration from environment variables")
    try:
        config = TrainingConfig()
        logger.info("Successfully loaded configuration from environment variables")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from environment variables: {e}")
        raise


def load_config_from_file(config_path: Path) -> TrainingConfig:
    """
    Load configuration from a JSON or YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        TrainingConfig: Configuration loaded from file
    """
    logger.info(f"Loading configuration from file: {config_path}")
    config_path = Path(config_path)
    try:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = TrainingConfig.from_config_file(config_path)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from file {config_path}: {e}")
        raise


def load_config_from_args() -> TrainingConfig:
    """
    Load configuration from command line arguments only.

    Returns:
        TrainingConfig: Configuration loaded from command line arguments
    """
    logger.info("Loading configuration from command line arguments")

    parser = argparse.ArgumentParser(description="DiLoCo Training Script")

    # Add all configuration arguments
    parser.add_argument("--model", type=str, required=True, help="Model name.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--local_steps", type=int, default=500, help="Local steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--outer_lr", type=float, default=1e-3, help="Outer learning rate."
    )
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps.")
    parser.add_argument("--total_steps", type=int, default=88000, help="Total steps.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per device.",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Total batch size.")
    parser.add_argument(
        "--optim_method",
        type=str,
        default="sgd",
        choices=["demo", "sgd", "ddp"],
        help="Optimizer.",
    )
    parser.add_argument(
        "--quantization", action="store_true", help="Enable quantization."
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="checkpoint.pth", help="Checkpoint path."
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=512, help="Checkpoint interval."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device."
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="diloco_training",
        help="WandB project name.",
    )
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID.")
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB group.")
    parser.add_argument(
        "--heterogeneous", action="store_true", help="Enable heterogeneous profiling."
    )
    parser.add_argument(
        "--compression_decay", type=float, default=0.9, help="Compression decay."
    )
    parser.add_argument(
        "--compression_topk", type=int, default=32, help="Compression top-k."
    )
    parser.add_argument(
        "--experiment_description",
        type=str,
        default="DiLoCo distributed training experiment",
        help="Description of the experiment run.",
    )
    parser.add_argument(
        "--experiment_tags",
        type=str,
        nargs="*",
        default=[],
        help="Tags for the experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--async_communication",
        action="store_true",
        help="Enable asynchronous communication.",
    )

    args = parser.parse_args()

    try:
        config_dict = {k: v for k, v in vars(args).items() if v is not None}
        config = TrainingConfig.from_args_and_env(**config_dict)
        logger.info("Successfully loaded configuration from command line arguments")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from command line arguments: {e}")
        raise


def load_config() -> TrainingConfig:
    """
    Load configuration based on command line arguments:
    - If --config is specified: load from file
    - If no arguments provided: load from environment variables
    - If other arguments provided: load from command line arguments

    Returns:
        TrainingConfig: Loaded configuration
    """
    parser = argparse.ArgumentParser(description="DiLoCo Training Script")
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (JSON/YAML)"
    )

    # Parse only the --config argument first
    args, remaining = parser.parse_known_args()

    # If --config is specified, load from file
    if args.config:
        return load_config_from_file(args.config)

    # If no other arguments provided, load from environment variables
    if len(remaining) == 0:
        return load_config_from_env()

    # Otherwise, load from command line arguments
    return load_config_from_args()


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
