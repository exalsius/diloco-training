import json
import argparse
import logging.config
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


class TrainingConfig(BaseSettings):
    """
    Pydantic model for DiLoCo training configuration arguments.

    Supports loading from:
    - Environment variables (prefixed with DILOCO_)
    - Config files (JSON/YAML)
    - Command line arguments
    """

    # Required arguments
    model: str = Field(default="gpt-neo-x", description="Model name")
    dataset: str = Field(default="c4", description="Dataset name")

    # Training hyperparameters
    local_steps: int = Field(default=500, ge=1, description="Local steps")
    lr: float = Field(default=1e-4, gt=0, description="Learning rate")
    outer_lr: float = Field(default=1e-3, gt=0, description="Outer learning rate")
    warmup_steps: int = Field(default=50, ge=0, description="Warmup steps")
    total_steps: int = Field(default=88000, ge=1, description="Total steps")

    # Batch size configuration
    per_device_train_batch_size: int = Field(
        default=32, ge=1, description="Batch size per device"
    )
    batch_size: int = Field(default=512, ge=1, description="Total batch size")
    min_batch_size: int = Field(default=16, ge=1, description="Minimum batch size")
    max_batch_size: int = Field(default=512, ge=1, description="Maximum batch size")

    # Optimization and training method
    optim_method: Literal["demo", "sgd", "ddp"] = Field(
        default="sgd", description="Optimizer method"
    )
    quantization: bool = Field(default=False, description="Enable quantization")
    async_communication: bool = Field(
        default=False, description="Enable asynchronous communication"
    )

    # Checkpoint configuration
    checkpoint_path: Path = Field(
        default=Path("checkpoint.pth"), description="Checkpoint path"
    )
    checkpoint_interval: int = Field(
        default=512, ge=1, description="Checkpoint interval"
    )

    # Device configuration
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device to use")

    # WandB configuration
    wandb_project_name: str = Field(
        default="diloco_training", description="WandB project name"
    )
    wandb_run_id: Optional[str] = Field(default=None, description="WandB run ID")
    wandb_group: Optional[str] = Field(default=None, description="WandB group")

    # Advanced features
    heterogeneous: bool = Field(
        default=False, description="Enable heterogeneous profiling"
    )
    compression_decay: float = Field(
        default=0.9, ge=0, le=1, description="Compression decay"
    )
    compression_topk: int = Field(default=32, ge=1, description="Compression top-k")
    group_perc_variance: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description="Percentage variance for grouping GPUs during heterogeneous profiling",
    )

    # Experiment metadata
    experiment_description: str = Field(
        default="DiLoCo distributed training experiment",
        description="Description of the experiment run",
    )
    experiment_tags: List[str] = Field(
        default_factory=list,
        description="Tags for the experiment",
        json_schema_extra={"type": "array", "items": {"type": "string"}},
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    wandb_logging: bool = Field(default=True, description="Enable WandB logging")

    # torch.compile settings
    compile_model: bool = Field(
        default=False, description="Enable torch.compile for the model"
    )
    compile_backend: str = Field(
        default="inductor", description="torch.compile backend"
    )
    compile_mode: str = Field(
        default="default",
        description="torch.compile mode (default, reduce-overhead, max-autotune)",
    )

    class Config:
        env_prefix = "DILOCO_"  # Environment variables will be prefixed with DILOCO_
        case_sensitive = False  # Allow case-insensitive env vars
        extra = "ignore"  # Ignore extra fields in config files

    @field_validator("experiment_tags", mode="before")
    @classmethod
    def parse_experiment_tags(cls, v):
        """Parse experiment tags from string if needed."""
        if isinstance(v, str):
            # Handle comma-separated string from environment variable
            return [tag.strip() for tag in v.split(",") if tag.strip()]
        elif isinstance(v, list):
            # Already a list, return as is
            return v
        else:
            # Default to empty list for other types
            return []

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        """Validate that the model name exists in MODEL_REGISTRY."""
        if v not in MODEL_REGISTRY:
            valid_models = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(f"Invalid model: {v}. Must be one of: {valid_models}")
        return v

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v):
        """Validate that the dataset name exists in DATASET_REGISTRY."""
        if v not in DATASET_REGISTRY:
            valid_datasets = ", ".join(DATASET_REGISTRY.keys())
            raise ValueError(f"Invalid dataset: {v}. Must be one of: {valid_datasets}")
        return v

    @classmethod
    def from_config_file(cls, config_path: Path) -> "TrainingConfig":
        """Load configuration from a JSON or YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.suffix.lower() == ".json":
            with open(config_path, "r") as f:
                config_data = json.load(f)
        elif config_path.suffix.lower() in [".yml", ".yaml"]:
            try:
                import yaml

                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML config files. Install with: pip install pyyaml"
                )
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        return cls(**config_data)

    @classmethod
    def from_args_and_env(cls, **kwargs) -> "TrainingConfig":
        """Load configuration from command line args and environment variables.

        This method allows you to pass command line arguments as kwargs,
        and they will override environment variables.
        """
        # Filter out None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Convert config to dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def save_config(self, config_path: Path) -> None:
        """Save configuration to a JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def get_argparse_args(self) -> List[str]:
        """Convert config to argparse-style arguments for compatibility."""
        args = []
        config_dict = self.to_dict()

        for key, value in config_dict.items():
            if key == "config_file":  # Skip config_file itself
                continue
            if isinstance(value, bool):
                if value:
                    args.extend([f"--{key}"])
            elif isinstance(value, list):
                for item in value:
                    args.extend([f"--{key}", str(item)])
            else:
                args.extend([f"--{key}", str(value)])

        return args


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
    # torch.compile options
    parser.add_argument(
        "--compile_model", action="store_true", help="Enable torch.compile."
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        help="torch.compile backend (e.g. inductor, aot_eager).",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        help="torch.compile mode (default, reduce-overhead, max-autotune).",
    )
    parser.add_argument(
        "--min_batch_size", type=int, default=16, help="Minimum batch size."
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=512, help="Maximum batch size."
    )
    parser.add_argument(
        "--wandb_logging", action="store_true", help="Enable WandB logging."
    )
    parser.add_argument(
        "--no-wandb_logging",
        action="store_false",
        dest="wandb_logging",
        help="Disable WandB logging.",
    )
    parser.set_defaults(wandb_logging=True)

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
