import json
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY


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

    # Hugging Face upload configuration
    hf_upload: bool = Field(
        default=False, description="Upload model to Hugging Face Hub"
    )
    trained_model_hf_name: Optional[str] = Field(
        default=None, description="Hugging Face Hub model name for upload"
    )

    # Device configuration
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device to use")
    gpu_type: Literal["nvidia", "amd"] = Field(
        default="nvidia", description="GPU type (nvidia or amd)"
    )
    pgroup_backend: Optional[str] = Field(
        default=None,
        description="PyTorch distributed backend (nccl, gloo, ucc, etc.). If None, auto-selects based on device.",
    )

    # Dataset configuration
    dataset_cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for caching datasets. If None, uses HuggingFace default (~/.cache/huggingface/datasets)",
    )

    # Model configuration
    model_cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for caching models. If None, uses HuggingFace default (~/.cache/huggingface/hub)",
    )

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
    min_batch_size: int = Field(
        default=16,
        ge=1,
        description="Minimum batch size for GPU profiling during heterogeneous training",
    )
    max_batch_size: int = Field(
        default=512,
        ge=1,
        description="Maximum batch size limit for GPU profiling during heterogeneous training",
    )
    group_perc_variance: float = Field(
        default=0.15,
        gt=0,
        le=1,
        description="Percentage variance threshold for grouping GPUs in heterogeneous training (0.15 = 15%)",
    )
    compression_decay: float = Field(
        default=0.9, ge=0, le=1, description="Compression decay"
    )
    compression_topk: int = Field(default=32, ge=1, description="Compression top-k")

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

    @model_validator(mode="after")
    def validate_batch_sizes(self):
        """Validate that max_batch_size >= min_batch_size."""
        if self.max_batch_size < self.min_batch_size:
            raise ValueError(
                f"max_batch_size ({self.max_batch_size}) must be >= min_batch_size ({self.min_batch_size})"
            )
        return self

    @model_validator(mode="after")
    def set_pgroup_backend_default(self):
        """Set pgroup_backend default based on device if not specified."""
        if self.pgroup_backend is None:
            self.pgroup_backend = "nccl" if self.device == "cuda" else "gloo"
        return self

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
