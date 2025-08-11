import time
import psutil
import platform
import subprocess
import torch
import torch.distributed as dist
from collections import defaultdict
from typing import Dict, Any
import wandb
from diloco_training.utils.exalsius_logger import get_logger

logger = get_logger("metrics_logger")


class MetricsLogger:
    """Comprehensive metrics logger for distributed training"""

    def __init__(self, local_rank: int, global_rank: int, world_size: int, device: str):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = device

        # Timing trackers
        self.timers = defaultdict(list)
        self.start_times = {}

        # Communication metrics
        self.communication_metrics = {
            "total_bytes_sent": 0,
            "total_bytes_received": 0,
            "sync_count": 0,
            "reduce_wait_times": [],
            "reduce_processing_times": [],
        }

        # Performance metrics
        self.performance_metrics = defaultdict(list)

        # System metrics baseline
        self.process = psutil.Process()
        self.experiment_start_time = time.time()

    def start_timer(self, name: str):
        """Start a named timer"""
        self.start_times[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End a named timer and return duration"""
        if name not in self.start_times:
            logger.warning(f"Timer {name} was not started")
            return 0.0
        duration = time.time() - self.start_times[name]
        self.timers[name].append(duration)
        del self.start_times[name]
        return duration

    def log_communication_metrics(self, bytes_sent: int, sync_type: str = "outer_sync"):
        """Log communication overhead metrics"""
        self.communication_metrics["total_bytes_sent"] += bytes_sent
        self.communication_metrics["sync_count"] += 1

        wandb.log(
            {
                f"comm/{sync_type}_bytes_sent": bytes_sent,
                f"comm/{sync_type}_bytes_sent_mb": bytes_sent / (1024 * 1024),
                "comm/total_bytes_sent_mb": self.communication_metrics[
                    "total_bytes_sent"
                ]
                / (1024 * 1024),
                "comm/sync_count": self.communication_metrics["sync_count"],
                "rank": self.local_rank,
            }
        )

    def log_reduce_wait_time(self, wait_time: float):
        """Log time spent waiting for reduce step to start"""
        self.communication_metrics["reduce_wait_times"].append(wait_time)
        wandb.log(
            {
                "comm/reduce_wait_time": wait_time,
                "comm/avg_reduce_wait_time": sum(
                    self.communication_metrics["reduce_wait_times"]
                )
                / len(self.communication_metrics["reduce_wait_times"]),
                "rank": self.local_rank,
            }
        )

    def log_reduce_processing_time(self, processing_time: float):
        """Log time spent in reduce processing (excluding wait time)"""
        self.communication_metrics["reduce_processing_times"].append(processing_time)
        wandb.log(
            {
                "comm/reduce_processing_time": processing_time,
                "comm/avg_reduce_processing_time": sum(
                    self.communication_metrics["reduce_processing_times"]
                )
                / len(self.communication_metrics["reduce_processing_times"]),
                "rank": self.local_rank,
            }
        )

    def log_system_metrics(self):
        """Log system resource utilization"""
        try:
            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            metrics = {
                "system/cpu_percent": cpu_percent,
                "system/memory_usage_mb": memory_info.rss / (1024 * 1024),
                "system/memory_percent": memory_percent,
                "rank": self.local_rank,
            }

            # GPU metrics if available
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                gpu_utilization = (
                    torch.cuda.utilization()
                    if hasattr(torch.cuda, "utilization")
                    else 0
                )

                metrics.update(
                    {
                        "system/gpu_memory_allocated_mb": gpu_memory_allocated,
                        "system/gpu_memory_reserved_mb": gpu_memory_reserved,
                        "system/gpu_utilization_percent": gpu_utilization,
                    }
                )

            wandb.log(metrics)

        except Exception as e:
            logger.warning(f"Failed to log system metrics: {e}")

    def log_throughput_metrics(self, samples_processed: int, time_elapsed: float):
        """Log throughput metrics"""
        samples_per_second = samples_processed / time_elapsed if time_elapsed > 0 else 0
        wandb.log(
            {
                "performance/samples_per_second": samples_per_second,
                "performance/time_per_sample": (
                    time_elapsed / samples_processed if samples_processed > 0 else 0
                ),
                "rank": self.local_rank,
            }
        )

    def log_training_metrics(
        self, step: int, loss: float, loss_type: str = "inner", **kwargs
    ):
        """Log training performance metrics"""
        metrics = {
            f"training/{loss_type}_loss": loss,
            f"training/{loss_type}_perplexity": (
                torch.exp(torch.tensor(loss)).item() if loss_type != "gan" else loss
            ),
            "step": step,
            "rank": self.local_rank,
        }

        # Add any additional metrics
        for key, value in kwargs.items():
            metrics[f"training/{key}"] = value

        wandb.log(metrics)

    def log_timing_summary(self, step: int):
        """Log timing summary for various operations"""
        if not self.timers:
            return

        timing_metrics = {"step": step, "rank": self.local_rank}

        for timer_name, times in self.timers.items():
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                timing_metrics.update(
                    {
                        f"timing/{timer_name}_avg": avg_time,
                        f"timing/{timer_name}_total": total_time,
                        f"timing/{timer_name}_count": len(times),
                    }
                )

        # Calculate total experiment duration
        total_duration = time.time() - self.experiment_start_time
        timing_metrics["timing/experiment_duration"] = total_duration

        wandb.log(timing_metrics)

    def reset_timers(self):
        """Reset all timing metrics"""
        self.timers.clear()
        self.start_times.clear()


def collect_environment_metadata(args) -> Dict[str, Any]:
    """Collect comprehensive environment and reproducibility metadata"""
    metadata = {
        # System information
        "system/platform": platform.platform(),
        "system/python_version": platform.python_version(),
        "system/cpu_count": psutil.cpu_count(),
        "system/memory_total_gb": psutil.virtual_memory().total / (1024**3),
        # PyTorch information
        "pytorch/version": torch.__version__,
        "pytorch/cuda_available": torch.cuda.is_available(),
        "pytorch/distributed_available": dist.is_available(),
        # Training configuration
        "training/seed": getattr(args, "seed", 42),
        "training/device": args.device,
        "training/world_size": getattr(args, "world_size", 1),
        "training/local_steps": args.local_steps,
        "training/total_steps": args.total_steps,
        "training/batch_size": args.batch_size,
        "training/per_device_batch_size": args.per_device_train_batch_size,
        "training/lr": args.lr,
        "training/outer_lr": args.outer_lr,
        "training/optim_method": args.optim_method,
        "training/quantization": args.quantization,
        # Model and dataset
        "model/name": args.model,
        "dataset/name": args.dataset,
    }

    # GPU information if available
    if torch.cuda.is_available():
        metadata.update(
            {
                "system/gpu_count": torch.cuda.device_count(),
                "system/gpu_name": torch.cuda.get_device_name(0),
                "pytorch/cuda_version": torch.version.cuda,
                "pytorch/cudnn_version": torch.backends.cudnn.version(),
            }
        )

    # Git information if available
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        git_branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )
        metadata.update(
            {
                "git/commit": git_commit,
                "git/branch": git_branch,
            }
        )
    except Exception:
        pass  # Git not available or not in a git repository

    return metadata


def log_model_config(model_config: Any, model_name: str):
    """Log model configuration details"""
    if hasattr(model_config, "__dict__"):
        config_dict = {
            f"model_config/{k}": v
            for k, v in model_config.__dict__.items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        }
        config_dict["model_config/name"] = model_name
        wandb.log(config_dict)
    else:
        wandb.log({"model_config/name": model_name})


def log_dataset_config(dataset_name: str, **dataset_kwargs):
    """Log dataset configuration details"""
    config_dict = {"dataset_config/name": dataset_name}
    for key, value in dataset_kwargs.items():
        if isinstance(value, (int, float, str, bool)):
            config_dict[f"dataset_config/{key}"] = value
    wandb.log(config_dict)
