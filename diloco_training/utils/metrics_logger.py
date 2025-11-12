import platform
import subprocess
import time
from collections import defaultdict
from typing import Any, Dict

import psutil
import torch
import torch.distributed as dist

import wandb
from diloco_training.utils.exalsius_logger import get_logger

logger = get_logger("metrics_logger")


class MetricsLogger:
    """Comprehensive metrics logger for distributed training"""

    def __init__(
        self,
        local_rank: int,
        global_rank: int,
        world_size: int,
        device: str,
        wandb_logging: bool = True,
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = device
        self.wandb_logging = wandb_logging

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

        if self.wandb_logging:
            wandb.log(
                {
                    f"comm/{sync_type}_bytes_sent": bytes_sent,
                    f"comm/{sync_type}_bytes_sent_mb": bytes_sent / (1024 * 1024),
                    "comm/total_bytes_sent_mb": self.communication_metrics[
                        "total_bytes_sent"
                    ]
                    / (1024 * 1024),
                    "comm/sync_count": self.communication_metrics["sync_count"],
                    "local_rank": self.local_rank,
                    "global_rank": self.global_rank,
                }
            )
        else:
            logger.info("Skipping WandB logging for communication metrics")
            logger.info(f"comm/{sync_type}_bytes_sent: {bytes_sent}")
            logger.info(f"comm/{sync_type}_bytes_sent_mb: {bytes_sent / (1024 * 1024)}")
            logger.info(
                f"comm/total_bytes_sent_mb: {self.communication_metrics['total_bytes_sent'] / (1024 * 1024)}"
            )
            logger.info(f"comm/sync_count: {self.communication_metrics['sync_count']}")
            logger.info(f"local_rank: {self.local_rank}")
            logger.info(f"global_rank: {self.global_rank}")

    def log_reduce_wait_time(self, wait_time: float):
        """Log time spent waiting for reduce step to start"""
        self.communication_metrics["reduce_wait_times"].append(wait_time)
        if self.wandb_logging:
            wandb.log(
                {
                    "comm/reduce_wait_time": wait_time,
                    "comm/avg_reduce_wait_time": sum(
                        self.communication_metrics["reduce_wait_times"]
                    )
                    / len(self.communication_metrics["reduce_wait_times"]),
                    "local_rank": self.local_rank,
                    "global_rank": self.global_rank,
                }
            )
        else:
            logger.info("Skipping WandB logging for reduce wait time")
            logger.info(f"comm/reduce_wait_time: {wait_time}")
            logger.info(
                f"comm/avg_reduce_wait_time: {sum(self.communication_metrics['reduce_wait_times']) / len(self.communication_metrics['reduce_wait_times'])}"
            )
            logger.info(f"local_rank: {self.local_rank}")
            logger.info(f"global_rank: {self.global_rank}")

    def log_reduce_processing_time(self, processing_time: float):
        """Log time spent in reduce processing (excluding wait time)"""
        self.communication_metrics["reduce_processing_times"].append(processing_time)
        if self.wandb_logging:
            wandb.log(
                {
                    "comm/reduce_processing_time": processing_time,
                    "comm/avg_reduce_processing_time": sum(
                        self.communication_metrics["reduce_processing_times"]
                    )
                    / len(self.communication_metrics["reduce_processing_times"]),
                    "local_rank": self.local_rank,
                    "global_rank": self.global_rank,
                }
            )
        else:
            logger.info("Skipping WandB logging for reduce processing time")
            logger.info(f"comm/reduce_processing_time: {processing_time}")
            logger.info(
                f"comm/avg_reduce_processing_time: {sum(self.communication_metrics['reduce_processing_times']) / len(self.communication_metrics['reduce_processing_times'])}"
            )
            logger.info(f"local_rank: {self.local_rank}")
            logger.info(f"global_rank: {self.global_rank}")

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
                "local_rank": self.local_rank,
                "global_rank": self.global_rank,
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

            if self.wandb_logging:
                wandb.log(metrics)
            else:
                logger.info("Skipping WandB logging for system metrics")
                logger.info(f"system/cpu_percent: {cpu_percent}")
                logger.info(
                    f"system/memory_usage_mb: {memory_info.rss / (1024 * 1024)}"
                )
                logger.info(f"system/memory_percent: {memory_percent}")

        except Exception as e:
            logger.warning(f"Failed to log system metrics: {e}")

    def log_throughput_metrics(self, samples_processed: int, time_elapsed: float):
        """Log throughput metrics"""
        samples_per_second = samples_processed / time_elapsed if time_elapsed > 0 else 0
        if self.wandb_logging:
            wandb.log(
                {
                    "performance/samples_per_second": samples_per_second,
                    "performance/time_per_sample": (
                        time_elapsed / samples_processed if samples_processed > 0 else 0
                    ),
                    "local_rank": self.local_rank,
                    "global_rank": self.global_rank,
                }
            )
        else:
            logger.info("Skipping WandB logging for throughput metrics")
            logger.info(f"performance/samples_per_second: {samples_per_second}")
            logger.info(
                f"performance/time_per_sample: {time_elapsed / samples_processed if samples_processed > 0 else 0}"
            )
            logger.info(f"local_rank: {self.local_rank}")
            logger.info(f"global_rank: {self.global_rank}")

    def log_training_metrics(
        self, step: int, loss: float, loss_type: str = "inner", **kwargs
    ):
        """Log training performance metrics"""
        metrics = {
            f"training/{loss_type}_loss": loss,
            f"training/{loss_type}_perplexity": (
                torch.exp(torch.tensor(loss)).item() if loss_type != "gan" else loss
            ),
            "real_step": step,
            "local_rank": self.local_rank,
            "global_rank": self.global_rank,
        }

        # Add any additional metrics
        for key, value in kwargs.items():
            metrics[f"training/{key}"] = value

        if self.wandb_logging:
            wandb.log(metrics)
        else:
            logger.info("Skipping WandB logging for training metrics")
            logger.info(f"training/{loss_type}_loss: {loss}")
            logger.info(
                f"training/{loss_type}_perplexity: {torch.exp(torch.tensor(loss)).item() if loss_type != 'gan' else loss}"
            )
            logger.info(f"real_step: {step}")
            logger.info(f"local_rank: {self.local_rank}")
            logger.info(f"global_rank: {self.global_rank}")

    def log_outer_step_metrics(
        self,
        real_step: int,
        loss: float,
        world_size: int,
        batch_size: int,
        optim_method: str,
        sync_count: int,
        total_bytes_sent: float,
        val_stats: Dict = None,
        local_steps: int = None,
        per_device_train_batch_size: int = None,
    ):
        """Log comprehensive outer loop metrics"""
        try:
            loss_value = loss.item() if hasattr(loss, "item") else loss
        except AttributeError:
            loss_value = loss

        metrics = {
            "training/outer_loss": loss_value,
            "training/outer_perplexity": torch.exp(torch.tensor(loss_value)).item(),
            "real_step": real_step,
            "training/total_steps_all_workers": real_step * world_size,
            "training/total_samples_all_workers": real_step * batch_size * world_size,
            "training/optim_method": optim_method,
            "comm/sync_count": sync_count,
            "comm/total_bytes_sent_mb": total_bytes_sent / (1024 * 1024),
            "training/local_steps": local_steps,
            "training/batch_size": batch_size,
            "training/per_device_train_batch_size": per_device_train_batch_size,
            "local_rank": self.local_rank,
            "global_rank": self.global_rank,
        }

        # Add validation stats if available
        if val_stats:
            for key, value in val_stats.items():
                if isinstance(value, (int, float, bool, str)):
                    metrics[f"eval/{key}"] = value

        if self.wandb_logging:
            wandb.log(metrics)
        else:
            logger.info("Skipping WandB logging for outer step metrics")
            logger.info(f"Outer step metrics: {metrics}")

    def log_timing_summary(self, step: int):
        """Log timing summary for various operations"""
        if not self.timers:
            return

        timing_metrics = {"real_step": step, "rank": self.local_rank}

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

        if self.wandb_logging:
            wandb.log(timing_metrics)
        else:
            logger.info("Skipping WandB logging for timing summary")
            logger.info(f"timing/experiment_duration: {total_duration}")
            for timer_name, times in self.timers.items():
                logger.info(f"timing/{timer_name}_avg: {sum(times) / len(times)}")
                logger.info(f"timing/{timer_name}_total: {sum(times)}")
                logger.info(f"timing/{timer_name}_count: {len(times)}")

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


def log_model_config(model_config: Any, model_name: str, wandb_logging: bool = True):
    """Log model configuration details"""
    if hasattr(model_config, "__dict__"):
        config_dict = {
            f"model_config/{k}": v
            for k, v in model_config.__dict__.items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        }
        config_dict["model_config/name"] = model_name
        if wandb_logging:
            wandb.log(config_dict)
        else:
            logger.info("Skipping WandB logging for model config")
            logger.info(f"model config dict: {config_dict}")
    else:
        if wandb_logging:
            wandb.log({"model_config/name": model_name})
        else:
            logger.info("Skipping WandB logging for model config")
            logger.info(f"model_config/name: {model_name}")


def log_dataset_config(
    dataset_name: str,
    wandb_logging: bool = True,
    **dataset_kwargs,
):
    """Log dataset configuration details"""
    config_dict = {"dataset_config/name": dataset_name}
    for key, value in dataset_kwargs.items():
        if isinstance(value, (int, float, str, bool)):
            config_dict[f"dataset_config/{key}"] = value
    if wandb_logging:
        wandb.log(config_dict)
    else:
        logger.info("Skipping WandB logging for dataset config")
        logger.info(f"dataset config dict: {config_dict}")
