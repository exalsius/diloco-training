import logging
import logging.config
import os

import torch
import torch.distributed as dist

import wandb
from diloco_training.utils.demo_optimizer import DeMo
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


def ddp_setup(
    master_addr="localhost",
    master_port="12355",
    world_size=1,
    local_rank=0,
    device="cuda",
):
    logger.info(
        "Training on %s with local rank: %s, world size: %s",
        device,
        local_rank,
        world_size,
    )
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=local_rank,
    )
    if device == "cuda":
        torch.cuda.set_device(local_rank)

    if local_rank == 0:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="diloco")


def get_offloaded_param(outer_optimizer: torch.optim.Optimizer, device="cuda"):
    """
    Get the offloaded parameters from the outer optimizer.
    """

    if device == "cuda":
        return [
            param.data.detach().clone().to("cpu")
            for group in outer_optimizer.param_groups
            for param in group["params"]
        ]
    else:
        return [
            param.data.detach().clone()
            for group in outer_optimizer.param_groups
            for param in group["params"]
        ]


def initialize_model(model_class, device):
    config, model = model_class()
    model = model.to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    return config, model


def get_optimizers(model, lr, outer_lr, optim_method="demo"):
    inner_optimizer = torch.optim.AdamW(
        model.parameters(), weight_decay=0.1, lr=lr, betas=(0.9, 0.95)
    )

    optimizer_config = {
        "params": model.parameters(),
        "lr": outer_lr,
        "weight_decay": 0.1,
    }

    if optim_method == "demo":
        optimizer_config.update(
            {
                "compression_decay": 0.999,
                "compression_topk": 32,
                "compression_chunk": 64,
            }
        )
        outer_optimizer = DeMo(**optimizer_config)
    elif optim_method in ["sgd", "sgd_quantized"]:
        optimizer_config.update(
            {
                "momentum": 0.9,
                "nesterov": True,
            }
        )
        outer_optimizer = torch.optim.SGD(**optimizer_config)
    else:
        raise ValueError(f"Unknown optimization method: {optim_method}")

    return inner_optimizer, outer_optimizer


def log_stats(
    local_rank,
    step,
    loss_batch,
    world_size,
    batch_size,
    optim_method,
    sync_count,
    bytes_sent,
    bytes_received,
    total_bytes_sent,
    total_bytes_received,
):
    if local_rank == 0:
        mb_sent = bytes_sent / (1024 * 1024)
        mb_received = bytes_received / (1024 * 1024)
        total_mb_sent = total_bytes_sent / (1024 * 1024)
        total_mb_received = total_bytes_received / (1024 * 1024)
        dict_to_log = {
            "Loss": loss_batch.item(),
            "step": step,
            "Perplexity": torch.exp(loss_batch).item(),
            "effective_step": step * world_size,
            "total_samples": step * batch_size * world_size,
            "optim_method": optim_method,
            "sync_count": sync_count,
            "bytes_sent_mb": mb_sent,
            "bytes_received_mb": mb_received,
            "total_bytes_sent_mb": total_mb_sent,
            "total_bytes_received_mb": total_mb_received,
        }
        logger.info("Stats: %s", dict_to_log)
        wandb.log(dict_to_log)


def save_checkpoint(
    model,
    inner_optimizer,
    outer_optimizer,
    scheduler,
    step,
    checkpoint_path,
    local_rank,
    global_rank,
    model_name,
    dataset_name,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "inner_optimizer_state_dict": inner_optimizer.state_dict(),
        "outer_optimizer_state_dict": outer_optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
    }
    checkpoint_file = f"{checkpoint_path}_{model_name}_{dataset_name}_node_{global_rank}_rank_{local_rank}.pth"
    torch.save(checkpoint, checkpoint_file)
    logger.info(
        f"Checkpoint saved at step {step} for global rank {global_rank} and local rank {local_rank}"
    )


def load_checkpoint(
    model,
    inner_optimizer,
    outer_optimizer,
    scheduler,
    checkpoint_path,
    local_rank,
    global_rank,
    model_name,
    dataset_name,
):
    checkpoint_file = f"{checkpoint_path}_{model_name}_{dataset_name}_node_{global_rank}_rank_{local_rank}.pth"
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        inner_optimizer.load_state_dict(checkpoint["inner_optimizer_state_dict"])
        outer_optimizer.load_state_dict(checkpoint["outer_optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = checkpoint["step"]
        logger.info(
            f"Checkpoint loaded from step {step} for global rank {global_rank} and local rank {local_rank}"
        )
        return step
    else:
        logger.info(
            f"No checkpoint found for global rank {global_rank} and local rank {local_rank}, starting from scratch"
        )
        return 0
