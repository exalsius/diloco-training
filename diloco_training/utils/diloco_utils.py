import logging
import logging.config
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

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
        timeout=timedelta(hours=10),
    )
    if device == "cuda":
        torch.cuda.set_device(local_rank)

    if local_rank == 0:
        wandb.login(key="6800d2a81420c3adf2b8f658e79f63bd4003b3e1")
        wandb.init(project="diloco_training")


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


def evaluate_model(eval_dataloader, model, global_rank, local_rank):
    if global_rank == 0 and local_rank == 0:
        logger.info("Starting evaluation...")
        loss_eval = 0
        step_eval = 0
        eval_start_time = time.time()
        for step, batch_eval in enumerate(eval_dataloader):
            for key in batch_eval.keys():
                batch_eval[key] = batch_eval[key].to("cuda")

            with torch.no_grad():
                model.eval()
                outputs = model(**batch_eval)
                loss_eval += outputs.loss
            step_eval += 1
            if step >= 1000:
                break
        eval_end_time = time.time()
        model.train()

        logger.info(f"Evaluation time: {eval_end_time - eval_start_time:.2f} seconds")
        loss_eval /= step_eval
        return {"eval_loss": loss_eval, "eval_perplexity": torch.exp(loss_eval)}
    else:
        return None


def initialize_model(model_class, device, optim_method=None, local_rank=None):
    config, model = model_class()
    model = model.to(device)

    if optim_method is None:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    else:
        model = DDP(model, device_ids=[local_rank] if device == "cuda" else None)
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
    effective_step,
    loss_batch,
    world_size,
    batch_size,
    optim_method,
    sync_count,
    total_bytes_sent,
    total_bytes_received,
    val_stats,
    local_steps,
    per_device_train_batch_size,
):
    if local_rank == 0:
        total_mb_sent = total_bytes_sent / (1024 * 1024)
        total_mb_received = total_bytes_received / (1024 * 1024)
        try:
            loss_b = loss_batch.item()
        except AttributeError:
            loss_b = loss_batch
        dict_to_log = {
            "Loss": loss_b,
            "effective_step": effective_step,
            "Perplexity": torch.exp(loss_batch).item(),
            "total_steps": effective_step * world_size,
            "total_samples": effective_step * batch_size * world_size,
            "optim_method": optim_method,
            "sync_count": sync_count,
            "total_bytes_sent_mb": total_mb_sent,
            "total_bytes_received_mb": total_mb_received,
            "val_stats": val_stats,
            "local_steps": local_steps,
            "batch_size": batch_size,
            "per_device_train_batch_size": per_device_train_batch_size,
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
    optim_method,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "inner_optimizer_state_dict": inner_optimizer.state_dict(),
        "outer_optimizer_state_dict": (
            outer_optimizer.state_dict() if outer_optimizer is not None else None
        ),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "optim_method": optim_method,
    }
    checkpoint_file = f"{checkpoint_path}_{model_name}_{dataset_name}_node_{global_rank}_rank_{local_rank}_optim_{optim_method}.pth"
    torch.save(checkpoint, checkpoint_file)
    logger.info(
        f"Checkpoint saved at step {step} for global rank {global_rank} and local rank {local_rank}, optim method {optim_method}"
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
