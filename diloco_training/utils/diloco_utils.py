import logging
import os

import torch
import torch.distributed as dist
import wandb
from torch.distributed import init_process_group

from diloco_training.utils.demo_optimizer import DeMo
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


def ddp_setup():
    logger.info(
        "Local rank: %s, world size: %s",
        os.environ["LOCAL_RANK"],
        os.environ["WORLD_SIZE"],
    )
    init_process_group(
        backend="nccl",
        rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_offloaded_param(outer_optimizer: torch.optim.Optimizer):
    return [
        param.data.detach().clone().to("cpu")
        for group in outer_optimizer.param_groups
        for param in group["params"]
    ]


def initialize_model(model_class, device):
    model = model_class().to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    return model


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
