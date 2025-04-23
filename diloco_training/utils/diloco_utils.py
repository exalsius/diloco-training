import logging
import logging.config
import math
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.amp import autocast

import wandb
from diloco_training.utils.demo_optimizer import DeMo
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger
from diloco_training.utils.quantization import distributed_reduce_quantized

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


def wandb_setup(
    local_rank, user_key, project_name, run_id=None, group="diloco_workers"
):
    if user_key is None:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=project_name)
    else:
        wandb.login(key=user_key)
        wandb.init(
            project=project_name,
            group=group,
            name=f"{group}-worker-{local_rank}",
            id=run_id,
            resume="allow",
        )


def compute_l2_norm(current_params, reference_params, normalize=True):
    """Compute the L2 norm of the difference between current and reference parameters."""
    l2_norm = 0.0
    total_params = 0
    for current, reference in zip(current_params, reference_params):
        l2_norm += torch.norm(current - reference).item() ** 2
        total_params += current.numel()
    l2_norm = l2_norm**0.5  # Take the square root
    if normalize:
        l2_norm /= (
            total_params**0.5
        )  # Normalize by the square root of the number of parameters
    return l2_norm


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


def evaluate_model(eval_dataloader, model, global_rank, local_rank, device):
    if global_rank == 0 and local_rank == 0:
        logger.info("Starting evaluation...")
        loss_eval: float = 0.0
        step_eval: int = 0
        eval_start_time = time.time()
        model.eval()

        for step, batch_eval in enumerate(eval_dataloader):
            for key in batch_eval.keys():
                batch_eval[key] = batch_eval[key].to(device)
            with torch.no_grad():
                with autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(**batch_eval)
                    loss_eval += outputs.loss
            step_eval += 1
            if step >= 1000:
                break
        eval_end_time = time.time()
        model.train()

        logger.info(f"Evaluation time: {eval_end_time - eval_start_time:.2f} seconds")
        loss_eval /= float(step_eval)
        return {
            "eval_loss": loss_eval,
            "eval_perplexity": torch.exp(torch.tensor(loss_eval)).item(),
        }
    else:
        return None


def initialize_model(model_class, device, optim_method=None, local_rank=None):
    config, model = model_class()
    model = model.to(device)

    # if optim_method is None:
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    # else:
    #     model = DDP(model, device_ids=[local_rank] if device == "cuda" else None)
    return config, model


def cosine_schedule_inverse_with_warmup(
    local_steps, target_steps, warmup_steps, total_steps
):
    """
    Generates a schedule that starts with a fixed value during warmup and then increases
    using an inverse cosine schedule.

    Args:
        local_steps (int): Fixed value during warmup.
        target_steps (int): Target value after warmup.
        warmup_steps (int): Number of warm-up steps.
        total_steps (int): Total number of training steps.

    Returns:
        list: A list of values for each step.
    """
    schedule = []
    for step in range(total_steps):
        if step < warmup_steps:
            # Fixed value during warm-up phase
            value = local_steps
        else:
            # Inverse cosine increase phase
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            value = local_steps + (target_steps - local_steps) * 0.5 * (
                1 - math.cos(math.pi * progress)
            )
        schedule.append(int(value))
    return schedule


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


def log_inner_stats(
    local_rank, real_step, loss_batch, sync_count, l2_norm, normalized_l2_norm
):
    dict_to_log = {
        "local_rank": local_rank,
        "inner_loss": loss_batch.item(),
        "real_step": real_step,
        "inner_perplexity": torch.exp(loss_batch).item(),
        "sync_count": sync_count,
        "param_drift_l2_norm": l2_norm,
        "param_drift_normalized_l2_norm": normalized_l2_norm,
    }

    wandb.log(dict_to_log)


def log_stats(
    local_rank,
    real_step,
    loss_batch,
    world_size,
    batch_size,
    optim_method,
    sync_count,
    total_bytes_sent,
    val_stats,
    local_steps,
    per_device_train_batch_size,
):
    if local_rank == 0:
        total_mb_sent = total_bytes_sent / (1024 * 1024)
        try:
            loss_b = loss_batch.item()
        except AttributeError:
            loss_b = loss_batch
        dict_to_log = {
            "Loss": loss_b,
            "real_step": real_step,
            "Perplexity": torch.exp(loss_batch).item(),
            "total_steps_all_workers": real_step * world_size,
            "total_samples_all_workers": real_step * batch_size * world_size,
            "optim_method": optim_method,
            "sync_count": sync_count,
            "total_bytes_sent_mb": total_mb_sent,
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
        f"Checkpoint saved for global rank {global_rank} and local rank {local_rank}, optim method {optim_method}"
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
    optim_method,
):
    checkpoint_file = f"{checkpoint_path}_{model_name}_{dataset_name}_node_{global_rank}_rank_{local_rank}_optim_{optim_method}.pth"
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
        return step, model, inner_optimizer, outer_optimizer, scheduler
    else:
        logger.info(
            f"No checkpoint found for global rank {global_rank} and local rank {local_rank}, starting from scratch"
        )
        return 0, None, None, None, None


def prepare_batch(batch, device="cuda"):
    for key in batch.keys():
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def forward_and_compute_loss(model, batch, gradient_accumulation_steps):
    outputs = model(**batch)
    loss = outputs.loss / gradient_accumulation_steps
    return loss


def update_inner_optimizer(inner_optimizer, scheduler, model, scaler):
    scaler.unscale_(optimizer=inner_optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
    scaler.step(optimizer=inner_optimizer)
    scaler.update()
    scheduler.step()
    inner_optimizer.zero_grad()


def update_outer_optimizer(
    params_offloaded,
    main_param,
    optim_method,
    world_size,
    outer_optimizer,
    device="cuda",
):
    bytes_sent = 0
    for param_offloaded, param in zip(params_offloaded, main_param):
        param_offloaded_on_device = param_offloaded.data.to(param.device)
        param.grad = param_offloaded_on_device - param.data
        # ReduceOp.AVG with Gloo is not supported, so we use SUM instead and manually average later
        op = dist.ReduceOp.AVG if device == "cuda" else dist.ReduceOp.SUM
        if optim_method != "demo":
            is_quantized = optim_method == "sgd_quantized"
            nbytes = param.grad.nbytes

            if is_quantized:
                # Assuming 8x compression from quantization
                # TODO: This needs to be done in distributed_reduce_quantized
                nbytes = nbytes // 8

                param.grad = distributed_reduce_quantized(param.grad, op=op)
                if device == "cpu":
                    # Manual averaging after SUM since dist.ReduceOp.AVG is not supported with gloo
                    # and we use dist.ReduceOp.SUM instead
                    param.grad.div_(world_size)

            else:
                dist.all_reduce(param.grad, op=op)
                # Manual averaging after SUM since dist.ReduceOp.AVG is not supported with gloo
                # and we use dist.ReduceOp.SUM instead
                if device == "cpu":
                    param.grad.div_(world_size)

            # NCCL and gloo are running ring-all reduce
            # In ring all-reduce, each node sends/receives 2(n-1)/n times the data
            # -> bytes sent is typically equal to bytes received
            if world_size > 1:
                bytes_sent += 2 * nbytes * (world_size - 1) / world_size
            else:
                bytes_sent += nbytes

            param.data = param_offloaded_on_device
    outer_optimizer.step()
    if optim_method == "demo":
        if world_size > 1:
            bytes_sent = 2 * outer_optimizer.nbytes * (world_size - 1) / world_size
        else:
            bytes_sent = outer_optimizer.nbytes
    outer_optimizer.zero_grad()

    return bytes_sent
