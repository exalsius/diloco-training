import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import autocast

import wandb
from diloco_training.utils.exalsius_logger import get_logger
from diloco_training.utils.quantization import distributed_reduce_quantized

logger = get_logger("diloco_training")


def ddp_setup(
    master_addr="localhost",
    master_port="12355",
    world_size=1,
    global_rank=0,
    local_rank=0,
    device="cuda",
):
    logger.info(
        "Training on %s with global rank: %s, local rank: %s, world size: %s",
        device,
        global_rank,
        local_rank,
        world_size,
    )
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=global_rank,
        timeout=timedelta(minutes=10),
    )
    if device == "cuda":
        torch.cuda.set_device(local_rank)


def wandb_setup(
    local_rank,
    global_rank,
    user_key,
    project_name,
    run_id=None,
    group="diloco_workers",
    experiment_description=None,
    metadata=None,
    args=None,
):

    # Prepare wandb configuration
    wandb_config = {
        "description": experiment_description
        or "DiLoCo distributed training experiment",
    }

    # Add metadata to config
    if metadata:
        wandb_config.update(metadata)

    # Add all args to config
    if args:
        # Create a serializable version of args
        args_dict = vars(args)
        serializable_args = {}
        for key, value in args_dict.items():
            if isinstance(value, Path):
                serializable_args[key] = str(value)
            else:
                serializable_args[key] = value

        wandb_config.update({f"args/{k}": v for k, v in serializable_args.items()})

    # Set up tags
    tags = getattr(args, "experiment_tags", []) if args else []
    tags.extend(
        [
            f"optim_{args.optim_method}" if args else "unknown",
            f"device_{args.device}" if args else "unknown",
        ]
    )

    if user_key is None:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=project_name,
            config=wandb_config,
            tags=tags,
            notes=experiment_description,
        )
    else:
        wandb.login(key=user_key)
        wandb.init(
            project=project_name,
            group=group,
            name=f"{group}-worker-{global_rank}-{local_rank}",
            id=f"{run_id}{global_rank}",
            resume="allow",
            config=wandb_config,
            tags=tags,
            notes=experiment_description,
        )

    # Enable system monitoring
    if args and args.device == "cuda":
        wandb.watch_called = False  # Reset watch state

    logger.info(f"WandB initialized with description: {experiment_description}")


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

        # Check if this is a GAN model
        is_gan = hasattr(model, "generator") and hasattr(model, "discriminator")

        for step, batch_eval in enumerate(eval_dataloader):
            for key in batch_eval.keys():
                batch_eval[key] = batch_eval[key].to(device)
            with torch.no_grad():
                with autocast(device_type=device, dtype=torch.bfloat16):
                    if is_gan:
                        # For GAN, evaluate discriminator loss
                        model.set_training_mode("discriminator")
                        outputs = model(batch_eval["image"], batch_eval["label"])
                    else:
                        outputs = model(**batch_eval)
                    loss_eval += outputs.loss
            step_eval += 1
            if step > 1000:
                break
        eval_end_time = time.time()
        eval_duration = eval_end_time - eval_start_time
        model.train()

        logger.info(f"Evaluation time: {eval_duration:.2f} seconds")
        loss_eval /= float(step_eval)

        # Log evaluation metrics
        eval_metrics = {
            "duration": eval_duration,
            "steps": step_eval,
            "samples_per_second": (
                step_eval / eval_duration if eval_duration > 0 else 0
            ),
        }

        if is_gan:
            eval_metrics.update(
                {
                    "loss": loss_eval,
                    "d_loss": loss_eval,
                }
            )
            return {"loss": loss_eval, "d_loss": loss_eval, **eval_metrics}
        else:
            perplexity = torch.exp(loss_eval.detach().clone()).item()
            eval_metrics.update(
                {
                    "loss": loss_eval,
                    "perplexity": perplexity,
                }
            )
            return {
                "loss": loss_eval,
                "perplexity": perplexity,
                **eval_metrics,
            }
    else:
        return None


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
    local_steps,
    backend="gloo",
    quantization=False,
    metrics_logger=None,
    sum_local_steps=10,
    async_communication=False,
):
    # Start timing for reduce operation
    reduce_start_time = time.time()

    bytes_sent = 0

    # Time the wait for reduce to start (if there's synchronization overhead)
    reduce_processing_start = time.time()
    if metrics_logger:
        wait_time = reduce_processing_start - reduce_start_time
        metrics_logger.log_reduce_wait_time(wait_time)

    for param_offloaded, param in zip(params_offloaded, main_param):
        param_offloaded_on_device = param_offloaded.data.to(param.device)
        param.grad = (param_offloaded_on_device - param.data) * (
            local_steps / (sum_local_steps / world_size)
        )
        # ReduceOp.AVG with Gloo is not supported, so we use SUM instead and manually average later
        op = dist.ReduceOp.SUM if backend == "gloo" else dist.ReduceOp.AVG
        if optim_method != "demo":
            nbytes = param.grad.nbytes

            if quantization is True:
                # Assuming 8x compression from quantization
                # TODO: This needs to be done in distributed_reduce_quantized
                nbytes = nbytes // 8
                param.grad = distributed_reduce_quantized(param.grad, op=op)
                if backend == "gloo":
                    # Manual averaging after SUM since dist.ReduceOp.AVG is not supported with gloo
                    # and we use dist.ReduceOp.SUM instead
                    param.grad.div_(world_size)

            else:
                dist.all_reduce(param.grad, op=op, async_op=async_communication)
                # Manual averaging after SUM since dist.ReduceOp.AVG is not supported with gloo
                # and we use dist.ReduceOp.SUM instead
                if backend == "gloo":
                    param.grad.div_(world_size)

            # NCCL and gloo are running ring-all reduce
            # In ring all-reduce, each node sends/receives 2(n-1)/n times the data
            # -> bytes sent is typically equal to bytes received
            if world_size > 1:
                bytes_sent += 2 * nbytes * (world_size - 1) / world_size
            else:
                bytes_sent += nbytes
        param.data = param_offloaded_on_device
    if quantization is True and optim_method == "demo":
        outer_optimizer.quantization = True
    outer_optimizer.step()
    if optim_method == "demo":
        if world_size > 1:
            bytes_sent = 2 * outer_optimizer.nbytes * (world_size - 1) / world_size
        else:
            bytes_sent = outer_optimizer.nbytes
    outer_optimizer.zero_grad()

    # Log reduce processing time
    reduce_end_time = time.time()
    processing_time = reduce_end_time - reduce_processing_start
    if metrics_logger:
        metrics_logger.log_reduce_processing_time(processing_time)
        metrics_logger.log_communication_metrics(bytes_sent, "outer_sync")

    return bytes_sent
