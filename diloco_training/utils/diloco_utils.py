import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import autocast
from tqdm import tqdm

import wandb
from diloco_training.utils.exalsius_logger import get_logger
from diloco_training.utils.quantization import (
    distributed_reduce_quantized,
    distributed_reduce_lattice,
)
from diloco_training.utils.lattice_compression import compress_tensor, decompress_tensor

logger = get_logger("diloco_training")

# Error feedback buffers: keyed by (worker_rank, param_index) → residual tensor.
# Persists across outer steps to accumulate compression residuals.
_error_feedback_buffers: dict[tuple[int, int], torch.Tensor] = {}


def reset_error_feedback():
    """Clear all error feedback buffers (e.g., at start of training)."""
    _error_feedback_buffers.clear()


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

        for step, batch_eval in enumerate(eval_dataloader):
            for key in batch_eval.keys():
                batch_eval[key] = batch_eval[key].to(device)
            with torch.no_grad():
                with autocast(device_type=device, dtype=torch.bfloat16):
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

    # Check for NaN/Inf loss and log warning
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning(
            f"Detected NaN/Inf loss: {loss.item()}. This batch will be skipped."
        )
        # Return zero loss to skip this batch's gradient contribution
        return torch.zeros_like(loss, requires_grad=True)

    return loss


def update_inner_optimizer(inner_optimizer, scheduler, model, scaler):
    scaler.unscale_(optimizer=inner_optimizer)

    # Check for NaN gradients before clipping
    has_nan_grad = False
    for param in model.parameters():
        if param.grad is not None and (
            torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
        ):
            has_nan_grad = True
            break

    if has_nan_grad:
        logger.warning("Detected NaN/Inf gradients. Skipping optimizer step.")
        inner_optimizer.zero_grad()
        scaler.update()  # Still need to update scaler state
        return

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
    compression_method="none",
    compression_error_feedback=False,
    metrics_logger=None,
    sum_local_steps=10,
    async_communication=False,
):
    # Start timing for reduce operation
    reduce_start_time = time.time()

    bytes_sent = 0
    compression_cos_sims = []
    compression_time_total = 0.0

    # Time the wait for reduce to start (if there's synchronization overhead)
    reduce_processing_start = time.time()
    if metrics_logger:
        wait_time = reduce_processing_start - reduce_start_time
        metrics_logger.log_reduce_wait_time(wait_time)

    # Get total number of parameters for progress tracking
    total_params = len(params_offloaded)
    logger.info(f"Starting outer optimizer sync: {total_params} parameters to process")

    # Phase 1: Computing gradients and performing all_reduce
    logger.info(
        "Phase 1/3: Computing parameter gradients and performing all_reduce communication..."
    )
    grad_compute_start = time.time()

    for param_idx, (param_offloaded, param) in enumerate(tqdm(
        zip(params_offloaded, main_param),
        total=total_params,
        desc="Syncing parameters",
        disable=False,
        unit="param",
        ncols=80,
        mininterval=5.0,
        maxinterval=10.0,
        file=sys.stdout,
        position=0,
        leave=True,
    )):
        param_offloaded_on_device = param_offloaded.data.to(param.device)
        param.grad = (param_offloaded_on_device - param.data) * (
            local_steps / (sum_local_steps / world_size)
        )
        # ReduceOp.AVG with Gloo is not supported, so we use SUM instead and manually average later
        op = dist.ReduceOp.SUM if backend == "gloo" else dist.ReduceOp.AVG
        if optim_method != "demo":
            nbytes = param.grad.nbytes

            # Resolve effective compression: new compression_method takes priority
            effective_compression = compression_method
            if effective_compression == "none" and quantization is True:
                effective_compression = "int8"

            if effective_compression == "lattice":
                nbytes = nbytes // 8  # ~8x compression
                compress_start = time.time()

                # Error feedback: add accumulated residual before compression
                if compression_error_feedback:
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    ef_key = (rank, param_idx)
                    if ef_key in _error_feedback_buffers:
                        param.grad.add_(_error_feedback_buffers[ef_key].to(param.grad.device))
                    # Compress locally to compute residual
                    pre_compress = param.grad.clone()
                    compressed_local = compress_tensor(param.grad)
                    reconstructed_local = decompress_tensor(compressed_local, device=param.grad.device)
                    _error_feedback_buffers[ef_key] = (pre_compress - reconstructed_local).cpu()

                # Track compression quality (sample every 10th param to avoid overhead)
                if param_idx % 10 == 0 and param.grad.numel() >= 8:
                    orig_grad = param.grad.clone()
                    c = compress_tensor(param.grad)
                    r = decompress_tensor(c, device=param.grad.device)
                    cos = torch.nn.functional.cosine_similarity(
                        orig_grad.flatten().unsqueeze(0), r.flatten().unsqueeze(0)
                    ).item()
                    compression_cos_sims.append(cos)

                param.grad = distributed_reduce_lattice(param.grad, op=op)
                if backend == "gloo":
                    param.grad.div_(world_size)
                compression_time_total += time.time() - compress_start

            elif effective_compression == "int8":
                nbytes = nbytes // 4  # ~4x compression
                param.grad = distributed_reduce_quantized(param.grad, op=op)
                if backend == "gloo":
                    param.grad.div_(world_size)

            else:
                if backend == "gloo":
                    logger.debug(
                        f"Using gloo backend with CPU offload - gradient shape: {param.grad.shape}, "
                        f"size: {param.grad.nbytes / (1024**2):.4f} MB, async: {async_communication}"
                    )
                    # Create CPU copy for all_reduce
                    grad_cpu = param.grad.cpu()
                    dist.all_reduce(grad_cpu, op=op, async_op=async_communication)
                    # Copy result back to original CUDA gradient
                    param.grad.copy_(grad_cpu)
                    # Manual averaging after SUM
                    param.grad.div_(world_size)
                    logger.debug(
                        f"Gloo all_reduce completed - applied manual averaging over {world_size} workers"
                    )
                else:
                    dist.all_reduce(param.grad, op=op, async_op=async_communication)

            # NCCL and gloo are running ring-all reduce
            # In ring all-reduce, each node sends/receives 2(n-1)/n times the data
            # -> bytes sent is typically equal to bytes received
            if world_size > 1:
                bytes_sent += 2 * nbytes * (world_size - 1) / world_size
            else:
                bytes_sent += nbytes
        param.data = param_offloaded_on_device

    grad_compute_time = time.time() - grad_compute_start
    logger.info(
        f"Phase 1 completed in {grad_compute_time:.2f}s - All parameters synced"
    )

    # Phase 2: Handle quantization if needed
    if quantization is True and optim_method == "demo":
        logger.info("Phase 2/3: Setting quantization for DeMo optimizer...")
        outer_optimizer.quantization = True
    else:
        logger.info("Phase 2/3: Skipping quantization setup")

    # Phase 3: Running outer optimizer step
    logger.info("Phase 3/3: Running outer optimizer step...")
    optimizer_step_start = time.time()
    outer_optimizer.step()
    optimizer_step_time = time.time() - optimizer_step_start
    logger.info(
        f"Phase 3 completed in {optimizer_step_time:.2f}s - Optimizer step finished"
    )

    if optim_method == "demo":
        if world_size > 1:
            bytes_sent = 2 * outer_optimizer.nbytes * (world_size - 1) / world_size
        else:
            bytes_sent = outer_optimizer.nbytes
    outer_optimizer.zero_grad()

    # Log reduce processing time
    reduce_end_time = time.time()
    processing_time = reduce_end_time - reduce_processing_start
    total_time = reduce_end_time - reduce_start_time

    logger.info(
        f"Outer optimizer sync completed - Total time: {total_time:.2f}s, "
        f"Data transferred: {bytes_sent / (1024**2):.2f} MB"
    )

    if metrics_logger:
        metrics_logger.log_reduce_processing_time(processing_time)
        metrics_logger.log_communication_metrics(bytes_sent, "outer_sync")

        # Log compression quality metrics
        if compression_cos_sims:
            avg_cos_sim = sum(compression_cos_sims) / len(compression_cos_sims)
            effective = compression_method if compression_method != "none" else ("int8" if quantization else "none")
            ratio = 8.0 if effective == "lattice" else (4.0 if effective == "int8" else 1.0)
            metrics_logger.log_compression_metrics(
                compression_method=effective,
                compression_ratio=ratio,
                cosine_similarity=avg_cos_sim,
                error_feedback=compression_error_feedback,
                compression_time=compression_time_total,
            )

    return bytes_sent
