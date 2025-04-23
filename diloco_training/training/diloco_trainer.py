import argparse
import logging.config
import os
import time  # Add this import
from typing import Callable, Optional
import torch
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

import wandb
from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.utils.args_types import (
    dataset_type,
    model_type,
    validate_optimizer,
)
from diloco_training.utils.diloco_utils import (
    compute_l2_norm,
    cosine_schedule_inverse_with_warmup,
    ddp_setup,
    evaluate_model,
    forward_and_compute_loss,
    get_offloaded_param,
    get_optimizers,
    initialize_model,
    load_checkpoint,
    log_inner_stats,
    log_stats,
    prepare_batch,
    save_checkpoint,
    update_inner_optimizer,
    update_outer_optimizer,
    wandb_setup,
)
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


def train(
    model,
    train_dataloader,
    val_dataloader,
    inner_optimizer,
    outer_optimizer,
    scheduler,
    local_rank,
    global_rank,
    world_size,
    local_steps,
    batch_size,
    per_device_train_batch_size,
    total_steps=-1,
    optim_method="demo",
    checkpoint_path="checkpoint.pth",
    checkpoint_interval=1000,
    model_name="model",
    dataset_name="dataset",
    device="cuda",
    start_step=0,
    warmup_steps=1000,
):
    model.train()
    loss_batch = 0
    gradient_accumulation_steps = batch_size // per_device_train_batch_size
    total_bytes_sent, sync_count = 0, 0
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    params_offloaded = get_offloaded_param(outer_optimizer, device=device)
    reference_params = [
        param.clone().detach() for param in model.parameters()
    ]  # Initialize reference parameters

    scaler = GradScaler(device=device)  # Initialize GradScaler for mixed precision training

    if optim_method == "demo":
        local_steps_scheduler = cosine_schedule_inverse_with_warmup(
            local_steps, local_steps * 4, warmup_steps, total_steps
        )
    else:
        local_steps_scheduler = cosine_schedule_inverse_with_warmup(
            local_steps, local_steps, warmup_steps, total_steps
        )
    count_inner_optimizer_steps = 0
    for step, batch in enumerate(train_dataloader):

        if step < start_step:
            continue

        if step == start_step:
            logger.info(f"Starting training from step {start_step}...")

        real_step = (step + 1) // gradient_accumulation_steps
        step_within_grad_acc = (step + 1) % gradient_accumulation_steps

        # Measure time for preparing the batch
        batch = prepare_batch(batch, device=device)
    
        # Measure time for forward pass and loss computation
        with autocast(device_type=device, dtype=torch.bfloat16 if device=="cuda" else torch.float16):  # Enable mixed precision for forward pass
            loss = forward_and_compute_loss(model, batch, gradient_accumulation_steps)
        loss_batch += loss.detach()

        # Measure time for backward pass
        scaler.scale(loss).backward()  # Scale the loss before backward pass

        if step_within_grad_acc == 0:
            # Measure time for optimizer update
            logger.info(
                f"Local rank {local_rank} - Real Step {real_step} - Loss: {loss_batch.item()}, current step sync interval: {local_steps_scheduler[real_step - 1]}"
            )
            update_inner_optimizer(inner_optimizer,scheduler, model, scaler)
            count_inner_optimizer_steps += 1
            # Measure time for parameter drift computation
            current_params = [param.clone().detach() for param in model.parameters()]
            l2_norm = compute_l2_norm(current_params, reference_params, normalize=False)
            normalized_l2_norm = compute_l2_norm(
                current_params, reference_params, normalize=True
            )

            log_inner_stats(
                local_rank,
                real_step,
                loss_batch,
                sync_count,
                l2_norm,
                normalized_l2_norm,
            )

            if count_inner_optimizer_steps % local_steps_scheduler[real_step - 1] == 0:
                count_inner_optimizer_steps = 0
                # Measure time for outer optimizer sync
                logger.info(
                    f"Local rank {local_rank} - Syncing outer optimizer at step {real_step}"
                )
                main_param = [
                    param
                    for group in inner_optimizer.param_groups
                    for param in group["params"]
                ]
                bytes_sent = update_outer_optimizer(
                    params_offloaded,
                    main_param,
                    optim_method,
                    world_size,
                    outer_optimizer,
                    device=device,
                )
                params_offloaded = get_offloaded_param(outer_optimizer, device=device)

                # Update reference parameters after outer optimizer sync
                reference_params = [
                    param.clone().detach() for param in model.parameters()
                ]

                # Update the total bytes sent and received
                total_bytes_sent += bytes_sent
                sync_count += 1

            if real_step % checkpoint_interval == 0:
                # Measure time for checkpointing
                val_stats = evaluate_model(
                    val_dataloader, model, local_rank, global_rank
                )
                dist.barrier()
                log_stats(
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
                )
                save_checkpoint(
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
                )

            loss_batch = 0

        if total_steps != -1 and total_steps <= real_step:
            logger.info(
                    f"Performing final sync of the outer optimizer at step {real_step}"
                )
            main_param = [
                    param
                    for group in inner_optimizer.param_groups
                    for param in group["params"]
                ]
            bytes_sent = update_outer_optimizer(
                params_offloaded,
                main_param,
                optim_method,
                world_size,
                outer_optimizer,
                device=device,
            )
            params_offloaded = get_offloaded_param(outer_optimizer, device=device)

            # Update reference parameters after outer optimizer sync
            reference_params = [
                param.clone().detach() for param in model.parameters()
            ]

            # Update the total bytes sent and received
            total_bytes_sent += bytes_sent
            sync_count += 1
            break


def main(args):
    # Setup distributed training
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    ddp_setup(
        master_addr=master_addr,
        master_port=master_port,
        local_rank=local_rank,
        world_size=world_size,
        device=args.device,
    )

    # Initialize model and optimizers
    model_class: Optional[Callable] = MODEL_REGISTRY.get(args.model)
    assert model_class is not None, f"Model {args.model} not found"

    get_dataset: Optional[Callable] = DATASET_REGISTRY.get(args.dataset)
    assert get_dataset is not None, f"Dataset {args.dataset} not found"

    model_config, model = initialize_model(model_class, args.device, args.optim_method, local_rank)
    inner_optimizer, outer_optimizer = get_optimizers(
        model, lr=args.lr, outer_lr=args.outer_lr, optim_method=args.optim_method
    )
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    # Load checkpoint if it exists
    START_STEP = 0
    if os.path.exists(args.checkpoint_path):
        START_STEP, model, inner_optimizer, outer_optimizer, scheduler = (
            load_checkpoint(
                model,
                inner_optimizer,
                outer_optimizer,
                scheduler,
                args.checkpoint_path,
                local_rank,
                global_rank,
                args.model,
                args.dataset,
                args.optim_method,
            )
        )
        wandb_setup(
            local_rank=local_rank,
            user_key=args.wandb_user_key,
            project_name=args.wandb_project_name,
            run_id=args.wandb_run_id,
            group=args.wandb_group,
        )
        logger.info(f"Resuming training from step {START_STEP}")
    else:
        logger.info("No checkpoint found, starting from scratch.")
        START_STEP = 0
        wandb_setup(
            local_rank=local_rank,
            user_key=args.wandb_user_key,
            project_name=args.wandb_project_name,
            run_id=None,
            group=args.wandb_group,
        )
    logger.info("Model initialized on rank %s", local_rank)

    # Load dataset
    if args.dataset == "test_squence_dataset":
        _, train_dataloader = get_dataset(
            args.batch_size,
            vocab_size=model_config.vocab_size,
            num_samples=-1,
        )
        val_dataloader = train_dataloader
    else:
        train_dataloader, val_dataloader = get_dataset(
            world_size, local_rank, args.per_device_train_batch_size, split="train"
        )

    logger.info("Dataset initialized on rank %s", local_rank)

    # Start training
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        inner_optimizer=inner_optimizer,
        outer_optimizer=outer_optimizer,
        scheduler=scheduler,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        total_steps=args.total_steps,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        optim_method=args.optim_method,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        model_name=args.model,
        dataset_name=args.dataset,
        device=args.device,
        start_step=START_STEP,
        warmup_steps=args.warmup_steps,
    )

    wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    logger.info("Starting training script...")
    parser = argparse.ArgumentParser(description="DiLoCo Training Script")
    parser.add_argument(
        "--model",
        type=model_type,
        required=True,
        help="Model to train. Options: " + ", ".join(MODEL_REGISTRY.keys()) + ".",
    )
    parser.add_argument(
        "--dataset",
        type=dataset_type,
        required=True,
        help="Dataset to use. Options: " + ", ".join(DATASET_REGISTRY.keys()) + ".",
    )
    parser.add_argument(
        "--local_steps",
        type=int,
        default=500,
        help="Steps before syncing outer optimizer.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--outer_lr",
        type=float,
        default=1e-3,
        help="Outer learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=88000,
        help="Total training steps.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Total batch size across all devices.",
    )
    parser.add_argument(
        "--optim_method",
        type=validate_optimizer,
        default="sgd",
        choices=["demo", "sgd", "sgd_quantized"],
        help="Optimizer: demo (DeMo), sgd (standard), or sgd_quantized.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint.pth",
        help="File path for saving/loading checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=512,
        help="Steps between saving checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use: cuda (GPU) or cpu.",
    )
    parser.add_argument(
        "--wandb_user_key",
        type=str,
        default=None,
        help="WandB user key for authentication.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="diloco_training",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="WandB run ID for resuming or tracking experiments.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="WandB group for resuming or tracking experiments.",
    )

    args = parser.parse_args()
    main(args)
