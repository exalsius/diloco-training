import argparse
import logging.config
import os
from typing import Callable, Optional

import torch
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup

from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.utils.args_types import (
    dataset_type,
    model_type,
)
from diloco_training.utils.diloco_utils import (
    ddp_setup,
    evaluate_model,
    forward_and_compute_loss,
    initialize_model,
    load_checkpoint,
    log_inner_stats,
    log_stats,
    prepare_batch,
    save_checkpoint,
    wandb_setup,
)
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


def calculate_ddp_communication(model):
    total_bytes = 0
    for param in model.parameters():
        if param.requires_grad:
            total_bytes += param.nbytes
    return total_bytes


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    local_rank,
    global_rank,
    world_size,
    batch_size,
    per_device_train_batch_size,
    total_steps,
    checkpoint_path,
    checkpoint_interval,
    model_name,
    dataset_name,
    device="cuda",
):
    model.train()
    gradient_accumulation_steps = batch_size // per_device_train_batch_size
    loss_batch = 0
    total_bytes_sent = 0
    total_bytes_received = 0
    total_bytes = 0
    for step, batch in enumerate(train_dataloader):
        batch = prepare_batch(batch, device=device)
        loss = forward_and_compute_loss(model, batch, gradient_accumulation_steps)
        loss_batch += loss.detach()
        loss.backward()
        total_bytes = calculate_ddp_communication(model)
        total_bytes_sent += total_bytes * (world_size - 1)  # Sent to other processes
        total_bytes_received += total_bytes * (
            world_size - 1
        )  # Received from other processes
        real_step = (step + 1) // gradient_accumulation_steps
        step_within_grad_acc = (step + 1) % gradient_accumulation_steps
        if step_within_grad_acc == 0:
            logger.info(
                f"Local rank {local_rank} - Step {real_step} - Loss: {loss_batch.item()}"
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            log_inner_stats(
                local_rank,
                real_step,
                loss_batch,
                None,
                None,
                None,
            )
            if real_step % checkpoint_interval == 0:
                val_stats = evaluate_model(
                    val_dataloader, model, local_rank, global_rank
                )
                log_stats(
                    local_rank,
                    real_step,
                    loss_batch,
                    world_size,
                    batch_size,
                    "ddp",
                    None,
                    total_bytes_sent,
                    total_bytes_received,
                    val_stats,
                    None,
                    per_device_train_batch_size,
                )
                save_checkpoint(
                    model,
                    optimizer,
                    None,  # No outer optimizer
                    scheduler,
                    step,
                    checkpoint_path,
                    local_rank,
                    global_rank,
                    model_name,
                    dataset_name,
                    "ddp",
                )
            loss_batch = 0
        if total_steps != -1 and total_steps < (real_step):
            break


def main(args):
    # Parse command-line arguments
    model_class: Optional[Callable] = MODEL_REGISTRY.get(args.model)
    assert model_class is not None, f"Model {args.model} not found"

    get_dataset: Optional[Callable] = DATASET_REGISTRY.get(args.dataset)
    assert get_dataset is not None, f"Dataset {args.dataset} not found"

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

    logger.info("Model initialized on rank %s", local_rank)

    # Initialize model and optimizer
    model_config, model = initialize_model(
        model_class, args.device, optim_method="ddp", local_rank=local_rank
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    # Load checkpoint if it exists
    START_STEP = 0
    placeholder_for_none_optimizer = None
    if os.path.exists(args.checkpoint_path):
        START_STEP, model, optimizer, placeholder_for_none_optimizer, scheduler = (
            load_checkpoint(
                model,
                optimizer,
                placeholder_for_none_optimizer,  # No outer optimizer
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

    # Load dataset
    train_dataloader, val_dataloader = get_dataset(
        world_size, local_rank, args.per_device_train_batch_size, split="train"
    )

    # Start training
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        model_name=args.model,
        dataset_name=args.dataset,
        device=args.device,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard DDP Training Script")
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
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
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
        type=str,
        default="ddp",
        choices=["ddp"],
        help="Optimizer: ddp",
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
