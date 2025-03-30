import argparse
import logging
import os

import torch
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup

from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.utils.diloco_utils import (
    ddp_setup,
    evaluate_model,
    initialize_model,
    log_stats,
    save_checkpoint,
)
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


def prepare_batch(batch, device="cuda"):
    for key in batch.keys():
        batch[key] = batch[key].to(device)
    return batch


def compute_loss(model, batch, gradient_accumulation_steps):
    outputs = model(**batch)
    loss = outputs.loss / gradient_accumulation_steps
    return loss


def calculate_ddp_communication(model):
    total_bytes = 0
    for param in model.parameters():
        if param.requires_grad:
            total_bytes += param.numel() * param.element_size()
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
        loss = compute_loss(model, batch, gradient_accumulation_steps)
        loss_batch += loss.detach()
        loss.backward()
        total_bytes = calculate_ddp_communication(model)
        total_bytes_sent += total_bytes * (world_size - 1)  # Sent to other processes
        total_bytes_received += total_bytes * (
            world_size - 1
        )  # Received from other processes
        if (step + 1) % gradient_accumulation_steps == 0:
            logger.info(
                f"Local rank {local_rank} - Step {(step + 1)* world_size / gradient_accumulation_steps} - Loss: {loss_batch.item()}"
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ((step + 1) // gradient_accumulation_steps) % checkpoint_interval == 0:
                val_stats = evaluate_model(
                    val_dataloader, model, global_rank, local_rank
                )
                # barrier not needed here, since sync happens in backward
                # if we add barrier, we will end up in deadlock
                log_stats(
                    local_rank,
                    (step + 1) // gradient_accumulation_steps,
                    loss_batch,
                    world_size,
                    batch_size,
                    "ddp",
                    None,  # No sync count for standard DDP
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
        if total_steps != -1 and total_steps < (
            (step + 1) // gradient_accumulation_steps
        ):
            break


def main(args):
    # Parse command-line arguments
    model_class = MODEL_REGISTRY.get(args.model)
    if model_class is None:
        raise ValueError(f"Model {args.model} not found in registry.")

    get_dataset = DATASET_REGISTRY.get(args.dataset)
    if get_dataset is None:
        raise ValueError(f"Dataset {args.dataset} not found in registry.")

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
        type=str,
        required=True,
        help="Model to train (e.g., ResNet-50, GPT-Neo, etc.)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to use. Must be one of: " + ", ".join(DATASET_REGISTRY.keys()),
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Global batch size")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=50,
        help="Warmup steps",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=88000,
        help="Total steps",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint.pth",
        help="Path to save/load checkpoints",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=4,
        help="Interval steps to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the training on: cuda (GPU) or cpu",
    )
    args = parser.parse_args()
    main(args)
