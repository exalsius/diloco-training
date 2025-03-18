import argparse
import logging.config
import os
import warnings

import torch
import torch.distributed as dist
import wandb
from transformers import get_cosine_schedule_with_warmup

from diloco_training.datasets import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.utils.diloco_utils import (
    ddp_setup,
    get_offloaded_param,
    get_optimizers,
    initialize_model,
)
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger
from diloco_training.utils.quantization import distributed_reduce_quantized

warnings.filterwarnings("ignore")  # Suppresses all warnings

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


def train(
    model,
    train_dataloader,
    inner_optimizer,
    outer_optimizer,
    scheduler,
    local_rank,
    world_size,
    total_steps,
    local_steps,
    batch_size,
    per_device_train_batch_size,
    optim_method="demo",
):
    model.train()
    loss_batch = 0
    params_offloaded = get_offloaded_param(outer_optimizer)
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    # Initialize data transmission tracking
    total_bytes_sent = 0
    total_bytes_received = 0
    bytes_sent = 0
    bytes_received = 0
    sync_count = 0
    effective_total_steps = 0
    for step, batch in enumerate(train_dataloader):
        effective_total_steps += 1
        if total_steps < effective_total_steps * world_size:
            print("Total steps reached")
            break
        for key in batch.keys():
            batch[key] = batch[key].to("cuda")
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss_batch += loss.detach()
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inner_optimizer.step()
            scheduler.step()
            inner_optimizer.zero_grad()
            # logger.info(f"Rank {local_rank} Step local: {step}, Loss: {loss_batch.item()}")
            if (step + 1) // gradient_accumulation_steps % local_steps == 0:
                main_param = [
                    param
                    for group in inner_optimizer.param_groups
                    for param in group["params"]
                ]

                # Track data transmission for this sync
                bytes_sent = 0
                bytes_received = 0
                sync_count += 1

                # Apply parameter updates for all optimization methods
                for param_offloaded, param in zip(params_offloaded, main_param):
                    param_offloaded_on_device = param_offloaded.data.to(param.device)
                    param.grad = param_offloaded_on_device - param.data
                    # Method-specific gradient synchronization
                    if optim_method != "demo":
                        # Calculate data size for tracking
                        is_quantized = optim_method == "sgd_quantized"
                        param_size = param.grad.numel() * (
                            1 if is_quantized else param.grad.element_size()
                        )

                        # Add overhead for quantization parameters if needed
                        if is_quantized:
                            param_size += 8  # 2 float32 values (4 bytes each)
                            param.grad = distributed_reduce_quantized(
                                param.grad, op=dist.ReduceOp.AVG
                            )
                        else:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

                        bytes_sent += param_size
                        bytes_received += param_size * (world_size - 1)
                        param.data = param_offloaded_on_device

                outer_optimizer.step()

                # Get DeMo-specific transmission stats if applicable
                if optim_method == "demo":
                    bytes_sent = outer_optimizer.data_transmit
                    bytes_received = outer_optimizer.data_receive

                # Update total bytes
                total_bytes_sent += bytes_sent
                total_bytes_received += bytes_received

                outer_optimizer.zero_grad()

                params_offloaded = get_offloaded_param(outer_optimizer)

                if local_rank == 0:
                    # Log communication stats
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
                    print("Stats: ", dict_to_log)
                    wandb.log(dict_to_log)
            loss_batch = 0


def main(args):
    # Parse command-line arguments
    # Get model from registry

    model_class = MODEL_REGISTRY.get(args.model)
    if model_class is None:
        raise ValueError(f"Model {args.model} not found in registry.")
    # Get dataset from registry
    get_dataset = DATASET_REGISTRY.get(args.dataset)
    if get_dataset is None:
        raise ValueError(f"Dataset {args.dataset} not found in registry.")

    if args.optim_method == "demo":
        outer_lr = 1e-3
    else:
        outer_lr = 0.7
    # Run the training pipeline
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if local_rank == 0:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="diloco")
    model = initialize_model(model_class, local_rank)
    inner_optimizer, outer_optimizer = get_optimizers(
        model, lr=4e-4, outer_lr=outer_lr, optim_method=args.optim_method
    )
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )
    print("Model initialized")
    train_dataset, train_dataloader = get_dataset(
        world_size, local_rank, args.per_device_train_batch_size, split="train"
    )
    print("Dataset initialized")
    train(
        model,
        train_dataloader,
        inner_optimizer,
        outer_optimizer,
        scheduler,
        local_rank,
        world_size,
        args.total_steps,
        args.local_steps,
        args.batch_size,
        args.per_device_train_batch_size,
        optim_method=args.optim_method,
        outer_lr=outer_lr,
    )
    wandb.finish()


if __name__ == "__main__":
    logger.info("Starting training script...")
    parser = argparse.ArgumentParser(description="DiLoCo Training Script")
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
        help="Dataset to use (e.g., ImageNet, LibriSpeech, etc.)",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Global batch size")
    parser.add_argument(
        "--local_steps",
        type=int,
        default=500,
        help="Local steps for outer-loop optimization",
    )
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
        "--optim_method",
        type=str,
        default="sgd",
        choices=["demo", "sgd", "sgd_quantized"],
        help="Optimization method: demo (DeMo optimizer), sgd (standard SGD), sgd_quantized (SGD with quantization)",
    )
    args = parser.parse_args()
    ddp_setup()
    main(args)
    dist.destroy_process_group()
