import argparse
import logging.config
import os
from argparse import ArgumentTypeError

import torch
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup

import wandb
from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.utils.diloco_utils import (
    ddp_setup,
    get_offloaded_param,
    get_optimizers,
    initialize_model,
    log_stats,
    save_checkpoint,
)
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger
from diloco_training.utils.quantization import distributed_reduce_quantized

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


def update_inner_optimizer(inner_optimizer, scheduler, model):
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    inner_optimizer.step()
    scheduler.step()
    inner_optimizer.zero_grad()


def sync_outer_optimizer_params(
    params_offloaded,
    main_param,
    optim_method,
    world_size,
    outer_optimizer,
    device="cuda",
):
    bytes_sent = 0
    bytes_received = 0
    for param_offloaded, param in zip(params_offloaded, main_param):
        param_offloaded_on_device = param_offloaded.data.to(param.device)
        param.grad = param_offloaded_on_device - param.data
        # ReduceOp.AVG with Gloo is not supported, so we use SUM instead and manually average later
        op = dist.ReduceOp.AVG if device == "cuda" else dist.ReduceOp.SUM
        if optim_method != "demo":
            is_quantized = optim_method == "sgd_quantized"
            param_size = param.grad.numel() * (
                1 if is_quantized else param.grad.element_size()
            )
            if is_quantized:
                param_size += 8
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
            bytes_sent += param_size
            bytes_received += param_size * (world_size - 1)
            param.data = param_offloaded_on_device
    outer_optimizer.step()
    if optim_method == "demo":
        bytes_sent = outer_optimizer.data_transmit
        bytes_received = outer_optimizer.data_receive
    return bytes_sent, bytes_received


def train(
    model,
    train_dataloader,
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
):
    model.train()
    loss_batch = 0
    params_offloaded = get_offloaded_param(outer_optimizer, device=device)
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    total_bytes_sent = 0
    total_bytes_received = 0
    sync_count = 0

    print(batch_size, per_device_train_batch_size, gradient_accumulation_steps)

    for step, batch in enumerate(train_dataloader):
        batch = prepare_batch(batch, device=device)
        loss = compute_loss(model, batch, gradient_accumulation_steps)
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss_batch += loss.detach()
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            update_inner_optimizer(inner_optimizer, scheduler, model)

            if (step + 1) // gradient_accumulation_steps % local_steps == 0:
                main_param = [
                    param
                    for group in inner_optimizer.param_groups
                    for param in group["params"]
                ]
                bytes_sent, bytes_received = sync_outer_optimizer_params(
                    params_offloaded,
                    main_param,
                    optim_method,
                    world_size,
                    outer_optimizer,
                    device=device,
                )
                total_bytes_sent += bytes_sent
                total_bytes_received += bytes_received
                outer_optimizer.zero_grad()
                params_offloaded = get_offloaded_param(outer_optimizer, device=device)
                log_stats(
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
                )
                sync_count += 1
            loss_batch = 0

        if (step + 1) % checkpoint_interval == 0:
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
            )
        # finish training
        # TODO: final outer optimizer sync needed
        if total_steps != -1 and total_steps == step + 1:
            break


def main(args):
    # Parse command-line arguments
    model_class = MODEL_REGISTRY.get(args.model)
    if model_class is None:
        raise ValueError(f"Model {args.model} not found in registry.")

    get_dataset = DATASET_REGISTRY.get(args.dataset)
    if get_dataset is None:
        raise ValueError(f"Dataset {args.dataset} not found in registry.")

    outer_lr = 1e-3 if args.optim_method == "demo" else 0.7

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
    model_config, model = initialize_model(model_class, args.device)
    inner_optimizer, outer_optimizer = get_optimizers(
        model, lr=4e-4, outer_lr=outer_lr, optim_method=args.optim_method
    )
    scheduler = get_cosine_schedule_with_warmup(
        inner_optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    logger.info("Model initialized on rank %s", local_rank)

    # Load dataset
    if args.dataset == "test_squence_dataset":
        _, train_dataloader = get_dataset(
            args.batch_size,
            vocab_size=model_config.vocab_size,
            num_samples=-1,
        )
    else:
        _, train_dataloader = get_dataset(
            world_size, local_rank, args.per_device_train_batch_size, split="train"
        )

    logger.info("Dataset initialized on rank %s", local_rank)

    # Start training
    train(
        model=model,
        train_dataloader=train_dataloader,
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
    )

    wandb.finish()
    dist.destroy_process_group()


def __dataset_type(value):
    if value not in DATASET_REGISTRY:
        valid_datasets = ", ".join(DATASET_REGISTRY.keys())
        raise ArgumentTypeError(
            f"Invalid dataset: {value}. Must be one of: {valid_datasets}"
        )
    return value


def __model_type(value):
    if value not in MODEL_REGISTRY:
        valid_models = ", ".join(MODEL_REGISTRY.keys())
        raise ArgumentTypeError(
            f"Invalid model: {value}. Must be one of: {valid_models}"
        )
    return value


if __name__ == "__main__":
    logger.info("Starting training script...")
    parser = argparse.ArgumentParser(description="DiLoCo Training Script")
    parser.add_argument(
        "--model",
        type=__model_type,
        required=True,
        help="Model to train (e.g., ResNet-50, GPT-Neo, etc.)",
    )
    parser.add_argument(
        "--dataset",
        type=__dataset_type,
        required=True,
        help="Dataset to use. Must be one of: " + ", ".join(DATASET_REGISTRY.keys()),
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
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint.pth",
        help="Path to save/load checkpoints",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
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
