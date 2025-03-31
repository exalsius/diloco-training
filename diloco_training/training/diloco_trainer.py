import argparse
import logging.config
import os
import torch
import torch.distributed as dist
import wandb

from argparse import ArgumentTypeError
from transformers import get_cosine_schedule_with_warmup

from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.utils.diloco_utils import (
    ddp_setup,
    evaluate_model,
    get_offloaded_param,
    get_optimizers,
    initialize_model,
    load_checkpoint,
    log_stats,
    log_inner_stats,
    save_checkpoint,
    update_outer_optimizer,
    update_inner_optimizer,
    prepare_batch,
    forward_and_compute_loss,
    wandb_setup,
    compute_l2_norm
)
from diloco_training.utils.args_types import (
    __dataset_type,
    __model_type,
    __validate_optimizer
)
from diloco_training.utils.exalsius_logger import (
    LOG_CONFIG,
    get_logger
)

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
):
    model.train()
    loss_batch = 0
    gradient_accumulation_steps = batch_size // per_device_train_batch_size
    total_bytes_sent, total_bytes_received, sync_count = 0, 0, 0
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    params_offloaded = get_offloaded_param(outer_optimizer, device=device)
    reference_params = [param.clone().detach() for param in model.parameters()]  # Initialize reference parameters

    for step, batch in enumerate(train_dataloader):
        if step < start_step:
            continue

        if step == start_step:
            logger.info(f"Starting training from step {start_step}...")
            
        real_step = (step + 1) // gradient_accumulation_steps
        step_within_grad_acc = (step + 1) % gradient_accumulation_steps

        batch = prepare_batch(batch, device=device)
        loss = forward_and_compute_loss(model, batch, gradient_accumulation_steps)
        loss_batch += loss.detach()
        loss.backward()

        if step_within_grad_acc == 0:
            logger.info(
                f"Local rank {local_rank} - Total Step {(step + 1)* world_size / gradient_accumulation_steps} - Loss: {loss_batch.item()}"
            )
            update_inner_optimizer(inner_optimizer, scheduler, model)

            # Compute and log parameter drift
            current_params = [param.clone().detach() for param in model.parameters()]
            l2_norm = compute_l2_norm(current_params, reference_params, normalize=False)
            normalized_l2_norm = compute_l2_norm(current_params, reference_params, normalize=True)
            log_inner_stats(local_rank, real_step, loss_batch, sync_count, l2_norm, normalized_l2_norm)

            if real_step % local_steps == 0:
                logger.info(
                    f"Local rank {local_rank} - Syncing outer optimizer at step {real_step}"
                )
                # Sync outer optimizer parameters
                main_param = [
                    param
                    for group in inner_optimizer.param_groups
                    for param in group["params"]
                ]
                bytes_sent, bytes_received = update_outer_optimizer(
                    params_offloaded,
                    main_param,
                    optim_method,
                    world_size,
                    outer_optimizer,
                    device=device,
                )
                params_offloaded = get_offloaded_param(outer_optimizer, device=device)

                # Update reference parameters after outer optimizer sync
                reference_params = [param.clone().detach() for param in model.parameters()]

                # Update the total bytes sent and received
                total_bytes_sent += bytes_sent
                total_bytes_received += bytes_received
                sync_count += 1

            if real_step % checkpoint_interval == 0:
                val_stats = evaluate_model(val_dataloader, model, local_rank, global_rank)
                dist.barrier()
                log_stats(
                    local_rank,
                    (step + 1) // gradient_accumulation_steps,
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
    
        if total_steps != -1 and total_steps < (real_step * world_size):
            # TODO: final outer optimizer sync needed
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
    model_class = MODEL_REGISTRY.get(args.model)
    get_dataset = DATASET_REGISTRY.get(args.dataset)

    model_config, model = initialize_model(model_class, args.device)
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
        wandb_setup(local_rank=local_rank, 
            user_key=args.wandb_user_key, 
            project_name=args.wandb_project_name, 
            run_id=args.wandb_run_id
        )
        logger.info(f"Resuming training from step {START_STEP}")
    else:
        logger.info("No checkpoint found, starting from scratch.")
        START_STEP = 0
        wandb_setup(local_rank=local_rank, 
            user_key=args.wandb_user_key, 
            project_name=args.wandb_project_name, 
            run_id=None
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
    )

    wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    logger.info("Starting training script...")
    parser = argparse.ArgumentParser(description="DiLoCo Training Script")
    parser.add_argument(
        "--model",
        type=__model_type,
        required=True,
        help="Model to train. Options: " + ", ".join(MODEL_REGISTRY.keys()) + ".",
    )
    parser.add_argument(
        "--dataset",
        type=__dataset_type,
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
        type=__validate_optimizer,
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
        required=True,
        help="WandB user key for authentication.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default='diloco_training',
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="WandB run ID for resuming or tracking experiments.",
    )
    args = parser.parse_args()
    main(args)
