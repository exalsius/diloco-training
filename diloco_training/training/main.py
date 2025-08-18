import argparse
import os
import torch.distributed as dist
from huggingface_hub import login
from diloco_training.utils.diloco_utils import ddp_setup, wandb_setup
from diloco_training.training.distributed_trainer import DistributedTrainer
from diloco_training.utils.metrics_logger import collect_environment_metadata


def main(args):
    # Setup distributed training
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    wandb_user_key = os.environ.get("WANDB_USER_KEY", None)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
    setattr(args, "local_rank", local_rank)
    setattr(args, "global_rank", global_rank)
    setattr(args, "world_size", world_size)
    ddp_setup(
        master_addr=master_addr,
        master_port=master_port,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        device=args.device,
    )

    # Collect environment metadata for logging
    env_metadata = collect_environment_metadata(args)

    # Initialize WandB with enhanced metadata
    wandb_setup(
        local_rank=args.local_rank,
        global_rank=args.global_rank,
        user_key=wandb_user_key,
        project_name=args.wandb_project_name,
        run_id=args.wandb_run_id,
        group=args.wandb_group,
        experiment_description=args.experiment_description,
        metadata=env_metadata,
        args=args,
    )

    # hf login
    login(token=hf_token)

    # Initialize and run trainer
    trainer = DistributedTrainer(args)
    if args.heterogeneous:
        trainer.heterogeneous_profiling()
    trainer.load_checkpoint()
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiLoCo Training Script")
    parser.add_argument("--model", type=str, required=True, help="Model name.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--local_steps", type=int, default=500, help="Local steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--outer_lr", type=float, default=1e-3, help="Outer learning rate."
    )
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps.")
    parser.add_argument("--total_steps", type=int, default=88000, help="Total steps.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per device.",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Total batch size.")
    parser.add_argument(
        "--optim_method",
        type=str,
        default="sgd",
        choices=["demo", "sgd", "ddp"],
        help="Optimizer.",
    )
    parser.add_argument(
        "--quantization", type=bool, default=False, help="Enable quantization."
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="checkpoint.pth", help="Checkpoint path."
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=512, help="Checkpoint interval."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device."
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="diloco_training",
        help="WandB project name.",
    )
    parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID.")
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB group.")
    parser.add_argument(
        "--heterogeneous",
        type=bool,
        default=False,
        help="Enable heterogeneous profiling.",
    )
    parser.add_argument(
        "--compression_decay", type=float, default=0.9, help="Compression decay."
    )
    parser.add_argument(
        "--compression_topk", type=int, default=32, help="Compression top-k."
    )

    # New experiment metadata arguments
    parser.add_argument(
        "--experiment_description",
        type=str,
        default="DiLoCo distributed training experiment",
        help="Description of the experiment run.",
    )
    parser.add_argument(
        "--experiment_tags",
        type=str,
        nargs="*",
        default=[],
        help="Tags for the experiment (e.g., 'baseline', 'optimization', 'scaling').",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    main(args)
