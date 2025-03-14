import argparse
import logging.config
import os
import warnings

import torch
import torch.distributed as dist
import wandb
from torch.distributed import init_process_group
from transformers import get_cosine_schedule_with_warmup

from diloco_training.datasets import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.utils.demo_optimizer import DeMo
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger

warnings.filterwarnings("ignore")  # Suppresses all warnings

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


def ddp_setup():
    logger.info(
        "Local rank: %s, world size: %s",
        os.environ["LOCAL_RANK"],
        os.environ["WORLD_SIZE"],
    )
    init_process_group(
        backend="nccl",
        rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_offloaded_param(outer_optimizer: torch.optim.Optimizer):
    return [
        param.data.detach().clone().to("cpu")
        for group in outer_optimizer.param_groups
        for param in group["params"]
    ]


def initialize_model(model_class, device):
    model = model_class().to(device)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    return model


def get_optimizers(model, lr, outer_lr, optim_method="demo"):
    inner_optimizer = torch.optim.AdamW(
        model.parameters(), weight_decay=0.1, lr=lr, betas=(0.9, 0.95)
    )

    if optim_method == "demo":
        outer_optimizer = DeMo(
            model.parameters(),
            lr=outer_lr,
            compression_decay=0.999,
            compression_topk=32,
            compression_chunk=64,
            weight_decay=0.1,
        )
    elif optim_method in ["sgd", "sgd_quantized"]:
        outer_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=outer_lr,
            weight_decay=0.1,
        )
    else:
        raise ValueError(f"Unknown optimization method: {optim_method}")

    return inner_optimizer, outer_optimizer


def quantize_tensor(tensor):
    """Quantize tensor to int8 using mean and 6-sigma range."""
    # Compute mean and standard deviation
    mean = tensor.mean()
    std = tensor.std()

    # Define quantization range [mean - 6*std, mean + 6*std]
    qmin = mean - 6 * std
    qmax = mean + 6 * std

    # Scale factor for quantization
    scale = 255.0 / (qmax - qmin)

    # Quantize to int8
    tensor_q = torch.clamp((tensor - qmin) * scale, 0, 255).round().to(torch.uint8)

    # Return quantized tensor and quantization parameters
    return tensor_q, qmin, qmax


def dequantize_tensor(tensor_q, qmin, qmax):
    """Dequantize int8 tensor back to fp32."""
    scale = 255.0 / (qmax - qmin)
    return tensor_q.float() / scale + qmin


def distributed_reduce_quantized(tensor, op=dist.ReduceOp.AVG):
    """Perform distributed reduction with int8 quantization for communication efficiency."""
    # Create a copy of the original tensor for the final result
    result_tensor = tensor.clone()
    world_size = dist.get_world_size()

    # For rank 0, we'll gather quantized tensors from all other ranks
    if dist.get_rank() == 0:
        # Quantize our own tensor
        local_tensor_q, qmin, qmax = quantize_tensor(tensor)

        # Create list to store gathered quantized tensors and quantization params
        gathered_tensors_q = [
            torch.zeros_like(local_tensor_q) for _ in range(world_size)
        ]
        gathered_qmins = [
            torch.zeros(1, device=tensor.device) for _ in range(world_size)
        ]
        gathered_qmaxs = [
            torch.zeros(1, device=tensor.device) for _ in range(world_size)
        ]

        # Place our tensor in the list
        gathered_tensors_q[0] = local_tensor_q
        gathered_qmins[0] = torch.tensor([qmin], device=tensor.device)
        gathered_qmaxs[0] = torch.tensor([qmax], device=tensor.device)

        # Gather quantized tensors and params from all other ranks
        for i in range(1, world_size):
            dist.recv(gathered_tensors_q[i], src=i, tag=0)
            dist.recv(gathered_qmins[i], src=i, tag=1)
            dist.recv(gathered_qmaxs[i], src=i, tag=2)

        # Dequantize all tensors back to full precision
        dequantized_tensors = []
        for i in range(world_size):
            dequantized = dequantize_tensor(
                gathered_tensors_q[i],
                gathered_qmins[i].item(),
                gathered_qmaxs[i].item(),
            )
            dequantized_tensors.append(dequantized)

        # Perform reduction in full precision
        if op == dist.ReduceOp.SUM:
            result = torch.stack(dequantized_tensors).sum(dim=0)
        elif op == dist.ReduceOp.AVG:
            result = torch.stack(dequantized_tensors).mean(dim=0)
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")

        # Update the result tensor
        result_tensor.copy_(result)

        # Broadcast the result to all other ranks
        for i in range(1, world_size):
            dist.send(result_tensor, dst=i, tag=3)
    else:
        # Other ranks: quantize and send to rank 0
        local_tensor_q, qmin, qmax = quantize_tensor(tensor)
        qmin_tensor = torch.tensor([qmin], device=tensor.device)
        qmax_tensor = torch.tensor([qmax], device=tensor.device)

        # Send quantized tensor and params to rank 0
        dist.send(local_tensor_q, dst=0, tag=0)
        dist.send(qmin_tensor, dst=0, tag=1)
        dist.send(qmax_tensor, dst=0, tag=2)

        # Receive the reduced result from rank 0
        dist.recv(result_tensor, src=0, tag=3)

    # Copy the result back to the input tensor
    tensor.copy_(result_tensor)

    return tensor


def train(
    model,
    train_dataloader,
    inner_optimizer,
    outer_optimizer,
    scheduler,
    local_rank,
    world_size,
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

    for step, batch in enumerate(train_dataloader):
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
                print("Outer loop optimization started")
                main_param = [
                    param
                    for group in inner_optimizer.param_groups
                    for param in group["params"]
                ]

                # Track data transmission for this sync
                bytes_sent = 0
                bytes_received = 0
                sync_count += 1

                if optim_method == "demo":
                    # DeMo-specific logic
                    for param_offloaded, param in zip(params_offloaded, main_param):
                        param_offloaded_on_device = param_offloaded.data.to(
                            param.device
                        )
                        param.grad = param_offloaded_on_device - param.data
                        param.data = param_offloaded_on_device

                    # After DeMo step, get transmission stats
                    outer_optimizer.step()
                    bytes_sent = outer_optimizer.data_transmit
                    bytes_received = outer_optimizer.data_receive
                else:
                    # For SGD methods, we need to synchronize gradients
                    for param_offloaded, param in zip(params_offloaded, main_param):
                        param_offloaded_on_device = param_offloaded.data.to(
                            param.device
                        )
                        param.grad = param_offloaded_on_device - param.data
                        # Synchronize gradients across processes
                        if optim_method == "sgd_quantized":
                            # Track data size for quantized tensors
                            param_size = (
                                param.grad.numel() * 1
                            )  # uint8 = 1 byte per element
                            # Add small overhead for quantization parameters (qmin, qmax)
                            param_size += 8  # 2 float32 values (4 bytes each)
                            bytes_sent += param_size
                            bytes_received += param_size * (world_size - 1)

                            param.grad = distributed_reduce_quantized(
                                param.grad, op=dist.ReduceOp.AVG
                            )
                            param.data = param_offloaded_on_device
                        else:  # sgd without quantization
                            # Track data size for full precision tensors
                            param_size = param.grad.numel() * param.grad.element_size()
                            bytes_sent += param_size
                            bytes_received += param_size * (world_size - 1)

                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                            param.data = param_offloaded_on_device

                    outer_optimizer.step()

                # Update total bytes
                total_bytes_sent += bytes_sent
                total_bytes_received += bytes_received

                outer_optimizer.zero_grad()

                params_offloaded = get_offloaded_param(outer_optimizer)
                print("Outer loop optimization finished")

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

    # Run the training pipeline
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if local_rank == 0:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="diloco")
    model = initialize_model(model_class, local_rank)
    inner_optimizer, outer_optimizer = get_optimizers(
        model, lr=4e-4, outer_lr=4e-4, optim_method=args.optim_method
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
        args.local_steps,
        args.batch_size,
        args.per_device_train_batch_size,
        optim_method=args.optim_method,
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
        default=10,
        help="Total number of steps",
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
