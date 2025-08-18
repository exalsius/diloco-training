import os
import socket
import torch.distributed as dist
from datetime import timedelta


def main():
    print(f"Hostname: {socket.gethostname()}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Local rank: {local_rank}")
    print(f"Global rank: {global_rank}")
    print(f"World size: {world_size}")

    print("About to initialize process group...")

    # Initialize DDP process group
    dist.init_process_group(
        backend="nccl",  # Try gloo first for debugging
        init_method="tcp://10.42.0.52:29500",
        rank=global_rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
    )

    print(f"[Rank {global_rank}/{world_size}] Successfully initialized!")
    print(f"[Rank {global_rank}/{world_size}] Hello from process!")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
