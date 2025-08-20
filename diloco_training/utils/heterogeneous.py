import torch
import time
import torch.distributed as dist
from argparse import Namespace
import gc


def profile_gpu(
    trainer_class,
    args,
    max_batch_size=256,
    n_trials=10,
):
    # Convert Namespace to a dictionary and back to Namespace for copying
    copy_args = Namespace(**vars(args))
    # Find max batch size that fits in memory using the real train loop
    device_batch_size = 32
    found = 1
    avg_time = None
    while device_batch_size <= max_batch_size:
        try:
            setattr(copy_args, "local_steps", n_trials // 5)
            setattr(copy_args, "per_device_train_batch_size", device_batch_size)
            setattr(copy_args, "total_steps", n_trials)
            setattr(copy_args, "checkpoint_path", "/tmp/dummy.pth")
            setattr(copy_args, "checkpoint_interval", 1000)
            setattr(copy_args, "heterogeneous", True)

            trainer = trainer_class(copy_args)
            start = time.time()
            trainer.train()
            elapsed = time.time() - start
            avg_time = elapsed / n_trials
            found = device_batch_size
            device_batch_size *= 2
            with torch.no_grad():
                torch.cuda.empty_cache()
            del trainer
            gc.collect()
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print(str(e))
                break
            else:
                print(str(e))
                raise
    return found, avg_time


def synchronize_batch_and_steps(trainer_class, args):
    # Profile each GPU to get its time_per_batch
    batch_size, time_per_batch = profile_gpu(trainer_class, args)
    dist.barrier()  # Ensure all processes are synchronized before gathering
    world_size = dist.get_world_size()
    local_info = {"batch_size": batch_size, "time_per_batch": time_per_batch}
    all_info = [None for _ in range(world_size)]
    dist.all_gather_object(all_info, local_info)

    # Find the fastest GPU's time_per_batch
    min_time_per_batch = min(info["time_per_batch"] for info in all_info)
    target_sync_time = args.local_steps * min_time_per_batch

    # Compute local_steps for each GPU
    local_steps_list = []
    for info in all_info:
        steps = int(round(target_sync_time / info["time_per_batch"]))
        local_steps_list.append(steps)

    # Calculate speed ratios (inverse of time_per_batch, normalized)
    speeds = [min_time_per_batch / info["time_per_batch"] for info in all_info]
    total_speed = sum(speeds)

    # Distribute total work proportionally based on speed
    total_steps_list = []
    for speed in speeds:
        steps = int(round(args.total_steps * speed / total_speed * world_size))
        total_steps_list.append(steps)

    # Compute checkpoint_interval for each GPU (proportional to speed, same as local_steps)
    checkpoint_interval_list = []
    for info in all_info:
        interval = int(
            round(
                args.checkpoint_interval * (min_time_per_batch / info["time_per_batch"])
            )
        )
        checkpoint_interval_list.append(max(1, interval))

    # Group GPUs within 5% of each other's time_per_batch and assign max values in group
    assigned_steps = None
    assigned_total_steps = None
    assigned_checkpoint_interval = None

    for i, info in enumerate(all_info):
        group = [
            j
            for j, other in enumerate(all_info)
            if abs(other["time_per_batch"] - info["time_per_batch"])
            / info["time_per_batch"]
            <= 0.05
        ]
        group_local_steps = max(local_steps_list[j] for j in group)
        group_total_steps = max(total_steps_list[j] for j in group)
        group_checkpoint_interval = max(checkpoint_interval_list[j] for j in group)

        if i == dist.get_rank():
            assigned_steps = group_local_steps
            assigned_total_steps = group_total_steps
            assigned_checkpoint_interval = group_checkpoint_interval

    return (
        batch_size,
        assigned_steps,
        assigned_total_steps,
        assigned_checkpoint_interval,
        all_info,
    )
