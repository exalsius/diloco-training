import gc
import time

import torch
import torch.distributed as dist

from diloco_training.training.training_config import TrainingConfig


def profile_gpu(
    config: TrainingConfig,
    local_rank: int,
    global_rank: int,
    world_size: int,
    n_trials: int = 10,
):
    """Profile GPU performance to find optimal batch size."""
    from diloco_training.training.distributed_trainer import DistributedTrainer

    device_batch_size = config.min_batch_size
    found = 1
    avg_time = None

    while device_batch_size <= config.max_batch_size:
        print(config.max_batch_size)
        try:
            # Create a copy of config with profiling parameters
            profiling_config = config.model_copy(
                update={
                    "local_steps": n_trials // 5,
                    "per_device_train_batch_size": device_batch_size,
                    "total_steps": n_trials,
                    "checkpoint_path": "/tmp/dummy.pth",
                    "checkpoint_interval": 1000,
                    "heterogeneous": True,
                }
            )

            trainer = DistributedTrainer(
                profiling_config, local_rank, global_rank, world_size
            )

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


def synchronize_batch_and_steps(
    config: TrainingConfig,
    local_rank: int,
    global_rank: int,
    world_size: int,
):
    """Synchronize batch sizes and steps across heterogeneous GPUs."""
    # Profile each GPU to get its time_per_batch
    batch_size, time_per_batch = profile_gpu(
        config, local_rank, global_rank, world_size
    )

    dist.barrier()  # Ensure all processes are synchronized before gathering
    local_info = {"batch_size": batch_size, "time_per_batch": time_per_batch}
    all_info = [None for _ in range(world_size)]
    dist.all_gather_object(all_info, local_info)

    # Find the fastest GPU's time_per_batch
    min_time_per_batch = min(info["time_per_batch"] for info in all_info)
    target_sync_time = config.local_steps * min_time_per_batch

    # Compute local_steps for each GPU
    local_steps_list = []
    for info in all_info:
        steps = int(round(target_sync_time / info["time_per_batch"]))
        local_steps_list.append(steps)

    # Calculate speed ratios (inverse of time_per_batch, normalized)
    speeds = [min_time_per_batch / info["time_per_batch"] for info in all_info]
    total_speed = sum(speeds)

    # Distribute total work proportionally based on speed
    # Each worker gets a portion of the total_steps based on its relative speed
    # The sum of all worker steps should equal the original total_steps
    total_steps_list = []
    for speed in speeds:
        steps = int(round(config.total_steps * speed / total_speed))
        total_steps_list.append(steps)

    # Group GPUs within the configured percentage variance of each other's time_per_batch
    groups = []
    for i, info in enumerate(all_info):
        if info is None or "time_per_batch" not in info:
            continue

        group = [
            j
            for j, other in enumerate(all_info)
            if other is not None
            and "time_per_batch" in other
            and abs(other["time_per_batch"] - info["time_per_batch"])
            / info["time_per_batch"]
            <= config.group_perc_variance
        ]
        if group not in groups:
            groups.append(group)

    # Calculate group assignments and ensure sync compatibility
    group_assignments = {}
    sync_multipliers = []

    for group in groups:
        group_local_steps = max(local_steps_list[j] for j in group)
        group_total_steps = max(total_steps_list[j] for j in group)

        # Calculate how many sync rounds this would be
        sync_multiplier = group_total_steps // group_local_steps
        sync_multipliers.append(sync_multiplier)

        for gpu_idx in group:
            group_assignments[gpu_idx] = (
                group_local_steps,
                group_total_steps,
                sync_multiplier,
            )

    # Use the maximum sync multiplier across all groups to ensure consistency
    max_sync_multiplier = max(sync_multipliers)

    # Recalculate total_steps for all groups using the same multiplier
    assigned_steps = None
    assigned_total_steps = None

    for i, info in enumerate(all_info):
        if info is None or "time_per_batch" not in info:
            continue

        group = [
            j
            for j, other in enumerate(all_info)
            if other is not None
            and "time_per_batch" in other
            and abs(other["time_per_batch"] - info["time_per_batch"])
            / info["time_per_batch"]
            <= config.group_perc_variance
        ]
        group_local_steps = max(local_steps_list[j] for j in group)
        # Ensure total_steps is a multiple of local_steps and uses consistent multiplier
        group_total_steps = group_local_steps * max_sync_multiplier

        if i == dist.get_rank():
            assigned_steps = group_local_steps
            assigned_total_steps = group_total_steps

    return batch_size, assigned_steps, assigned_total_steps, all_info
