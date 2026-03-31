import torch
import torch.distributed as dist

from diloco_training.utils.lattice_compression import (
    compress_tensor,
    decompress_tensor,
)


def quantize_tensor(tensor):
    """Quantize tensor to int8 using mean and 6-sigma range."""
    # Compute mean and standard deviation
    mean = tensor.mean()
    std = tensor.std()

    # Define quantization range [mean - 6*std, mean + 6*std]
    qmin = mean - 6 * std
    qmax = mean + 6 * std

    # Scale factor for quantization (guard against zero range)
    q_range = qmax - qmin
    if q_range == 0:
        q_range = torch.tensor(1.0, device=tensor.device)
    scale = 255.0 / q_range

    # Quantize to int8
    tensor_q = torch.clamp((tensor - qmin) * scale, 0, 255).round().to(torch.uint8)

    # Return quantized tensor and quantization parameters
    return tensor_q, qmin, qmax


def dequantize_tensor(tensor_q, qmin, qmax):
    """Dequantize int8 tensor back to fp32."""
    q_range = qmax - qmin
    if q_range == 0:
        q_range = torch.tensor(1.0)
    scale = 255.0 / q_range
    return tensor_q.float() / scale + qmin


def distributed_reduce_quantized(tensor, op=dist.ReduceOp.AVG):
    """Distributed reduction with 8-bit quantization for all communication, including broadcast."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Quantize local tensor
    tensor_q, qmin, qmax = quantize_tensor(tensor)
    qmin_tensor = torch.tensor([qmin], device=tensor.device)
    qmax_tensor = torch.tensor([qmax], device=tensor.device)

    # Prepare buffers for gathering
    gathered_tensors_q = [torch.zeros_like(tensor_q) for _ in range(world_size)]
    gathered_qmins = [torch.zeros(1, device=tensor.device) for _ in range(world_size)]
    gathered_qmaxs = [torch.zeros(1, device=tensor.device) for _ in range(world_size)]

    # Gather quantized tensors and params from all ranks
    dist.all_gather(gathered_tensors_q, tensor_q, async_op=True)
    dist.all_gather(gathered_qmins, qmin_tensor, async_op=True)
    dist.all_gather(gathered_qmaxs, qmax_tensor, async_op=True)

    # Only rank 0 performs reduction and broadcasts result in 8-bit
    if rank == 0:
        # Dequantize all tensors
        dequantized_tensors = [
            dequantize_tensor(
                gathered_tensors_q[i],
                gathered_qmins[i].item(),
                gathered_qmaxs[i].item(),
            )
            for i in range(world_size)
        ]
        # Reduce in float32
        if op == dist.ReduceOp.SUM:
            reduced = torch.stack(dequantized_tensors).sum(dim=0)
        elif op == dist.ReduceOp.AVG:
            reduced = torch.stack(dequantized_tensors).mean(dim=0)
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")

        # Quantize reduced tensor for broadcast
        reduced_q, rqmin, rqmax = quantize_tensor(reduced)
        rqmin_tensor = torch.tensor([rqmin], device=tensor.device)
        rqmax_tensor = torch.tensor([rqmax], device=tensor.device)
    else:
        reduced_q = torch.zeros_like(tensor_q)
        rqmin_tensor = torch.zeros(1, device=tensor.device)
        rqmax_tensor = torch.zeros(1, device=tensor.device)

    # Broadcast quantized reduced tensor and params
    dist.broadcast(reduced_q, src=0, async_op=True)
    dist.broadcast(rqmin_tensor, src=0, async_op=True)
    dist.broadcast(rqmax_tensor, src=0, async_op=True)

    # Dequantize locally
    result = dequantize_tensor(reduced_q, rqmin_tensor.item(), rqmax_tensor.item())
    tensor.copy_(result)
    return tensor


def distributed_reduce_lattice(tensor, op=dist.ReduceOp.AVG):
    """Distributed reduction with E8P12 lattice compression (~8x compression).

    Each worker compresses its pseudo-gradient to lattice indices + block norms,
    then AllGathers the compressed representations. Each worker decompresses
    and averages locally in fp32.

    Communication per worker: ~0.5 bytes/param (vs 4 bytes/param uncompressed).

    Args:
        tensor: gradient tensor on the worker's device
        op: reduction operation (AVG or SUM)

    Returns:
        tensor with the reduced result (modified in-place)
    """
    world_size = dist.get_world_size()
    device = tensor.device

    # Compress locally
    compressed = compress_tensor(tensor, device="cpu")

    # AllGather compressed components:
    #   - indices: int32 (n_blocks,) — E8 lattice indices
    #   - block_norms: fp16 (n_blocks,) — normalized per-block magnitudes
    #   - norm_scale: fp32 scalar — per-worker normalization factor
    indices_local = compressed.indices.to(device)
    norms_local = compressed.block_norms.to(device)
    norm_scale_local = compressed.norm_scale.to(device).reshape(1)

    gathered_indices = [torch.zeros_like(indices_local) for _ in range(world_size)]
    gathered_norms = [torch.zeros_like(norms_local) for _ in range(world_size)]
    gathered_norm_scales = [torch.zeros_like(norm_scale_local) for _ in range(world_size)]

    idx_handle = dist.all_gather(gathered_indices, indices_local, async_op=True)
    norm_handle = dist.all_gather(gathered_norms, norms_local, async_op=True)
    scale_handle = dist.all_gather(gathered_norm_scales, norm_scale_local, async_op=True)

    idx_handle.wait()
    norm_handle.wait()
    scale_handle.wait()

    # Decompress each worker's pseudo-gradient and reduce in fp32
    from diloco_training.utils.lattice_compression import LatticeCompressed

    decompressed = []
    for i in range(world_size):
        remote_compressed = LatticeCompressed(
            indices=gathered_indices[i].cpu().int(),
            block_norms=gathered_norms[i].cpu().half(),
            norm_scale=gathered_norm_scales[i].cpu().squeeze(),
            e8_scale=compressed.e8_scale,
            original_shape=compressed.original_shape,
            original_numel=compressed.original_numel,
        )
        decompressed.append(decompress_tensor(remote_compressed, device="cpu"))

    stacked = torch.stack(decompressed)
    if op == dist.ReduceOp.SUM:
        result = stacked.sum(dim=0)
    else:
        result = stacked.mean(dim=0)

    tensor.copy_(result.to(device))
    return tensor
