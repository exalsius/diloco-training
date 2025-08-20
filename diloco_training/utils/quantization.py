import torch
import torch.distributed as dist


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
