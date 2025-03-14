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
