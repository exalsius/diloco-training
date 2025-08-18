#!/bin/bash
# Force IPv4 and select the correct network interface
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=eth0     # change to your actual NIC name (ip a)
export NCCL_IB_DISABLE=1           # disable InfiniBand if not used
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Debug logs
export TORCHELASTIC_LOG_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=ERROR

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=10.42.0.52 \
  --master_port=29500 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=10.42.0.52:29500 \
  diloco_training/training/simple.py
