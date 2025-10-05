#!/bin/bash

# Environment variables
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export PYTHONWARNINGS=default
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_USE_V6=0
export GLOO_SOCKET_IFNAME=eth0 
# soeren: this is my wandb api key
export WANDB_USER_KEY=X
export HUGGINGFACE_TOKEN=X
export TORCH_CPP_LOG_LEVEL=ERROR
export HF_HOME="/workspace/datasets"
export MASTER_ADDR="10.42.0.52"
export MASTER_PORT="29500"
# Torchrun command
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --node_rank=0 \
  diloco_training/training/start_training.py \
  --config config.yaml
